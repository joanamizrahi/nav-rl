"""Fusion-window sweep: coverage, sharpness and (on-walk) label agreement as a
function of heading / lateral offset from the walk, for several windows.

Two modes. Rasterizer only (default): coverage + raster sharpness + rasterized-label
agreement, minutes per scene. With --diffuse --live_ckpt <semantic ckpt>: ALSO runs the
generator at every pose/window and records what the policy would actually see --
alignment (generated vs raster on covered pixels), generated sharpness, the coverage
the gate reads (windowed when --coverage_window > 0), and on-walk accuracy of the
GENERATED labels vs SAM3 on the real frame. The window decision is made in this mode.
5 frames x 11 yaws x 3 windows = 165 generations, a few minutes on one GPU.
Answers, per scene:
  * how far off the walk (in heading and sideways) coverage stays above the
    gate, for windows of 81 (whole walk), 21, 9 frames -- and for the FULL-scene
    alpha next to the WINDOWED alpha, which is the plot that shows the full
    statistic saturates while the windowed one falls;
  * how much sharpness each window costs (mean gradient magnitude of the
    raster over covered pixels);
  * at the recorded poses, how well the rasterized labels agree with the
    walkthrough's own SAM3 labels (a sanity number per window).

Outputs in --out_dir/<scene>/:
  sweep.csv     scene, window, frame, yaw_deg, lat_m, cov_full, cov_win,
                sharp, label_agree, [gen_sharp, align, gen_label_acc, cov_gate]
  panel_f<frame>.png   raster strips: rows = windows, cols = yaw offsets
  summary.json  per window: yaw reach (deg) and lateral reach (m) at which
                mean windowed coverage crosses --tau, mean sharpness

    python scripts/coverage_sweep.py --scenes campusA_00 campusA_02 \
        --clips_dir ... --poses_dir ... --labels_dir ... --out_dir out/sweep \
        [--windows 81 21 9] [--yaws -90 -60 -45 -30 -15 0 15 30 45 60 90] \
        [--lats 0 0.5 1 2] [--frames 10 25 40 55 70] [--static_scene]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.env.real_calibrated import CalibratedBackendConfig, CalibratedRealWorldBackend  # noqa: E402
from src.env.window import window_gaussians  # noqa: E402


def gradient_energy(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean gradient magnitude of a uint8 image over `mask` (all pixels if None)."""
    g = rgb.astype(np.float32).mean(-1)
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    m = np.sqrt(gx ** 2 + gy ** 2)
    if mask is not None:
        mk = mask[:-1, :-1].astype(bool)
        if mk.sum() < 100:
            return float("nan")
        return float(m[mk].mean())
    return float(m.mean())


def offset_pose(cal, frame: int, yaw_deg: float, lat_m: float) -> np.ndarray:
    """Robot pose at recorded frame `frame`, shifted `lat_m` to the LEFT of the
    walk direction and turned by `yaw_deg` (positive = left)."""
    pose = cal.robot_pose_nav(frame, heading_from_walk=True).astype(np.float64)
    left = pose[:3, 1]
    pose[:3, 3] += lat_m * left
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pose[:3, :3] = Rz @ pose[:3, :3]
    return pose.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--windows", type=int, nargs="+", default=[81, 21, 9])
    ap.add_argument("--yaws", type=float, nargs="+",
                    default=[-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])
    ap.add_argument("--lats", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--frames", type=int, nargs="+", default=[10, 25, 40, 55, 70])
    ap.add_argument("--tau", type=float, default=0.4, help="gate threshold to report reach against")
    ap.add_argument("--static_scene", action="store_true", default=True)
    ap.add_argument("--dynamic_scene", dest="static_scene", action="store_false")
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--diffuse", action="store_true",
                    help="also run the generator at every pose/window (needs --live_ckpt)")
    ap.add_argument("--live_ckpt", default="", help="semantic co-generation checkpoint")
    ap.add_argument("--coverage_window", type=int, default=0,
                    help="window the gate's coverage is computed from in --diffuse mode (0 = full)")
    ap.add_argument("--sem_palette", type=int, default=4)
    args = ap.parse_args()
    if args.diffuse and not args.live_ckpt:
        raise SystemExit("--diffuse needs --live_ckpt")

    import torch
    from diffsynth.utils.auxiliary import homo_matrix_inverse

    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in args.scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in args.scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in args.scenes},
        render_mode="rasterizer_only", model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        static_scene=bool(args.static_scene), H=args.height, W=args.width,
    )
    if args.diffuse:
        from src.env.live_backend import LiveDiffusedBackend
        cfg.coverage_window = int(args.coverage_window)
        cfg.sem_palette_version = int(args.sem_palette)
        world = LiveDiffusedBackend(cfg, checkpoint=args.live_ckpt, live_frames=5, alpha_gate=False)
    else:
        world = CalibratedRealWorldBackend(cfg)
    # the backend loads the reconstructor lazily, on the first load_scene
    # (2026-09-15: reading it before that crashed the first campus sweep)
    for scene in args.scenes:
        print(f"\n=== {scene} ===", flush=True)
        world.load_scene(scene)
        raster = world._reconstructor.gs_renderer.rasterizer
        sc = world._cache[scene]
        cal = world._calib[scene]
        sam3 = np.load(f"{args.labels_dir}/{scene}.npz")["labels"]      # [T,H,W] walkthrough labels
        out = args.out_dir / scene
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        panels = {}
        for frame in args.frames:
            frame = int(np.clip(frame, 0, len(cal.positions) - 1))
            for lat in args.lats:
                for yaw in args.yaws:
                    pose = offset_pose(cal, frame, yaw, lat)
                    pose_recon, t_idx = world._pose_nav_to_recon(pose)
                    w2c, K1, ts1 = world._single_frame_inputs(sc, pose_recon, t_idx)
                    # full-scene alpha once
                    with torch.no_grad():
                        _, _, a_full = raster.forward(sc["gaussians"], render_viewmats=[w2c],
                                                      render_Ks=[K1], render_timestamps=[ts1],
                                                      sh_degree=0, width=args.width, height=args.height)
                    cov_full = float(a_full.float().mean())
                    for w in args.windows:
                        gs = [window_gaussians(sc["gaussians"][0], int(t_idx), w)] if w < 81 else sc["gaussians"]
                        if not gs[0]:
                            rows.append((scene, w, frame, yaw, lat, cov_full, 0.0, float("nan"), float("nan")))
                            continue
                        with torch.no_grad():
                            rgb, _, a_w = raster.forward(gs, render_viewmats=[w2c], render_Ks=[K1],
                                                         render_timestamps=[ts1], sh_degree=0,
                                                         width=args.width, height=args.height)
                            lab_agree = float("nan")
                            if yaw == 0.0 and lat == 0.0:
                                sem, _, _ = raster.forward(gs, render_viewmats=[w2c], render_Ks=[K1],
                                                           render_timestamps=[ts1], sh_degree=0,
                                                           width=args.width, height=args.height,
                                                           feature="labels")
                                pred = sem[0, 0].argmax(-1).cpu().numpy()
                                ref = sam3[min(frame, len(sam3) - 1)]
                                if ref.shape != pred.shape:
                                    import cv2
                                    ref = cv2.resize(ref.astype(np.uint8), (pred.shape[1], pred.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)
                                valid = ref != 0
                                lab_agree = float((pred[valid] == ref[valid]).mean()) if valid.any() else float("nan")
                        rgb_np = (rgb[0, 0].clamp(0, 1).float().cpu().numpy()[..., :3] * 255).astype(np.uint8)
                        a_np = a_w[0, 0].float().cpu().numpy().squeeze(-1) if a_w.ndim == 5 else a_w[0, 0].float().cpu().numpy()
                        cov_w = float(a_np.mean())
                        sharp = gradient_energy(rgb_np, a_np > 0.5)
                        extra = ()
                        if args.diffuse:
                            # what the policy would see: generate at this pose with this window
                            world.cfg.render_window = (w if w < 81 else 0)
                            world._pose_hist = []                     # independent query (cold walk-in)
                            gen_rgb, _, _ = world.render(pose)
                            gen_lab = np.asarray(getattr(world, "_last_semantic_raw", None))
                            cov_gate = float(getattr(world, "last_coverage", float("nan")))
                            covered = a_np > 0.5
                            align = float(np.abs(gen_rgb.astype(np.float32) - rgb_np.astype(np.float32))[covered].mean() / 255.0) if covered.sum() > 100 else float("nan")
                            gen_sharp = gradient_energy(gen_rgb, covered)
                            gen_acc = float("nan")
                            if yaw == 0.0 and lat == 0.0 and gen_lab.ndim == 2:
                                ref = sam3[min(frame, len(sam3) - 1)]
                                if ref.shape != gen_lab.shape:
                                    import cv2
                                    ref = cv2.resize(ref.astype(np.uint8), (gen_lab.shape[1], gen_lab.shape[0]), interpolation=cv2.INTER_NEAREST)
                                valid = ref != 0
                                gen_acc = float((gen_lab[valid] == ref[valid]).mean()) if valid.any() else float("nan")
                            extra = (gen_sharp, align, gen_acc, cov_gate)
                            if lat == 0.0:
                                panels.setdefault(("gen", frame), {}).setdefault(w, {})[yaw] = np.asarray(gen_rgb)
                        rows.append((scene, w, frame, yaw, lat, cov_full, cov_w, sharp, lab_agree) + extra)
                        if lat == 0.0:
                            panels.setdefault(frame, {}).setdefault(w, {})[yaw] = rgb_np
            print(f"  frame {frame}: {len(rows)} rows so far", flush=True)

        with open(out / "sweep.csv", "w") as fh:
            fh.write("scene,window,frame,yaw_deg,lat_m,cov_full,cov_win,sharp,label_agree"
                     + (",gen_sharp,align,gen_label_acc,cov_gate" if args.diffuse else "") + "\n")
            for r in rows:
                fh.write(",".join(str(v) for v in r) + "\n")

        # summary: reach in yaw / lateral at which mean windowed coverage crosses tau
        summary = {}
        R = np.array([(r[1], r[3], r[4], r[5], r[6], r[7], r[8]) for r in rows], dtype=float)
        RG = np.array([r[9:13] if len(r) > 9 else (np.nan,) * 4 for r in rows], dtype=float)
        for w in args.windows:
            m = R[:, 0] == w
            yaw_reach, lat_reach = 0.0, 0.0
            for y in sorted(set(abs(v) for v in args.yaws)):
                sel = m & (np.abs(R[:, 1]) == y) & (R[:, 2] == 0.0)
                if sel.any() and np.nanmean(R[sel, 4]) >= args.tau:
                    yaw_reach = y
            for l in sorted(args.lats):
                sel = m & (R[:, 1] == 0.0) & (R[:, 2] == l)
                if sel.any() and np.nanmean(R[sel, 4]) >= args.tau:
                    lat_reach = l
            on = m & (R[:, 1] == 0.0) & (R[:, 2] == 0.0)
            summary[str(w)] = {
                "yaw_reach_deg": yaw_reach, "lat_reach_m": lat_reach,
                "sharp_mean": float(np.nanmean(R[m, 5])),
                "cov_full_at_90": float(np.nanmean(R[m & (np.abs(R[:, 1]) == 90) & (R[:, 2] == 0), 3])) if (m & (np.abs(R[:, 1]) == 90)).any() else None,
                "cov_win_at_90": float(np.nanmean(R[m & (np.abs(R[:, 1]) == 90) & (R[:, 2] == 0), 4])) if (m & (np.abs(R[:, 1]) == 90)).any() else None,
                "label_agree_on_walk": float(np.nanmean(R[on, 6])) if on.any() else None,
            }
            if args.diffuse:
                summary[str(w)].update({
                    "gen_sharp_mean": float(np.nanmean(RG[m, 0])),
                    "align_mean": float(np.nanmean(RG[m, 1])),
                    "gen_label_acc_on_walk": float(np.nanmean(RG[on, 2])) if on.any() else None,
                    "cov_gate_at_90": float(np.nanmean(RG[m & (np.abs(R[:, 1]) == 90) & (R[:, 2] == 0), 3])) if (m & (np.abs(R[:, 1]) == 90)).any() else None,
                })
        with open(out / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2), flush=True)

        try:
            import cv2
            for key, per_w in panels.items():
                tag, frame = (key if isinstance(key, tuple) else ("raster", key))
                strips = []
                for w in args.windows:
                    row = [per_w.get(w, {}).get(y) for y in args.yaws]
                    row = [r if r is not None else np.zeros((args.height, args.width, 3), np.uint8) for r in row]
                    tiles = []
                    for y, r in zip(args.yaws, row):
                        t_ = r.copy()
                        cv2.putText(t_, f"w={w} yaw={y:+.0f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        tiles.append(t_)
                    strips.append(np.concatenate(tiles, 1))
                panel = np.concatenate(strips, 0)
                cv2.imwrite(str(out / f"panel_{tag}_f{frame:02d}.png"), panel[..., ::-1])
        except Exception as e:
            print(f"[sweep] panels skipped ({e})", flush=True)
        print(f"[sweep] wrote {out}/sweep.csv, summary.json, panels", flush=True)


if __name__ == "__main__":
    main()
