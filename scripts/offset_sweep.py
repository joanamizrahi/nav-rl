"""Offset sweep (2026-09-18, Joana): what do the co-generated semantics do where
the world model has little coverage, and do they agree with SAM3 run on the
generated image?

No frames are withheld. From every --base_every-th recorded pose of the FULL
scene, the camera is moved sideways (0 .. lat_max in lat_step, heading kept)
and, separately, turned in place (0 .. yaw_max in yaw_step, position kept).
Each query is rendered cold with the semantic world model: generated RGB,
generated labels, the rasterized hint labels, and the per-pixel opacity.
The sideways sign alternates per base pose so both sides of the walk are
visited; the turn sign likewise.

Writes to --out:
  queries.csv          idx, base_frame, kind (lat|yaw), offset, sign, coverage
  semantic_labels.npz  labels [N,H,W] int8  (generated labels)
  hint_labels.npz      labels [N,H,W] int8  (rasterized hint)
  alpha.npz            alpha  [N,H,W] float32
  frames/q_%04d.png    generated RGB, one per query (lossless, SAM3 reads this
                       directory with --static_scene = every file, in order)
  raster/q_%04d.png    rasterized colour at the same pose (alignment input)
  rgb.mp4              the same frames as a video, for the eye
Then:  SAM3 on frames/ -> grade_offsets.py

    python scripts/offset_sweep.py --scene quad2_01 --clips_dir <clips> \
        --poses_dir <poses> --labels_dir <sam3 v14 labels> --live_ckpt <ckpt> \
        --sem_palette 6 --render_window 21 --coverage_window 21 --out out/offsets/quad2_01
"""
from __future__ import annotations
import os as _os
_S = _os.environ.get("SCRATCH_ROOT", "/scratch/m000204-pm07b/joana")  # 2026-11-05: Cycle 7 scratch; override with SCRATCH_ROOT

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))
from src.env.real_calibrated import CalibratedBackendConfig  # noqa: E402
from withheld_render import pose_nav_from_xy_yaw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", required=True); ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default=_S + "/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default=_S + "/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--live_ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--render_window", type=int, default=21)
    ap.add_argument("--coverage_window", type=int, default=21)
    ap.add_argument("--sem_palette", type=int, default=6)
    ap.add_argument("--static_movers", default="", help="e.g. 12,13 -- per-frame movers, as in training")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560); ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--base_every", type=int, default=8, help="use every k-th recorded frame as a base pose")
    ap.add_argument("--base_lo", type=int, default=10); ap.add_argument("--base_hi", type=int, default=70)
    ap.add_argument("--lat_max", type=float, default=2.0); ap.add_argument("--lat_step", type=float, default=0.2)
    ap.add_argument("--yaw_max", type=float, default=90.0); ap.add_argument("--yaw_step", type=float, default=10.0)
    ap.add_argument("--history", choices=["gradual", "cold"], default="gradual",
                    help="clip history for each query: 4 poses stepping from the base pose to the query (as a robot would), or the cold recorded-frame history")
    args = ap.parse_args()
    import cv2, torch
    from src.env.live_backend import LiveDiffusedBackend

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only", model_path=args.model_path,
        reconstructor_path=args.reconstructor_path, static_scene=True,
        static_movers=str(args.static_movers or ""),
        H=args.height, W=args.width, num_frames=int(args.num_frames),
        render_window=int(args.render_window),
        coverage_window=int(args.coverage_window), sem_palette_version=int(args.sem_palette))
    world = LiveDiffusedBackend(cfg, checkpoint=args.live_ckpt, live_frames=5, alpha_gate=False)
    world.load_scene(args.scene)
    sc = world._cache[args.scene]; cal = world._calib[args.scene]
    raster = world._reconstructor.gs_renderer.rasterizer
    pos = np.asarray(cal.positions)[:, :2]
    yaw_rec = np.arctan2(cal.headings[:, 1], cal.headings[:, 0])

    # the query list: (base_frame, kind, offset, sign)
    lats = np.round(np.arange(0.0, args.lat_max + 1e-6, args.lat_step), 3)
    yaws = np.round(np.arange(0.0, args.yaw_max + 1e-6, args.yaw_step), 3)
    bases = list(range(int(args.base_lo), min(int(args.base_hi), len(pos) - 1) + 1, int(args.base_every)))
    queries = []
    for bi, b in enumerate(bases):
        sgn = 1.0 if bi % 2 == 0 else -1.0
        for o in lats:
            queries.append((b, "lat", float(o), sgn))
        for o in yaws[1:]:          # yaw 0 is the lat 0 query already
            queries.append((b, "yaw", float(o), sgn))
    print(f"[offsets] {args.scene}: {len(bases)} base poses x ({len(lats)} lateral + {len(yaws) - 1} yaw) "
          f"= {len(queries)} renders", flush=True)

    args.out.mkdir(parents=True, exist_ok=True); (args.out / "frames").mkdir(exist_ok=True); (args.out / "raster").mkdir(exist_ok=True); (args.out / "raster_depth").mkdir(exist_ok=True)
    preds, alphas, hints, gens, covs = [], [], [], [], []
    for qi, (b, kind, o, sgn) in enumerate(queries):
        yaw0 = float(yaw_rec[b]); x0, y0 = float(pos[b, 0]), float(pos[b, 1])
        if kind == "lat":
            x = x0 + sgn * o * (-math.sin(yaw0)); y = y0 + sgn * o * math.cos(yaw0); yaw = yaw0
        else:
            x, y = x0, y0; yaw = yaw0 + sgn * math.radians(o)
        pose = pose_nav_from_xy_yaw(x, y, yaw)
        if args.history == "gradual":
            # 2026-09-19 (Joana: "the yaw is not really working"): with a COLD
            # history (recorded frames behind the spawn, all facing the walk)
            # the 5-frame clip is four forward frames plus one turned 90 deg;
            # the video model keeps the clip continuous and the last frame
            # comes out un-turned (alignment error 0.06 -> 0.39 on quad2_01).
            # A robot never jumps like that: it turns 17 deg per step. Give the
            # query the history it would really have -- four poses stepping
            # from the base pose to the query.
            world._pose_hist = []
            for i in range(1, 5):
                f = i / 5.0
                ph = pose_nav_from_xy_yaw(x0 + f * (x - x0), y0 + f * (y - y0), yaw0 + f * (yaw - yaw0))
                pr, _ = world._pose_nav_to_recon(ph)
                world._pose_hist.append(pr.astype(np.float32))
        else:
            world._pose_hist = []
        gen_rgb, K, w2c = world.render(pose)
        pred = np.asarray(world._last_semantic_raw)
        pose_recon, t_idx = world._pose_nav_to_recon(pose)
        w2c_r, K1, ts1 = world._single_frame_inputs(sc, pose_recon, t_idx)
        with torch.no_grad():
            _, d, a = raster.forward(world._render_gaussians(sc, t_idx), render_viewmats=[w2c_r], render_Ks=[K1],
                                     render_timestamps=[ts1], sh_degree=0, width=args.width, height=args.height)
        alpha = a[0, 0].float().cpu().numpy().squeeze(-1) if a.ndim == 5 else a[0, 0].float().cpu().numpy()
        # 2026-11-05 (Fig. 3): the rasterized DEPTH at the same pose, the NeoVerse
        # condition the figure shows next to colour and opacity. Saved as an 8-bit
        # png normalised over the observed pixels (alpha > 0.5), holes black.
        dep = d[0, 0].float().cpu().numpy().squeeze(-1) if d.ndim == 5 else d[0, 0].float().cpu().numpy()
        _obs = alpha > 0.5
        if _obs.any():
            lo, hi = np.percentile(dep[_obs], 2), np.percentile(dep[_obs], 98)
            d8 = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1); d8 = (255 * (1.0 - d8)).astype(np.uint8); d8[~_obs] = 0
        else:
            d8 = np.zeros(dep.shape, np.uint8)
        cv2.imwrite(str(args.out / "raster_depth" / f"q_{qi:04d}.png"), d8)
        # 2026-09-18 (Joana caught it): LiveDiffusedBackend._rasterize_labels
        # returns the GENERATED labels right after a render (that is how the
        # reward reads what the policy saw), so calling it here gave a copy of
        # `pred` and the "hint" column was the generated labels. Clear the
        # pending labels first so the parent's raster pass runs.
        _pend = world._pending_labels; world._pending_labels = None
        hint = world._rasterize_labels(sc, pose_recon, t_idx)
        world._pending_labels = _pend
        g = np.asarray(gen_rgb)[..., :3].astype(np.uint8)
        cv2.imwrite(str(args.out / "frames" / f"q_{qi:04d}.png"), g[:, :, ::-1])
        # ALIGNMENT input (2026-09-18): the rasterized colour at the same pose,
        # so the grader can measure |generated - rasterized| on observed pixels.
        r_rgb, _, _ = world._rasterize_single(sc, pose_recon, t_idx)
        r8 = np.asarray(r_rgb)[..., :3]
        r8 = (r8 * 255.0).astype(np.uint8) if r8.dtype != np.uint8 and float(np.max(r8)) <= 1.0 else r8.astype(np.uint8)
        cv2.imwrite(str(args.out / "raster" / f"q_{qi:04d}.png"), r8[:, :, ::-1])
        preds.append(pred.astype(np.int8)); alphas.append(alpha.astype(np.float32))
        hints.append(np.asarray(hint).astype(np.int8)); gens.append(g); covs.append(float(world.last_coverage))
        print(f"  q{qi:04d} base {b:2d} {kind} {sgn * o:+6.2f}  cov {covs[-1]:.2f}", flush=True)

    np.savez_compressed(args.out / "semantic_labels.npz", labels=np.stack(preds))
    np.savez_compressed(args.out / "hint_labels.npz", labels=np.stack(hints))
    np.savez_compressed(args.out / "alpha.npz", alpha=np.stack(alphas))
    with open(args.out / "queries.csv", "w") as fh:
        fh.write("idx,base_frame,kind,offset,sign,coverage\n")
        for qi, ((b, kind, o, sgn), c) in enumerate(zip(queries, covs)):
            fh.write(f"{qi},{b},{kind},{o},{int(sgn)},{c:.4f}\n")
    h, wd = gens[0].shape[:2]
    vw = cv2.VideoWriter(str(args.out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (wd, h))
    for f in gens:
        vw.write(f[:, :, ::-1])
    vw.release()
    print(f"[offsets] wrote {args.out} ({len(queries)} queries) -> SAM3 on {args.out}/frames, then grade_offsets.py")


if __name__ == "__main__":
    main()
