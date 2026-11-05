"""Withheld-frame protocol, step 2 (GPU): render the held-out poses inside the
SUBSET scene and write the files the grader consumes.

For one subset scene (reconstructed from <scene>_wh_<pattern>.mp4):
  1. fit the subset's reconstructed camera path onto the odometry (Umeyama, as in
     pose_vs_odom.py) so odometry poses can be placed in the subset's nav frame;
  2. for each withheld frame, take its odometry pose, map it into the nav frame,
     render there with the semantic world model (generated RGB + labels), the
     rasterized hint labels, and the per-pixel opacity;
  3. deviation of each withheld pose from the kept walk = Mahalanobis distance
     under the fit residual covariance.
Writes to --out:
  semantic_labels.npz (labels [T,H,W]), alpha.npz (alpha [T,H,W]),
  ref_labels.npz (labels [T,H,W]  = SAM3 of the withheld REAL frames),
  hint_labels.npz (labels [T,H,W]), deviation.csv (frame,mahalanobis),
  rgb.mp4 (generated), real.mp4 (the withheld real frames, same order)
Then:  label_acc_vs_coverage.py --pred out/semantic_labels.npz --alpha out/alpha.npz
       --ref out/ref_labels.npz --hint out/hint_labels.npz --deviation out/deviation.csv

    python scripts/withheld_render.py --scene campusA_00_wh_win10 \
        --withheld clips_withheld/campusA_00_wh_win10_withheld.json \
        --clips_dir clips_withheld --poses_dir poses --labels_dir clips_withheld \
        --live_ckpt <semantic ckpt> --out out/withheld/campusA_00_win10 \
        [--render_window 21 --coverage_window 5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from pose_vs_odom import odom_at_frames, umeyama_2d  # noqa: E402
from src.env.real_calibrated import CalibratedBackendConfig  # noqa: E402


def pose_nav_from_xy_yaw(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.eye(4, dtype=np.float32)
    P[:3, 0] = (c, s, 0.0); P[:3, 1] = (-s, c, 0.0); P[:3, 2] = (0.0, 0.0, 1.0)
    P[:3, 3] = (x, y, 0.0)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="subset scene id, e.g. campusA_00_wh_win10")
    ap.add_argument("--withheld", required=True, type=Path, help="*_withheld.json from make_withheld_clips")
    ap.add_argument("--clips_dir", required=True); ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--live_ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--render_window", type=int, default=0)
    ap.add_argument("--coverage_window", type=int, default=0)
    ap.add_argument("--sem_palette", type=int, default=4)
    ap.add_argument("--static_scene", action="store_true", default=True)
    ap.add_argument("--static_movers", default="",
                    help="classes kept per-frame under static fusion, e.g. 12,13 (as in training)")
    ap.add_argument("--width", type=int, default=560); ap.add_argument("--height", type=int, default=336)
    args = ap.parse_args()
    import cv2, torch
    from src.env.live_backend import LiveDiffusedBackend

    meta = json.load(open(args.withheld))
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only", model_path=args.model_path,
        reconstructor_path=args.reconstructor_path, static_scene=bool(args.static_scene),
        static_movers=str(args.static_movers or ""),
        H=args.height, W=args.width, num_frames=int(meta.get("num_frames", 81)),
        render_window=int(args.render_window),
        coverage_window=int(args.coverage_window), sem_palette_version=int(args.sem_palette))
    world = LiveDiffusedBackend(cfg, checkpoint=args.live_ckpt, live_frames=5, alpha_gate=False)
    world.load_scene(args.scene)
    sc = world._cache[args.scene]; cal = world._calib[args.scene]
    raster = world._reconstructor.gs_renderer.rasterizer

    # 1. fit subset nav frame onto odometry (kept frames), residual covariance
    od = np.load(f"{args.clips_dir}/{args.scene}_odom.npz")
    Tk = len(cal.positions)
    _, xy_od, yaw_od = odom_at_frames(od, Tk)
    pos = np.asarray(cal.positions)[:, :2]
    yaw_rec = np.arctan2(cal.headings[:, 1], cal.headings[:, 0])
    s_, R_, t_ = umeyama_2d(pos, xy_od, with_scale=True)      # nav -> odom
    rot = math.atan2(R_[1, 0], R_[0, 0])
    pos_al = (s_ * (R_ @ pos.T)).T + t_
    e = xy_od - pos_al
    fwd = np.stack([np.cos(yaw_od), np.sin(yaw_od)], 1); lat = np.stack([-np.sin(yaw_od), np.cos(yaw_od)], 1)
    E = np.stack([(e * fwd).sum(1), (e * lat).sum(1),
                  (yaw_rec + rot - yaw_od + np.pi) % (2 * np.pi) - np.pi], 1)
    mu, icov = E.mean(0), np.linalg.inv(np.cov(E.T) + 1e-9 * np.eye(3))
    Rinv = R_.T / s_
    print(f"[withheld] fit: scale {s_:.3f} rot {math.degrees(rot):.1f} deg, "
          f"residual std along {E[:,0].std():.2f} m lateral {E[:,1].std():.2f} m yaw {math.degrees(E[:,2].std()):.1f} deg")

    # 2. render each withheld pose
    full_labels = np.load(meta["full_labels"])["labels"]
    cap = cv2.VideoCapture(meta["full_clip"]); real_all = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        real_all.append(f)
    cap.release()
    preds, alphas, refs, hints, devs, gens, reals = [], [], [], [], [], [], []
    for w in meta["withheld"]:
        xy_nav = Rinv @ (np.array([w["x"], w["y"]]) - t_)
        yaw_nav = w["yaw"] - rot
        pose = pose_nav_from_xy_yaw(xy_nav[0], xy_nav[1], yaw_nav)
        # deviation from the kept walk: residual-space distance to the nearest kept pose
        j = int(np.argmin(np.linalg.norm(pos - xy_nav, axis=1)))
        dvec = np.array([((xy_nav - pos[j]) @ np.array([math.cos(yaw_rec[j]), math.sin(yaw_rec[j])])),
                         ((xy_nav - pos[j]) @ np.array([-math.sin(yaw_rec[j]), math.cos(yaw_rec[j])])),
                         (yaw_nav - yaw_rec[j] + np.pi) % (2 * np.pi) - np.pi])
        devs.append(float(np.sqrt((dvec - mu) @ icov @ (dvec - mu))))
        world._pose_hist = []
        gen_rgb, K, w2c = world.render(pose)
        pred = np.asarray(world._last_semantic_raw)
        pose_recon, t_idx = world._pose_nav_to_recon(pose)
        w2c_r, K1, ts1 = world._single_frame_inputs(sc, pose_recon, t_idx)
        with torch.no_grad():
            _, _, a = raster.forward(world._render_gaussians(sc, t_idx), render_viewmats=[w2c_r], render_Ks=[K1],
                                     render_timestamps=[ts1], sh_degree=0, width=args.width, height=args.height)
        alpha = a[0, 0].float().cpu().numpy().squeeze(-1) if a.ndim == 5 else a[0, 0].float().cpu().numpy()
        # 2026-09-18 (Joana caught it): LiveDiffusedBackend._rasterize_labels
        # returns the GENERATED labels right after a render (that is how the
        # reward reads what the policy saw), so calling it here gave a copy of
        # `pred` and the "hint" column was the generated labels. Clear the
        # pending labels first so the parent's raster pass runs.
        _pend = world._pending_labels; world._pending_labels = None
        hint = world._rasterize_labels(sc, pose_recon, t_idx)
        world._pending_labels = _pend
        ref = full_labels[w["frame"]]
        if ref.shape != pred.shape:
            ref = cv2.resize(ref.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        preds.append(pred.astype(np.int8)); alphas.append(alpha.astype(np.float32)); refs.append(ref.astype(np.int8))
        hints.append(np.asarray(hint).astype(np.int8)); gens.append(np.asarray(gen_rgb)[..., :3]); reals.append(real_all[w["frame"]])
        print(f"  frame {w['frame']:3d} dev {devs[-1]:.2f} cov {world.last_coverage:.2f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "semantic_labels.npz", labels=np.stack(preds))
    np.savez_compressed(args.out / "alpha.npz", alpha=np.stack(alphas))
    np.savez_compressed(args.out / "ref_labels.npz", labels=np.stack(refs))
    np.savez_compressed(args.out / "hint_labels.npz", labels=np.stack(hints))
    with open(args.out / "deviation.csv", "w") as fh:
        fh.write("frame,mahalanobis\n")
        for w, d in zip(meta["withheld"], devs):
            fh.write(f"{w['frame']},{d:.3f}\n")
    for name, fr in (("rgb.mp4", gens), ("real.mp4", reals)):
        h, wd = fr[0].shape[:2]
        vw = cv2.VideoWriter(str(args.out / name), cv2.VideoWriter_fourcc(*"mp4v"), 8, (wd, h))
        for f in fr:
            vw.write(f[:, :, ::-1] if name == "rgb.mp4" else f)
        vw.release()

    # COMPARISON PANEL (2026-09-16, Joana): one row per withheld frame --
    #   real photo | generated | SAM3 on the real photo | generated labels | rasterized hint | coverage
    # so "what did the model invent here, and was it right" is answerable by eye,
    # with the coverage and the pose deviation of that frame written on the row.
    try:
        from src.eval.palette import display_palette
        _pal = display_palette(int(args.sem_palette))
        _h, _w = gens[0].shape[:2]
        _rows = []
        for _i, _w_meta in enumerate(meta["withheld"]):
            _real = cv2.resize(reals[_i], (_w, _h))[:, :, ::-1]
            _gen = np.asarray(gens[_i])
            _ref = cv2.resize(_pal[np.clip(np.asarray(refs[_i]).astype(np.int64), 0, 13)], (_w, _h), interpolation=cv2.INTER_NEAREST)
            _prd = cv2.resize(_pal[np.clip(np.asarray(preds[_i]).astype(np.int64), 0, 13)], (_w, _h), interpolation=cv2.INTER_NEAREST)
            _hnt = cv2.resize(_pal[np.clip(np.asarray(hints[_i]).astype(np.int64), 0, 13)], (_w, _h), interpolation=cv2.INTER_NEAREST)
            _a = np.asarray(alphas[_i], dtype=np.float32)
            _cov = cv2.applyColorMap((np.clip(cv2.resize(_a, (_w, _h)), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            _row = np.concatenate([_real, _gen, _ref, _prd, _hnt, _cov], axis=1).astype(np.uint8)
            _row = np.ascontiguousarray(_row)
            for _k, _nm in enumerate(["REAL withheld photo", "GENERATED (no frame here)", "SAM3 on the real photo",
                                      "GENERATED labels", "rasterized hint", "coverage (dark = invented)"]):
                cv2.putText(_row, _nm, (_k * _w + 8, _h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(_row, _nm, (_k * _w + 8, _h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(_row, f"frame {_w_meta['frame']}  coverage {float(_a.mean()):.2f}  deviation {devs[_i]:.2f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            _rows.append(_row)
        _vw = cv2.VideoWriter(str(args.out / "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2,
                              (_rows[0].shape[1], _rows[0].shape[0]))
        for _r in _rows:
            _vw.write(_r[:, :, ::-1])
        _vw.release()
        _sel = _rows[:: max(1, len(_rows) // 6)][:6]
        cv2.imwrite(str(args.out / "compare.png"), np.concatenate(_sel, axis=0)[:, :, ::-1])
        print(f"[withheld] wrote {args.out}/compare.mp4 and compare.png ({len(_rows)} rows)", flush=True)
    except Exception as _e:
        print(f"[withheld] comparison panel skipped: {type(_e).__name__}: {_e}", flush=True)
    print(f"[withheld] wrote {args.out} ({len(preds)} withheld frames) -> run label_acc_vs_coverage.py")


if __name__ == "__main__":
    main()
