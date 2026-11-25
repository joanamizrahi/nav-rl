"""Held-out sweep (2026-11-25, Jing's idea for the threshold question): render AT a recorded camera, but from a
window of OTHER recorded frames, so the real frame is ground truth for what the world model had to invent.

For base frames b (every --base_every-th of the clip) the fusion window is ONE-SIDED and never contains b
(src/env/window.py RANGE), starting next to b and moving away in small steps (Joana: no big jumps):
  behind, gap g: frames b-g-(w-1) .. b-g        ahead, gap g: frames b+g .. b+g+(w-1)
plus the in-sample control (side "in", gap 0: the normal centred window, frame b included).
quad2_01 test (job 606911, centred shifts of 12-36 frames): behind windows lose coverage FASTER than ahead ones
(far frames see b's ground only as sparse dark speckles that still count as coverage and the model paints them in
as dark ground); ahead windows miss only the near ground, and at +28/+36 the model invented a car interior.
The camera is b's own reconstructed camera (pixel-aligned with the real frame and its SAM 3 labels); the clip
history is the 4 recorded frames before b, as a robot walking the path would have.

Writes to --out:
  queries.csv          idx, base_frame, side (in|behind|ahead), gap, win_lo, win_hi, dist_m (b to nearest kept frame), coverage
  semantic_labels.npz  generated labels [N,H,W] int8       gt_labels.npz  SAM 3 labels of real frame b [N,H,W]
  hint_labels.npz      rasterized labels from the window    alpha.npz      window opacity [N,H,W]
  frames/q_%04d.png    generated RGB      gt/q_%04d.png  the real frame     raster/q_%04d.png  rasterized colour
Then grade_heldout.py (CPU).

    python scripts/heldout_sweep.py --scene quad2_01 --clips_dir <clips> --poses_dir <poses> --labels_dir <v14 labels> \
        --live_ckpt <ckpt> --out out/heldout/quad2_01
"""
from __future__ import annotations
import os as _os
_S = _os.environ.get("SCRATCH_ROOT", "/scratch/m000204-pm07b/joana")

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))
from src.env.real_calibrated import CalibratedBackendConfig  # noqa: E402
import src.env.window as window_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", required=True); ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default=_S + "/NeoVerse/models")
    ap.add_argument("--reconstructor_path", default=_S + "/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--live_ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--render_window", type=int, default=21)
    ap.add_argument("--coverage_window", type=int, default=21)
    ap.add_argument("--sem_palette", type=int, default=6)
    ap.add_argument("--static_movers", default="")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560); ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--base_every", type=int, default=4)
    ap.add_argument("--base_lo", type=int, default=6); ap.add_argument("--base_hi", type=int, default=74)
    ap.add_argument("--gaps", default="1,2,3,4,6,8,10,13,16", help="frames between b and the nearest window frame")
    ap.add_argument("--sides", default="behind,ahead")
    ap.add_argument("--min_frames", type=int, default=8, help="skip a query whose window keeps fewer recorded frames")
    args = ap.parse_args()
    import cv2, torch
    from src.env.live_backend import LiveDiffusedBackend
    from diffsynth.utils.auxiliary import load_video

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
    c2w_all = sc["cam2world"].detach().cpu().numpy() if hasattr(sc["cam2world"], "detach") else np.asarray(sc["cam2world"])
    nrec = int(args.num_frames)                      # cameras 0..80 are the clip frames (pano side views follow)

    # ground truth: the real frames, loaded exactly as the reconstruction loaded them, and their SAM 3 labels
    real = load_video(cfg.scene_video_paths[args.scene], nrec, resolution=(args.width, args.height),
                      resize_mode="center_crop", static_scene=False)
    gt_lab = np.load(cfg.scene_labels_paths[args.scene])["labels"]
    assert gt_lab.shape[-2:] == (args.height, args.width) and len(gt_lab) == nrec, gt_lab.shape

    # the render picks its scene time / window centre as the NEAREST camera to the pose; a pano side view at the
    # same position would win the tie and move the window. Say so instead of shifting silently.
    if len(c2w_all) > nrec:
        dup = [b for b in range(nrec) if (np.linalg.norm(c2w_all[nrec:, :3, 3] - c2w_all[b, :3, 3], axis=1) < 1e-6).any()]
        print(f"[heldout] {len(c2w_all) - nrec} pano side cameras; {len(dup)} clip frames share a position with one "
              f"(first {dup[:5]}) -- window centre may follow the side view there", flush=True)
    w = int(args.render_window); half = w // 2
    gaps = [int(g) for g in args.gaps.split(",")]; sides = args.sides.split(",")
    queries = []                                     # (b, side, gap, lo, hi)
    for b in range(int(args.base_lo), min(int(args.base_hi), nrec - 1) + 1, int(args.base_every)):
        queries.append((b, "in", 0, None, None))
        for side in sides:
            for g in gaps:
                lo, hi = (b - g - (w - 1), b - g) if side == "behind" else (b + g, b + g + (w - 1))
                lo, hi = max(lo, 0), min(hi, nrec - 1)
                if hi - lo + 1 >= int(args.min_frames):
                    queries.append((b, side, g, lo, hi))
    print(f"[heldout] {args.scene}: {len(queries)} queries (window {w}, sides {sides}, gaps {gaps})", flush=True)

    for d in ("frames", "gt", "raster"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    preds, hints, alphas, gts, rows = [], [], [], [], []
    for qi, (b, side, gap, lo, hi) in enumerate(queries):
        window_mod.RANGE = None if lo is None else (lo, hi)
        pose_recon = c2w_all[b].astype(np.float32)
        world._pose_hist = []                         # -> cold_history: the 4 recorded frames before b
        world.gen_seed = 0
        gen_rgb, K, w2c = world._rasterize_and_diffuse(sc, pose_recon)
        pred = np.asarray(world._last_semantic_raw).astype(np.int8)
        cov = float(world.last_coverage)
        w2c_r, K1, ts1 = world._single_frame_inputs(sc, pose_recon, b)
        with torch.no_grad():
            r_rgb, _, a = raster.forward(world._render_gaussians(sc, b), render_viewmats=[w2c_r], render_Ks=[K1],
                                         render_timestamps=[ts1], sh_degree=0, width=args.width, height=args.height)
        alpha = a[0, 0].float().cpu().numpy().squeeze(-1) if a.ndim == 5 else a[0, 0].float().cpu().numpy()
        _pend = world._pending_labels; world._pending_labels = None
        hint = np.asarray(world._rasterize_labels(sc, pose_recon, b)).astype(np.int8)
        world._pending_labels = _pend
        r8 = r_rgb[0, 0].float().clamp(0, 1).cpu().numpy()[..., :3]
        cv2.imwrite(str(args.out / "raster" / f"q_{qi:04d}.png"), (r8 * 255).astype(np.uint8)[:, :, ::-1])
        g = np.asarray(gen_rgb)[..., :3].astype(np.uint8)
        cv2.imwrite(str(args.out / "frames" / f"q_{qi:04d}.png"), g[:, :, ::-1])
        im = real[b]; im = np.asarray(im.convert("RGB")) if hasattr(im, "convert") else np.asarray(im)[..., :3]
        cv2.imwrite(str(args.out / "gt" / f"q_{qi:04d}.png"), im[:, :, ::-1])
        near = b if lo is None else (hi if side == "behind" else lo)      # nearest window frame
        dist = float(np.linalg.norm(pos[near] - pos[b]))
        preds.append(pred); hints.append(hint); alphas.append(alpha.astype(np.float16)); gts.append(gt_lab[b].astype(np.int8))
        rows.append((qi, b, side, gap, -1 if lo is None else lo, -1 if hi is None else hi, dist, cov))
        print(f"  q{qi:04d} base {b:2d} {side:6s} gap {gap:2d} (nearest kept frame {dist:4.1f} m)  cov {cov:.2f}", flush=True)
    window_mod.RANGE = None

    np.savez_compressed(args.out / "semantic_labels.npz", labels=np.stack(preds))
    np.savez_compressed(args.out / "hint_labels.npz", labels=np.stack(hints))
    np.savez_compressed(args.out / "gt_labels.npz", labels=np.stack(gts))
    np.savez_compressed(args.out / "alpha.npz", alpha=np.stack(alphas))
    with open(args.out / "queries.csv", "w") as fh:
        fh.write("idx,base_frame,side,gap,win_lo,win_hi,dist_m,coverage\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]:.3f},{r[7]:.4f}\n")
    print(f"[heldout] wrote {args.out} ({len(queries)} queries) -> grade_heldout.py --sweep {args.out}")


if __name__ == "__main__":
    main()
