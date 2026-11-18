"""Training-style frames along a camera path, for the supplementary video (2026-11-18).

The smooth clips from video_paths.py + inference_semantic.py are ONE 81-frame generation; the policy never saw
that. This walks the same path the way the training environment does, with the same machinery as the paper's
offset_sweep.py: the path is cut into policy-sized steps (<= --step_m forward, <= --yaw_step turning), and at every
step LiveDiffusedBackend.render() generates a 5-frame clip from the rolling pose history and keeps the last frame.
The coverage saved per step is the backend's own last_coverage, i.e. the C_w the reward read in training.

Input: path csvs written by video_paths.py (frame, x, y, yaw_deg, ...), nav frame.
Writes to --out_dir/<path name>/:
  rgb.mp4, semantic.mp4, rough_rgb.mp4   one frame per step (generated image, generated labels, raster)
  steps.csv                              step, x, y, yaw_deg, coverage
  rgb/, semantic/, raster/               the same frames as pngs

    python scripts/video_steps.py --scene quad2_00 --clips_dir <clips> --poses_dir <poses> --labels_dir <v14 labels> \
        --live_ckpt <ckpt> --out_dir <out> --paths a.csv b.csv
"""
from __future__ import annotations
import os as _os
_S = _os.environ.get("SCRATCH_ROOT", "/scratch/m000204-pm07b/joana")

import argparse
import csv
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


def policy_steps(P, step_m, yaw_step):
    """Resample a dense path [[x, y, yaw], ...] to steps no longer than step_m and no sharper than yaw_step."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    a = np.r_[0.0, np.cumsum(np.abs(np.diff(P[:, 2])))]
    s = d / step_m + a / yaw_step                       # progress in "step units"; one step covers at most 1 unit,
    if s[-1] < 1e-9:                                    # so it never exceeds step_m forward NOR yaw_step turning
        return P[:1]
    n = int(math.ceil(s[-1]))
    q = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(3)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", required=True); ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default=_S + "/NeoVerse/models")
    ap.add_argument("--reconstructor_path", default=_S + "/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--live_ckpt", required=True)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--paths", nargs="+", required=True, help="path csvs from video_paths.py")
    ap.add_argument("--step_m", type=float, default=0.25, help="training step_size_m")
    ap.add_argument("--yaw_step", type=float, default=0.3, help="training yaw_step_rad")
    ap.add_argument("--render_window", type=int, default=21)
    ap.add_argument("--coverage_window", type=int, default=21)
    ap.add_argument("--sem_palette", type=int, default=6)
    ap.add_argument("--static_movers", default="", help="training (env_config_j503294) used none")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560); ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    import cv2
    from src.env.live_backend import LiveDiffusedBackend
    from diffsynth.utils.class_taxonomy import v14_palette

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
    sc = world._cache[args.scene]
    pal = (v14_palette(args.sem_palette).numpy() * 255).astype(np.uint8)

    for pcsv in args.paths:
        rows = list(csv.DictReader(open(pcsv)))
        P = np.array([[float(r["x"]), float(r["y"]), math.radians(float(r["yaw_deg"]))] for r in rows])
        S = policy_steps(P, args.step_m, args.yaw_step)
        name = Path(pcsv).stem
        od = args.out_dir / name
        for sub in ("rgb", "semantic", "raster"):
            (od / sub).mkdir(parents=True, exist_ok=True)
        world._pose_hist = []                            # episode start: cold history, as at a training reset
        covs = []
        for i, (x, y, yw) in enumerate(S):
            pose = pose_nav_from_xy_yaw(float(x), float(y), float(yw))
            gen_rgb, _, _ = world.render(pose)
            lab = np.asarray(world._last_semantic_raw).astype(np.int64)
            covs.append(float(world.last_coverage))
            pose_recon, t_idx = world._pose_nav_to_recon(pose)
            r_rgb, _, _ = world._rasterize_single(sc, pose_recon, t_idx)
            r8 = np.asarray(r_rgb)[..., :3]
            r8 = (r8 * 255.0).astype(np.uint8) if r8.dtype != np.uint8 and float(np.max(r8)) <= 1.0 else r8.astype(np.uint8)
            g = np.asarray(gen_rgb)[..., :3].astype(np.uint8)
            s = pal[np.clip(lab, 0, len(pal) - 1)]
            cv2.imwrite(str(od / "rgb" / f"{i:03d}.png"), g[:, :, ::-1])
            cv2.imwrite(str(od / "semantic" / f"{i:03d}.png"), s[:, :, ::-1])
            cv2.imwrite(str(od / "raster" / f"{i:03d}.png"), r8[:, :, ::-1])
            print(f"  {name} step {i:3d}/{len(S) - 1}  x {x:7.2f} y {y:7.2f} yaw {math.degrees(yw):7.1f}  cov {covs[-1]:.2f}", flush=True)
        with open(od / "steps.csv", "w") as fh:
            fh.write("step,x,y,yaw_deg,coverage\n")
            for i, ((x, y, yw), c) in enumerate(zip(S, covs)):
                fh.write(f"{i},{x:.3f},{y:.3f},{math.degrees(yw):.1f},{c:.4f}\n")
        for sub, mp4 in (("rgb", "rgb.mp4"), ("semantic", "semantic.mp4"), ("raster", "rough_rgb.mp4")):
            fr = sorted((od / sub).glob("*.png"))
            h, w = cv2.imread(str(fr[0])).shape[:2]
            vw = cv2.VideoWriter(str(od / mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            for f in fr:
                vw.write(cv2.imread(str(f)))
            vw.release()
        c = np.array(covs)
        print(f"[steps] {name}: {len(S)} steps, C_w {c[0]:.2f} -> {c[-1]:.2f} (min {c.min():.2f})", flush=True)


if __name__ == "__main__":
    main()
