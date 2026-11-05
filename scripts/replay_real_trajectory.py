"""Replay the REAL recorded trajectory through the RL environment and record it.

Shows what a GOOD agent looks like in our env: the robot follows the path the
real robot drove, the env renders each view, and the reward is computed exactly
as in RL training (Gaussian-label footprint). Output video has the same HUD +
top-down map as the PPO rollout, so the two are directly comparable:
"expert replay" (this) vs "current policy" (rollout.mp4).

Proves env/reward correctness independent of policy quality — the reward along
the real path should be consistently favorable.

Usage (Marlowe, GPU):
    python scripts/replay_real_trajectory.py \
        --scene rugd_trail_00 \
        --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
        --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
        --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
        --out /scratch/m000204-pm07b/joana/outputs/replay_trail00.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.env.real_calibrated import (
    CalibratedRealWorldBackend, CalibratedBackendConfig, GaussianLabelBackend,
)
from src.eval.reward_2d import compute_reward, RewardWeights, GO2_BODY_LENGTH, GO2_BODY_WIDTH
from src.eval.traversability import load_traversability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--look_ahead_dist", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=1, help="replay every Nth recorded pose")
    args = ap.parse_args()

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    world.load_scene(args.scene)
    cal = world._calib[args.scene]
    trav = load_traversability()
    non_trav = trav <= 0.1
    weights = RewardWeights()
    goal = cal.positions[-1].copy(); goal[2] = 0.0

    from PIL import Image, ImageDraw
    import imageio.v3 as iio

    T = len(cal.positions)
    frames, path_xy, prev_pos = [], [], None
    for t in range(0, T, args.stride):
        pose = cal.robot_pose_nav(t)
        rgb, K, w2c = world.render(pose)
        labels = sem.segment(rgb)
        position = pose[:3, 3].copy()
        heading = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
        bd = compute_reward(
            semantic_image=labels, K=K, w2c=w2c,
            robot_position=position, robot_heading=heading, goal=goal,
            traversability_scores=trav, non_traversable_mask=non_trav,
            previous_position=prev_pos, look_ahead_dist=args.look_ahead_dist,
            body_length=GO2_BODY_LENGTH, body_width=GO2_BODY_WIDTH, weights=weights,
        )
        prev_pos = position
        path_xy.append(position[:2])

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img, "RGBA")
        H, W = img.height, img.width
        m = 110
        all_pts = np.vstack([cal.positions[:, :2], [goal[:2]]])
        lo, hi = all_pts.min(0) - 1.0, all_pts.max(0) + 1.0
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-3))
        def to_px(p):
            return (W - m - 6 + (p[0] - lo[0]) / span * m,
                    H - 6 - (p[1] - lo[1]) / span * m)
        draw.rectangle([W - m - 10, H - m - 10, W - 2, H - 2], fill=(0, 0, 0, 170))
        draw.line([to_px(p) for p in cal.positions[::4, :2]], fill=(160, 160, 160, 255), width=1)
        if len(path_xy) > 1:
            draw.line([to_px(p) for p in path_xy], fill=(255, 255, 255, 255), width=2)
        gx, gy = to_px(goal[:2]); draw.ellipse([gx-3, gy-3, gx+3, gy+3], fill=(0, 255, 0, 255))
        ax, ay = to_px(path_xy[-1]); draw.ellipse([ax-2, ay-2, ax+2, ay+2], fill=(255, 80, 80, 255))
        draw.rectangle([0, 0, W, 14], fill=(0, 0, 0, 180))
        dist = float(np.linalg.norm(position - goal))
        draw.text((4, 2), f"REAL TRAJECTORY REPLAY t={t:3d} r={bd.total:+.2f} "
                          f"sem={bd.semantic:+.2f} coll={bd.collision:+.2f} dist={dist:.1f}m",
                  fill=(255, 255, 255, 255))
        frames.append(np.array(img.convert("RGB")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(args.out), np.stack(frames), fps=8,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    totals = "replayed"
    print(f"[replay] wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
