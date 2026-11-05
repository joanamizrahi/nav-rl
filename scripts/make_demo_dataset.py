"""Build a behavior-cloning dataset from the REAL trajectories (offline RL data).

The recorded robot drives ARE demonstrations: for every consecutive pose pair we can recover
the action the env would have needed ((v, omega) from the pose delta), and the
env can render the observation the policy would have seen at each pose.

For each scene:
  for t in 0..T-2:
      obs_rgb[t]  = env render at real pose t (rasterized, same as training obs)
      goal_vec[t] = goal in robot frame at pose t (goal = position ~goal_frame)
      action[t]   = ((step distance)/step_size_m, (yaw delta)/yaw_step_rad), clipped to [-1,1]

Saves one compressed npz: obs uint8 [N,H,W,3], goal float32 [N,3], act float32 [N,2].
Runs on Marlowe (GPU; ~30 s/scene).

Usage:
    python scripts/make_demo_dataset.py \
        --scenes rugd_trail_00 rugd_trail_01 ... \
        --clips_dir ... --poses_dir ... --labels_dir ... \
        --out /scratch/.../outputs/demos_v1.npz
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

from src.env.real_calibrated import CalibratedRealWorldBackend, CalibratedBackendConfig

STEP_SIZE_M = 0.25      # MUST match SceneEnvConfig in train_ppo_real.py
                        # (0.15 saturated 43% of demo actions; real steps run up to ~0.25 m)
YAW_STEP_RAD = 0.3


def derive_action(pos_t, yaw_t, pos_t1, yaw_t1):
    """Recover the env action that moves pose t -> t+1 (approximately)."""
    v = float(np.linalg.norm(pos_t1[:2] - pos_t[:2])) / STEP_SIZE_M
    dyaw = float(np.arctan2(np.sin(yaw_t1 - yaw_t), np.cos(yaw_t1 - yaw_t)))
    w = dyaw / YAW_STEP_RAD
    return np.clip([v, w], -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--goal_frame", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--obs_cache", default=None,
                    help="ribbon-cache root: demo observations come from the "
                         "cached diffused views (match cached-obs training)")
    ap.add_argument("--live", action="store_true",
                    help="live per-action diffusion observations (MUST be used "
                         "for BC ahead of --live training: demo obs and training "
                         "obs must come from the same source)")
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm07b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors")
    args = ap.parse_args()

    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in args.scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in args.scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in args.scenes},
        goal_frame=args.goal_frame,
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    if args.live:
        # Rendering the recorded poses IN ORDER fills the live backend's rolling
        # history exactly like training does — demos match the training
        # observation distribution frame for frame.
        from src.env.live_backend import LiveDiffusedBackend
        world = LiveDiffusedBackend(cfg, checkpoint=args.live_ckpt)
    elif args.obs_cache:
        from src.env.cached_backend import CachedDiffusedBackend
        world = CachedDiffusedBackend(cfg, args.obs_cache)
    else:
        world = CalibratedRealWorldBackend(cfg)

    all_obs, all_goal, all_act, all_scene = [], [], [], []
    for scene in args.scenes:
        print(f"=== {scene} ===", flush=True)
        world.load_scene(scene)
        cal = world._calib[scene]
        goal = world.goal_position(scene)
        T = len(cal.positions)
        yaws = np.arctan2(cal.headings[:, 1], cal.headings[:, 0])
        n = 0
        for t in range(T - 1):
            pose = cal.robot_pose_nav(t)
            rgb, K, w2c = world.render(pose)
            # goal vector in robot frame (same math as SceneEnv._goal_in_robot_frame)
            dp = goal - pose[:3, 3]
            dp_r = pose[:3, :3].T @ dp
            gvec = np.array([dp_r[0], dp_r[1],
                             np.arctan2(dp_r[1], dp_r[0])], dtype=np.float32)
            act = derive_action(cal.positions[t], yaws[t],
                                cal.positions[t + 1], yaws[t + 1])
            all_obs.append(rgb.astype(np.uint8))
            all_goal.append(gvec)
            all_act.append(act)
            all_scene.append(scene)
            n += 1
        print(f"  {n} transitions", flush=True)
        getattr(world, "_cache", {}).pop(scene, None)
        import torch; torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        obs=np.stack(all_obs), goal=np.stack(all_goal),
                        act=np.stack(all_act),
                        scene=np.array(all_scene),
                        step_size_m=np.float32(STEP_SIZE_M),
                        yaw_step_rad=np.float32(YAW_STEP_RAD))
    a = np.stack(all_act)
    print(f"\nwrote {args.out}: {len(all_obs)} transitions from {len(args.scenes)} scenes")
    print(f"action stats: v mean {a[:,0].mean():.2f} (should be ~0.6-1.0), "
          f"|w| mean {np.abs(a[:,1]).mean():.2f}")
    print(f"v saturated at 1.0: {(a[:,0] >= 0.999).mean():.0%} "
          f"(if high, STEP_SIZE_M is too small for real step lengths)")


if __name__ == "__main__":
    main()
