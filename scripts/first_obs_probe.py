"""What did the policy ACTUALLY see at episode start? (Joana, 2026-08-29)

Reproduces the training-time first observation at several spawn poses,
rendered two ways side by side:
  OLD: static history (5 identical poses) — what every episode began with
       until the cold-start fix
  NEW: synthesized walk-in history (the fix)

Output: FIRST_OBS_OLD_VS_NEW.png — one row per spawn, [old rgb | old sem |
new rgb | new sem]. The direct evidence for/against the hallucination story.

Usage (GPU node):
    python scripts/first_obs_probe.py --clips_dir ... --poses_dir ... \
        --labels_dir ... --height 224 --width 336 --out ...
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

from src.env.real_calibrated import CalibratedBackendConfig
from src.env.vec_live_env import BatchedLiveDiffusedBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm07b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors")
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=336)
    ap.add_argument("--num_steps", type=int, default=4)
    ap.add_argument("--spawn_frames", default="5,15,25,35,45,60")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    from diffsynth.utils.class_taxonomy import v14_palette

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.height,
        W=args.width,
    )
    world = BatchedLiveDiffusedBackend(cfg, checkpoint=args.live_ckpt)
    world.num_inference_steps = args.num_steps
    world.load_scene(args.scene)
    cal = world._calib[args.scene]
    pal = (v14_palette().numpy() * 255).astype(np.uint8)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in [int(x) for x in args.spawn_frames.split(",")]:
        pose = cal.robot_pose_nav(frame)
        cells = []
        for mode in ("OLD static", "NEW walk-in"):
            if mode.startswith("OLD"):
                pose_recon, _ = world._pose_nav_to_recon(pose)
                world._hists[0] = [pose_recon.astype(np.float32)] * world.live_frames
            else:
                world._hists.pop(0, None)
            rgb, K, w2c, lab = world.render_batch([(0, pose)])[0]
            sem = pal[np.clip(lab, 0, 13).astype(int)][:, :, ::-1]
            for img, txt in ((rgb[:, :, ::-1], f"{mode} rgb f{frame}"),
                             (sem, f"{mode} sem")):
                img = np.ascontiguousarray(img)
                cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cells.append(img)
        rows.append(np.hstack(cells))
        print(f"spawn frame {frame} done", flush=True)
    cv2.imwrite(str(out_dir / "FIRST_OBS_OLD_VS_NEW.png"), np.vstack(rows))
    print(f"==> {out_dir / 'FIRST_OBS_OLD_VS_NEW.png'}", flush=True)


if __name__ == "__main__":
    main()
