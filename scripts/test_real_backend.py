"""Integration test for RealWorldBackend on Marlowe.

Loads a real RUGD clip through NeoVerse's pipeline via RealWorldBackend and
renders at 4 poses to verify:
  1. Identity pose -> reproduces something close to source frame 0
  2. Small forward step (+x=0.05 recon units) -> slightly different view
  3. Bigger forward step (+x=0.2)               -> significantly novel view
  4. Yaw right by 30 degrees                    -> rotated view

Saves 4 PNGs. If the images look reasonable AND consistent with the coord
convention (forward step should show the scene "coming toward" the camera,
yaw right should show the scene moving left in the frame), our RealWorldBackend
is wired correctly.

Usage (on Marlowe):
    python scripts/test_real_backend.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
sys.path.insert(0, str(REPO_ROOT))
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", default="/scratch/m000204-pm07b/joana/data/rugd_clips/rugd_park-1_00.mp4")
    ap.add_argument("--output_dir", default="outputs/backend_test")
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--render_mode", default="rasterizer_plus_diffusion",
                    choices=["rasterizer_plus_diffusion", "rasterizer_only"])
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--use_lora", type=int, default=0,
                    help="1 = merge lightx2v 4-step distill LoRA (fast, +VRAM spike). "
                         "0 = 50-step schedule (slower, no merge). Default 0 to avoid "
                         "OOM during LoRA merge on 80GB GPUs.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scene_id = "test_scene"

    from src.env.real_backend import RealWorldBackend, RealWorldBackendConfig

    cfg = RealWorldBackendConfig(
        scene_video_paths={scene_id: args.input_path},
        scene_goals={scene_id: np.array([1.0, 0.0, 0.0], dtype=np.float32)},
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        render_mode=args.render_mode,
        num_frames=args.num_frames,
        use_lora=bool(args.use_lora),
    )

    world = RealWorldBackend(cfg=cfg)
    print(f"[test] loading scene from {args.input_path}", flush=True)
    world.load_scene(scene_id)

    # ---- 4 test poses in SceneEnv frame (x=fwd, y=right, z=up) ----
    test_poses = {}

    # 1. Identity — expect ~source frame 0
    test_poses["01_identity"] = np.eye(4, dtype=np.float32)

    # 2. Small forward step (0.05 recon units along +x=fwd in scene frame)
    p2 = np.eye(4, dtype=np.float32)
    p2[0, 3] = 0.05
    test_poses["02_fwd_0.05"] = p2

    # 3. Bigger forward step (0.2)
    p3 = np.eye(4, dtype=np.float32)
    p3[0, 3] = 0.2
    test_poses["03_fwd_0.2"] = p3

    # 4. Yaw right by 30 degrees.
    # In SceneEnv frame (x=fwd, y=right, z=up), a right turn rotates the robot's
    # +x (fwd) TOWARD +y (right). R_z(theta) applied to (1,0,0) gives
    # (cos theta, sin theta, 0); to get positive y (=right), we need theta > 0.
    # Verified empirically: prior sign choice made the world pan RIGHT in frame
    # (i.e., the camera turned LEFT). Positive angle here is a right turn.
    angle = +np.pi / 6
    c, s = np.cos(angle), np.sin(angle)
    p4 = np.eye(4, dtype=np.float32)
    p4[:3, :3] = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float32)
    test_poses["04_yaw_right_30deg"] = p4

    for name, pose_scene in test_poses.items():
        print(f"[test] rendering {name} ...", flush=True)
        rgb, K, w2c = world.render(pose_scene)
        out_path = os.path.join(args.output_dir, f"{name}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[test] wrote {out_path}  (rgb shape {rgb.shape})", flush=True)

    print(f"\n[test] done. Outputs in {args.output_dir}/")
    print("\nExpected results:")
    print("  01_identity        — should look like source video's frame 0")
    print("  02_fwd_0.05        — scene slightly 'closer' (same content, small forward step)")
    print("  03_fwd_0.2         — scene noticeably 'closer' (likely some holes at edges)")
    print("  04_yaw_right_30deg — scene appears rotated LEFT in the frame (right turn = world moves left)")


if __name__ == "__main__":
    main()
