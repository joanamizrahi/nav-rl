"""Probe VIDEOS: moving-camera raster|diffused pairs to judge temporal consistency.

The static pairs show what the diffusion invents at single poses; these videos
show whether its inventions are STABLE as the camera moves — "consistently
accurate images" in the literal sense. The diffusion is a video model, so each
mode is generated in ONE 81-frame diffusion pass (~30 s), giving its native
temporal behavior rather than stitched single frames.

Modes (per scene):
  spin    full 360 rotation at the mid-trail point
  slide   lateral drift 0 -> 2 m off-path, facing along the path
  walk1m  traverse the whole trajectory at a fixed 1 m lateral offset
          (deployment-relevant; compare with the on-path expert replay)

Output per scene+mode: <scene>_<mode>_pair.mp4 (left=raster, right=diffused).
Runs on Marlowe GPU; ~2-3 min per scene+mode after the one-time pipeline load.
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
from probe_world_model import yaw_pose

N_FRAMES = 81


def mode_poses(cal, mode: str):
    """Return a list of N_FRAMES nav-frame robot poses for the given motion."""
    T = len(cal.positions)
    mid = min(40, T - 1)
    if mode == "spin":
        return [yaw_pose(cal, mid, ang) for ang in np.linspace(0, 360, N_FRAMES)]
    if mode == "slide":
        return [yaw_pose(cal, mid, 0, off) for off in np.linspace(0, 2.0, N_FRAMES)]
    if mode == "walk1m":
        frames = np.linspace(0, T - 1, N_FRAMES).astype(int)
        return [yaw_pose(cal, int(f), 0, 1.0) for f in frames]
    raise ValueError(mode)


def render_sequence(world, scene: str, poses):
    """Rasterize the pose sequence AND diffuse it in one video pass.

    Returns (raster_frames, diffused_frames) as lists of HxWx3 uint8."""
    import torch
    from diffsynth.utils.auxiliary import homo_matrix_inverse
    from src.env.real_backend import _move_tree_to

    cal = world._calib[scene]
    device = next(world._reconstructor.parameters()).device
    world._cache[scene] = _move_tree_to(world._cache[scene], device)
    scene_cache = world._cache[scene]

    # nav robot poses -> recon camera poses (same math as the env's render)
    R_cam_local = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    c2ws = []
    t_idxs = []
    for pose in poses:
        c2w_nav = pose.astype(np.float64).copy()
        c2w_nav[:3, 3] += [0, 0, cal.camera_height_m]
        c2w_nav[:3, :3] = c2w_nav[:3, :3] @ R_cam_local
        c2ws.append(cal.nav_cam_to_recon_cam(c2w_nav))
        t_idxs.append(int(np.argmin(np.linalg.norm(
            cal.positions[:, :2] - pose[:2, 3], axis=1))))
    c2w_t = torch.as_tensor(np.stack(c2ws), dtype=scene_cache["cam2world"].dtype,
                            device=device)
    w2c_t = homo_matrix_inverse(c2w_t)
    K_rep = scene_cache["K"][0:1].repeat(len(poses), 1, 1)
    ts = scene_cache["timestamps"][torch.as_tensor(t_idxs, device=device)]

    rgb, depth, alpha = world._reconstructor.gs_renderer.rasterizer.forward(
        scene_cache["gaussians"], render_viewmats=[w2c_t], render_Ks=[K_rep],
        render_timestamps=[ts], sh_degree=0, width=world.W, height=world.H)
    raster_frames = [(rgb[0, i].detach().clamp(0, 1).float().cpu().numpy() * 255
                      ).astype(np.uint8) for i in range(len(poses))]

    # One video-diffusion pass over the whole sequence (mirrors the backend's
    # _rasterize_and_diffuse, but keeps ALL frames instead of frame 0).
    pipe = world._pipe
    target_mask = (alpha > 1.0).float()
    wrapped = {
        "source_views": scene_cache["views"],
        "target_rgb": rgb, "target_depth": depth, "target_mask": target_mask,
        "target_poses": c2w_t.unsqueeze(0), "target_intrs": K_rep.unsqueeze(0),
    }
    with torch.no_grad():
        generated = pipe(
            prompt=world.cfg.prompt, negative_prompt=world.cfg.negative_prompt,
            seed=0, rand_device=device,
            height=world.H, width=world.W, num_frames=len(poses),
            cfg_scale=world.cfg.cfg_scale,
            num_inference_steps=4 if world.cfg.use_lora else 50,
            tiled=False, **wrapped)
    diffused_frames = [np.asarray(im) for im in generated]
    return raster_frames, diffused_frames


def main():
    import imageio.v3 as iio

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["spin", "slide", "walk1m"])
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/scratch/m000204-pm07b/joana/outputs/probe"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in args.scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in args.scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in args.scenes},
        render_mode="rasterizer_plus_diffusion",
        model_path=args.model_path, reconstructor_path=args.reconstructor_path)
    world = CalibratedRealWorldBackend(cfg)
    print("loading diffusion pipeline first (empty GPU)...", flush=True)
    world._ensure_pipe_loaded()

    import torch
    for scene in args.scenes:
        print(f"=== {scene} ===", flush=True)
        world.load_scene(scene)
        cal = world._calib[scene]
        for mode in args.modes:
            poses = mode_poses(cal, mode)
            r_frames, d_frames = render_sequence(world, scene, poses)
            pair = [np.concatenate([r, d], axis=1) for r, d in zip(r_frames, d_frames)]
            out = args.out_dir / f"{scene}_{mode}_pair.mp4"
            iio.imwrite(str(out), np.stack(pair), fps=8, codec="libx264",
                        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
            print(f"  wrote {out} (left=raster, right=diffused)", flush=True)
        world._cache.pop(scene, None)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
