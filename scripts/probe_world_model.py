"""World-model boundary probe: 360-degree spins + lateral offsets, measured by void%.

The question ("how far from the source views can we trust the world model, and
how much would the diffusion have to hallucinate?") operationalized:

At probe points along the real trajectory:
  SPIN:    render every 15 degrees of a full rotation at the point
  OFFSET:  step 0 / 0.5 / 1 / 2 m sideways off the path, facing along it
For every rendered view record COVERAGE = fraction of pixels with rasterizer
alpha > 0.5 (1 - void%). Low coverage = the diffusion would be inventing most
of that view = outside the trust boundary.

Outputs (per scene):
  outputs/probe/<scene>_probe.npz          coverage arrays + angles/offsets
  outputs/probe/<scene>_spin_strip.png     raster views every 45 deg (probe midpoint)
  outputs/probe/<scene>_offset_strip.png   raster views at each lateral offset
  outputs/probe/probe_curves.png           coverage vs angle + vs offset, all scenes

Runs on Marlowe GPU (~1 min/scene, rasterizer only).
Usage: python scripts/probe_world_model.py --scenes s1 s2 ... --clips_dir ... --poses_dir ... --labels_dir ...
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

SPIN_STEP_DEG = 15
OFFSETS_M = [0.0, 0.5, 1.0, 2.0]
PROBE_FRAMES = [20, 40, 60]


def render_with_alpha(world, scene_cache, cal, pose_nav):
    """Like the env's render, but also returns coverage from the alpha channel."""
    import torch
    c2w_nav = pose_nav.astype(np.float64).copy()
    c2w_nav[:3, 3] += [0, 0, cal.camera_height_m]
    R_cam_local = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    c2w_nav[:3, :3] = c2w_nav[:3, :3] @ R_cam_local
    pose_recon = cal.nav_cam_to_recon_cam(c2w_nav)
    t_idx = int(np.argmin(np.linalg.norm(
        cal.positions[:, :2] - pose_nav[:2, 3], axis=1)))
    w2c, K1, ts1 = world._single_frame_inputs(scene_cache, pose_recon, t_idx)
    rgb, _, alpha = world._reconstructor.gs_renderer.rasterizer.forward(
        scene_cache["gaussians"], render_viewmats=[w2c], render_Ks=[K1],
        render_timestamps=[ts1], sh_degree=0, width=world.W, height=world.H)
    rgb_np = (rgb[0, 0].detach().clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
    cov = float((alpha[0, 0].detach().float().cpu().numpy() > 0.5).mean())
    return rgb_np, cov


def yaw_pose(cal, frame, yaw_deg, lateral_m=0.0):
    """Robot pose at trajectory frame, rotated to yaw_deg (0 = path heading),
    optionally shifted sideways by lateral_m."""
    base = cal.robot_pose_nav(frame).astype(np.float64)
    fwd = base[:3, 0]
    left = base[:3, 1]
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    base[:3, :3] = Rz @ base[:3, :3]
    base[:3, 3] += lateral_m * left
    return base.astype(np.float32)


# Stage 2: (yaw_deg, lateral_m) pairs spanning trustworthy -> pure invention,
# chosen from stage 1's coverage map.
S2_POSES = [(0, 0.0), (0, 0.5), (0, 1.0), (0, 2.0), (45, 0.0), (90, 0.0), (180, 0.0)]


def run_diffusion_stage(args):
    """Single-backend stage 2 (attempts 1-3 OOM'd with two backends / late pipe
    load). Order matters: load the ~30GB diffusion pipeline FIRST onto an empty
    GPU, then cycle scenes around it — the configuration the backend was
    designed for. Raster twins come from the same backend's rasterizer."""
    import torch
    from PIL import Image, ImageDraw
    from src.env.real_backend import _move_tree_to

    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in args.scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in args.scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in args.scenes},
        render_mode="rasterizer_plus_diffusion",
        model_path=args.model_path, reconstructor_path=args.reconstructor_path)
    world = CalibratedRealWorldBackend(cfg)
    print("loading diffusion pipeline first (empty GPU)...", flush=True)
    world._ensure_pipe_loaded()          # 30GB resident from here on; scenes cycle around it

    for scene in args.scenes:
        print(f"=== {scene} (diffusion stage) ===", flush=True)
        world.load_scene(scene)          # reconstruct; cache lands on CPU (offload fix)
        cal = world._calib[scene]
        f = min(PROBE_FRAMES[len(PROBE_FRAMES) // 2], len(cal.positions) - 1)
        # Move the scene to GPU ONCE for the whole pose set (raster + diffused).
        device = next(world._reconstructor.parameters()).device
        world._cache[scene] = _move_tree_to(world._cache[scene], device)
        cache = world._cache[scene]

        pairs = []
        for yaw, off in S2_POSES:
            pose = yaw_pose(cal, f, yaw, off)
            r_rgb, cov = render_with_alpha(world, cache, cal, pose)
            d_rgb, _, _ = world.render(pose)
            pairs.append((yaw, off, cov, r_rgb, np.asarray(d_rgb)))
            print(f"  yaw {yaw:>3} off {off:>3} cov {cov:.0%} rendered", flush=True)

        W, H = world.W // 2, world.H // 2
        sheet = Image.new("RGB", (W * len(pairs), 2 * H + 20), (0, 0, 0))
        dr = ImageDraw.Draw(sheet)
        for n, (yaw, off, cov, r_rgb, d_rgb) in enumerate(pairs):
            sheet.paste(Image.fromarray(r_rgb).resize((W, H)), (n * W, 20))
            sheet.paste(Image.fromarray(d_rgb).resize((W, H)), (n * W, 20 + H))
            dr.text((n * W + 4, 2), f"yaw{yaw} off{off}m cov{cov:.0%}",
                    fill=(255, 255, 255))
        out = args.out_dir / f"{scene}_diffusion_pairs.png"
        sheet.save(out)
        print(f"  wrote {out} (top=raster, bottom=diffused)", flush=True)

        world._cache.pop(scene, None)
        torch.cuda.empty_cache()


def main():
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/scratch/m000204-pm07b/joana/outputs/probe"))
    ap.add_argument("--diffuse", action="store_true",
                    help="stage 2: paired raster/diffused renders at S2_POSES")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.diffuse:
        run_diffusion_stage(args)
        return

    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in args.scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in args.scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in args.scenes},
        render_mode="rasterizer_only",
        model_path=args.model_path, reconstructor_path=args.reconstructor_path)
    world = CalibratedRealWorldBackend(cfg)

    angles = list(range(0, 360, SPIN_STEP_DEG))
    for scene in args.scenes:
        print(f"=== {scene} ===", flush=True)
        world.load_scene(scene)
        cache = world._cache[scene]
        cal = world._calib[scene]
        frames_avail = [f for f in PROBE_FRAMES if f < len(cal.positions)]

        spin_cov = np.zeros((len(frames_avail), len(angles)))
        off_cov = np.zeros((len(frames_avail), len(OFFSETS_M)))
        spin_imgs, off_imgs = [], []
        for i, f in enumerate(frames_avail):
            for a, ang in enumerate(angles):
                rgb, cov = render_with_alpha(world, cache, cal, yaw_pose(cal, f, ang))
                spin_cov[i, a] = cov
                if f == frames_avail[len(frames_avail) // 2] and ang % 45 == 0:
                    spin_imgs.append((ang, rgb, cov))
            for o, off in enumerate(OFFSETS_M):
                rgb, cov = render_with_alpha(world, cache, cal, yaw_pose(cal, f, 0, off))
                off_cov[i, o] = cov
                if f == frames_avail[len(frames_avail) // 2]:
                    off_imgs.append((off, rgb, cov))
        np.savez_compressed(args.out_dir / f"{scene}_probe.npz",
                            spin_cov=spin_cov, off_cov=off_cov,
                            angles=angles, offsets=OFFSETS_M, frames=frames_avail)
        print(f"  spin coverage: front {spin_cov[:, 0].mean():.0%} "
              f"side {spin_cov[:, len(angles)//4].mean():.0%} "
              f"back {spin_cov[:, len(angles)//2].mean():.0%}", flush=True)
        print(f"  offset coverage: " + " ".join(
            f"{o}m={off_cov[:, i].mean():.0%}" for i, o in enumerate(OFFSETS_M)), flush=True)

        def strip(items, name, label):
            W, H = world.W // 2, world.H // 2
            sheet = Image.new("RGB", (W * len(items), H + 18), (0, 0, 0))
            d = ImageDraw.Draw(sheet)
            for n, (key, rgb, cov) in enumerate(items):
                im = Image.fromarray(rgb).resize((W, H))
                sheet.paste(im, (n * W, 18))
                d.text((n * W + 4, 2), f"{label}{key}  cov {cov:.0%}", fill=(255, 255, 255))
            sheet.save(args.out_dir / f"{scene}_{name}_strip.png")
        strip(spin_imgs, "spin", "yaw ")
        strip(off_imgs, "offset", "off ")

        world._cache.pop(scene, None)
        import torch; torch.cuda.empty_cache()

    # summary curves over all scenes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for scene in args.scenes:
        d = np.load(args.out_dir / f"{scene}_probe.npz")
        a1.plot(d["angles"], d["spin_cov"].mean(0), alpha=0.6, label=scene)
        a2.plot(d["offsets"], d["off_cov"].mean(0), marker="o", alpha=0.6, label=scene)
    a1.set_xlabel("yaw from path heading (deg)"); a1.set_ylabel("coverage (1 - void%)")
    a1.set_title("360 spin: how much of the view exists"); a1.grid(alpha=0.3)
    a2.set_xlabel("lateral offset from path (m)")
    a2.set_title("stepping off the path"); a2.grid(alpha=0.3)
    a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out_dir / "probe_curves.png", dpi=120)
    print(f"\nwrote {args.out_dir}/probe_curves.png")


if __name__ == "__main__":
    main()
