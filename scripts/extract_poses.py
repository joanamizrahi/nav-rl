"""Extract REAL per-frame robot trajectories from NeoVerse's reconstructor.

Milestone A / TODO item: "Extract REAL trajectories from NeoVerse reconstructor
per clip". Replaces validate_reward.py's synthetic straight-line trajectory with
the path the camera actually took — which for RUGD IS the robot's ground-truth
trajectory (the clip was recorded from the robot).

Runs on Marlowe (needs GPU + reconstructor checkpoint). One reconstruction per
clip (~1-2 min each on an H100 after the one-time model load); no diffusion.

Per clip it saves `<stem>_poses.npz` with exactly the keys load_clip's
`pose_source="npz"` branch expects:

    positions  (T, 3)  robot feet-on-ground positions, meters, z-up world
    headings   (T, 3)  robot forward unit vectors, horizontal
    w2c        (T, 4, 4) world -> camera, consistent with `positions`' frame
    K          (3, 3)  intrinsics from the reconstructor (not the pinhole guess)

plus diagnostics (c2w, raw camera positions, fitted plane, scale factor).

The geometry, step by step (conventions verified in src/env/real_backend.py):

  1. Reconstructor outputs camera-to-world poses in ITS frame: x=right,
     y=DOWN, z=forward — and in NORMALIZED per-clip units, not meters.
  2. Rotate into the eval/scene frame (x=fwd, y=right, z=up) with the fixed
     R_SCENE_TO_RECON^T from real_backend (single source of truth).
  3. That only makes z *approximately* up (exact if the frame-0 camera was
     level). So we RANSAC-fit a ground plane to the Gaussian means
     (src/eval/ground_plane.py) and rotate the world so the plane normal is
     exactly +z, with the ground at z=0.
  4. Metric scale: the camera's median height above the fitted plane, in recon
     units, must equal the physical mount height (--camera_height_m; RUGD was
     recorded on a Clearpath Husky; paper Sec III-A: viewpoint <25 cm off the
     ground -> 0.25 m). scale = mount_m / median.
     Scaling world points AND c2w translations by the same factor leaves pixel
     projections bit-identical, so the reward's meter-based footprint
     (GO2 body 0.6x0.3 m, look_ahead 0.5 m) becomes meaningful without
     changing what projects where.
  5. Robot position = camera position dropped to the ground (z=0); robot
     heading = camera forward projected to horizontal.

Usage (on Marlowe):
    python scripts/extract_poses.py \
        --videos /scratch/m000204-pm07b/joana/data/rugd_clips/*.mp4 \
        --output_dir /scratch/m000204-pm07b/joana/outputs/poses

Then (on Mac, after scp):
    python scripts/validate_reward.py --video ... --labels ... \
        --pose_source npz --poses_npz outputs/poses/<stem>_poses.npz ...
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.env.real_backend import R_SCENE_TO_RECON
from src.eval.ground_plane import fit_ground_plane_ransac, GroundPlane


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy — unit-testable on Mac without torch)
# ---------------------------------------------------------------------------

def rotation_aligning_to_z(normal: np.ndarray) -> np.ndarray:
    """Rodrigues rotation taking `normal` (unit, up-ish) exactly onto +z."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(n @ z, -1.0, 1.0))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:                      # anti-parallel: flip around x
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(n, z)
    axis /= np.linalg.norm(axis)
    s = np.sqrt(1.0 - c * c)
    Kx = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + s * Kx + (1.0 - c) * (Kx @ Kx)


def poses_from_c2w_recon(
    c2w_recon: np.ndarray,          # (T, 4, 4) camera-to-world, recon frame, recon units
    gaussian_means_recon: np.ndarray,   # (N, 3) recon frame, recon units
    camera_height_m: float,
    path_length_m: "float | None" = None,   # GT path length (from odometry)
    odom_xy_m: "np.ndarray | None" = None,   # (T,2) odometry position at each frame, metres
    scale_from: str = "height",              # height | symmetric | umeyama | pathlen
) -> dict:
    """Recon-frame camera poses -> z-up, ground-at-z0, metric-scale robot poses.

    Returns dict with positions/headings/w2c/c2w plus diagnostics. Pure numpy.
    """
    T = len(c2w_recon)
    R_rs = R_SCENE_TO_RECON.T.astype(np.float64)     # recon -> scene(z-up-ish)

    # --- step 2: fixed rotation into the approx z-up scene frame ---
    c2w = c2w_recon.astype(np.float64).copy()
    c2w[:, :3, :3] = R_rs @ c2w[:, :3, :3]
    c2w[:, :3, 3] = c2w[:, :3, 3] @ R_rs.T
    means = gaussian_means_recon.astype(np.float64) @ R_rs.T

    # --- step 3: exact gravity alignment from a RANSAC ground plane ---
    # inlier_thresh is documented as meters but we're still in recon units here,
    # so make it scale-relative: 1% of the scene's vertical extent.
    extent = float(np.percentile(means[:, 2], 98) - np.percentile(means[:, 2], 2))
    plane: GroundPlane = fit_ground_plane_ransac(
        means, inlier_thresh=max(1e-6, 0.01 * extent),
        cameras_above=c2w[:, :3, 3])
    R_up = rotation_aligning_to_z(plane.normal)
    c2w[:, :3, :3] = R_up @ c2w[:, :3, :3]
    c2w[:, :3, 3] = c2w[:, :3, 3] @ R_up.T
    means = means @ R_up.T
    # After R_up the plane is horizontal at z = -offset (normal@p+offset=0 with
    # normal now +z). Shift so the ground sits at z=0.
    ground_z = -plane.offset
    c2w[:, 2, 3] -= ground_z
    means[:, 2] -= ground_z

    # --- step 4: metric scale from camera mount height ---
    cam_heights = c2w[:, 2, 3]
    h_median = float(np.median(cam_heights))
    if h_median <= 0:
        # 2026-09-15: two RealSense campus clips (quad2_09_rs, quad2_10_rs)
        # failed here with the same value (-0.035 units), i.e. the winning
        # RANSAC plane passes almost through the camera path. Hand the state
        # after the plane step to the caller so it can be saved and looked at
        # (a side view of the gaussians vs the cameras) instead of guessed at.
        err = RuntimeError(
            f"median camera height above fitted ground is {h_median:.4f} <= 0; "
            "the plane fit or the recon->scene rotation is wrong for this clip. "
            "Inspect the saved diagnostics before trusting anything.")
        inl = float((np.abs(means[:, 2]) < max(1e-6, 0.01 * extent)).mean())
        err.diag = {
            "cam_positions_after_plane": c2w[:, :3, 3].copy(),
            "means_after_plane": means.astype(np.float32),
            "plane_normal_scene": plane.normal, "plane_offset": float(plane.offset),
            "inlier_frac": inl, "inlier_thresh_units": float(max(1e-6, 0.01 * extent)),
            "h_median_units": h_median, "extent_units": extent,
        }
        raise err
    # Prefer GT odometry path length when the clip has it: monocular scale
    # from an assumed mount height drifted 2-4x on the GND ZED clips (odom
    # 33 m vs "extracted 119 m"), and the error varies per clip.
    L_units = float(np.linalg.norm(np.diff(c2w[:, :2, 3], axis=0), axis=1).sum())
    scale_height = camera_height_m / h_median
    scale_pathlen = (path_length_m / L_units) if (path_length_m is not None and L_units > 1e-6) else None
    # 2026-09-15: the path-length ratio is NOT robust -- one jump in the
    # reconstructed camera path (campus clips: max step 2 m in a 0.37 m/frame
    # walk) inflates L_units and shrinks the scale for the whole scene; the
    # estimated camera heights spread 0.28-1.69 m for a 0.69 m camera. A
    # similarity fit over all T frame positions (Umeyama) is barely moved by
    # one outlier, so it is the scale of record when odometry is available.
    scale_umeyama = None
    if odom_xy_m is not None and len(odom_xy_m) == T:
        # The scene frame's y axis is mirrored relative to the robot's (ROS: y
        # left); NavCalibration applies diag(1,-1,1) on load. Apply the same
        # flip here or a curved path fits its mirror image and the scale is
        # biased low (2026-09-15).
        src_all = c2w[:, :2, 3] * np.array([1.0, -1.0])
        dst_all = np.asarray(odom_xy_m, float)

        def _fit(src, dst):
            n = len(src)
            mu_s, mu_d = src.mean(0), dst.mean(0)
            S_, D_ = src - mu_s, dst - mu_d
            U, sig, Vt = np.linalg.svd(D_.T @ S_ / n)
            dsign = np.sign(np.linalg.det(U @ Vt))
            Dm = np.diag([1.0, dsign])
            R2 = U @ Dm @ Vt
            var_s = (S_ ** 2).sum() / n
            if var_s <= 1e-12:
                return None, None
            sc = float(np.trace(np.diag(sig) @ Dm) / var_s)
            resid = np.linalg.norm(dst - ((sc * (R2 @ src.T)).T + (mu_d - sc * R2 @ mu_s)), axis=1)
            return sc, resid

        # two passes: fit, drop frames whose residual is > 3x the median (the
        # reconstruction's jump frames), refit on the rest
        sc0, res0 = _fit(src_all, dst_all)
        if sc0 is not None:
            keep = res0 <= 3.0 * max(np.median(res0), 1e-6)
            if keep.sum() >= 8 and keep.sum() < T:
                sc1, _ = _fit(src_all[keep], dst_all[keep])
                scale_umeyama = sc1 if sc1 is not None else sc0
            else:
                scale_umeyama = sc0
    # Symmetric odometry estimate: the forward fit (recon -> odom) is biased LOW
    # by path noise, the reverse fit (odom -> recon) biased HIGH in the same
    # way; their geometric mean cancels most of it. Diagnostic only.
    scale_sym = None
    if scale_umeyama is not None and odom_xy_m is not None and len(odom_xy_m) == T:
        sc_rev, _ = _fit(np.asarray(odom_xy_m, float), c2w[:, :2, 3])
        if sc_rev is not None and sc_rev > 0:
            scale_sym = float(np.sqrt(scale_umeyama / sc_rev))
    # 2026-09-15 (campus clips): path-length 18.6 / similarity 9.3 / tape 13.3 on
    # the same clip -- the two odometry fits bracket the tape value because the
    # reconstructed path is noisy. The tape-height scale uses the median of 81
    # camera heights over a tightly fitted plane and is immune to path noise,
    # so it is the scale of record; odometry is the check.
    if scale_from == "height" or scale_umeyama is None:
        scale, scale_source = scale_height, "mount-height"
    elif scale_from == "symmetric" and scale_sym is not None:
        scale, scale_source = scale_sym, "symmetric(odometry)"
    elif scale_from == "pathlen" and scale_pathlen is not None:
        scale, scale_source = scale_pathlen, "path-length(odometry)"
    else:
        scale, scale_source = scale_umeyama, "umeyama(odometry)"
    c2w[:, :3, 3] *= scale
    means *= scale

    # --- step 5: robot poses on the ground ---
    cam_pos = c2w[:, :3, 3].copy()
    positions = cam_pos.copy()
    positions[:, 2] = 0.0
    fwd = c2w[:, :3, 2].copy()                        # camera z-axis in world
    fwd[:, 2] = 0.0                                    # horizontal projection
    norms = np.linalg.norm(fwd, axis=1, keepdims=True)
    if (norms < 1e-3).any():
        raise RuntimeError("near-vertical camera in some frame; heading undefined")
    headings = fwd / norms

    w2c = np.zeros_like(c2w)
    for t in range(T):
        R = c2w[t, :3, :3]
        p = c2w[t, :3, 3]
        w2c[t, :3, :3] = R.T
        w2c[t, :3, 3] = -R.T @ p
        w2c[t, 3, 3] = 1.0

    step_sizes = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    return {
        "positions": positions.astype(np.float32),
        "headings": headings.astype(np.float32),
        "w2c": w2c.astype(np.float32),
        "c2w": c2w.astype(np.float32),
        "cam_positions": cam_pos.astype(np.float32),
        "scale_m_per_unit": np.float32(scale),
        "scale_source": scale_source,
        "scale_umeyama": np.float32(scale_umeyama if scale_umeyama is not None else np.nan),
        "scale_pathlen": np.float32(scale_pathlen if scale_pathlen is not None else np.nan),
        "scale_height": np.float32(scale_height),
        "scale_symmetric": np.float32(scale_sym if scale_sym is not None else np.nan),
        "camera_height_units_median": np.float32(h_median),
        "plane_normal_scene": plane.normal.astype(np.float32),
        "plane_offset_scene": np.float32(plane.offset),
        "step_sizes_m": step_sizes.astype(np.float32),
    }


def _extract_gaussian_means(splats) -> np.ndarray:
    """Pull (N, 3) position means out of the reconstructor's splats structure.

    Actual layout (rasterization.py Rasterizer.forward + rasterize_splats):
    predictions["splats"] is a LIST over batch; each element is a LIST of
    Gaussians objects (constant + dynamic), each with a `.means` tensor.
    We recurse through any list/dict nesting and collect every means tensor.
    """
    import torch

    def _to_np(x):
        return x.detach().float().cpu().numpy().reshape(-1, 3)

    collected = []

    def _collect(node):
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _collect(item)
            return
        for key in ("means", "means3d", "xyz", "positions"):
            val = None
            if isinstance(node, dict) and key in node:
                val = node[key]
            elif hasattr(node, key):
                val = getattr(node, key)
            if torch.is_tensor(val) and val.numel() and val.shape[-1] == 3:
                collected.append(_to_np(val))
                return

    _collect(splats)
    if not collected:
        raise RuntimeError(
            f"couldn't find gaussian means; splats type={type(splats)}, "
            f"keys/attrs={list(splats.keys()) if isinstance(splats, dict) else dir(splats)}")
    return np.concatenate(collected, axis=0)


# ---------------------------------------------------------------------------
# GPU side: run the reconstructor (mirrors real_backend._reconstruct_scene)
# ---------------------------------------------------------------------------

def load_reconstructor(reconstructor_path: str):
    import torch
    from diffsynth.utils import ModelConfig
    from diffsynth.models import ModelManager

    print(f"[extract_poses] loading reconstructor from {reconstructor_path} ...", flush=True)
    mm = ModelManager()
    cfg = ModelConfig(path=reconstructor_path, offload_device="cuda")
    cfg.download_if_necessary()
    mm.load_model(cfg.path, device="cuda", torch_dtype=torch.bfloat16)
    return mm.fetch_model("reconstructor")


def reconstruct_clip(reconstructor, video_path: Path, num_frames: int, width: int, height: int,
                     K_prior: "np.ndarray | None" = None):
    """K_prior (3x3, pixels at width x height): the camera's intrinsics given
    to the reconstructor as its rays prior (2026-09-15). Without it every clip
    guesses its own focal length, and on long open sightlines (Packard) the
    guess came out 25-70% above the same camera's other clips."""
    import torch
    from torchvision.transforms import functional as F
    from diffsynth.utils.auxiliary import load_video

    device = next(reconstructor.parameters()).device
    dtype = next(reconstructor.parameters()).dtype

    # Same loader + center_crop as SAM3 labeling and load_clip, so frame i here
    # is frame i everywhere.
    images = load_video(str(video_path), num_frames,
                        resolution=(width, height), resize_mode="center_crop",
                        static_scene=False)
    views = {
        "img": torch.stack([F.to_tensor(im)[None] for im in images], dim=1).to(device),
        "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "timestamp": torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0),
    }
    # The transformer reads the flags as [pose, depth, rays]
    # (visual_transformer._process_conditioning), whatever the docstring in
    # worldmirror.py says; rays = intrinsics normalised by the image size.
    cond_flags = [0, 0, 0]
    if K_prior is not None:
        Kt = torch.as_tensor(np.asarray(K_prior, dtype=np.float32), device=device)
        views["camera_intrs"] = Kt[None, None].repeat(1, len(images), 1, 1)
        cond_flags = [0, 0, 1]
    # no_grad is CRITICAL — see real_backend._reconstruct_scene (activation
    # memory otherwise blows past VRAM before anything can be reclaimed).
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = reconstructor(views, cond_flags=cond_flags, is_inference=True, use_motion=False)

    c2w = pred["rendered_extrinsics"][0].detach().float().cpu().numpy()   # (T,4,4)
    K_all = pred["rendered_intrinsics"][0].detach().float().cpu().numpy()  # (T,3,3)
    means = _extract_gaussian_means(pred["splats"])

    del pred, views
    torch.cuda.empty_cache()
    return c2w, K_all, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, type=Path,
                    help="one or more clip MP4s (shell glob ok)")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--scale_from", choices=["symmetric", "umeyama", "height", "pathlen"], default="symmetric",
                    help="metric scale. 2026-09-15: the reconstruction's vertical and horizontal scales differ "
                         "(tape-height vs odometry scale ratio 0.5-1.2 across campus scenes, worst on long open "
                         "sightlines); the robot's steps, goals and footprint are HORIZONTAL, so the odometry "
                         "similarity fit (symmetric) is the scale of record when odometry exists; tape height is "
                         "the diagnostic. Falls back to height without odometry.")
    ap.add_argument("--camera_height_m", type=float, default=0.25,
                    help="physical camera mount height used ONLY for metric scale "
                         "(RUGD paper Sec III-A: viewpoint <25 cm off ground -> 0.25). "
                         "Getting this off by 20%% "
                         "scales the footprint/look-ahead by 20%% — mild, not fatal.")
    ap.add_argument("--max_gaussians_for_plane", type=int, default=200_000)
    ap.add_argument("--intrinsics_prior", default="",
                    help="fx,fy,cx,cy in pixels at --width x --height, given to the reconstructor "
                         "as its rays prior for every clip (one camera, one calibration). Take it "
                         "from the median K of the clips the reconstructor already agrees on.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reconstructor = load_reconstructor(args.reconstructor_path)
    K_prior = None
    if args.intrinsics_prior:
        fx, fy, cx, cy = [float(v) for v in args.intrinsics_prior.split(",")]
        K_prior = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        print(f"[extract_poses] intrinsics prior for every clip: fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f}", flush=True)

    for video in args.videos:
        print(f"\n[extract_poses] === {video.name} ===", flush=True)
        # Subsample seeded by the clip's NAME, not by a generator that advances
        # through the batch (2026-09-15): the plane fit's outcome depended on
        # a clip's position in the job. Now a rerun of one clip is repeatable.
        rng = np.random.default_rng(zlib.crc32(video.stem.encode()))
        c2w_recon, K_all, means = reconstruct_clip(
            reconstructor, video, args.num_frames, args.width, args.height, K_prior=K_prior)
        print(f"[extract_poses] {len(c2w_recon)} poses, {len(means)} gaussians"
              f" | fy {float(K_all[0][1, 1]):.1f} cy {float(K_all[0][1, 2]):.1f}"
              + (f" (prior fy {float(K_prior[1, 1]):.1f})" if K_prior is not None else ""), flush=True)

        if len(means) > args.max_gaussians_for_plane:
            means = means[rng.choice(len(means), args.max_gaussians_for_plane, replace=False)]

        # GT scale from the recorder's own odometry when the clip has it
        # (<stem>_odom.npz written by prepare_rosbag_clips next to the mp4).
        path_length_m = None
        odom_xy = None
        odom_path = video.parent / f"{video.stem}_odom.npz"
        if odom_path.exists():
            od = np.load(odom_path)
            path_length_m = float(np.linalg.norm(
                np.diff(od["xyz"][:, :2], axis=0), axis=1).sum())
            T_ = len(c2w_recon)
            t_od = np.asarray(od["t"], float)
            fs = np.asarray(od["frame_stamps"], float) if "frame_stamps" in od and len(od["frame_stamps"]) == T_ \
                else np.linspace(t_od[0], t_od[-1], T_)
            odom_xy = np.stack([np.interp(fs, t_od, od["xyz"][:, 0]), np.interp(fs, t_od, od["xyz"][:, 1])], 1)
            print(f"[extract_poses] odometry found: {len(t_od)} samples, path {path_length_m:.1f} m, "
                  f"{'per-frame stamps' if 'frame_stamps' in od else 'stamps assumed uniform'}", flush=True)

        try:
            out = poses_from_c2w_recon(c2w_recon, means,
                                       camera_height_m=args.camera_height_m,
                                       path_length_m=path_length_m, odom_xy_m=odom_xy,
                                       scale_from=args.scale_from)
        except RuntimeError as e:
            # One bad clip must not kill the batch (2026-09-15: quad2_09_rs took
            # eight scenes down with it). Save what the plane step produced and
            # a side-view picture, print FAILED, go on to the next clip.
            print(f"[extract_poses] FAILED {video.stem}: {e}", flush=True)
            diag = getattr(e, "diag", None)
            if diag is not None:
                dpath = args.output_dir / f"{video.stem}_plane_fail.npz"
                np.savez_compressed(dpath, **diag)
                print(f"[extract_poses] diagnostics -> {dpath}", flush=True)
                _plane_fail_figure(diag, args.output_dir / f"{video.stem}_plane_fail.png", video.stem)
            continue
        out["K"] = K_all[0].astype(np.float32)
        out["K_all"] = K_all.astype(np.float32)
        out["video"] = str(video)
        # The render camera is lifted above the robot's ground pose by
        # camera_height_m. It must be the RECONSTRUCTION's own camera height in
        # the chosen metres (h_median x scale), not the tape value: with a
        # vertically squashed reconstruction the tape height would put every
        # render too high. The tape value is kept for the diagnostic ratio.
        out["camera_height_m"] = np.float32(float(out["camera_height_units_median"]) * float(out["scale_m_per_unit"]))
        out["camera_height_tape_m"] = np.float32(args.camera_height_m)
        out["intrinsics_prior"] = (K_prior if K_prior is not None else np.zeros((3, 3), np.float32))

        # Sanity numbers for eyeballing before trusting the npz downstream.
        steps = out["step_sizes_m"]
        length_m = float(steps.sum())
        print(f"[extract_poses] scale = {float(out['scale_m_per_unit']):.4f} m/unit from {out['scale_source']} "
              f"| tape-height {float(out['scale_height']):.4f} symmetric-odom {float(out['scale_symmetric']):.4f} "
              f"umeyama {float(out['scale_umeyama']):.4f} path-length {float(out['scale_pathlen']):.4f} "
              f"(median cam height {float(out['camera_height_units_median']):.4f} units)", flush=True)
        _sh, _ss = float(out['scale_height']), float(out['scale_symmetric'])
        if np.isfinite(_ss):
            print(f"[extract_poses] vertical/horizontal ratio (odometry scale / tape-height scale) = {_ss / _sh:.2f}; "
                  f"render camera height = {float(out['camera_height_m']):.2f} m (tape {args.camera_height_m:.2f} m)", flush=True)
            if abs(_ss / _sh - 1.0) > 0.50:
                print(f"[extract_poses] WARNING: reconstruction vertical scale off by {abs(_ss / _sh - 1.0) * 100:.0f}% "
                      f"vs horizontal -- long sightlines? check this scene's renders before training on it", flush=True)
        print(f"[extract_poses] trajectory: {length_m:.1f} m total, "
              f"step {steps.mean():.3f} m/frame (min {steps.min():.3f}, max {steps.max():.3f})", flush=True)
        cam_z = out["cam_positions"][:, 2]
        print(f"[extract_poses] camera height over ground: "
              f"mean {cam_z.mean():.2f} m, std {cam_z.std():.3f} m "
              f"(std >> 0.1 m means bad plane fit or non-flat terrain)", flush=True)

        out_path = args.output_dir / f"{video.stem}_poses.npz"
        np.savez_compressed(out_path, **out)
        print(f"[extract_poses] wrote {out_path}", flush=True)


def _plane_fail_figure(diag: dict, path, name: str) -> None:
    """Side views of the gaussians (sample) and the camera path after the plane
    step: the fitted plane is z=0. Shows whether the cameras sit on the plane
    (plane through the path) or the ground mass lies elsewhere."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = diag["means_after_plane"]; c = diag["cam_positions_after_plane"]
        rng = np.random.default_rng(0)
        if len(m) > 200000:
            m = m[rng.choice(len(m), 200000, replace=False)]
        lim = np.percentile(np.abs(m), 98)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        for k, (a, b, lab) in enumerate([(0, 2, "x-z (side)"), (1, 2, "y-z (side)"), (0, 1, "x-y (top)")]):
            ax[k].scatter(m[:, a], m[:, b], s=0.2, c="0.6", alpha=0.4)
            ax[k].plot(c[:, a], c[:, b], "r.-", ms=3, label="cameras")
            if b == 2:
                ax[k].axhline(0.0, color="b", lw=1, label="fitted plane (z=0)")
            ax[k].set_xlim(-lim, lim); ax[k].set_ylim(-lim, lim); ax[k].set_aspect("equal")
            ax[k].set_title(lab); ax[k].grid(alpha=0.3); ax[k].legend(loc="upper right")
        fig.suptitle(f"{name}: plane fit failed, median camera height {diag['h_median_units']:.4f} units, "
                     f"inliers {100 * diag['inlier_frac']:.1f}% within {diag['inlier_thresh_units']:.4f}")
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"[extract_poses] figure -> {path}", flush=True)
    except Exception as ex:
        print(f"[extract_poses] no figure ({ex})", flush=True)


if __name__ == "__main__":
    main()
