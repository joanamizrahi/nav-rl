"""Dump a scene's Gaussian cloud to npz for Mac-side 3D viz + Test A (ground shape).

What it does, in plain terms:
  1. Rebuilds the scene exactly like the RL env does (CalibratedRealWorldBackend:
     reconstruct the clip -> Gaussians with SAM3 labels attached).
  2. Pulls out each Gaussian's 3D center (its "mean"), plus its semantic label
     and color where available.
  3. Converts the centers from the reconstructor's normalized frame into our
     metric nav frame (meters, z-up, ground~0) with the SAME calibration the
     env uses — so the cloud, the trajectory, and the reward all live in one
     frame and can be overlaid directly.
  4. Subsamples (the full cloud is ~15M points; 400k is plenty for viz/stats)
     and saves ONE npz per scene next to the poses files.

Runs on Marlowe (GPU). ~30 s per scene on a warm node.

Usage:
    python scripts/dump_scene_cloud.py \
        --scenes rugd_trail_00 rugd_creek_00 rugd_park-1_00 rugd_park-2_00 \
        --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
        --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
        --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
        --out_dir /scratch/m000204-pm07b/joana/outputs/scene_clouds
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


def collect_gaussian_fields(splats):
    """Walk the nested splats structure (list[batch] -> list[Gaussians]) and pull
    per-gaussian arrays: means (N,3), labels (N,), colors (N,3), opacities (N,),
    sizes (N,) = mean linear scale, timestamps (N,) where present.
    Defensive: missing fields come back filled with sentinels."""
    import torch

    fields = {k: [] for k in ("means", "labels", "colors", "opacities", "sizes",
                              "times", "scales3", "quats", "sh_dc")}

    def _np(x):
        return x.detach().float().cpu().numpy()

    def _collect(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                _collect(item)
            return
        m = getattr(node, "means", None)
        if not torch.is_tensor(m):
            return
        n = m.shape[0]
        fields["means"].append(_np(m).reshape(-1, 3))

        lab = getattr(node, "labels", None)
        if torch.is_tensor(lab) and lab.shape[0] == n:
            l = _np(lab)
            fields["labels"].append(l.argmax(-1) if l.ndim == 2 else l)
        else:
            fields["labels"].append(np.full(n, -1, dtype=np.int16))

        sh = getattr(node, "sh", None) if getattr(node, "sh", None) is not None \
            else getattr(node, "harmonics", None)
        if torch.is_tensor(sh):
            s = _np(sh).reshape(n, -1, 3)
            fields["colors"].append(np.clip(0.2820948 * s[:, 0, :] + 0.5, 0, 1))
        else:
            fields["colors"].append(np.full((n, 3), 0.5, dtype=np.float32))

        op = getattr(node, "opacities", None)
        fields["opacities"].append(_np(op).reshape(-1) if torch.is_tensor(op)
                                   else np.full(n, 1.0, dtype=np.float32))

        sc = getattr(node, "scales", None)
        if torch.is_tensor(sc):
            s3 = _np(sc).reshape(n, -1)
            if s3.shape[1] != 3:
                s3 = np.repeat(s3.mean(-1, keepdims=True), 3, axis=1)
        else:
            s3 = np.full((n, 3), 1e-3, dtype=np.float32)
        fields["scales3"].append(s3)
        fields["sizes"].append(s3.mean(-1))

        q = getattr(node, "quats", None)
        if q is None:
            q = getattr(node, "rotations", None)
        if torch.is_tensor(q) and q.shape[0] == n:
            fields["quats"].append(_np(q).reshape(n, 4))
        else:
            fields["quats"].append(np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32))

        # raw SH DC coefficients (for the 3DGS ply; `colors` above is the
        # human-viewable version of the same thing)
        if torch.is_tensor(sh):
            fields["sh_dc"].append(_np(sh).reshape(n, -1, 3)[:, 0, :])
        else:
            fields["sh_dc"].append(np.zeros((n, 3), dtype=np.float32))

        ts = getattr(node, "timestamps", None)
        if ts is None:
            ts = getattr(node, "timestamp", None)
        if torch.is_tensor(ts) and ts.numel() in (n, 1):
            t = _np(ts).reshape(-1)
            fields["times"].append(np.full(n, float(t[0])) if t.size == 1 else t)
        else:
            fields["times"].append(np.full(n, -1.0, dtype=np.float32))

    _collect(splats)
    if not fields["means"]:
        raise RuntimeError(f"no means found in splats structure {type(splats)}")
    out = {k: np.concatenate(v) for k, v in fields.items()}
    out["labels"] = out["labels"].astype(np.int16)
    return out


def write_3dgs_ply(path: Path, means, scales3, quats, opacities, sh_dc):
    """Standard 3D-Gaussian-Splatting PLY (INRIA layout) for viewers like
    SuperSplat (https://superspl.at/editor — drag & drop the file).

    Convention notes:
      - scales stored as LOG of the actual lengths
      - opacity stored as LOGIT (inverse sigmoid) of alpha
      - rotation = normalized quaternion, w first
      - colors = raw SH DC coefficients (f_dc_*)
    """
    n = len(means)
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    a = np.clip(opacities, 1e-4, 1 - 1e-4)
    props = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
             ("opacity", "f4"),
             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    v = np.zeros(n, dtype=props)
    v["x"], v["y"], v["z"] = means[:, 0], means[:, 1], means[:, 2]
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
    v["opacity"] = np.log(a / (1 - a))
    logs = np.log(np.maximum(scales3, 1e-8))
    v["scale_0"], v["scale_1"], v["scale_2"] = logs[:, 0], logs[:, 1], logs[:, 2]
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {name}\n" for name, _ in props)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        v.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--max_points", type=int, default=400_000)
    ap.add_argument("--ply", action="store_true",
                    help="also write <scene>.ply (3DGS format) for SuperSplat")
    ap.add_argument("--ply_points", type=int, default=1_500_000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in args.scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in args.scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in args.scenes},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    world = CalibratedRealWorldBackend(cfg)
    rng = np.random.default_rng(0)

    for scene in args.scenes:
        print(f"\n=== {scene} ===", flush=True)
        world.load_scene(scene)
        cache = world._cache[scene]
        cal = world._calib[scene]

        g = collect_gaussian_fields(cache["gaussians"])
        print(f"  {len(g['means'])} gaussians (labels: {(g['labels'] >= 0).mean():.0%}, "
              f"opacity range {g['opacities'].min():.2f}-{g['opacities'].max():.2f}, "
              f"times present: {(g['times'] >= 0).mean():.0%})", flush=True)

        # recon -> metric nav frame: same map the env/reward uses.
        means_nav = (cal.A @ g["means"].T).T
        means_nav[:, 2] -= cal.ground_z
        means_nav *= cal.scale
        sizes_m = g["sizes"] * cal.scale        # scales are lengths -> also scale to meters

        if len(means_nav) > args.max_points:
            idx = rng.choice(len(means_nav), args.max_points, replace=False)
        else:
            idx = np.arange(len(means_nav))

        d = np.load(f"{args.poses_dir}/{scene}_poses.npz")
        out = args.out_dir / f"{scene}_cloud.npz"
        np.savez_compressed(
            out, points=means_nav[idx].astype(np.float32), labels=g["labels"][idx],
            colors=(g["colors"][idx] * 255).astype(np.uint8),
            opacities=g["opacities"][idx].astype(np.float32),
            sizes=sizes_m[idx].astype(np.float32),
            times=g["times"][idx].astype(np.float32),
            traj_positions=d["positions"], traj_cam_z=d["cam_positions"][:, 2],
            camera_height_m=np.float32(0.6),
        )
        print(f"  wrote {out}", flush=True)

        if args.ply:
            # Subsample opacity-weighted-ish (drop near-transparent floaters first).
            solid = np.nonzero(g["opacities"] >= 0.02)[0]
            take = solid if len(solid) <= args.ply_points else \
                rng.choice(solid, args.ply_points, replace=False)
            # NOTE: ply uses the RECON frame scaled to meters (not the nav frame):
            # the nav conversion includes a mirror, which quaternions cannot
            # represent — mixing frames would misorient every ellipsoid.
            ply_path = args.out_dir / f"{scene}.ply"
            write_3dgs_ply(ply_path, (g["means"][take] * cal.scale).astype(np.float32),
                           (g["scales3"][take] * cal.scale).astype(np.float32),
                           g["quats"][take].astype(np.float32),
                           g["opacities"][take].astype(np.float32),
                           g["sh_dc"][take].astype(np.float32))
            print(f"  wrote {ply_path} ({len(take)} splats) — open at superspl.at/editor",
                  flush=True)
        # Free the scene before the next one (each holds ~GBs on GPU).
        world._cache.pop(scene, None)
        import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
