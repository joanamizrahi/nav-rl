"""Benchmark the live per-action diffusion render — the Phase-0 decision gate.

Walks a simulated robot along the recorded trail in 0.25 m steps and times
every phase (raster / diffusion / decode) of LiveDiffusedBackend.render for a
matrix of frame-counts and resolutions. Saves a short observation clip per
config so the quality of each setting can be judged by eye (the fine-tune
never trained below 560x336 — low-res quality is measured, not assumed).

Output: a printed table (config -> s/step, phase breakdown) + <out>/*.mp4.

    python scripts/live_benchmark.py --scene rugd_trail_00 \
        --checkpoint /scratch/.../runs/train_semantic_v10/checkpoint-epoch-30.safetensors \
        --frames 5,9,21 --resolutions 560x336,392x224 --steps 10 \
        --clips_dir ... --poses_dir ... --labels_dir ... --out .../live_bench
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
SCRATCH_NEOVERSE = Path("/scratch/m000204-pm07b/joana/NeoVerse")
if SCRATCH_NEOVERSE.exists():
    sys.path.insert(0, str(SCRATCH_NEOVERSE))

from src.env.real_calibrated import CalibratedBackendConfig, NavCalibration
from src.env.live_backend import LiveDiffusedBackend


def walk_poses(cal: NavCalibration, n: int, step_m: float = 0.25):
    """Nav-frame robot poses marching along the recorded path in step_m hops."""
    p = cal.positions[:, :2]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.arange(1, n + 1) * step_m + s[-1] * 0.15   # start 15% in
    poses = []
    for st in targets:
        i = int(np.searchsorted(s, min(st, s[-1] - 1e-6))) - 1
        i = max(0, min(i, len(p) - 2))
        f = (st - s[i]) / max(s[i + 1] - s[i], 1e-9)
        xy = p[i] + f * (p[i + 1] - p[i])
        d = p[i + 1] - p[i]
        yaw = np.arctan2(d[1], d[0])
        P = np.eye(4, dtype=np.float64)
        c, si = np.cos(yaw), np.sin(yaw)
        P[0, 0], P[0, 1], P[1, 0], P[1, 1] = c, -si, si, c
        P[:2, 3] = xy
        poses.append(P)
    return poses


def resized_labels(src: Path, wh, out_dir: Path) -> Path:
    import cv2
    w, h = wh
    d = np.load(src)
    lab = d["labels"]
    if lab.shape[1:] == (h, w):
        return src
    out = out_dir / f"{src.stem}_{w}x{h}.npz"
    if not out.exists():
        rs = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST)
                       for x in lab])
        np.savez_compressed(out, labels=rs.astype(np.int8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--frames", default="5,9,21")
    ap.add_argument("--resolutions", default="560x336")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import cv2
    frame_counts = [int(x) for x in args.frames.split(",")]
    resolutions = [tuple(int(v) for v in r.split("x")) for r in args.resolutions.split(",")]
    rows, shared_pipe = [], None

    for (W, H) in resolutions:
        labels_path = resized_labels(
            Path(args.labels_dir) / f"{args.scene}.npz", (W, H), args.out)
        cfg = CalibratedBackendConfig(
            scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
            scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
            scene_labels_paths={args.scene: str(labels_path)},
            model_path=args.model_path,
            reconstructor_path=args.reconstructor_path,
        )
        cfg.W, cfg.H = W, H
        backend = LiveDiffusedBackend(cfg, checkpoint=args.checkpoint)
        if shared_pipe is not None:      # one 40 GB pipe load for the whole run
            backend._pipe = shared_pipe
            backend._reconstructor = shared_pipe.reconstructor
        backend.load_scene(args.scene)
        shared_pipe = backend._pipe
        cal = backend._calib[args.scene]
        poses = walk_poses(cal, args.steps + 2)

        for k in frame_counts:
            backend.live_frames = k
            backend._pose_hist = []
            times, frames = [], []
            for i, P in enumerate(poses):
                rgb, _, _ = backend.render(P)
                if i >= 2:               # skip warmup calls
                    times.append(dict(backend.last_timings))
                    frames.append(np.ascontiguousarray(rgb[:, :, ::-1]))
            tot = np.array([t["total"] for t in times])
            row = dict(W=W, H=H, k=k, mean=tot.mean(), p50=np.median(tot),
                       raster=np.mean([t["raster"] for t in times]),
                       diff=np.mean([t["diffusion"] for t in times]),
                       dec=np.mean([t["decode"] for t in times]))
            rows.append(row)
            print(f"[bench] {W}x{H} k={k}: {row['mean']:.2f}s/step "
                  f"(raster {row['raster']:.2f} diffusion {row['diff']:.2f} "
                  f"decode {row['dec']:.2f})", flush=True)
            vw = cv2.VideoWriter(str(args.out / f"sample_{W}x{H}_k{k}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 4, (W, H))
            for f in frames:
                vw.write(f)
            vw.release()

    print("\n==== LIVE RENDER BENCHMARK ====")
    print(f"{'config':>14} {'s/step':>7} {'raster':>7} {'diffusion':>9} {'decode':>7}")
    for r in rows:
        print(f"{r['W']}x{r['H']} k={r['k']:>2} {r['mean']:7.2f} {r['raster']:7.2f} "
              f"{r['diff']:9.2f} {r['dec']:7.2f}")
    print("\nbudget math: 50k fine-tune steps at X s/step = X*13.9 hours; "
          "200k pure run = X*55.6 hours")


if __name__ == "__main__":
    main()
