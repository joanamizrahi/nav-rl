"""Camera paths for the supplementary video (2026-11-17).

Builds smooth 81-frame camera paths in the NAV frame (metres), converts every pose to the reconstruction frame
with the training environment's own math (CalibratedRealWorldBackend._pose_nav_to_recon), and measures the
paper's coverage C_w (coverage window, default 21) at every pose. Rasterizer only: no generation, a few seconds
per path. Each path is written as an inference_semantic --trajectory_file JSON (mode=global, recon frame), so the
world model renders the SAME poses the coverage was measured at.

Path kinds (one or more per call):
  lat   from recorded frame B, slide sideways 0 -> D metres (sign +1 = robot's left), heading kept
  turn  at recorded frame B, turn in place 0 -> D degrees (sign +1 = left)
  xy    resample a nav-frame waypoint list [[x, y, yaw_rad], ...] (JSON file) to 81 poses by arc length
Motion is eased (slow start and stop).

Writes to --out_dir:
  <name>.json        trajectory file for inference_semantic.py --trajectory_file
  <name>.csv         frame, x, y, yaw_deg, t_idx, coverage
  coverage.png       C_w along every path, with tau = 0.4 and tau_term = 0.1
  map.png            top-down: recorded path (grey), every path (colour), start dots

    python scripts/video_paths.py --scene quad2_00 --clips_dir <clips> --poses_dir <poses> --labels_dir <v14 labels> \
        --out_dir <out> --lat 42:+1:9 42:-1:9 --turn 42:+1:180
"""
from __future__ import annotations
import os as _os
_S = _os.environ.get("SCRATCH_ROOT", "/scratch/m000204-pm07b/joana")

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))
from src.env.real_calibrated import CalibratedBackendConfig, CalibratedRealWorldBackend  # noqa: E402
from withheld_render import pose_nav_from_xy_yaw  # noqa: E402

TAU, TAU_TERM = 0.4, 0.1


def ease(n):
    """n values 0..1 with a smooth start and stop (cosine)."""
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, n))


def lat_path(pos, yaw_rec, b, sign, dist, n):
    x0, y0, yw = float(pos[b, 0]), float(pos[b, 1]), float(yaw_rec[b])
    s = ease(n) * dist * sign
    return np.stack([x0 - s * math.sin(yw), y0 + s * math.cos(yw), np.full(n, yw)], 1)


def turn_path(pos, yaw_rec, b, sign, deg, n):
    x0, y0, yw = float(pos[b, 0]), float(pos[b, 1]), float(yaw_rec[b])
    return np.stack([np.full(n, x0), np.full(n, y0), yw + sign * math.radians(deg) * ease(n)], 1)


def xy_path(wps, n):
    """Waypoints [[x, y, yaw], ...] resampled to n poses, evenly spaced by arc length; yaw interpolated unwrapped."""
    w = np.asarray(wps, dtype=np.float64)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(w[:, :2], axis=0), axis=1))]
    if seg[-1] < 1e-6:
        seg = np.linspace(0.0, 1.0, len(w))
    s = np.linspace(0.0, seg[-1], n)
    yaw = np.unwrap(w[:, 2])
    return np.stack([np.interp(s, seg, w[:, 0]), np.interp(s, seg, w[:, 1]), np.interp(s, seg, yaw)], 1)


def parse_spec(spec):
    b, sgn, d = spec.split(":")
    return int(b), float(sgn), float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", required=True); ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default=_S + "/NeoVerse/models")
    ap.add_argument("--reconstructor_path", default=_S + "/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--lat", nargs="*", default=[], help="B:SIGN:METRES, e.g. 42:+1:9")
    ap.add_argument("--turn", nargs="*", default=[], help="B:SIGN:DEGREES, e.g. 42:+1:180")
    ap.add_argument("--xy", nargs="*", default=[], help="JSON files of nav waypoints [[x, y, yaw_rad], ...]")
    ap.add_argument("--coverage_window", type=int, default=21)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--static_movers", default="12,13")
    ap.add_argument("--width", type=int, default=560); ap.add_argument("--height", type=int, default=336)
    args = ap.parse_args()
    import cv2

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only", model_path=args.model_path,
        reconstructor_path=args.reconstructor_path, static_scene=True, static_movers=str(args.static_movers or ""),
        H=args.height, W=args.width, num_frames=int(args.num_frames), coverage_window=int(args.coverage_window))
    world = CalibratedRealWorldBackend(cfg)
    world.load_scene(args.scene)
    sc = world._cache[args.scene]; cal = world._calib[args.scene]
    pos = np.asarray(cal.positions)[:, :2]
    yaw_rec = np.arctan2(cal.headings[:, 1], cal.headings[:, 0])
    n = int(args.num_frames)

    paths = []
    for spec in args.lat:
        b, sg, d = parse_spec(spec); paths.append((f"{args.scene}_lat_f{b}_{'L' if sg > 0 else 'R'}{d:g}m", lat_path(pos, yaw_rec, b, sg, d, n)))
    for spec in args.turn:
        b, sg, d = parse_spec(spec); paths.append((f"{args.scene}_turn_f{b}_{'L' if sg > 0 else 'R'}{d:g}deg", turn_path(pos, yaw_rec, b, sg, d, n)))
    for fpath in args.xy:
        paths.append((f"{args.scene}_xy_{Path(fpath).stem}", xy_path(json.load(open(fpath)), n)))
    assert paths, "give at least one of --lat / --turn / --xy"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = {}
    for name, P in paths:
        mats, tids, covs = [], [], []
        for x, y, yw in P:
            pose_recon, t_idx = world._pose_nav_to_recon(pose_nav_from_xy_yaw(float(x), float(y), float(yw)))
            w2c, K1, ts1 = world._single_frame_inputs(sc, pose_recon, t_idx)
            cov = world._window_coverage(sc, w2c[0], K1[0], ts1[0], t_idx)   # one view, as live_backend passes it
            mats.append(np.asarray(pose_recon, dtype=np.float64)); tids.append(int(t_idx)); covs.append(float(cov))
        with open(args.out_dir / f"{name}.json", "w") as fh:
            json.dump({"mode": "global", "num_frames": n, "name": name,
                       "trajectory": {"frame_indices": tids, "frame_matrices": np.stack(mats).tolist()}}, fh)
        with open(args.out_dir / f"{name}.csv", "w") as fh:
            fh.write("frame,x,y,yaw_deg,t_idx,coverage\n")
            for i, ((x, y, yw), t, c) in enumerate(zip(P, tids, covs)):
                fh.write(f"{i},{x:.3f},{y:.3f},{math.degrees(yw):.1f},{t},{c:.4f}\n")
        c = np.array(covs); curves[name] = c
        first = lambda thr: next((i for i, v in enumerate(c) if v < thr), None)
        print(f"[paths] {name}: C_w {c[0]:.2f} -> {c[-1]:.2f} (min {c.min():.2f}); "
              f"first below tau at frame {first(TAU)}, below tau_term at frame {first(TAU_TERM)}", flush=True)

    # coverage curves
    Wc, Hc, pad = 900, 420, 50
    img = np.full((Hc, Wc, 3), 255, np.uint8)
    X = lambda i: int(pad + i / (n - 1) * (Wc - 2 * pad)); Y = lambda v: int(Hc - pad - v * (Hc - 2 * pad))
    for thr, col in ((TAU, (0, 140, 255)), (TAU_TERM, (0, 0, 220))):
        cv2.line(img, (pad, Y(thr)), (Wc - pad, Y(thr)), col, 1)
    cv2.rectangle(img, (pad, Y(1.0)), (Wc - pad, Y(0.0)), (120, 120, 120), 1)
    rng = np.random.default_rng(0)
    for k, (name, c) in enumerate(curves.items()):
        col = tuple(int(v) for v in rng.integers(0, 200, 3))
        pts = np.array([[X(i), Y(v)] for i, v in enumerate(c)], np.int32)
        cv2.polylines(img, [pts], False, col, 2)
        cv2.putText(img, name, (pad + 5, pad + 16 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    cv2.putText(img, "C_w per frame (orange: tau 0.4, red: tau_term 0.1)", (pad, Hc - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.imwrite(str(args.out_dir / "coverage.png"), img)

    # top-down map
    allxy = np.concatenate([pos] + [P[:, :2] for _, P in paths])
    lo, hi = allxy.min(0) - 1.0, allxy.max(0) + 1.0; S = 700 / max(hi - lo)
    M = lambda p: (int(30 + (p[0] - lo[0]) * S), int(730 - (p[1] - lo[1]) * S))
    mp = np.full((760, 760, 3), 255, np.uint8)
    for i in range(len(pos) - 1):
        cv2.line(mp, M(pos[i]), M(pos[i + 1]), (170, 170, 170), 3)
    for i in range(0, len(pos), 10):
        cv2.putText(mp, str(i), M(pos[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    rng = np.random.default_rng(0)
    for name, P in paths:
        col = tuple(int(v) for v in rng.integers(0, 200, 3))
        for i in range(len(P) - 1):
            cv2.line(mp, M(P[i]), M(P[i + 1]), col, 2)
        cv2.circle(mp, M(P[0]), 5, col, -1)
        for i in range(0, len(P), 20):
            e = (P[i, 0] + 0.6 * math.cos(P[i, 2]), P[i, 1] + 0.6 * math.sin(P[i, 2]))
            cv2.line(mp, M(P[i]), M(e), col, 1)
    cv2.imwrite(str(args.out_dir / "map.png"), mp)
    print(f"[paths] wrote {len(paths)} paths to {args.out_dir}")


if __name__ == "__main__":
    main()
