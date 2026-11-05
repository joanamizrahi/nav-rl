"""Spawn-pose check (Joana, 2026-09-07: "the world model gets orientation
quite wrong sometimes; apart from the spawn poses it does not matter").

The env spawns at the RECORDED pose of a walk frame: position from the poses
file and heading from its recorded camera heading (cal.headings), then adds
the jitter. If a recorded heading is off, the robot spawns looking the wrong
way, the goal cone (centred on that heading) points the wrong way, and the
goal sampling along the walk shifts. This compares, per frame, the recorded
heading with the direction the walk actually moves (positions[f+1] -
positions[f-1]); both come from the same file so no calibration is needed.
Login node, numpy only.

    python scripts/spawn_pose_check.py --scenes gnd_AUd210 gnd_G2c2d330 ... [--flag_deg 30]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--poses_dir", default="/scratch/m000204-pm07b/joana/outputs/poses")
    ap.add_argument("--flag_deg", type=float, default=30.0)
    ap.add_argument("--min_step_m", type=float, default=0.05, help="frames moving less than this are skipped (standing)")
    ap.add_argument("--frames", default="", help="comma list of spawn frames to print in detail (empty = flagged only)")
    args = ap.parse_args()
    detail = set(int(v) for v in args.frames.split(",") if v.strip())
    print(f"{'scene':16s} {'moving':>6s} {'median|d|':>9s} {'p90|d|':>7s} {'>flag':>6s}  flagged frames (heading vs walk direction, deg)")
    for sc in args.scenes:
        p = Path(args.poses_dir) / f"{sc}_poses.npz"
        if not p.exists():
            print(f"{sc:16s} no poses"); continue
        d = np.load(p)
        pos = np.asarray(d["positions"], float)[:, :2]
        hd = np.asarray(d["headings"], float)[:, :2]
        n = len(pos); diffs = []; flagged = []
        for f in range(n):
            a, b = max(0, f - 1), min(n - 1, f + 1)
            mv = pos[b] - pos[a]
            if np.linalg.norm(mv) < args.min_step_m * (b - a):
                diffs.append(np.nan); continue
            h = hd[f] / (np.linalg.norm(hd[f]) + 1e-9); m = mv / (np.linalg.norm(mv) + 1e-9)
            ang = np.degrees(np.arctan2(h[0] * m[1] - h[1] * m[0], h[0] * m[0] + h[1] * m[1]))
            diffs.append(ang)
            if abs(ang) > args.flag_deg:
                flagged.append((f, ang))
        diffs = np.asarray(diffs, float); ok = ~np.isnan(diffs)
        med = float(np.nanmedian(np.abs(diffs))) if ok.any() else float("nan")
        p90 = float(np.nanpercentile(np.abs(diffs), 90)) if ok.any() else float("nan")
        fl = " ".join(f"{f}:{a:+.0f}" for f, a in flagged[:14]) + (" ..." if len(flagged) > 14 else "")
        print(f"{sc:16s} {int(ok.sum()):6d} {med:9.1f} {p90:7.1f} {len(flagged):6d}  {fl}")
        if detail:
            for f in sorted(detail):
                if f < n:
                    print(f"    frame {f:2d}: heading vs walk {diffs[f]:+.1f} deg" if not np.isnan(diffs[f]) else f"    frame {f:2d}: standing")
    print("Rule of thumb: |d| < 15 deg = fine; 15-30 = the jitter covers it; > 30 = the spawn looks the wrong way there -- keep spawns off those frames or drop the scene.")


if __name__ == "__main__":
    main()
