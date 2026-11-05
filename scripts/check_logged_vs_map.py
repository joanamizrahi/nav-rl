"""Which recorded pose does the eval's logged collision fraction belong to?

For every CRASH episode of an eval, recompute the crash box on the map at the
last three recorded poses and print them next to the logged fraction. The
row that matches is the reward pose. 2026-09-04: on gnd_AUw360 ep 15 logged
0.36 while the near box at traj[-2] read 0.00 -- this script says where the
0.36 came from.

    python scripts/check_logged_vs_map.py --metrics <eval dir>/metrics.json \
        --scene gnd_AUw360 --collahead 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid, footprint_samples  # noqa: E402
from src.eval.traversability import load_traversability  # noqa: E402

BODY_L, BODY_W = 0.6, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--collahead", type=float, default=0.6)
    ap.add_argument("--inflate", type=float, default=0.1)
    ap.add_argument("--inflate_classes", default="")
    ap.add_argument("--fill", type=float, default=0.3)
    ap.add_argument("--fill_area", type=float, default=10.0)
    ap.add_argument("--walk", type=float, default=0.4)
    ap.add_argument("--outcome", default="CRASH", help="episodes to check; ALL for every episode")
    args = ap.parse_args()

    scores = load_traversability(args.trav)
    nontrav = scores <= 0.1
    c = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    walk = (np.asarray(c["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
    g = build_label_grid(c["points"], c["labels"].astype(int), nontrav, res=0.1, inflate_m=args.inflate,
                         fill_m=args.fill, fill_max_area_m2=args.fill_area, walk_xy=walk, walk_halfwidth_m=args.walk,
                         inflate_classes=tuple(int(v) for v in args.inflate_classes.split(",") if v.strip()))

    def frac(pos, yaw, ahead):
        h = np.array([np.cos(yaw), np.sin(yaw)])
        fp = footprint_samples(pos + ahead * h, h, BODY_L, BODY_W, g.res / 2.0)
        cl = g.lookup(fp).astype(int)
        cl = np.where(cl < 0, 0, cl)
        cnt = np.bincount(cl, minlength=14)
        top = ",".join(f"{k}x{cnt[k]}" for k in np.argsort(-cnt)[:2] if cnt[k] > 0)
        return float((nontrav[cl] & (cl != 0)).mean()), top

    m = json.load(open(args.metrics))
    print(f"{'ep':>3} {'out':<7} {'row':>4} {'logged':>7} {'near':>6} {'far':>6}  {'yaw':>6}  classes(near)   "
          f"[near = box {args.collahead} m ahead; row -1 = pose AFTER the last action, -2 = reward pose]")
    for e in m["episodes"]:
        if args.outcome != "ALL" and e["outcome"] != args.outcome:
            continue
        tr = np.asarray(e["traj"], float)
        for k in (-3, -2, -1):
            if len(tr) + k < 0:
                continue
            r = tr[k]
            near, top = frac(r[:2], r[2], args.collahead)
            far, _ = frac(r[:2], r[2], 1.5)
            print(f"{e['episode']:>3} {e['outcome']:<7} {k:>4} {r[3]:>7.2f} {near:>6.2f} {far:>6.2f}  {np.degrees(r[2]):>6.0f}  {top}")
        print()


if __name__ == "__main__":
    main()
