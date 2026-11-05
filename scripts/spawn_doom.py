"""Doomed-spawn share per scene: how often the crash box is already >= 35%
non-traversable ON THE MAP at the spawn pose, before the policy acts.

Samples spawns the way the env does: a recorded frame in [lo, hi), heading =
walk direction at that frame (the recorded camera yaw is within ~30 deg of
it on the campus walks), then the fleet's jitter (yaw U(-jy, jy) deg,
lateral U(-jl, jl) m). Run it ON THE CLUSTER against the clouds the env
loads -- the 2026-09-04 sampler ran on an Aug-20 laptop copy of
gnd_AU_180 and said 0% where the eval showed 30%.

    python scripts/spawn_doom.py --scenes gnd_AU_180 gnd_AUd210 gnd_AUw210 \
        gnd_AUw360 gnd_AUw330 gnd_AUw60 --lo 15 --hi 70
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid, footprint_samples  # noqa: E402
from src.eval.traversability import load_traversability  # noqa: E402

BODY_L, BODY_W = 0.6, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--lo", type=int, default=15, help="first spawn frame (spawn_min_frame)")
    ap.add_argument("--hi", type=int, default=70, help="one past the last spawn frame")
    ap.add_argument("--yaw_jitter_deg", type=float, default=20.0)
    ap.add_argument("--lat_jitter_m", type=float, default=0.4)
    ap.add_argument("--crash_frac", type=float, default=0.35)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--inflate", type=float, default=0.1)
    ap.add_argument("--inflate_classes", default="")
    ap.add_argument("--fill", type=float, default=0.3)
    ap.add_argument("--fill_area", type=float, default=10.0)
    ap.add_argument("--walk", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scores = load_traversability(args.trav)
    nontrav = scores <= args.collision_threshold
    rng = np.random.default_rng(args.seed)
    icls = tuple(int(v) for v in args.inflate_classes.split(",") if v.strip())

    def frac(g, pos, yaw, ahead):
        h = np.array([np.cos(yaw), np.sin(yaw)])
        fp = footprint_samples(pos + ahead * h, h, BODY_L, BODY_W, g.res / 2.0)
        cl = g.lookup(fp).astype(int)
        cl = np.where(cl < 0, 0, cl)
        return float((nontrav[cl] & (cl != 0)).mean())

    print(f"doomed-spawn share: crash box >= {args.crash_frac:.2f} non-traversable on the map at spawn; "
          f"frames [{args.lo},{args.hi}), yaw +-{args.yaw_jitter_deg:.0f} deg, lateral +-{args.lat_jitter_m} m, "
          f"inflation {args.inflate}{' on classes ' + args.inflate_classes if icls else ''}, n={args.n}")
    print(f"  {'scene':<12} {'frames':>6} {'far 1.5 m':>9} {'near 0.6 m':>10} {'far, no jitter':>14} {'near, no jitter':>15}   cloud md5[:8]")
    for sc in args.scenes:
        p = Path(args.clouds_dir) / f"{sc}_cloud.npz"
        md5 = hashlib.md5(p.read_bytes()).hexdigest()[:8]
        c = np.load(p)
        P = (np.asarray(c["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
        g = build_label_grid(c["points"], c["labels"].astype(int), nontrav, res=args.res,
                             inflate_m=args.inflate, fill_m=args.fill, fill_max_area_m2=args.fill_area,
                             walk_xy=P, walk_halfwidth_m=args.walk, inflate_classes=icls)
        d = np.diff(P, axis=0)
        wd = np.append(np.arctan2(d[:, 1], d[:, 0]), 0.0)
        wd[-1] = wd[-2]
        hi = min(args.hi, len(P) - 6)
        res = {}
        for jit in (True, False):
            hits = {1.5: 0, 0.6: 0}
            for _ in range(args.n):
                f = int(rng.integers(args.lo, hi))
                yaw = wd[f] + (np.radians(rng.uniform(-args.yaw_jitter_deg, args.yaw_jitter_deg)) if jit else 0.0)
                h = np.array([np.cos(yaw), np.sin(yaw)])
                pos = P[f] + ((rng.uniform(-args.lat_jitter_m, args.lat_jitter_m) if jit else 0.0)
                              * np.array([-h[1], h[0]]))
                for ahead in hits:
                    hits[ahead] += frac(g, pos, yaw, ahead) >= args.crash_frac
            res[jit] = {k: v / args.n for k, v in hits.items()}
        print(f"  {sc:<12} {len(P):>6} {res[True][1.5]:>9.0%} {res[True][0.6]:>10.0%} "
              f"{res[False][1.5]:>14.0%} {res[False][0.6]:>15.0%}   {md5}")


if __name__ == "__main__":
    main()
