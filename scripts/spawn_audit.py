"""Do our spawns start the robot on terrain it isn't allowed to stand on?

Measured 2026-09-01 with pick_goal: on gnd_AUw360 the grass comes within
0.40 m of the recorded path. Training spawns use `sjl0.4` — up to 0.4 m of
lateral jitter — and the reward's footprint is a 0.7 x 0.31 m box centred
1.5 m AHEAD of the robot, so it reaches further off-path than the spawn point
does, especially with the +-20 deg yaw jitter turning it outward.

If the footprint is already >= collision_terminate_frac non-traversable at
step 0, the episode dies before the policy acts, and the crash tells us
nothing about the world model or the policy. That would explain "32% of
crashes at steps 0-2" without any hallucination involved.

This is a pure GEOMETRY check against the scene cloud's SAM3 labels — no
diffusion, no rendering, no GPU. It answers the question the crash snapshots
could not: was the ground genuinely non-traversable, or did we put the robot
there?

Usage (login node):
    python scripts/spawn_audit.py --scene gnd_AUw360
    python scripts/spawn_audit.py --scene gnd_AUw360 --sweep_lat 0.0,0.2,0.4,0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.reward_2d import (
    _footprint_corners_world, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)
from src.eval.traversability import load_traversability

V14 = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk", "road",
       "pavement", "stairs", "obstacle", "vegetation", "person", "vehicle"]


def in_quad(pts_xy: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Point-in-convex-quad: every edge cross-product agrees with the polygon's
    own winding.

    The winding MUST come from the quad (shoelace), not from a sample point.
    Taking it from the extreme point of edge 0 tests which side of that one
    edge the farthest point lies on, which is unrelated to orientation — and
    for clockwise quads it inverts the whole test, returning the empty
    exterior wedge. Caught 2026-09-01: it reported 79% of spawn footprints as
    sitting over unreconstructed ground.
    """
    q = np.asarray(quad)[:, :2]
    shoelace = float(sum(q[i, 0] * q[(i + 1) % 4, 1] - q[(i + 1) % 4, 0] * q[i, 1]
                         for i in range(4)))
    s = 1.0 if shoelace >= 0 else -1.0
    inside = np.ones(len(pts_xy), dtype=bool)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        cr = ((b[0] - a[0]) * (pts_xy[:, 1] - a[1])
              - (b[1] - a[1]) * (pts_xy[:, 0] - a[0]))
        inside &= (cr * s) >= 0
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav_path", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--lat_jitter", type=float, default=0.4)
    ap.add_argument("--yaw_jitter", type=float, default=20.0)
    ap.add_argument("--spawn_min", type=int, default=10)
    ap.add_argument("--look_ahead", type=float, default=1.5)
    ap.add_argument("--crash_frac", type=float, default=0.35)
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--z_max", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep_lat", default="",
                    help="comma list of lateral jitters to compare, e.g. 0,0.2,0.4")
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    cloud = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    pts, labs = cloud["points"], cloud["labels"].astype(int)
    # Same frame trap as pick_goal: cloud points are nav-frame (y-mirrored),
    # traj_positions is stored raw.
    path = np.asarray(cloud["traj_positions"], dtype=float) * np.array([1., -1., 1.])
    path = path[:, :2]

    ground = pts[:, 2] < args.z_max
    gxy, glab = pts[ground][:, :2], labs[ground]
    trav = load_traversability(Path(args.trav_path))
    nontrav = trav <= args.collision_threshold
    valid = (glab >= 0) & (glab < len(trav))
    gxy, glab = gxy[valid], glab[valid]
    print(f"{args.scene}: {len(gxy)} ground-level labelled points "
          f"(z < {args.z_max}), path {len(path)} frames")
    print(f"  trav table: {args.trav_path}")

    def audit(lat, rng):
        bad, fracs, spawns, doms = 0, [], [], {}
        n_empty = 0
        for _ in range(args.n):
            f = int(rng.integers(args.spawn_min, max(len(path) - 6,
                                                     args.spawn_min + 1)))
            fw = path[min(f + 1, len(path) - 1)] - path[max(f - 1, 0)]
            base = float(np.arctan2(fw[1], fw[0]))
            p = path[f] + rng.uniform(-1, 1) * lat * np.array(
                [-np.sin(base), np.cos(base)])
            yaw = base + np.deg2rad(rng.uniform(-1, 1) * args.yaw_jitter)
            hd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            quad = _footprint_corners_world(
                np.array([p[0], p[1], 0.0]), hd, look_ahead_dist=args.look_ahead,
                length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
            m = in_quad(gxy, quad)
            if not m.any():
                n_empty += 1
                continue
            cls = glab[m]
            frac = float(nontrav[cls].mean())
            fracs.append(frac)
            spawns.append((p[0], p[1], frac))
            if frac >= args.crash_frac:
                bad += 1
                top = V14[int(np.bincount(cls[nontrav[cls]],
                                          minlength=len(V14)).argmax())]
                doms[top] = doms.get(top, 0) + 1
        return bad, np.array(fracs), np.array(spawns), doms, n_empty

    lats = ([float(v) for v in args.sweep_lat.split(",")] if args.sweep_lat
            else [args.lat_jitter])
    last = None
    print(f"\n  lat_jitter   spawns   footprint non-trav (mean/p90)   "
          f"START-ON-BAD-GROUND")
    for lat in lats:
        rng = np.random.default_rng(args.seed)
        bad, fr, sp, doms, empty = audit(lat, rng)
        if len(fr) == 0:
            print(f"  {lat:9.2f}   no footprint had any cloud points under it")
            continue
        print(f"  {lat:9.2f}   {len(fr):5d}   {fr.mean():.3f} / "
              f"{np.percentile(fr, 90):.3f}              "
              f"{bad}/{len(fr)} = {100.0 * bad / len(fr):.1f}%"
              + (f"   [{', '.join(f'{k} {v}' for k, v in doms.items())}]" if doms else ""))
        if empty:
            print(f"              ({empty} spawns had no cloud points under the "
                  f"footprint — unreconstructed ground)")
        last = (lat, sp)

    print(f"\n  'START-ON-BAD-GROUND' = footprint already >= {args.crash_frac} "
          f"non-traversable at step 0,\n  i.e. the episode crashes before the "
          f"policy acts. Geometry only — no world model involved.")

    if args.png and last is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lat, sp = last
        fig, ax = plt.subplots(figsize=(9, 9))
        for cid, col, name in ((6, "0.75", "sidewalk"), (8, "0.75", None),
                               (3, "tab:green", "grass")):
            m = glab == cid
            if m.any():
                ax.scatter(gxy[m][::15, 0], gxy[m][::15, 1], s=1, c=col,
                           linewidths=0, label=name)
        ax.plot(path[:, 0], path[:, 1], "-", c="tab:blue", lw=2, label="recorded walk")
        ok, bad = sp[sp[:, 2] < args.crash_frac], sp[sp[:, 2] >= args.crash_frac]
        if len(ok):
            ax.scatter(ok[:, 0], ok[:, 1], s=4, c="k", alpha=0.25, label="spawn ok")
        if len(bad):
            ax.scatter(bad[:, 0], bad[:, 1], s=14, c="tab:red", label="spawn CRASHES at step 0")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{args.scene}: spawn footprints vs terrain "
                     f"(lat_jitter {lat} m, yaw {args.yaw_jitter} deg)")
        fig.savefig(args.png, dpi=130, bbox_inches="tight")
        print(f"\n==> {args.png}")


if __name__ == "__main__":
    main()
