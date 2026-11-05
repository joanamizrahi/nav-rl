"""Are the goals we train on actually reachable, or is the task unwinnable?

`spawn_audit.py` answered the same question for the START of an episode. This
answers it for the END, and the motivation is a number: at 285k steps, arm A
(460539) reaches its goal in ~9.5% of episodes and arm B (460540) in 0%. Before
concluding anything about either policy, we have to know what fraction of goals
COULD be reached at all.

The goal sampler with `--goal_dir_360` (real_calibrated.py:376-393) is pure
geometry:

    d  ~ U(goal_dist_range)                       e.g. U(5, 10) m
    th = path tangent at the nearest frame + U(-cone/2, +cone/2)
    goal = spawn_xy + d * (cos th, sin th)

Nothing consults the cloud, the SAM3 labels, or the reconstruction extent. So a
goal can legitimately land:

  * on grass, where the traversability table scores 0.0 and the correct
    behaviour is to STOP at the boundary and never arrive;
  * inside a building or a hedge;
  * past the edge of the reconstruction, where there is no world to walk
    through -- the drive_preview sweep measured 0% coverage beyond +-105deg and
    the reconstructed wedge is only so deep.

This is pure GEOMETRY against the scene cloud -- no diffusion, no rendering, no
GPU. It reports, per scene, the share of sampled episodes whose goal is
off-cloud, on non-traversable terrain, or behind a non-traversable corridor,
and draws every one of them on a top-down map.

Frame convention follows spawn_audit/pick_goal: cloud points are nav-frame,
`traj_positions` is stored RAW, so the path is y-mirrored into nav frame and
spawn/goal/tangent are all computed there. Everything stays in one frame.

Usage (login node, no GPU):
    python scripts/goal_audit.py
    python scripts/goal_audit.py --goal_dist_range 3,10 --png_dir /scratch/.../goal_audit
    python scripts/goal_audit.py --scenes gnd_AUw360 --goal_cone_deg 50 --n 4000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.reward_2d import (
    _footprint_corners_world, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)
from src.eval.traversability import load_traversability
from scripts.spawn_audit import in_quad, V14

# The six the 2026-09-02 arms actually train on. sitex_* have a narrower
# camera (blind closer than 2.15 m) and gtown2c1_* / gnd_AUw240 are
# reconstruction failures -- all were dropped from training, so auditing them
# describes nothing that is running.
TRAIN_SCENES = ("gnd_AU_180", "gnd_AUd210", "gnd_AUw210", "gnd_AUw360",
                "gnd_AUw330", "gnd_AUw60")


class GroundGrid:
    """Uniform-cell spatial index over the ground points, pure numpy.

    The first version of this script tested every footprint against the WHOLE
    ground cloud -- a few million points -- once per 0.3 m of corridor, i.e.
    ~300k full scans. That is hours. The footprint is under a metre across, so
    all but a handful of those points can never be inside it.

    Points are bucketed into `cell`-metre squares and sorted by flat cell id
    (`i * ny + j`), so for a fixed column i the cells j0..j1 are CONTIGUOUS in
    the sorted order and a whole column of the query box is one slice. A query
    then gathers a few thousand candidates instead of millions, and the exact
    in_quad test runs on those. Same answer, ~100x less work.
    """

    def __init__(self, xy: np.ndarray, cell: float = 1.0):
        self.cell = float(cell)
        self._xy = xy
        self.min = xy.min(axis=0) - 1e-6
        ij = np.floor((xy - self.min) / self.cell).astype(np.int64)
        self.nx = int(ij[:, 0].max()) + 1
        self.ny = int(ij[:, 1].max()) + 1
        flat = ij[:, 0] * self.ny + ij[:, 1]
        self.order = np.argsort(flat, kind="stable")
        self.starts = np.searchsorted(flat[self.order],
                                      np.arange(self.nx * self.ny + 1))

    def candidates(self, lo_xy, hi_xy) -> np.ndarray:
        """Indices of every point whose cell overlaps the box. A superset of
        the true hits -- the caller still runs the exact test."""
        i0, j0 = np.floor((np.asarray(lo_xy) - self.min) / self.cell).astype(int)
        i1, j1 = np.floor((np.asarray(hi_xy) - self.min) / self.cell).astype(int)
        i0, i1 = max(i0, 0), min(i1, self.nx - 1)
        j0, j1 = max(j0, 0), min(j1, self.ny - 1)
        if i0 > i1 or j0 > j1:
            return np.empty(0, dtype=np.int64)
        chunks = []
        for i in range(i0, i1 + 1):
            a = self.starts[i * self.ny + j0]
            b = self.starts[i * self.ny + j1 + 1]
            if b > a:
                chunks.append(self.order[a:b])
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def within(self, centre, radius: float) -> np.ndarray:
        """Exact radius query, box-prefiltered."""
        c = np.asarray(centre, dtype=float)
        cand = self.candidates(c - radius, c + radius)
        if len(cand) == 0:
            return cand
        return cand[np.linalg.norm(self._xy[cand] - c, axis=1) <= radius]

# Verdicts, most-fatal first. An episode gets the FIRST one that applies, so the
# columns partition the sample rather than double-counting.
OFF_CLOUD = "off-cloud"          # no reconstruction under the goal at all
ON_BAD = "on-non-trav"           # goal sits on grass/obstacle/water
BLOCKED = "corridor-blocked"     # straight line crosses non-traversable ground
REACHABLE = "reachable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=list(TRAIN_SCENES))
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav_path", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--goal_dist_range", default="3,8",
                    help="matches --goal_dist_range in training (the 09-02 arms use 3,8)")
    ap.add_argument("--goal_cone_deg", type=float, default=50.0)
    ap.add_argument("--goal_radius", type=float, default=0.5,
                    help="arrival radius; terrain under the goal is judged inside it")
    ap.add_argument("--lat_jitter", type=float, default=0.4)
    ap.add_argument("--yaw_jitter", type=float, default=20.0)
    ap.add_argument("--old_cone", action="store_true",
                    help="audit the PRE-FIX sampler (cone on the nearest-frame "
                         "tangent). The before/after bearing table is printed "
                         "either way; this only changes which goals are then "
                         "checked for reachability.")
    ap.add_argument("--spawn_min", type=int, default=10)
    ap.add_argument("--look_ahead", type=float, default=1.5)
    ap.add_argument("--crash_frac", type=float, default=0.35)
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--step_m", type=float, default=0.3,
                    help="corridor sampling interval; matches training's step_size_m")
    ap.add_argument("--z_max", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goal_support_radius", type=float, default=0.6,
                    help="apply the training support check: resample goals with "
                         "fewer than min_frac of the recorded path's local "
                         "ground-point density. 0 = audit raw draws.")
    ap.add_argument("--goal_support_min_frac", type=float, default=0.25)
    ap.add_argument("--goal_support_tries", type=int, default=12)
    ap.add_argument("--png_dir", default="")
    args = ap.parse_args()

    lo_d, hi_d = (float(v) for v in args.goal_dist_range.split(","))
    trav = load_traversability(Path(args.trav_path))
    nontrav = trav <= args.collision_threshold

    print(f"goal sampler: d ~ U({lo_d}, {hi_d}) m, "
          f"theta = tangent +- {args.goal_cone_deg / 2.0:.0f}deg, "
          f"arrival radius {args.goal_radius} m")
    print(f"trav table: {args.trav_path}  "
          f"(non-traversable = score <= {args.collision_threshold})")
    print(f"{args.n} sampled episodes per scene, seed {args.seed}\n")

    hdr = (f"{'scene':<16}{'off-cloud':>11}{'on-non-trav':>13}"
           f"{'blocked':>10}{'REACHABLE':>11}   dominant class under bad goals")
    print(hdr)
    print("-" * len(hdr))

    totals = {k: 0 for k in (OFF_CLOUD, ON_BAD, BLOCKED, REACHABLE)}
    bearing_tot = {"old": [], "new": []}
    for scene in args.scenes:
        cloud_path = Path(args.clouds_dir) / f"{scene}_cloud.npz"
        if not cloud_path.exists():
            print(f"{scene:<16}  no cloud at {cloud_path}")
            continue
        cloud = np.load(cloud_path)
        pts, labs = cloud["points"], cloud["labels"].astype(int)
        path = (np.asarray(cloud["traj_positions"], dtype=float)
                * np.array([1.0, -1.0, 1.0]))[:, :2]

        ground = pts[:, 2] < args.z_max
        gxy, glab = pts[ground][:, :2], labs[ground]
        valid = (glab >= 0) & (glab < len(trav))
        gxy, glab = gxy[valid], glab[valid]
        if len(gxy) == 0:
            print(f"{scene:<16}  no ground-level labelled points")
            continue

        # Progress goes to stderr so the table on stdout stays pasteable.
        print(f"  [{scene}] {len(gxy)} ground pts, indexing...",
              file=sys.stderr, flush=True)
        t0 = time.time()
        grid = GroundGrid(gxy, cell=1.0)
        # Reference density on the recorded path -- the same self-calibration
        # SceneEnv uses. "Enough reconstruction" is measured against places we
        # KNOW are reconstructed, not against a constant.
        sup_r = args.goal_support_radius
        sup_need = 0
        if sup_r > 0.0:
            cnts = [int((np.abs(gxy - q).max(axis=1) <= sup_r).sum())
                    for q in path[::max(1, len(path) // 40)]]
            ref = float(np.median(cnts)) if cnts else 0.0
            sup_need = max(1, int(args.goal_support_min_frac * ref))
            print(f"  [{scene}] support reference {ref:.0f} pts within "
                  f"{sup_r} m on the recorded path -> goals need >= "
                  f"{sup_need}", file=sys.stderr, flush=True)
        n_resampled = 0
        d_raw, d_kept = [], []
        rng = np.random.default_rng(args.seed)
        counts = {k: 0 for k in (OFF_CLOUD, ON_BAD, BLOCKED, REACHABLE)}
        bearing = {"old": [], "new": []}
        doms: dict[str, int] = {}
        drawn = []

        for _ in range(args.n):
            f = int(rng.integers(args.spawn_min,
                                 max(len(path) - 6, args.spawn_min + 1)))
            fw = path[min(f + 1, len(path) - 1)] - path[max(f - 1, 0)]
            base = float(np.arctan2(fw[1], fw[0]))
            spawn = path[f] + rng.uniform(-1, 1) * args.lat_jitter * np.array(
                [-np.sin(base), np.cos(base)])

            # --- the robot's ACTUAL heading at reset: spawn-frame tangent
            # plus the same yaw jitter the env applies (real_calibrated.py:310).
            yaw = base + np.deg2rad(float(rng.uniform(-1, 1)) * args.yaw_jitter)

            # --- goal. Two cone centres, drawn from the SAME d and the SAME
            # offset so the comparison is paired:
            #   OLD  the path tangent at the frame NEAREST THE SPAWN. Lateral
            #        jitter can move that frame off the spawn frame, and where
            #        the walk turns the two tangents disagree by up to 150 deg
            #        -- the goal then sits BEHIND a forward-only robot.
            #   NEW  the RECORDED heading at the spawn frame -- the same
            #        frame the spawn came from, so the two cannot disagree.
            #        The cone stays fixed and the yaw jitter moves the robot
            #        INSIDE it, which bounds the error at cone/2 + jitter.
            i = int(np.argmin(np.linalg.norm(path - spawn, axis=1)))
            gfw = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
            gbase = float(np.arctan2(gfw[1], gfw[0]))
            half = np.deg2rad(args.goal_cone_deg) / 2.0
            off = float(rng.uniform(-half, half))
            d = float(rng.uniform(lo_d, hi_d))
            for _nm, _b in (("old", gbase), ("new", base)):
                _g = spawn + d * np.array([np.cos(_b + off), np.sin(_b + off)])
                _e = np.arctan2(*(_g - spawn)[::-1]) - yaw
                bearing[_nm].append(
                    abs(np.degrees((_e + np.pi) % (2 * np.pi) - np.pi)))
            cone_base = gbase if args.old_cone else base
            th = cone_base + off
            goal = spawn + d * np.array([np.cos(th), np.sin(th)])
            d_raw.append(d)

            # --- the training support check, applied here so its EFFECT is
            # visible: does requiring reconstruction quietly bias goals closer,
            # or away from grass? Neither should happen -- support is about
            # whether the place EXISTS, not about distance or walkability.
            if sup_need:
                for _t in range(args.goal_support_tries):
                    if len(grid.within(goal, sup_r)) >= sup_need:
                        break
                    n_resampled += 1
                    d = float(rng.uniform(lo_d, hi_d))
                    th = cone_base + float(rng.uniform(-half, half))
                    goal = spawn + d * np.array([np.cos(th), np.sin(th)])
            d_kept.append(float(np.linalg.norm(goal - spawn)))

            near = grid.within(goal, args.goal_radius)
            if len(near) == 0:
                counts[OFF_CLOUD] += 1
                drawn.append((spawn[0], spawn[1], goal[0], goal[1], OFF_CLOUD))
                continue
            gcls = glab[near]
            if float(nontrav[gcls].mean()) >= args.crash_frac:
                counts[ON_BAD] += 1
                worst = V14[int(np.bincount(gcls[nontrav[gcls]],
                                            minlength=len(V14)).argmax())]
                doms[worst] = doms.get(worst, 0) + 1
                drawn.append((spawn[0], spawn[1], goal[0], goal[1], ON_BAD))
                continue

            # --- corridor: walk the straight line, footprint at each step.
            # This is the OPTIMISTIC test -- a real policy may need to detour,
            # so a blocked straight line means unreachable-by-the-simplest-route,
            # not strictly unreachable.
            hd_ang = np.arctan2(goal[1] - spawn[1], goal[0] - spawn[0])
            hd = np.array([np.cos(hd_ang), np.sin(hd_ang), 0.0])
            blocked = False
            for t in np.arange(0.0, d, args.step_m):
                p = spawn + t * hd[:2]
                quad = _footprint_corners_world(
                    np.array([p[0], p[1], 0.0]), hd,
                    look_ahead_dist=args.look_ahead,
                    length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
                q = np.asarray(quad)[:, :2]
                cand = grid.candidates(q.min(axis=0), q.max(axis=0))
                if len(cand) == 0:
                    continue
                m = in_quad(gxy[cand], quad)
                if m.any() and float(
                        nontrav[glab[cand][m]].mean()) >= args.crash_frac:
                    blocked = True
                    break
            key = BLOCKED if blocked else REACHABLE
            counts[key] += 1
            drawn.append((spawn[0], spawn[1], goal[0], goal[1], key))

        print(f"  [{scene}] {args.n} episodes in {time.time() - t0:.1f}s",
              file=sys.stderr, flush=True)
        # THE TEST FOR THE CONE FIX. |bearing to goal - robot heading| at
        # reset. With the cone centred on the heading this is bounded by
        # cone/2 BY CONSTRUCTION, so `max` must not exceed it and `behind`
        # must be 0. Any nonzero "behind" is an episode a forward-only robot
        # cannot win: it can only time out.
        for nm in ("old", "new"):
            b = np.asarray(bearing[nm])
            if not len(b):
                continue
            bearing_tot[nm].append(b)
            print(f"  [{scene}] cone {nm:<3} |goal bearing - heading|: "
                  f"mean {b.mean():5.1f}  p95 {np.percentile(b, 95):5.1f}  "
                  f"max {b.max():6.1f} deg | "
                  f">90deg (BEHIND) {100.0 * (b > 90).mean():5.2f}%  "
                  f">bound {100.0 * (b > args.goal_cone_deg / 2 + args.yaw_jitter + 1e-6).mean():5.2f}%",
                  file=sys.stderr, flush=True)
        if sup_need and d_raw:
            print(f"  [{scene}] support resampled {n_resampled} draws; "
                  f"distance mean {np.mean(d_raw):.2f} -> "
                  f"{np.mean(d_kept):.2f} m", file=sys.stderr, flush=True)
        tot = sum(counts.values())
        for k in counts:
            totals[k] += counts[k]
        top = ", ".join(f"{k} {v}" for k, v in
                        sorted(doms.items(), key=lambda kv: -kv[1])[:3])
        print(f"{scene:<16}{100.0 * counts[OFF_CLOUD] / tot:10.1f}%"
              f"{100.0 * counts[ON_BAD] / tot:12.1f}%"
              f"{100.0 * counts[BLOCKED] / tot:9.1f}%"
              f"{100.0 * counts[REACHABLE] / tot:10.1f}%   {top}")

        if args.png_dir:
            draw_scene(args, scene, gxy, glab, path, drawn)

    for nm in ("old", "new"):
        if bearing_tot[nm]:
            b = np.concatenate(bearing_tot[nm])
            print(f"ALL SCENES  cone {nm:<3} |goal bearing - heading|: "
                  f"mean {b.mean():5.1f}  max {b.max():6.1f} deg | "
                  f">90deg (BEHIND) {100.0 * (b > 90).mean():5.2f}%",
                  file=sys.stderr, flush=True)

    tot = sum(totals.values())
    if tot:
        print("-" * len(hdr))
        print(f"{'ALL SCENES':<16}{100.0 * totals[OFF_CLOUD] / tot:10.1f}%"
              f"{100.0 * totals[ON_BAD] / tot:12.1f}%"
              f"{100.0 * totals[BLOCKED] / tot:9.1f}%"
              f"{100.0 * totals[REACHABLE] / tot:10.1f}%")
        print(f"\nREACHABLE is a CEILING on the goal-reaching rate, not a "
              f"target: it assumes\na straight run with no detour and perfect "
              f"perception. A policy scoring well below\nit may still be fine; "
              f"a policy cannot score above it.")


def draw_scene(args, scene, gxy, glab, path, drawn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = Path(args.png_dir)
    od.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    for cid, col, name in ((6, "0.8", "sidewalk"), (8, "0.8", None),
                           (7, "0.65", "road"), (3, "tab:green", "grass"),
                           (10, "tab:brown", "obstacle")):
        m = glab == cid
        if m.any():
            ax.scatter(gxy[m][::15, 0], gxy[m][::15, 1], s=1, c=col,
                       linewidths=0, label=name)
    ax.plot(path[:, 0], path[:, 1], "-", c="tab:blue", lw=2.5,
            label="recorded walk")

    style = {REACHABLE: ("k", 4, 0.25), BLOCKED: ("tab:orange", 10, 0.7),
             ON_BAD: ("tab:red", 12, 0.8), OFF_CLOUD: ("tab:purple", 10, 0.6)}
    arr = np.array([(sx, sy, gx, gy) for sx, sy, gx, gy, _ in drawn],
                   dtype=float)
    kinds = np.array([k for *_, k in drawn])
    for k, (col, size, alpha) in style.items():
        m = kinds == k
        if m.any():
            ax.scatter(arr[m, 2], arr[m, 3], s=size, c=col, alpha=alpha,
                       linewidths=0, label=f"goal {k} ({int(m.sum())})")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{scene}: sampled GOALS vs terrain\n"
                 f"d ~ U({args.goal_dist_range}) m, cone {args.goal_cone_deg:.0f}deg, "
                 f"arrival radius {args.goal_radius} m")
    out = od / f"goals_{scene}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    ==> {out}")

    draw_cones(args, scene, gxy, glab, path, drawn, od)


def draw_cones(args, scene, gxy, glab, path, drawn, od, k_spawns=6):
    """The pooled scatter hides the structure: goals are NOT one cloud, they are
    one fan per spawn (Joana, 2026-09-02 -- "goals on cone shouldn't mix between
    spawn points"). Each episode draws its goal within +-cone/2 of the tangent
    at ITS OWN spawn, so the honest picture is a handful of spawns with their
    individual fans and a line from each spawn to each of its goals.

    Spawns are grouped by nearest of `k_spawns` evenly spaced path frames, and
    only those groups are drawn -- a subsample of the same episodes the table
    counted, not a re-sampling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array([(sx, sy, gx, gy) for sx, sy, gx, gy, _ in drawn], dtype=float)
    kinds = np.array([k for *_, k in drawn])
    if len(arr) == 0:
        return

    anchors_i = np.linspace(args.spawn_min, len(path) - 7, k_spawns).astype(int)
    anchors = path[anchors_i]
    # nearest anchor per episode, and keep only episodes whose spawn is close to
    # one -- otherwise every episode joins some anchor and the fans overlap again
    d_anchor = np.linalg.norm(arr[:, None, :2] - anchors[None, :, :], axis=2)
    which = d_anchor.argmin(axis=1)
    close = d_anchor.min(axis=1) <= 1.0

    fig, ax = plt.subplots(figsize=(11, 11))
    for cid, col, name in ((6, "0.85", "sidewalk"), (8, "0.85", None),
                           (7, "0.72", "road"), (3, "tab:green", "grass"),
                           (10, "tab:brown", "obstacle")):
        m = glab == cid
        if m.any():
            ax.scatter(gxy[m][::15, 0], gxy[m][::15, 1], s=1, c=col,
                       linewidths=0, label=name)
    ax.plot(path[:, 0], path[:, 1], "-", c="tab:blue", lw=2.5,
            label="recorded walk")

    seg_col = {REACHABLE: "k", BLOCKED: "tab:orange",
               ON_BAD: "tab:red", OFF_CLOUD: "tab:purple"}
    shown = 0
    for a in range(k_spawns):
        m = close & (which == a)
        if not m.any():
            continue
        sub = np.flatnonzero(m)
        # cap the fan so the lines stay readable
        if len(sub) > 40:
            sub = sub[np.linspace(0, len(sub) - 1, 40).astype(int)]
        for i in sub:
            ax.plot([arr[i, 0], arr[i, 2]], [arr[i, 1], arr[i, 3]],
                    "-", c=seg_col[kinds[i]], lw=0.6, alpha=0.55)
        ax.scatter(anchors[a, 0], anchors[a, 1], s=90, marker="*",
                   c="tab:cyan", edgecolors="k", linewidths=0.6, zorder=5)
        ax.annotate(f"spawn {anchors_i[a]}", anchors[a], fontsize=8,
                    xytext=(4, 4), textcoords="offset points", zorder=6)
        shown += len(sub)

    for k, c in seg_col.items():
        ax.plot([], [], "-", c=c, lw=1.4, label=f"goal {k}")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{scene}: goal cones, {k_spawns} spawns, {shown} episodes drawn\n"
                 f"each line is one episode: spawn -> its goal, "
                 f"cone {args.goal_cone_deg:.0f}deg about the tangent AT THAT SPAWN")
    out = od / f"cones_{scene}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    ==> {out}")


if __name__ == "__main__":
    main()
