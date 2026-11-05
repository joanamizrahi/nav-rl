"""Where did the policy actually GO? Episode trajectories on the terrain map.

eval_policy stores per-step [x, y, yaw, collision_frac, ground_class] for every
episode. Success rate collapses all of that into one bit — and when the goal
sits on grass, that bit is the WRONG question anyway (stopping at the boundary
is correct, arriving is a trespass). Drawing the paths over the scene cloud's
ground labels shows the behaviour directly: did it approach and stop, did it
walk onto the grass, or did it wander into unreconstructed space.

Overlaying two policies answers the lineage question in one picture — e.g.
459126 (warm-started, 200k steps under a reward where grass scored 0.75)
against 459127 (cold, J-spec only).

Usage (login node, no GPU):
    python scripts/viz_episodes.py --scene gnd_AUw360 --out eps.png \
        warm=/scratch/.../eval_..._warm_.../metrics.json \
        cold=/scratch/.../eval_..._cur_.../metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.eval.reward_2d import (
    _footprint_corners_world, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)

V14 = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk", "road",
       "pavement", "stairs", "obstacle", "vegetation", "person", "vehicle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=/path/to/metrics.json")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--z_max", type=float, default=0.15)
    ap.add_argument("--crash_frac", type=float, default=0.35)
    ap.add_argument("--goal_xy", default="",
                    help="'x,y'. If omitted it is parsed from the eval "
                         "directory name, which carries gxy<X>_<Y>.")
    ap.add_argument("--goal_radius", type=float, default=0.5)
    ap.add_argument("--look_ahead", type=float, default=1.5,
                    help="the reward scores a box this far AHEAD of the robot, "
                         "so the robot can stand on sidewalk while the graded "
                         "box is over grass. Drawing both is the only way to "
                         "tell which one a 'collision' actually came from.")
    ap.add_argument("--no_footprints", action="store_true")
    ap.add_argument("--out", default="episodes.png")
    ap.add_argument("--per_episode_dir", default="",
                    help="also write one plot PER EPISODE here — 20 overlaid "
                         "paths hide whether a single run approached and "
                         "stopped or wandered")
    args = ap.parse_args()

    cloud = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    pts, labs = cloud["points"], cloud["labels"].astype(int)
    # same nav-frame y-mirror trap as pick_goal / spawn_audit
    path = np.asarray(cloud["traj_positions"], dtype=float) * np.array([1., -1., 1.])
    path = path[:, :2]
    gnd = pts[:, 2] < args.z_max
    gxy, glab = pts[gnd][:, :2], labs[gnd]

    # The goal is the entire question ("did it stop SHORT of the grass?") so
    # it has to be on the picture. eval_policy encodes it in the output dir
    # name: ..._gxy10.75_-16.00_gnd_AUw360_live
    goal = None
    if args.goal_xy:
        goal = [float(v) for v in args.goal_xy.split(",")]
    else:
        for spec in args.runs:
            m = re.search(r"gxy(-?[\d.]+)_(-?[\d.]+)", spec)
            if m:
                goal = [float(m.group(1)), float(m.group(2))]
                break
    print(f"goal: {goal if goal else 'UNKNOWN — pass --goal_xy'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    for cid, col, name in ((6, "0.82", "sidewalk"), (8, "0.82", None),
                           (7, "0.68", "road"), (3, "tab:green", "grass")):
        m = glab == cid
        if m.any():
            ax.scatter(gxy[m][::12, 0], gxy[m][::12, 1], s=1, c=col,
                       linewidths=0, label=name, zorder=1)
    ax.plot(path[:, 0], path[:, 1], "-", c="tab:blue", lw=2.5,
            label="recorded walk", zorder=3)

    colors = ["tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    for k, spec in enumerate(args.runs):
        label, _, p = spec.partition("=")
        d = json.loads(Path(p).read_text())
        eps = d["episodes"]
        c = colors[k % len(colors)]
        n_cr = 0          # NB: do NOT rebind `goal` here — a leftover
                          # `goal = None` in this loop silently erased the
                          # parsed goal before anything was drawn (2026-09-01)
        for i, e in enumerate(eps):
            # eval_policy seeds traj with a 4-element row (x,y,yaw,0) then
            # appends 5-element rows (x,y,yaw,frac,ground_class), so the list
            # is RAGGED and np.asarray refuses it. Read the columns we need
            # defensively instead of assuming a rectangle.
            raw = e.get("traj") or []
            if len(raw) < 2:
                continue
            t = np.array([[r[0], r[1], r[2] if len(r) > 2 else 0.0,
                           r[3] if len(r) > 3 else 0.0] for r in raw],
                         dtype=float)
            ax.plot(t[:, 0], t[:, 1], "-", c=c, lw=1.2, alpha=0.65, zorder=4,
                    label=label if i == 0 else None)
            ax.plot(t[0, 0], t[0, 1], "o", c=c, ms=4, mec="k", mew=0.4, zorder=5)
            ax.plot(t[-1, 0], t[-1, 1], "s", c=c, ms=6, mec="k", mew=0.6, zorder=5)
            # PER-EPISODE goal (2026-09-02). --goal_xy draws one fixed goal,
            # which is right for a pinned-goal eval and wrong for the training
            # distribution, where every episode draws its own goal in a cone at
            # a random distance. eval_policy already records it per episode; a
            # path is unreadable without knowing where it was TRYING to go.
            g = e.get("goal_xy")
            if g and len(g) >= 2:
                ax.plot(g[0], g[1], "*", c=c, ms=11, mec="k", mew=0.5,
                        zorder=7, alpha=0.9,
                        label=f"{label} goals" if i == 0 else None)
                ax.plot([t[-1, 0], g[0]], [t[-1, 1], g[1]], ":", c=c, lw=0.7,
                        alpha=0.45, zorder=3)
                ax.add_patch(plt.Circle((g[0], g[1]), args.goal_radius,
                                        fill=False, ec=c, lw=0.7, alpha=0.5,
                                        zorder=6))
                # outcome at a glance, on the endpoint
                oc = (e.get("outcome") or "")[:1]
                if oc:
                    ax.annotate(oc, (t[-1, 0], t[-1, 1]), fontsize=6,
                                color="k", zorder=9,
                                xytext=(3, 3), textcoords="offset points")
            if t.shape[1] > 3:
                bad = t[t[:, 3] >= args.crash_frac]
                n_cr += len(bad)
                if len(bad):
                    ax.plot(bad[:, 0], bad[:, 1], "x", c="red", ms=7, mew=1.6,
                            zorder=6)
        s = d["summary"]
        gs = s.get("ground_share", {})
        top = ", ".join(f"{k2} {v:.0%}" for k2, v in
                        sorted(gs.items(), key=lambda kv: -kv[1])[:3])
        print(f"{label:<8} eps {len(eps):2d}  success {s.get('success_rate')}  "
              f"return {s.get('mean_return')}  crash-steps drawn {n_cr}  [{top}]")

    if goal is not None:
        ax.plot(goal[0], goal[1], "*", c="gold", ms=26, mec="k", mew=1.0,
                zorder=8, label="goal (on grass)")
        ax.add_patch(plt.Circle(tuple(goal), args.goal_radius, fill=False,
                                ec="gold", ls="--", lw=1.5, zorder=8))
    ax.plot([], [], "x", c="red", ms=7, mew=1.6,
            label=f"footprint >= {args.crash_frac} non-traversable")
    ax.plot([], [], "o", c="0.3", ms=4, label="spawn")
    ax.plot([], [], "s", c="0.3", ms=6, label="final pose")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(f"{args.scene}: where the policies went\n"
                 f"(goal on grass — stopping at the boundary is the CORRECT "
                 f"outcome, arriving is a trespass)", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"==> {args.out}")

    if args.per_episode_dir:
        od = Path(args.per_episode_dir)
        od.mkdir(parents=True, exist_ok=True)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        n = 0
        for k, spec in enumerate(args.runs):
            label, _, p2 = spec.partition("=")
            d = json.loads(Path(p2).read_text())
            # max_steps is not stored; the longest episode is the cap in
            # practice, since only a timeout runs the clock out.
            max_steps = max(int(e.get("steps", 0)) for e in d["episodes"])
            for i, e in enumerate(d["episodes"]):
                raw = e.get("traj") or []
                if len(raw) < 2:
                    continue
                tt = np.array([[r[0], r[1], r[3] if len(r) > 3 else 0.0]
                               for r in raw], dtype=float)
                f2, a2 = plt.subplots(figsize=(7, 7))
                for cid, col in ((6, "0.82"), (8, "0.82"), (7, "0.68"),
                                 (3, "tab:green")):
                    m = glab == cid
                    if m.any():
                        a2.scatter(gxy[m][::12, 0], gxy[m][::12, 1], s=1, c=col,
                                   linewidths=0)
                a2.plot(path[:, 0], path[:, 1], "-", c="tab:blue", lw=2)
                if goal is not None:
                    a2.plot(goal[0], goal[1], "*", c="gold", ms=22, mec="k",
                            mew=1.0, zorder=8)
                    a2.add_patch(plt.Circle(tuple(goal), args.goal_radius,
                                            fill=False, ec="gold", ls="--",
                                            lw=1.4, zorder=8))
                a2.plot(tt[:, 0], tt[:, 1], "-o", c=colors[k % len(colors)],
                        ms=2.5, lw=1.4)
                a2.plot(tt[0, 0], tt[0, 1], "o", c="k", ms=7)
                bad = tt[tt[:, 2] >= args.crash_frac]
                if len(bad):
                    a2.plot(bad[:, 0], bad[:, 1], "x", c="red", ms=9, mew=2)
                if not args.no_footprints:
                    # the ROBOT is the x; the QUAD is what the reward graded.
                    # If the quad sits on grass while the x sits on sidewalk,
                    # the "collision" is the look-ahead, not the robot.
                    for r_ in raw:
                        if len(r_) > 3 and r_[3] >= args.crash_frac:
                            hd = np.array([np.cos(r_[2]), np.sin(r_[2]), 0.0])
                            q = _footprint_corners_world(
                                np.array([r_[0], r_[1], 0.0]), hd,
                                look_ahead_dist=args.look_ahead,
                                length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
                            a2.plot(list(q[:, 0]) + [q[0, 0]],
                                    list(q[:, 1]) + [q[0, 1]],
                                    "-", c="red", lw=1.2, alpha=0.8, zorder=7)
                a2.set_xlim(xl); a2.set_ylim(yl); a2.set_aspect("equal")
                a2.grid(alpha=0.25)
                dist = (float(np.hypot(tt[-1, 0] - goal[0], tt[-1, 1] - goal[1]))
                        if goal is not None else float("nan"))
                # WHY did it end? The video HUD does not say, and "success:
                # false" covers three completely different outcomes. Infer it:
                # reached the goal / ended early (crash or a void-coherence
                # kill) / ran out of steps.
                if e.get("success"):
                    why = "GOAL"
                elif int(e.get("steps", 0)) >= max_steps:
                    why = "TIMEOUT"
                else:
                    why = "ENDED EARLY (crash/void)"
                a2.set_title(f"{label} ep{i}  [{why}]  steps {e.get('steps')}  "
                             f"return {e.get('return'):.2f}  "
                             f"crash-steps {len(bad)}  "
                             f"final dist to goal {dist:.2f} m", fontsize=9)
                f2.savefig(od / f"{label}_ep{i:02d}.png", dpi=110,
                           bbox_inches="tight")
                plt.close(f2)
                n += 1
        print(f"==> {n} per-episode plots: {od}/")


if __name__ == "__main__":
    main()
