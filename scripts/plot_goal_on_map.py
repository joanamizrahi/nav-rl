"""Show a proposed pinned goal on the scene map BEFORE spending a GPU on it.

Draws the scene's ground labels (grass green, other non-traversable blue,
walkable grey), the recorded walk with frame numbers, the proposed spawn
frames, the proposed goal, and a dashed straight line from each spawn to the
goal -- so you can see whether the straight line actually crosses grass, which
is the whole point of a goal-on-grass test. Prints the grass fraction under the
goal and the distance from each spawn.

    python scripts/plot_goal_on_map.py --scene gnd_AUw360 \
        --goal_xy 10.0,-15.6 --spawn_frames 64,68 --out goal_check.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NONTRAV = (0, 1, 3, 5, 10, 11, 12, 13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--goal_xy", required=True, help="x,y in the nav frame")
    ap.add_argument("--spawn_frames", default="", help="lo,hi frame range to highlight")
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--out", default="goal_check.png")
    ap.add_argument("--z_max", type=float, default=0.15)
    ap.add_argument("--radius", type=float, default=0.5, help="arrival radius")
    args = ap.parse_args()

    c = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    pts, labs = c["points"], c["labels"].astype(int)
    path = (np.asarray(c["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
    m = pts[:, 2] < args.z_max
    gxy, glab = pts[m][:, :2], labs[m]
    goal = np.array([float(v) for v in args.goal_xy.split(",")])
    lo, hi = (int(v) for v in args.spawn_frames.split(",")) if args.spawn_frames else (None, None)

    d = np.linalg.norm(gxy - goal[None, :], axis=1)
    near = glab[d < args.radius]
    gfrac = float(np.mean(near == 3)) if len(near) >= 5 else float("nan")
    print(f"goal ({goal[0]:.1f}, {goal[1]:.1f}): {len(near)} ground points within "
          f"{args.radius} m, grass fraction {100 * gfrac:.0f}%")
    if lo is not None:
        for f in range(lo, hi + 1):
            v = goal - path[f]
            dist = float(np.hypot(*v))
            # how much of the straight line lies over grass
            samp = path[f][None, :] + np.linspace(0, 1, 25)[:, None] * v[None, :]
            over = []
            for s in samp:
                dd = np.linalg.norm(gxy - s[None, :], axis=1)
                nb = glab[dd < 0.4]
                over.append(float(np.mean(nb == 3)) if len(nb) >= 5 else float("nan"))
            over = np.array(over)
            print(f"  spawn frame {f} at ({path[f][0]:.1f}, {path[f][1]:.1f}): "
                  f"{dist:.1f} m to goal; straight line is over grass for "
                  f"{100 * np.nanmean(over):.0f}% of its length")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    bad = np.isin(glab, NONTRAV); grass = glab == 3
    ax.scatter(gxy[~bad, 0], gxy[~bad, 1], s=1, c="#dedede", linewidths=0, label="walkable")
    ax.scatter(gxy[bad & ~grass, 0], gxy[bad & ~grass, 1], s=1, c="#9ecae1", linewidths=0,
               label="other non-traversable")
    ax.scatter(gxy[grass, 0], gxy[grass, 1], s=1, c="#74c476", linewidths=0, label="GRASS")
    ax.plot(path[:, 0], path[:, 1], "-", lw=1.2, c="0.35", label="recorded walk")
    for f in range(0, len(path), 10):
        ax.annotate(str(f), path[f], fontsize=7, color="0.3",
                    xytext=(3, 3), textcoords="offset points")
    if lo is not None:
        ax.plot(path[lo:hi + 1, 0], path[lo:hi + 1, 1], "o", ms=6, c="k", mec="w",
                mew=0.8, label=f"spawn frames {lo}-{hi}")
        for f in (lo, hi):
            ax.plot([path[f][0], goal[0]], [path[f][1], goal[1]], "--", lw=1, c="0.4")
    ax.plot(*goal, "*", ms=18, c="tab:red", mec="k", mew=0.6, label="proposed goal")
    circ = plt.Circle(goal, args.radius, fill=False, ec="tab:red", lw=1)
    ax.add_patch(circ)
    # zoom to the action
    allp = np.vstack([path[lo:hi + 1] if lo is not None else path, goal[None, :]])
    c0, span = allp.mean(0), max(np.ptp(allp, axis=0).max(), 6.0) + 6.0
    ax.set_xlim(c0[0] - span / 2, c0[0] + span / 2)
    ax.set_ylim(c0[1] - span / 2, c0[1] + span / 2)
    ax.set_aspect("equal")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="s", ls="", mfc="#dedede", mec="none", ms=8, label="walkable"),
          Line2D([], [], marker="s", ls="", mfc="#9ecae1", mec="none", ms=8, label="other non-traversable"),
          Line2D([], [], marker="s", ls="", mfc="#74c476", mec="none", ms=8, label="GRASS"),
          Line2D([], [], color="0.35", lw=1.2, label="recorded walk"),
          Line2D([], [], marker="o", ls="", mfc="k", mec="w", ms=6, label="spawn frames %s" % args.spawn_frames),
          Line2D([], [], marker="*", ls="", mfc="tab:red", mec="k", ms=10, label="proposed goal")]
    ax.legend(handles=hs, fontsize=8, loc="best")
    ax.set_title(f"{args.scene}: proposed goal ({goal[0]:.1f}, {goal[1]:.1f}), "
                 f"grass under it {100 * gfrac:.0f}%")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"==> {args.out}")


if __name__ == "__main__":
    main()
