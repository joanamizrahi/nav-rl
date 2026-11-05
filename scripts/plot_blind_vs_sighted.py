"""Overhead paths: what does the SAME policy do with and without vision?

`eval_policy` stores a full trajectory per episode, and both halves of a pair
run under the same `--eval_seed`, so episode N is the SAME spawn and the SAME
goal in both. That makes the honest picture a paired overhead plot: two paths
from one start to one goal, one blind, one sighted, drawn over the scene's own
ground labels.

It exists because the summary table cannot settle the question. On
gnd_AUw360 (2026-09-03) grass contact came back 0.000 for BOTH halves of every
pair, so the trespass metric was degenerate -- while the ancestor's outcomes
differed enormously (sighted 19/20 TIMEOUT, blind 20/20 GOAL). A number that
says "no difference" next to outcomes that say "completely different" means the
number is measuring the wrong thing. The paths show what actually happened.

    python scripts/plot_blind_vs_sighted.py \
        --sighted /scratch/.../eval_..._gnd_AUw360_fwd_live \
        --blind   /scratch/.../eval_..._blind_..._gnd_AUw360_fwd_live \
        --scene gnd_AUw360 --out /scratch/.../paths.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NONTRAV_DEFAULT = (0, 1, 3, 5, 10, 11, 12, 13)   # void sky grass water obstacle veg person vehicle

# How each episode ENDED, drawn at the last pose. Reading "12 GOAL 4 CRASH" off
# a summary tells you nothing about WHERE things went wrong; the symbol does.
V14_NAMES = {0: "void", 1: "sky", 2: "trail", 3: "grass", 4: "rough", 5: "water",
             6: "sidewalk", 7: "road", 8: "pavement", 9: "stairs", 10: "obstacle",
             11: "vegetation", 12: "person", 13: "vehicle", -1: "none"}

OUTCOME_MARK = {
    "GOAL":       ("*", "#1a9850", 15),
    "CRASH":      ("X", "#d73027", 11),
    "TIMEOUT":    ("s", "#7f7f2a", 8),
    "INCOHERENT": ("^", "#f46d43", 10),
    "HALTED":     ("D", "#4575b4", 9),
}


def mark_end(ax, xy, outcome, edge):
    m, c, sz = OUTCOME_MARK.get(outcome, ("o", "0.4", 8))
    ax.plot(xy[0], xy[1], m, ms=sz, c=c, mec=edge, mew=1.0, zorder=6)


def outcome_legend(ax, present):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=OUTCOME_MARK[o][0], color="none",
                markerfacecolor=OUTCOME_MARK[o][1], markeredgecolor="k",
                markersize=OUTCOME_MARK[o][2] * 0.8, label=o)
         for o in OUTCOME_MARK if o in present]
    if h:
        ax.legend(handles=h, fontsize=8, loc="lower right", framealpha=0.9,
                  title="how it ended", title_fontsize=8)


def load(run: Path):
    d = json.loads((run / "metrics.json").read_text())
    return d["summary"], d["episodes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sighted", required=True)
    ap.add_argument("--blind", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--out", default="blind_vs_sighted_paths.png")
    ap.add_argument("--episodes", type=int, default=20,
                    help="how many paired episodes to draw")
    ap.add_argument("--z_max", type=float, default=0.15)
    ap.add_argument("--overview_out", default="",
                    help="also draw ALL episodes on ONE whole-scene map. This "
                         "is what shows whether the spawns are CLUSTERED -- "
                         "Joana's 2026-09-03 hypothesis was that gnd_AUw360 "
                         "spawns sit at the start of the walk while the grass "
                         "only appears near the end, which would make a "
                         "grass-avoidance measurement impossible by "
                         "construction rather than by policy behaviour.")
    args = ap.parse_args()

    s_sum, s_eps = load(Path(args.sighted))
    b_sum, b_eps = load(Path(args.blind))

    gxy = glab = axy = alab = az = None
    cp = Path(args.clouds_dir) / f"{args.scene}_cloud.npz"
    if cp.exists():
        c = np.load(cp)
        pts, labs = c["points"], c["labels"].astype(int)
        axy, alab = pts[:, :2], labs          # every point, for the crash buckets
        az = pts[:, 2]
        m = pts[:, 2] < args.z_max
        gxy, glab = pts[m][:, :2], labs[m]
        # subsample: 150k points is unreadable and slow to render
        if len(gxy) > 60000:
            k = np.random.default_rng(0).choice(len(gxy), 60000, replace=False)
            gxy, glab = gxy[k], glab[k]
        _recorded_path = (np.asarray(c["traj_positions"], dtype=float)
                          * np.array([1.0, -1.0, 1.0]))[:, :2]
    else:
        _recorded_path = None
        print(f"[warn] no cloud at {cp}; drawing paths without terrain")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(args.episodes, len(s_eps), len(b_eps))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.1 * rows))
    axes = np.atleast_1d(axes).ravel()

    seen = set()
    for i in range(n):
        ax, se, be = axes[i], s_eps[i], b_eps[i]
        # Take x,y only and tolerate ragged rows: the first entry of `traj` was
        # written with 4 fields and the rest with 5, so np.array() on the whole
        # thing raises. Old metrics.json files still have that shape.
        st = np.array([r[:2] for r in se["traj"]], dtype=float)
        bt = np.array([r[:2] for r in be["traj"]], dtype=float)
        goal = np.array(se["goal_xy"], dtype=float)

        if gxy is not None:
            # frame the panel on the action, then draw only nearby terrain
            allxy = np.vstack([st[:, :2], bt[:, :2], goal[None, :]])
            lo, hi = allxy.min(0) - 4.0, allxy.max(0) + 4.0
            m = ((gxy[:, 0] > lo[0]) & (gxy[:, 0] < hi[0])
                 & (gxy[:, 1] > lo[1]) & (gxy[:, 1] < hi[1]))
            bad = np.isin(glab[m], NONTRAV_DEFAULT)
            ax.scatter(gxy[m][~bad, 0], gxy[m][~bad, 1], s=1, c="#d9d9d9",
                       linewidths=0)
            ax.scatter(gxy[m][bad, 0], gxy[m][bad, 1], s=1, c="#a8d5a2",
                       linewidths=0)

        ax.plot(st[:, 0], st[:, 1], "-", lw=2.0, c="tab:blue", label="sighted")
        ax.plot(bt[:, 0], bt[:, 1], "-", lw=2.0, c="tab:orange", label="blind")
        ax.plot(*st[0, :2], "ko", ms=6)
        ax.plot(*goal, "*", ms=16, c="tab:red", mec="k", mew=0.5, zorder=5)
        # end markers carry the OUTCOME, with the halo naming which policy
        mark_end(ax, st[-1, :2], se["outcome"], "tab:blue")
        mark_end(ax, bt[-1, :2], be["outcome"], "tab:orange")
        ax.set_title(f"ep{i}  sighted {se['outcome']} ({se['steps']}st)\n"
                     f"blind {be['outcome']} ({be['steps']}st)", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
        seen.update({se["outcome"], be["outcome"]})

    for j in range(n, len(axes)):
        axes[j].axis("off")
    # one legend, on the first spare panel if there is one, else on panel 0
    outcome_legend(axes[n] if n < len(axes) else axes[0], seen)

    if args.overview_out:
        overview(args, s_eps, b_eps, gxy, glab, _recorded_path, axy=axy, alab=alab, az=az)

    fig.suptitle(
        f"{args.scene}: same weights, same spawn, same goal — with and without vision\n"
        f"sighted {s_sum['outcomes']}   blind {b_sum['outcomes']}   "
        f"grass share {s_sum['ground_share'].get('grass', 0):.3f} vs "
        f"{b_sum['ground_share'].get('grass', 0):.3f}   "
        f"(black dot = spawn, red star = goal, square = where it stopped; "
        f"green = non-traversable)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"==> {args.out}")

    # the paired numbers, printed so the figure does not have to be squinted at
    print(f"{'ep':>3} {'sighted':>22} {'blind':>22}   d_start")
    for i in range(n):
        se, be = s_eps[i], b_eps[i]
        print(f"{i:>3} {se['outcome']:>10} {se['steps']:>3}st "
              f"c={se['closed_frac']!s:>6} "
              f"{be['outcome']:>10} {be['steps']:>3}st "
              f"c={be['closed_frac']!s:>6}   {se['d_start']}")


def overview(args, s_eps, b_eps, gxy, glab, path=None, axy=None, alab=None, az=None):
    """Every episode on one map of the whole scene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    if gxy is not None:
        bad = np.isin(glab, NONTRAV_DEFAULT)
        grass = glab == 3
        ax.scatter(gxy[~bad, 0], gxy[~bad, 1], s=1, c="#dedede", linewidths=0,
                   label="traversable")
        ax.scatter(gxy[bad & ~grass, 0], gxy[bad & ~grass, 1], s=1,
                   c="#9ecae1", linewidths=0, label="other non-traversable")
        ax.scatter(gxy[grass, 0], gxy[grass, 1], s=1, c="#74c476", linewidths=0,
                   label="GRASS")
    if path is not None and len(path):
        ax.plot(path[:, 0], path[:, 1], "-", lw=1.2, c="0.35",
                label="recorded walk")
        ax.plot(*path[0], "P", ms=11, c="k")
        ax.annotate("frame 0", path[0], fontsize=9,
                    xytext=(6, 6), textcoords="offset points")
        ax.annotate(f"frame {len(path) - 1}", path[-1], fontsize=9,
                    xytext=(6, 6), textcoords="offset points")

    seen = set()
    # Pair each spawn with ITS goal first, underneath everything: a faint
    # dashed tie plus a shared index. Without it a map of 20 spawns and 20
    # goals is unreadable -- you cannot tell which star belongs to which dot.
    for i, e in enumerate(s_eps):
        t0 = np.array(e["traj"][0][:2], dtype=float)
        g = np.array(e["goal_xy"], dtype=float)
        ax.plot([t0[0], g[0]], [t0[1], g[1]], "--", lw=0.7, c="0.55",
                alpha=0.8, zorder=1,
                label="spawn to its goal" if i == 0 else None)

    for eps, c, lbl in ((s_eps, "tab:blue", "sighted"),
                        (b_eps, "tab:orange", "blind")):
        for i, e in enumerate(eps):
            t = np.array([r[:2] for r in e["traj"]], dtype=float)
            ax.plot(t[:, 0], t[:, 1], "-", lw=1.4, c=c, alpha=0.75, zorder=3,
                    label=lbl if i == 0 else None)
            mark_end(ax, t[-1], e["outcome"], c)
            seen.add(e["outcome"])

    for i, e in enumerate(s_eps):
        t0 = np.array(e["traj"][0][:2], dtype=float)
        g = np.array(e["goal_xy"], dtype=float)
        ax.plot(*t0, "o", ms=7, c="k", mec="w", mew=0.8, zorder=4,
                label="spawn" if i == 0 else None)
        ax.plot(*g, "*", ms=13, c="tab:red", mec="k", mew=0.4, zorder=4,
                label="goal" if i == 0 else None)
        # the index ties the two ends together where the dashes cross
        ax.annotate(str(i), t0, fontsize=8, fontweight="bold", zorder=7,
                    xytext=(5, 4), textcoords="offset points")
        ax.annotate(str(i), g, fontsize=8, color="#b2182b", zorder=7,
                    xytext=(5, 4), textcoords="offset points")

    ax.set_aspect("equal")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="s", ls="", mfc="#dedede", mec="none", ms=8, label="traversable"),
          Line2D([], [], marker="s", ls="", mfc="#9ecae1", mec="none", ms=8, label="other non-traversable"),
          Line2D([], [], marker="s", ls="", mfc="#74c476", mec="none", ms=8, label="GRASS"),
          Line2D([], [], color="0.35", lw=1.2, label="recorded walk"),
          Line2D([], [], ls="--", color="0.55", lw=0.8, label="spawn to its goal"),
          Line2D([], [], color="tab:blue", lw=1.6, label="sighted"),
          Line2D([], [], color="tab:orange", lw=1.6, label="blind"),
          Line2D([], [], marker="o", ls="", mfc="k", mec="w", ms=6, label="spawn"),
          Line2D([], [], marker="*", ls="", mfc="tab:red", mec="k", ms=10, label="goal")]
    leg = ax.legend(handles=hs, fontsize=8, loc="upper left")
    ax.add_artist(leg)
    outcome_legend(ax, seen)
    ax.set_title(f"{args.scene}: every episode, whole scene\n"
                 f"are the spawns spread along the walk, and is there any "
                 f"grass where the robot actually goes?", fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.overview_out, dpi=150, bbox_inches="tight")
    print(f"==> {args.overview_out}")
    if gxy is not None:
        sp = np.array([np.array(e["traj"][0][:2], dtype=float) for e in s_eps])
        gl = np.array([e["goal_xy"] for e in s_eps], dtype=float)
        print(f"    spawn spread: x {sp[:, 0].min():.1f}..{sp[:, 0].max():.1f} m, "
              f"y {sp[:, 1].min():.1f}..{sp[:, 1].max():.1f} m")
        print(f"    goal  spread: x {gl[:, 0].min():.1f}..{gl[:, 0].max():.1f} m, "
              f"y {gl[:, 1].min():.1f}..{gl[:, 1].max():.1f} m")
        gr = gxy[glab == 3]
        print(f"    grass points in scene: {len(gr)} "
              f"({100.0 * len(gr) / max(len(gxy), 1):.1f}% of ground)")
        # WHERE THE CRASHES HAPPENED (Joana, 2026-09-03 22:20: "I don't know
        # how to start evaluating the behaviours"). The crash fires on the
        # FOOTPRINT, 1.5 m ahead along the heading, not under the robot; and a
        # wall is not a ground point. So: look up the footprint position in the
        # FULL cloud and name what the reconstruction has there. Two lines per
        # side: what the CLOUD has at the footprint, and what the GENERATOR
        # said there (traj row 4 at the last step) -- if the cloud says
        # walkable and the generator says obstacle, that crash is a phantom.
        cxy, clab = (axy, alab) if axy is not None else (gxy, glab)
        def crash_buckets(eps):
            cloud, gen = {}, {}
            for e in eps:
                if e["outcome"] != "CRASH":
                    continue
                row = e["traj"][-1]
                x, y, yaw = float(row[0]), float(row[1]), float(row[2]) if len(row) > 2 else 0.0
                fx, fy = x + 1.5 * np.cos(yaw), y + 1.5 * np.sin(yaw)
                inside = np.linalg.norm(cxy - np.array([[fx, fy]]), axis=1) < 0.6
                near = clab[inside]
                # a wall clipped by the footprint is a thin line in x,y and
                # loses the majority vote to dense ground; count it separately
                n_vert = int((az[inside] > 0.4).sum()) if az is not None else 0
                if len(near) < 8:
                    k = "edge of reconstruction"
                else:
                    dom = int(np.bincount(near).argmax())
                    if dom in NONTRAV_DEFAULT:
                        k = V14_NAMES.get(dom, str(dom))
                    elif n_vert >= 15:
                        k = V14_NAMES.get(dom, str(dom)) + " but a vertical structure in the disc"
                    else:
                        k = V14_NAMES.get(dom, str(dom)) + " (walkable, nothing vertical: PHANTOM if generator said obstacle)"
                cloud[k] = cloud.get(k, 0) + 1
                g = V14_NAMES.get(int(row[4]), "?") if len(row) > 4 else "?"
                gen[g] = gen.get(g, 0) + 1
            return cloud, gen
        for name, eps in (("sighted", s_eps), ("blind", b_eps)):
            cloud, gen = crash_buckets(eps)
            if not cloud:
                print(f"    {name}: no crashes")
                continue
            print(f"    {name} crashed where the CLOUD has: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(cloud.items(), key=lambda kv: -kv[1])))
            print(f"    {name} crashed where the GENERATOR said: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(gen.items(), key=lambda kv: -kv[1])))
        # WHAT IS UNDER EACH GOAL. "20/20 GOAL" is only informative if some
        # of those goals sat on ground the robot should have refused. Label
        # each goal by the dominant class within the arrival radius; if none
        # of them is grass, this eval could not have tested grass avoidance
        # no matter what the policy did (Joana's question, 2026-09-03).
        # Grass FRACTION under each goal, not just the dominant class: a goal
        # on the grass edge can be 40% grass and still vote "traversable",
        # which is how a figure showing stars on green got reported as
        # "GRASS 0" (Joana caught it by eye, 2026-09-03).
        r = 0.5
        gfrac = []
        for g in gl:
            d = np.linalg.norm(gxy - g[None, :], axis=1)
            near = glab[d < r]
            gfrac.append(float(np.mean(near == 3)) if len(near) >= 5 else float("nan"))
        gfrac = np.array(gfrac)
        on = np.where(gfrac > 0.2)[0]
        print("    grass fraction under each goal (cloud labels, %.1f m): "
              % r + " ".join("%d:%.0f%%" % (i, 100 * v) for i, v in enumerate(gfrac)))
        print("    goals with >20%% grass under them: %d/%d  -> episodes %s"
              % (len(on), len(gfrac), list(on)))
        if len(on) == 0:
            print("    !!! no goal sits on grass -- this eval cannot show grass "
                  "avoidance regardless of the policy. Pin goals on the grass "
                  "(GOAL_XY) or use a scene/range where the sampler lands there.")

        # THE WORLD-MODEL CHECK. The reward reads the GENERATED semantics; this
        # map shows the CLOUD's. traj row 4 is the generated dominant class in
        # the footprint at each step; the footprint sits 1.5 m ahead along the
        # heading. Look up what the cloud says at that same spot and count the
        # disagreements. "Walks on grass and does not crash" = cloud says grass,
        # generator says something else.
        agree = {"cloud grass, gen grass": 0, "cloud grass, gen OTHER": 0,
                 "cloud walkable, gen grass": 0, "cloud walkable, gen walkable": 0}
        # what the generator painted where the cloud has grass -- vegetation
        # (score 0) would still penalise; road/pavement means the reward is blind
        gen_at_grass: dict = {}
        for eps_ in (s_eps, b_eps):
            for e in eps_:
                for row in e["traj"]:
                    if len(row) < 5:
                        continue
                    x, y, yaw, _, gen = row[:5]
                    fx, fy = x + 1.5 * np.cos(yaw), y + 1.5 * np.sin(yaw)
                    d = np.linalg.norm(gxy - np.array([[fx, fy]]), axis=1)
                    near = glab[d < 0.4]
                    if len(near) < 5:
                        continue
                    cloud_grass = float(np.mean(near == 3)) > 0.5
                    gen_grass = int(gen) == 3
                    if cloud_grass:
                        gen_at_grass[int(gen)] = gen_at_grass.get(int(gen), 0) + 1
                    key = ("cloud grass, " if cloud_grass else "cloud walkable, ") + \
                          ("gen grass" if gen_grass else ("gen OTHER" if cloud_grass else "gen walkable"))
                    agree[key] = agree.get(key, 0) + 1
        tot_cg = agree["cloud grass, gen grass"] + agree["cloud grass, gen OTHER"]
        print("    footprint steps where the CLOUD says grass: %d; of those the "
              "GENERATED semantics also said grass: %d (%.0f%%)"
              % (tot_cg, agree["cloud grass, gen grass"],
                 100.0 * agree["cloud grass, gen grass"] / max(tot_cg, 1)))
        V14 = {0: "void", 1: "sky", 2: "trail", 3: "grass", 4: "rough", 5: "water",
               6: "sidewalk", 7: "road", 8: "pavement", 9: "stairs", 10: "obstacle",
               11: "vegetation", 12: "person", 13: "vehicle", -1: "none"}
        NONTRAV_SCORE0 = {0, 1, 3, 5, 10, 11, 12, 13}
        if gen_at_grass:
            top = sorted(gen_at_grass.items(), key=lambda kv: -kv[1])[:4]
            still_pen = sum(v for k, v in gen_at_grass.items() if k in NONTRAV_SCORE0)
            print("    what the generator painted there instead: "
                  + ", ".join("%s %d" % (V14.get(k, k), v) for k, v in top)
                  + "  -> still penalised (any score-0 class): %d/%d" % (still_pen, tot_cg))
        if tot_cg and agree["cloud grass, gen grass"] / tot_cg < 0.5:
            print("    at footprint positions the cloud labels grass, the generated "
                  "DOMINANT class was walkable. The generator does draw grass "
                  "elsewhere in the frame; at the verge its boundary sits inside "
                  "the cloud's, or the cloud over-segments (SAM3 labels, not GT). "
                  "Either way the reward at the footprint did not see grass here.")
        if len(gr):
            # distance from each spawn to the nearest grass point -- if this is
            # large for every spawn, the robot was never given the chance to
            # make a terrain decision, and a grass metric of 0.000 says nothing
            # about the policy.
            step = max(1, len(gr) // 20000)          # cap the pairwise cost
            d = np.min(np.linalg.norm(gr[::step][None, :, :] - sp[:, None, :],
                                      axis=2), axis=1)
            print(f"    nearest grass to each spawn: min {d.min():.1f} m, "
                  f"median {np.median(d):.1f} m, max {d.max():.1f} m")


if __name__ == "__main__":
    main()
