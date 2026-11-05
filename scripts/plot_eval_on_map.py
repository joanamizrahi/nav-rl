"""Overhead view of an eval on the reward map: every episode's path, its
spawn heading, the goal, and the CRASH BOX at the final pose, drawn over the
binary traversable / non-traversable / void map the map reward reads.

Use it when a video does not look like a crash: the picture shows whether the
box really sat on map non-traversable cells (map content) or whether the box
landed somewhere the video does not show (box placement / frame).

    python scripts/plot_eval_on_map.py \
        --metrics /scratch/.../outputs/eval_.../metrics.json --scene gnd_AU_180 \
        --collahead 1.5 --out_dir /scratch/.../outputs/eval_.../overhead

Writes overhead.png (whole scene, all episodes) and episodes.png (one zoomed
panel per episode) and prints, per episode, the collision fraction the eval
logged against the one recomputed here for the far (1.5 m) and near (0.6 m)
boxes at the final recorded pose.
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
OUTCOME_COL = {"GOAL": "#2ca02c", "CRASH": "#d62728", "HALTED": "#ff7f0e",
               "TIMEOUT": "#1f77b4", "INCOHERENT": "#9467bd"}


def box_corners(pos, yaw, ahead):
    h = np.array([np.cos(yaw), np.sin(yaw)])
    n = np.array([-h[1], h[0]])
    c = pos + ahead * h
    hl, hw = BODY_L / 2, BODY_W / 2
    return np.array([c + hl * h + hw * n, c + hl * h - hw * n,
                     c - hl * h - hw * n, c - hl * h + hw * n, c + hl * h + hw * n])


def box_frac(grid, nontrav, pos, yaw, ahead):
    """Non-traversable share of the box (void excluded, as the reward does)."""
    h = np.array([np.cos(yaw), np.sin(yaw)])
    fp = footprint_samples(pos + ahead * h, h, BODY_L, BODY_W, grid.res / 2.0)
    cl = grid.lookup(fp).astype(int)
    cl = np.where(cl < 0, 0, cl)
    void = cl == 0
    return float((nontrav[cl] & ~void).mean()), float(void.mean()), cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics.json written by eval_policy")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--collahead", type=float, default=1.5,
                    help="crash box centre distance the eval used (1.5 far, 0.6 near)")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--inflate", type=float, default=0.1)
    ap.add_argument("--fill", type=float, default=0.3)
    ap.add_argument("--fill_area", type=float, default=10.0)
    ap.add_argument("--walk", type=float, default=0.4)
    ap.add_argument("--inflate_classes", default="", help="comma list; empty = all non-traversable")
    ap.add_argument("--refusal_dist", type=float, default=2.5, help="refusal radius around the goal")
    ap.add_argument("--verge_dist", type=float, default=1.5, help="refusal radius around the verge point")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    m = json.load(open(args.metrics))
    eps = m["episodes"]
    scores = load_traversability(args.trav)
    nontrav = scores <= args.collision_threshold

    c = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    pts, labs = c["points"], c["labels"].astype(int)
    walk = (np.asarray(c["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
    g = build_label_grid(pts, labs, nontrav, res=args.res, inflate_m=args.inflate,
                         fill_m=args.fill, fill_max_area_m2=args.fill_area,
                         walk_xy=walk, walk_halfwidth_m=args.walk,
                         inflate_classes=tuple(int(v) for v in args.inflate_classes.split(",") if v.strip()))
    L = g.labels
    known = L >= 0
    nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
    ext = (g.x0, g.x0 + L.shape[1] * g.res, g.y0, g.y0 + L.shape[0] * g.res)
    b = np.full(L.shape, 0.5)
    b[known & ~nt] = 1.0
    b[nt] = 0.0
    # walkway strip: walkable cells within 3 m of the recorded walk (the env's verge candidates)
    wy, wx = np.nonzero(known & ~nt)
    _cells = np.c_[g.x0 + (wx + 0.5) * g.res, g.y0 + (wy + 0.5) * g.res].astype(np.float32)
    _keep = np.zeros(len(_cells), dtype=bool)
    for i0 in range(0, len(_cells), 4096):
        _blk = _cells[i0:i0 + 4096]
        _keep[i0:i0 + 4096] = ((_blk[:, None, :] - walk.astype(np.float32)[None, :, :]) ** 2).sum(-1).min(axis=1) <= 9.0
    strip = _cells[_keep] if _keep.any() else walk.astype(np.float32)

    # ---- table: logged vs recomputed collision fraction at the REWARD pose ----
    # The reward of step k is charged at the pose BEFORE action k. traj holds
    # the poses AFTER each action, so the last reward pose is traj[-2], and for
    # a one-step episode it is the spawn (recorded as 'spawn' since 09-04;
    # older metrics lack it -> marked 'late', box drawn one action late).
    def reward_pose(e, tr):
        if len(tr) >= 2:
            return tr[-2, :3], ""
        if e.get("spawn") is not None:
            return np.asarray(e["spawn"], float)[:3], ""
        return tr[-1, :3], "late"

    print(f"{'ep':>3} {'outcome':<8} {'steps':>5} {'logged':>7} {'far1.5':>7} {'near0.6':>8} "
          f"{'void':>5}  box classes (far)   [pose]")
    rows = []
    for e in eps:
        tr = np.asarray(e["traj"], float)
        (x, y, yaw), late = reward_pose(e, tr)
        logged = tr[-1, 3]
        far, vfar, cl = box_frac(g, nontrav, np.array([x, y]), yaw, 1.0 * 1.5)
        near, _, _ = box_frac(g, nontrav, np.array([x, y]), yaw, 0.6)
        cnt = np.bincount(cl, minlength=14)
        top = ", ".join(f"{int(k)}x{cnt[k]}" for k in np.argsort(-cnt)[:3] if cnt[k] > 0)
        print(f"{e['episode']:>3} {e['outcome']:<8} {e['steps']:>5} {logged:>7.2f} {far:>7.2f} "
              f"{near:>8.2f} {vfar:>5.2f}  {top}   {late}")
        rows.append((e, tr))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def draw_map(ax):
        ax.imshow(b, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.plot(walk[:, 0], walk[:, 1], "-", c="#e6550d", lw=1.0, alpha=0.8)
        ax.set_aspect("equal")

    def draw_episode(ax, e, tr, label=True):
        col = OUTCOME_COL.get(e["outcome"], "k")
        ax.plot(tr[:, 0], tr[:, 1], "-", c=col, lw=1.4)
        ax.plot(tr[0, 0], tr[0, 1], "o", c=col, ms=5, mec="w", mew=0.6)
        # heading at the first recorded pose
        ax.annotate("", xy=(tr[0, 0] + 0.8 * np.cos(tr[0, 2]), tr[0, 1] + 0.8 * np.sin(tr[0, 2])),
                    xytext=(tr[0, 0], tr[0, 1]), arrowprops=dict(arrowstyle="->", color=col, lw=1))
        (px, py, pyaw), late = reward_pose(e, tr)
        bx = box_corners(np.array([px, py]), pyaw, args.collahead)
        ax.plot(bx[:, 0], bx[:, 1], "--" if late else "-", c=col, lw=1.2)
        ax.fill(bx[:, 0], bx[:, 1], color=col, alpha=0.25)
        if e.get("spawn") is not None:
            sp = np.asarray(e["spawn"], float)
            ax.plot(sp[0], sp[1], "s", c=col, ms=4, mec="w", mew=0.5)
        gx, gy = e["goal_xy"]
        ax.plot(gx, gy, "*", c=col, ms=12, mec="k", mew=0.5)
        # lawn goal: mark the VERGE (walk point nearest the goal) and the refusal radius
        if e.get("goal_traversable") is False:
            vp = strip[int(np.argmin(np.linalg.norm(strip - np.array([gx, gy])[None, :], axis=1)))]
            ax.plot(vp[0], vp[1], "s", mfc="none", mec=col, ms=8, mew=1.2)
            ax.plot([gx, vp[0]], [gy, vp[1]], ":", c=col, lw=0.8)
            ax.add_patch(plt.Circle((vp[0], vp[1]), args.verge_dist, fill=False, ec=col, lw=0.8, ls="--", alpha=0.8))
        if label:
            ax.annotate(str(e["episode"]), (tr[0, 0], tr[0, 1]), fontsize=7, color=col,
                        xytext=(3, 3), textcoords="offset points")

    # ---- whole scene ----
    fig, ax = plt.subplots(figsize=(12, 12))
    draw_map(ax)
    for e, tr in rows:
        draw_episode(ax, e, tr)
    allxy = np.vstack([tr[:, :2] for _, tr in rows] + [np.array([e["goal_xy"] for e, _ in rows])])
    c0, span = allxy.mean(0), max(np.ptp(allxy, axis=0).max(), 8.0) + 6.0
    ax.set_xlim(c0[0] - span / 2, c0[0] + span / 2)
    ax.set_ylim(c0[1] - span / 2, c0[1] + span / 2)
    hs = [Line2D([], [], color=v, lw=2, label=k) for k, v in OUTCOME_COL.items()]
    hs += [Line2D([], [], color="#e6550d", lw=1, label="recorded walk"),
           Line2D([], [], marker="*", ls="", mfc="w", mec="k", ms=10, label="goal"),
           Line2D([], [], marker="s", ls="", mfc="k", mec="none", ms=8, label="map non-traversable"),
           Line2D([], [], marker="s", ls="", mfc="0.5", mec="none", ms=8, label="void (unknown)")]
    ax.legend(handles=hs, fontsize=8, loc="best")
    n = len(rows)
    ncr = sum(e["outcome"] == "CRASH" for e, _ in rows)
    ax.set_title(f"{args.scene}: {Path(args.metrics).parent.name[:70]}\n{n} episodes, {ncr} crashes; "
                 f"filled box = crash box ({args.collahead} m ahead) at the final pose")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "overhead.png", dpi=130, bbox_inches="tight")
    print(f"==> {out / 'overhead.png'}")

    # ---- one zoomed panel per episode ----
    cols = 5
    nrow = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(4.2 * cols, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, (e, tr) in enumerate(rows):
        ax = axes[k]
        draw_map(ax)
        draw_episode(ax, e, tr, label=False)
        cx, cy = tr[-1, 0], tr[-1, 1]
        r = 4.0
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        (px, py, pyaw), late = reward_pose(e, tr)
        far, vfar, _ = box_frac(g, nontrav, np.array([px, py]), pyaw, args.collahead)
        ax.set_title(f"ep {e['episode']} {e['outcome']} {e['steps']} steps\n"
                     f"logged {tr[-1, 3]:.2f}  map@{args.collahead}m {far:.2f}  void {vfar:.2f}"
                     + ("  (box 1 action LATE: spawn not recorded)" if late else ""), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "episodes.png", dpi=110, bbox_inches="tight")
    print(f"==> {out / 'episodes.png'}")


if __name__ == "__main__":
    main()
