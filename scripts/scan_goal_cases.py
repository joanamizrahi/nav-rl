"""Find OPEN / CORNER / NARROW goal cases on every scene's map (2026-09-07).

New training scope (Joana's mentor): every goal on traversable ground, but in
open spaces, narrow spaces, and around corners where the straight line to the
goal would cross non-traversable ground. This scanner reads only the scene
clouds (login node, no GPU) and, for spawns along the recorded walk, draws
walkable goals 4-8 m away inside the goal cone and classifies each pair:

  open     the straight line, widened to the body width, stays on walkable
           known ground
  corner   the straight line is blocked, but a walkable path exists on the
           map (BFS on the 0.1 m grid); the detour ratio says how much longer
  narrow   the walkable path passes through a corridor narrower than
           --narrow_m (distance transform on the walkable cells)
  blocked  no walkable path at all (never a goal)

Writes per scene: a JSON of the pairs (spawn, goal, class, detour ratio,
min corridor width) and an overhead picture with examples; prints a table.

    python scripts/scan_goal_cases.py --scenes gnd_AUw360 gnd_AUd210 \
        --out_dir /scratch/m000204-pm07b/joana/outputs/goal_cases
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid  # noqa: E402

BODY_W = 0.3


def load_scores(trav: str) -> np.ndarray:
    if trav == "none":
        s = np.ones(14); s[[0, 1, 3, 5, 10, 11, 12, 13]] = 0.0; s[7] = 0.85
        return s
    from src.eval.traversability import load_traversability
    return load_traversability(trav)


def distance_transform(free: np.ndarray, res: float) -> np.ndarray:
    """Distance (m) from each free cell to the nearest non-free cell. scipy if
    present, else a bounded brute-force erosion (fine for 0.1 m grids)."""
    try:
        from scipy import ndimage
        return ndimage.distance_transform_edt(free) * res
    except Exception:
        d = np.zeros(free.shape, np.float32)
        cur = free.copy(); k = 0
        while cur.any() and k < 60:
            k += 1
            nxt = cur.copy()
            nxt[1:] &= cur[:-1]; nxt[:-1] &= cur[1:]; nxt[:, 1:] &= cur[:, :-1]; nxt[:, :-1] &= cur[:, 1:]
            d[cur & ~nxt] = k * res
            cur = nxt
        d[cur] = k * res
        return d


def local_clearance(dist: np.ndarray, radius_m: float, res: float) -> np.ndarray:
    """For each cell, the LARGEST clearance within radius_m: the corridor's
    medial clearance near that cell, which does not depend on whether the
    path hugs a wall. Corridor width = 2 x this."""
    r = max(1, int(round(radius_m / res)))
    try:
        from scipy import ndimage
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return ndimage.maximum_filter(dist, footprint=(yy ** 2 + xx ** 2) <= r * r)
    except Exception:
        out = dist.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx > r * r:
                    continue
                sh = np.roll(np.roll(dist, dy, axis=0), dx, axis=1)
                out = np.maximum(out, sh)
        return out


def snap(mask: np.ndarray, ij: tuple, max_cells: int = 6):
    """The nearest True cell within max_cells of ij (ij itself if True)."""
    if mask[ij]:
        return ij
    best, bd = None, 1e9
    for dy in range(-max_cells, max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            n = (ij[0] + dy, ij[1] + dx)
            if 0 <= n[0] < mask.shape[0] and 0 <= n[1] < mask.shape[1] and mask[n]:
                d = dy * dy + dx * dx
                if d < bd:
                    best, bd = n, d
    return best


def bfs_path(free: np.ndarray, a: tuple, b: tuple, box: tuple):
    """Shortest 8-connected path on free cells inside box=(y0,y1,x0,x1); None if none."""
    y0, y1, x0, x1 = box
    if not (free[a] and free[b]):
        return None
    prev = {a: None}; q = deque([a])
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        c = q.popleft()
        if c == b:
            break
        for dy, dx in nbrs:
            n = (c[0] + dy, c[1] + dx)
            if y0 <= n[0] < y1 and x0 <= n[1] < x1 and free[n] and n not in prev:
                prev[n] = c; q.append(n)
    if b not in prev:
        return None
    path = []; c = b
    while c is not None:
        path.append(c); c = prev[c]
    return path[::-1]


def path_length(path, res):
    p = np.asarray(path, float)
    return float((np.linalg.norm(np.diff(p, axis=0), axis=1) * res).sum()) if len(p) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--poses_dir", default="/scratch/m000204-pm07b/joana/outputs/poses", help="draw the RECORDED camera heading (grey) beside the walk direction (orange) at each spawn frame")
    ap.add_argument("--heading", choices=["recorded", "walk"], default="recorded",
                    help="which heading centres the cone and orients the spawn: recorded = the env default (camera yaw from the poses file); walk = SPAWNHEADWALK=1")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frames", default="10,20,30,40,50,60,70", help="spawn frames along the walk")
    ap.add_argument("--window", default="4,8")
    ap.add_argument("--cone", type=float, default=50.0)
    ap.add_argument("--goals_per_spawn", type=int, default=40)
    ap.add_argument("--narrow_m", type=float, default=1.0, help="corridor width below which a case is NARROW")
    ap.add_argument("--detour_min", type=float, default=1.15, help="path/straight ratio above which a blocked line counts as CORNER")
    ap.add_argument("--inflate", type=float, default=0.1)
    ap.add_argument("--inflate_classes", default="10,11,13")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.trav); nontrav = scores <= 0.1
    lo, hi = (float(v) for v in args.window.split(","))
    rng = np.random.default_rng(args.seed)
    summary = {}
    for sc in args.scenes:
        p = Path(args.clouds_dir) / f"{sc}_cloud.npz"
        if not p.exists():
            print(f"{sc}: no cloud"); continue
        c = np.load(p)
        walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
        rec_hd = None
        pp = Path(args.poses_dir) / f"{sc}_poses.npz"
        if pp.exists():
            try:
                _d = np.load(pp); _h = np.asarray(_d["headings"], np.float32)
                rec_hd = (_h * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]     # same y-flip as the walk
                rec_hd = rec_hd / (np.linalg.norm(rec_hd, axis=1, keepdims=True) + 1e-9)
            except Exception:
                rec_hd = None
        g = build_label_grid(c["points"], c["labels"].astype(int), nontrav, res=0.1, inflate_m=args.inflate,
                             inflate_classes=tuple(int(v) for v in args.inflate_classes.split(",") if v.strip()), walk_xy=walk)
        L = g.labels; known = L >= 0
        free = known & ~nontrav[np.clip(L, 0, len(nontrav) - 1)]          # walkable known cells
        dist = distance_transform(free, g.res)                             # metres to the nearest non-walkable/unknown
        body_ok = dist >= BODY_W / 2.0                                     # a body centre may sit here
        clear = local_clearance(dist, 0.8, g.res)                          # medial clearance near each cell
        def cell(xy):
            return (int((xy[1] - g.y0) / g.res), int((xy[0] - g.x0) / g.res))
        def inb(ij):
            return 0 <= ij[0] < L.shape[0] and 0 <= ij[1] < L.shape[1]
        fy, fx = np.nonzero(free)
        free_xy = np.c_[g.x0 + (fx + 0.5) * g.res, g.y0 + (fy + 0.5) * g.res].astype(np.float32)
        pairs = []
        frame_notes = {}          # why a spawn frame produced nothing (Joana: "why no goals from frame 10?")
        req_frames = [int(v) for v in args.frames.split(",") if int(v) < len(walk) - 1]
        def spawn_yaw(f):
            dv = walk[min(f + 2, len(walk) - 1)] - walk[max(f - 2, 0)]
            yaw_w = float(np.arctan2(dv[1], dv[0]))
            if args.heading == "recorded" and rec_hd is not None and f < len(rec_hd):
                return float(np.arctan2(rec_hd[f][1], rec_hd[f][0])), yaw_w
            return yaw_w, yaw_w
        for f in req_frames:
            s = walk[f]; yaw, _ = spawn_yaw(f)
            sc_ij = cell(s)
            if not inb(sc_ij):
                frame_notes[f] = "spawn outside the map"; continue
            if not free[sc_ij]:
                # the walk point sits on a non-walkable/unknown cell (label error,
                # hole, inflation): snap to the nearest walkable cell within 0.5 m
                sn = snap(free, sc_ij, max_cells=5)
                if sn is None:
                    frame_notes[f] = "spawn cell not walkable (no walkable cell within 0.5 m)"; continue
                frame_notes[f] = f"spawn snapped {0.1 * np.hypot(sn[0] - sc_ij[0], sn[1] - sc_ij[1]):.1f} m to a walkable cell"
                sc_ij = sn; s = np.array([g.x0 + (sn[1] + 0.5) * g.res, g.y0 + (sn[0] + 0.5) * g.res], np.float32)
            d = np.linalg.norm(free_xy - s[None, :], axis=1)
            ang = np.arctan2(free_xy[:, 1] - s[1], free_xy[:, 0] - s[0])
            dth = np.abs((ang - yaw + np.pi) % (2 * np.pi) - np.pi)
            in_win = (d >= lo) & (d <= hi)
            cand = np.nonzero(in_win & (dth <= np.deg2rad(args.cone) / 2))[0]
            if len(cand) == 0:
                frame_notes[f] = f"no walkable cell in the cone ({int(in_win.sum())} in the window, none within {args.cone:.0f} deg)"; continue
            for gi in rng.choice(cand, size=min(args.goals_per_spawn, len(cand)), replace=False):
                gxy = free_xy[gi]; g_ij = cell(gxy)
                # straight line, widened to the body: every sample must be body_ok
                n = int(np.ceil(np.linalg.norm(gxy - s) / (g.res / 2)))
                line = s[None, :] + (gxy - s)[None, :] * np.linspace(0, 1, n)[:, None]
                ok = True
                for q in line:
                    ij = cell(q)
                    if not inb(ij) or not body_ok[ij]:
                        ok = False; break
                straight = float(np.linalg.norm(gxy - s))
                rec = {"frame": int(f), "spawn": [float(s[0]), float(s[1])], "yaw": yaw,
                       "goal": [float(gxy[0]), float(gxy[1])], "straight_m": round(straight, 2)}
                if ok:
                    rec.update(cls="open", detour=1.0)
                    # corridor width along the straight line (medial clearance, not edge distance)
                    rec["min_width_m"] = round(float(min(clear[cell(q)] for q in line)) * 2.0, 2)
                else:
                    # a walkable path around? BFS inside a box around the pair,
                    # endpoints snapped to body-clear cells (a goal near an edge is still a goal)
                    m = int(6.0 / g.res)
                    y0 = max(0, min(sc_ij[0], g_ij[0]) - m); y1 = min(L.shape[0], max(sc_ij[0], g_ij[0]) + m)
                    x0 = max(0, min(sc_ij[1], g_ij[1]) - m); x1 = min(L.shape[1], max(sc_ij[1], g_ij[1]) + m)
                    a_ij, b_ij = snap(body_ok, sc_ij), snap(body_ok, g_ij)
                    path = bfs_path(body_ok, a_ij, b_ij, (y0, y1, x0, x1)) if (a_ij is not None and b_ij is not None) else None
                    if path is None:
                        rec.update(cls="blocked", detour=float("inf"), min_width_m=0.0)
                    else:
                        plen = path_length(path, g.res); ratio = plen / max(straight, 1e-6)
                        wmin = float(min(clear[ij] for ij in path)) * 2.0
                        rec.update(cls=("corner" if ratio >= args.detour_min else "open"), detour=round(ratio, 2),
                                   min_width_m=round(wmin, 2), path=[[float(g.x0 + (j + 0.5) * g.res), float(g.y0 + (i + 0.5) * g.res)] for i, j in path[::5]])
                if rec["cls"] != "blocked" and rec["min_width_m"] < args.narrow_m:
                    rec["cls"] = "narrow" if rec["cls"] == "open" else "corner+narrow"
                pairs.append(rec)
        counts = {}
        for r in pairs:
            counts[r["cls"]] = counts.get(r["cls"], 0) + 1
        for f in req_frames:
            if f in frame_notes and not any(r["frame"] == f for r in pairs):
                print(f"    frame {f:2d}: NO goals -- {frame_notes[f]}", flush=True)
        n = max(1, len(pairs))
        summary[sc] = {k: counts.get(k, 0) for k in ("open", "corner", "narrow", "corner+narrow", "blocked")}
        summary[sc]["pairs"] = len(pairs)
        print(f"{sc:14s} pairs {len(pairs):4d}  " + "  ".join(f"{k} {counts.get(k, 0):4d} ({100 * counts.get(k, 0) / n:4.1f}%)" for k in ("open", "corner", "narrow", "corner+narrow", "blocked")), flush=True)
        with open(out / f"{sc}_goal_cases.json", "w") as fh:
            json.dump({"scene": sc, "window": [lo, hi], "cone": args.cone, "narrow_m": args.narrow_m, "pairs": pairs}, fh)
        if not args.no_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            b = np.full(L.shape, 0.5); b[free] = 1.0; b[known & ~free] = 0.0
            ext = (g.x0, g.x0 + L.shape[1] * g.res, g.y0, g.y0 + L.shape[0] * g.res)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(b, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.plot(walk[:, 0], walk[:, 1], "-", c="#e6550d", lw=1.2, label="recorded walk")
            col = {"open": "#2ca02c", "corner": "#ff7f0e", "narrow": "#1f77b4", "corner+narrow": "#9467bd", "blocked": "#7f7f7f"}
            # the GOAL VERSION (Joana, 2026-09-07): every spawn frame with its
            # cone wedge (window lo-hi), every sampled goal as a dot coloured by
            # class, the walkable path drawn for a few corner cases
            got = set(r["frame"] for r in pairs)
            for f in req_frames:
                sp = walk[f]; yaw, yaw_w = spawn_yaw(f)
                if f not in got:
                    ax.plot(sp[0], sp[1], "x", c="#7f7f7f", ms=9, mew=2)
                    ax.annotate(f"{f}: {frame_notes.get(f, 'no goals')}", (sp[0], sp[1]), xytext=(6, -12), textcoords="offset points", fontsize=7, color="#7f7f7f")
                    continue
                a0, a1 = yaw - np.deg2rad(args.cone) / 2, yaw + np.deg2rad(args.cone) / 2
                ang = np.linspace(a0, a1, 24)
                outer = np.c_[sp[0] + hi * np.cos(ang), sp[1] + hi * np.sin(ang)]
                inner = np.c_[sp[0] + lo * np.cos(ang[::-1]), sp[1] + lo * np.sin(ang[::-1])]
                wedge = np.vstack([outer, inner, outer[:1]])
                ax.plot(wedge[:, 0], wedge[:, 1], "-", c="#e6550d", lw=2.2, alpha=0.95)
                ax.fill(wedge[:, 0], wedge[:, 1], color="#e6550d", alpha=0.07)
                ax.plot(sp[0], sp[1], "o", c="#e6550d", ms=9, mec="black")
                # the two headings: orange = walk direction (what SPAWNHEADWALK spawns on),
                # grey = the recorded camera heading (the old spawn), angle between them
                ax.arrow(sp[0], sp[1], 2.5 * np.cos(yaw), 2.5 * np.sin(yaw), color="#e6550d", width=0.12, head_width=0.6, length_includes_head=True, zorder=5)
                lab = str(f)
                # grey = the OTHER heading (walk direction when spawning on the
                # recorded yaw, recorded yaw when spawning on the walk direction)
                other = yaw_w if args.heading == "recorded" else (float(np.arctan2(rec_hd[f][1], rec_hd[f][0])) if (rec_hd is not None and f < len(rec_hd)) else None)
                if other is not None:
                    ax.arrow(sp[0], sp[1], 2.5 * np.cos(other), 2.5 * np.sin(other), color="#333333", width=0.09, head_width=0.5, length_includes_head=True, alpha=0.9, zorder=5)
                    dang = np.degrees(np.arctan2(np.sin(other - yaw), np.cos(other - yaw)))
                    lab = f"{f} ({dang:+.0f} deg)"
                ax.annotate(lab, (sp[0], sp[1]), xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold", color="#e6550d", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
            for r in pairs:
                ax.plot(r["goal"][0], r["goal"][1], ".", c=col[r["cls"]], ms=5, alpha=0.7)
            shown = 0
            for r in pairs:
                if r["cls"].startswith("corner") and "path" in r and shown < 12:
                    shown += 1
                    pp = np.asarray(r["path"]); ax.plot(pp[:, 0], pp[:, 1], "--", c=col[r["cls"]], lw=1.0, alpha=0.9)
                    ax.plot([r["spawn"][0], r["goal"][0]], [r["spawn"][1], r["goal"][1]], "-", c=col[r["cls"]], lw=0.8, alpha=0.6)
                    ax.plot(r["goal"][0], r["goal"][1], "*", c=col[r["cls"]], ms=11)
            for k, cc in col.items():
                ax.plot([], [], ".", c=cc, ms=8, label=f"{k} {counts.get(k, 0)}")
            ax.legend(loc="upper right"); ax.set_aspect("equal")
            ax.set_title(f"{sc}: goals {lo}-{hi} m in the {args.cone:.0f} deg cone from each spawn frame; orange arrow = SPAWN heading ({args.heading}), grey = the other heading, (angle between); dots = goals by class")
            fig.savefig(out / f"{sc}_goal_cases.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"==> {out}")


if __name__ == "__main__":
    main()
