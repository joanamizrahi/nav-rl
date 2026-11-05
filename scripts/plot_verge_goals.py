"""Where do lawn goals land, and where is the verge the robot must stop at?

For each scene: draw the binary reward map, spawn the sampler at several walk
frames, draw the env's OWN map-direct lawn goals (SceneEnv._draw_map_goal,
extracted from scene_env.py so this is the same code the arms run) and, for
each, the verge point (walk point nearest the goal) with the refusal radius.
Pavement goals drawn in the same window and cone are shown as plain stars:
no verge, no circle -- the rule never fires for them.

    python scripts/plot_verge_goals.py --scenes gnd_AU_180 gnd_AUd210 gnd_AUw210 gnd_AUw360 gnd_AUw330 \
        --out_dir /scratch/m000204-pm07b/joana/outputs/verge_goals
"""
from __future__ import annotations

import argparse
import ast
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.reward_map import build_label_grid  # noqa: E402
try:
    from src.eval.traversability import load_traversability  # noqa: E402
except ModuleNotFoundError:      # no yaml on the laptop: use --trav none
    load_traversability = None


def env_methods():
    src = (ROOT / "src" / "env" / "scene_env.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "SceneEnv")
    ns = {"np": np}
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("_goal_supported", "_draw_map_goal", "_goal_walkable_share", "_refusal_point", "_disc_known_share"):
            exec(compile(ast.Module([n], []), "scene_env_extract", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--frames", default="15,30,45,60", help="spawn frames to draw from")
    ap.add_argument("--per_frame", type=int, default=6)
    ap.add_argument("--window", default="3,8")
    ap.add_argument("--cone", type=float, default=50.0)
    ap.add_argument("--lawn_cone", type=float, default=0.0, help="cone of the LAWN draw (GOALNTCONE; 0 = --cone)")
    ap.add_argument("--edge", type=float, default=0.0, help="goal_nontrav_edge_m (0 = off)")
    ap.add_argument("--refusal_dist", type=float, default=2.5)
    ap.add_argument("--verge_dist", type=float, default=1.5, help="refusal radius around the VERGE point")
    ap.add_argument("--classes", default="3,4,5")
    ap.add_argument("--tries", type=int, default=12, help="tries of the map-direct lawn draw (GOALNTTRIES)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.trav == "none":
        # the walkway table, for a laptop without yaml: everything walkable except
        # void/sky/grass/water/obstacle/vegetation/person/vehicle; road 0.85
        scores = np.ones(14); scores[[0, 1, 3, 5, 10, 11, 12, 13]] = 0.0; scores[7] = 0.85
    else:
        scores = load_traversability(args.trav)
    nontrav = scores <= 0.1
    ns = env_methods()
    lo, hi = (float(v) for v in args.window.split(","))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for sc in args.scenes:
        c = np.load(Path(args.clouds_dir) / f"{sc}_cloud.npz")
        pts, labs = c["points"], c["labels"].astype(int)
        walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
        g = build_label_grid(pts, labs, nontrav, res=0.1, inflate_m=0.1, inflate_classes=(10, 11, 13), walk_xy=walk)
        gr = pts[(pts[:, 2] < 0.15) & (labs >= 0)][:, :2][::4].astype(np.float32)
        ref = float(np.median([int((np.abs(gr - w[None, :]).max(axis=1) <= 0.6).sum()) for w in walk[10:70:5]]))

        class Cfg: pass
        env = types.SimpleNamespace(cfg=Cfg())
        env.cfg.goal_nontrav_classes = args.classes; env.cfg.goal_mix_tries = 12; env.cfg.goal_radius = 0.5
        env.cfg.goal_support_radius_m = 0.6; env.cfg.goal_support_min_frac = 0.6; env.cfg.goal_support_tries = 8
        env.cfg.goal_nontrav_edge_m = args.edge
        env.cfg.goal_nontrav_known_min = 0.5
        env.cfg.goal_nontrav_tries = int(args.tries)
        env.cfg.goal_nontrav_cone_deg = float(args.lawn_cone)
        env._lawn_draw_report = lambda st: None
        env.world_backend = types.SimpleNamespace(cfg=Cfg()); env.world_backend.cfg.goal_dist_range = (lo, hi); env.world_backend.cfg.goal_cone_deg = args.cone
        env._label_grids = {sc: g}; env._scene_id = sc; env._non_trav = nontrav; env._ground_pts = {sc: gr}
        env._support_ref = {sc: ref}; env._walk_xy = {sc: walk}; env.np_random = np.random.default_rng(0)
        env._goal_walkable_share = lambda goal, env=env: ns["_goal_walkable_share"](env, goal)
        env._goal_supported = lambda goal, env=env: ns["_goal_supported"](env, goal)
        env._disc_known_share = lambda goal, env=env: ns["_disc_known_share"](env, goal)

        L = g.labels; known = L >= 0; nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
        b = np.full(L.shape, 0.5); b[known & ~nt] = 1.0; b[nt] = 0.0
        ext = (g.x0, g.x0 + L.shape[1] * g.res, g.y0, g.y0 + L.shape[0] * g.res)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(b, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.plot(walk[:, 0], walk[:, 1], "-", c="#e6550d", lw=1.2)
        lawn_n, pave_n, d_verge, share_l, share_p = 0, 0, [], [], []
        # ---- funnel: which filter removes the lawn candidates on this scene ----
        cls = tuple(int(v) for v in args.classes.split(","))
        iy, ix = np.nonzero(np.isin(L, list(cls)))
        cells = np.c_[g.x0 + (ix + 0.5) * g.res, g.y0 + (iy + 0.5) * g.res].astype(np.float32)
        d2w = np.full(len(cells), np.inf)
        for i0 in range(0, len(cells), 4096):
            blk = cells[i0:i0 + 4096]
            d2w[i0:i0 + 4096] = ((blk[:, None, :] - walk[None, :, :]) ** 2).sum(-1).min(axis=1)
        # edge 0 = the edge rule is OFF: every lawn cell is a candidate (it used
        # to keep none, so the funnel printed zeros for every frame)
        near = cells if args.edge <= 0.0 else cells[d2w <= args.edge ** 2]
        cls_counts = {int(k): int(v) for k, v in zip(*np.unique(L[L >= 0], return_counts=True))}
        print(f"  {sc}: known cells by class {cls_counts}")
        print(f"  {sc}: {len(cells)} cells of classes {cls}; {len(near)} within {args.edge} m of the walk")
        for f in [int(v) for v in args.frames.split(",") if int(v) < len(walk) - 1]:
            dv = walk[min(f + 1, len(walk) - 1)] - walk[f]; yaw = float(np.arctan2(dv[1], dv[0]))
            d = np.linalg.norm(near - walk[f][None, :], axis=1)
            ang = np.arctan2(near[:, 1] - walk[f, 1], near[:, 0] - walk[f, 0])
            dth = np.abs((ang - yaw + np.pi) % (2 * np.pi) - np.pi)
            _lc = args.lawn_cone if args.lawn_cone > 0 else args.cone
            inwin = (d >= lo) & (d <= hi); incone = inwin & (dth <= np.deg2rad(_lc) / 2)
            cand = near[incone]
            sh_ok = sup_ok = 0
            for cxy in cand[:: max(1, len(cand) // 200)]:
                gl = np.array([cxy[0], cxy[1], 0.0], np.float32)
                sh = env._goal_walkable_share(gl)
                if sh == sh and sh <= 0.25:
                    sh_ok += 1
                    # the real draw's second test (no support rule since 09-05)
                    if env._disc_known_share(gl) >= env.cfg.goal_nontrav_known_min:
                        sup_ok += 1
            n_s = len(cand[:: max(1, len(cand) // 200)])
            print(f"    frame {f:2d}: in window {int(inwin.sum())}, in cone {int(incone.sum())}, "
                  f"of {n_s} sampled: share<=0.25 {sh_ok}, +known>=0.5 {sup_ok}  "
                  f"(pass rate {sup_ok / max(1, n_s):.2f}; P(12 tries fail) {(1 - sup_ok / max(1, n_s)) ** 12:.2f}, "
                  f"P({args.tries} fail) {(1 - sup_ok / max(1, n_s)) ** args.tries:.3f})")
        frames = [int(v) for v in args.frames.split(",") if int(v) < len(walk) - 1]
        for f in frames:
            dv = walk[min(f + 1, len(walk) - 1)] - walk[f]; yaw = float(np.arctan2(dv[1], dv[0]))
            P = np.eye(4); P[0, 0], P[1, 0], P[0, 1], P[1, 1] = np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw); P[:2, 3] = walk[f]
            env._robot_pose_world = P
            ax.plot(walk[f, 0], walk[f, 1], "o", c="k", ms=6, mec="w")
            # lawn goals: the env's own draw
            for _ in range(args.per_frame):
                gl = ns["_draw_map_goal"](env, yaw)
                if gl is None:
                    continue
                vp = ns["_refusal_point"](env, gl)
                lawn_n += 1; d_verge.append(float(np.linalg.norm(vp - gl[:2]))); share_l.append(env._goal_walkable_share(gl))
                ax.plot(gl[0], gl[1], "*", c="tab:red", ms=11, mec="k", mew=0.4)
                ax.plot(vp[0], vp[1], "s", mfc="none", mec="tab:red", ms=8, mew=1.3)
                ax.plot([gl[0], vp[0]], [gl[1], vp[1]], ":", c="tab:red", lw=0.8)
                ax.add_patch(plt.Circle((vp[0], vp[1]), args.verge_dist, fill=False, ec="tab:red", lw=0.8, ls="--", alpha=0.8))
            # pavement goals: the walk sampler's ring-and-cone draw, kept when the disc is >= 75% walkable
            for _ in range(args.per_frame * 3):
                d = env.np_random.uniform(lo, hi); th = yaw + np.deg2rad(env.np_random.uniform(-args.cone / 2, args.cone / 2))
                gp = np.array([walk[f, 0] + d * np.cos(th), walk[f, 1] + d * np.sin(th), 0.0], np.float32)
                sh = env._goal_walkable_share(gp)
                if sh == sh and sh >= 0.75 and env._goal_supported(gp):
                    pave_n += 1; share_p.append(sh)
                    ax.plot(gp[0], gp[1], "*", c="tab:green", ms=9, mec="k", mew=0.3)
                    if pave_n >= args.per_frame * len(frames):
                        break
        ax.set_aspect("equal")
        hs = [Line2D([], [], marker="*", ls="", mfc="tab:red", mec="k", ms=11, label=f"lawn goal ({lawn_n})"),
              Line2D([], [], marker="s", ls="", mfc="none", mec="tab:red", ms=8, label="verge = walkable cell nearest the goal (walkway strip)"),
              Line2D([], [], ls="--", c="tab:red", label=f"halt counts within {args.verge_dist} m of the verge"),
              Line2D([], [], marker="*", ls="", mfc="tab:green", mec="k", ms=9, label=f"pavement goal ({pave_n}), no verge rule"),
              Line2D([], [], marker="o", ls="", mfc="k", mec="w", ms=6, label="spawn frames " + args.frames)]
        ax.legend(handles=hs, fontsize=8, loc="best")
        dv_ = np.array(d_verge) if d_verge else np.array([np.nan])
        ax.set_title(f"{sc}: map-direct lawn goals (window {lo:.0f}-{hi:.0f} m, cone {args.cone:.0f}, edge {args.edge} m)\n"
                     f"goal-to-verge distance median {np.nanmedian(dv_):.2f} m, max {np.nanmax(dv_):.2f} m; "
                     f"lawn disc walkable share max {max(share_l) if share_l else float('nan'):.2f}; pavement share min {min(share_p) if share_p else float('nan'):.2f}")
        ax.grid(alpha=0.2); fig.tight_layout()
        fp = out / f"{sc}_verge_goals.png"; fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"{sc}: {lawn_n} lawn goals (verge distance median {np.nanmedian(dv_):.2f}, max {np.nanmax(dv_):.2f} m, "
              f"walkable share max {max(share_l) if share_l else float('nan'):.2f}), {pave_n} pavement goals (share min "
              f"{min(share_p) if share_p else float('nan'):.2f}) -> {fp}")


if __name__ == "__main__":
    main()
