"""Draw the reward map of a scene: the top-down label grid the map reward
reads (src/eval/reward_map.py), as classes and as the binary traversable /
non-traversable / void map. Look at this BEFORE trusting it as a reward: the
labels are SAM3's, and what SAM3 got wrong is now a wall the policy must avoid.

    python scripts/plot_scene_map.py --scenes gnd_AUw360 gnd_AU_180 \
        --trav config/traversability_v14_walkway.yaml --out_dir outputs/scene_maps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid, VOID  # noqa: E402
from src.eval.traversability import load_traversability  # noqa: E402

V14 = {0: "void", 1: "sky", 2: "trail", 3: "grass", 4: "rough", 5: "water", 6: "sidewalk",
       7: "road", 8: "pavement", 9: "stairs", 10: "obstacle", 11: "vegetation", 12: "person", 13: "vehicle"}
COL = {2: "#bde0b0", 3: "#74c476", 6: "#bdbdbd", 7: "#9e9ac8", 8: "#80cdc1", 9: "#9ecae1",
       10: "#252525", 11: "#fb6a4a", 12: "#e7298a", 13: "#8c6d31", 0: "#ffffff", 1: "#deebf7",
       4: "#8c8c00", 5: "#3182bd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--inflate", type=float, default=0.1)
    ap.add_argument("--fill", type=float, default=0.3)
    ap.add_argument("--fill_area", type=float, default=10.0, help="fill enclosed void regions up to this many m2")
    ap.add_argument("--walk", type=float, default=0.4, help="half-width of the walkable corridor along the recorded walk")
    ap.add_argument("--inflate_classes", default="", help="comma list of class ids that inflate; empty = every non-traversable class")
    ap.add_argument("--ignore", default="", help="classes that do not vote, e.g. 12,13 for person, vehicle")
    ap.add_argument("--out_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_maps")
    ap.add_argument("--suffix", default="", help="appended to the file names, e.g. filled")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    scores = load_traversability(args.trav)
    nontrav = scores <= args.collision_threshold

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    for sc in args.scenes:
        c = np.load(Path(args.clouds_dir) / f"{sc}_cloud.npz")
        pts, labs = c["points"], c["labels"].astype(int)
        path = (np.asarray(c["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
        ign = tuple(int(v) for v in args.ignore.split(",") if v.strip())
        g = build_label_grid(pts, labs, nontrav, res=args.res, inflate_m=args.inflate, fill_m=args.fill, fill_max_area_m2=args.fill_area, ignore_classes=ign,
                             walk_xy=path, walk_halfwidth_m=args.walk,
                             inflate_classes=tuple(int(v) for v in args.inflate_classes.split(",") if v.strip()))
        g_raw = build_label_grid(pts, labs, nontrav, res=args.res, inflate_m=0.0, fill_m=0.0, fill_max_area_m2=0.0, clean=False)
        # support along the recorded walk: how many cloud points per cell there,
        # the reference for 'enough points' anywhere else
        L = g.labels
        known = L >= 0
        nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
        # support along the recorded walk: cloud points per cell where we KNOW
        # the ground is walkable. The reference for 'enough points' elsewhere.
        ix = np.clip(((path[:, 0] - g.x0) / g.res).astype(int), 0, L.shape[1] - 1)
        iy = np.clip(((path[:, 1] - g.y0) / g.res).astype(int), 0, L.shape[0] - 1)
        on_path = g.n_points[iy, ix]
        print(f"    {sc}: points per cell on the recorded walk: median {np.median(on_path):.0f}, "
              f"10th pct {np.percentile(on_path, 10):.0f}, cells with none {(on_path == 0).mean():.0%}")
        ext = (g.x0, g.x0 + L.shape[1] * g.res, g.y0, g.y0 + L.shape[0] * g.res)

        fig, axes = plt.subplots(1, 3, figsize=(27, 9))
        # classes
        cmap = ListedColormap([COL.get(i, "#000000") for i in range(14)])
        img = np.ma.masked_where(~known, L)
        axes[0].imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=13.5, interpolation="nearest")
        axes[0].plot(path[:, 0], path[:, 1], "-", c="k", lw=1.2)
        present = sorted(set(int(v) for v in np.unique(L[known])))
        axes[0].legend(handles=[Patch(color=COL.get(i, "#000"), label=V14.get(i, str(i))) for i in present]
                       + [Patch(color="w", label="void (no reconstruction)")], fontsize=8, loc="best")
        axes[0].set_title(f"{sc}: reward map, classes ({g.res} m cells)")
        # binary
        b = np.full(L.shape, 0.5)
        b[known & ~nt] = 1.0
        b[nt] = 0.0
        axes[1].imshow(b, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1].plot(path[:, 0], path[:, 1], "-", c="tab:red", lw=1.2, label="recorded walk")
        axes[1].set_title(f"traversable (white) / non-traversable (black) / void = UNKNOWN, not obstacle (grey)\n"
                          f"known {known.mean():.0%} of cells, non-traversable {nt.sum() / max(known.sum(), 1):.0%} of known")
        axes[1].legend(fontsize=8)
        # what the cleanups changed, cell by cell
        Lr = g_raw.labels
        kr = Lr >= 0
        ntr = kr & nontrav[np.clip(Lr, 0, len(nontrav) - 1)]
        chg = np.full(L.shape, np.nan)
        chg[known & ~nt] = 0                                   # walkable, final
        chg[nt] = 1                                            # non-traversable, final
        chg[(~kr) & known] = 2                                 # FILLED: was void, now known
        chg[kr & ~ntr & nt] = 3                                # INFLATED or overridden: was walkable, now non-trav
        chg[ntr & known & ~nt] = 4                             # DESPECKLED: was non-trav, now walkable
        # cells the recorded-walk corridor made walkable (raw said non-trav or void)
        wk = np.zeros(L.shape, dtype=bool)
        ix_ = np.round((path[:, 0] - g.x0) / g.res).astype(int); iy_ = np.round((path[:, 1] - g.y0) / g.res).astype(int)
        rr = int(round(args.walk / g.res))
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx * dx + dy * dy <= rr * rr:
                    xx, yy = ix_ + dx, iy_ + dy; ok = (xx >= 0) & (yy >= 0) & (xx < L.shape[1]) & (yy < L.shape[0]); wk[yy[ok], xx[ok]] = True
        chg[wk & (ntr | ~kr) & known & ~nt] = 5                # WALK CORRIDOR: raw non-trav/void, now walkable
        cm = ListedColormap(["#ffffff", "#000000", "#00bcd4", "#ff9800", "#8bc34a", "#e91e63"])
        axes[2].imshow(np.ma.masked_invalid(chg), origin="lower", extent=ext, cmap=cm, vmin=-0.5, vmax=5.5, interpolation="nearest")
        axes[2].plot(path[:, 0], path[:, 1], "-", c="tab:red", lw=1.0)
        n_fill, n_infl, n_desp = int(((~kr) & known).sum()), int((kr & ~ntr & nt).sum()), int((ntr & known & ~nt).sum())
        axes[2].legend(handles=[Patch(color="#00bcd4", label=f"filled void ({n_fill} cells)"),
                                Patch(color="#ff9800", label=f"inflated / wall override ({n_infl} cells)"),
                                Patch(color="#8bc34a", label=f"despeckled to walkable ({n_desp} cells)"),
                                Patch(color="#e91e63", label=f"recorded-walk corridor ({int((wk & (ntr | ~kr) & known & ~nt).sum())} cells)")], fontsize=8, loc="best")
        axes[2].set_title("what the cleanups changed (raw cloud vote -> final map)")
        for ax in axes:
            ax.set_aspect("equal"); ax.grid(alpha=0.2)
        fig.tight_layout()
        sfx = ("_" + args.suffix) if args.suffix else ""
        out = Path(args.out_dir) / f"{sc}_reward_map{sfx}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        np.savez_compressed(Path(args.out_dir) / f"{sc}_reward_map{sfx}.npz", labels=L, x0=g.x0, y0=g.y0, res=g.res)
        print(f"==> {out}   known {known.mean():.0%}  non-trav {nt.sum() / max(known.sum(), 1):.0%}")


if __name__ == "__main__":
    main()
