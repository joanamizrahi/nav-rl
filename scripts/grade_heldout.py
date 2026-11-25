"""Grade a held-out sweep (scripts/heldout_sweep.py): generated labels and image against the REAL frame.
CPU only (numpy + PIL), runs on the login node next to the data. Writes <sweep>/graded.csv, one row per query:
  acc_class        generated class == SAM 3 on the real frame, over pixels SAM 3 labelled (person/vehicle left out)
  acc_trav         the same for walkable vs not (what the reward reads)
  acc_trav_seen / acc_trav_unseen   split by the window's opacity (alpha > 0.5 = some kept frame saw it)
  acc_trav_ahead   walkable vs not in the ground-ahead box (bottom 38 %, middle half: where the footprint lands)
  hint_trav_seen   the rasterized hint's walkable-vs-not accuracy on seen pixels (sanity: should be high)
  rgb_l1 / rgb_l1_unseen   mean |generated - real| per channel, 0-1
  frac_unseen      share of pixels with alpha <= 0.5
Ground truth is SAM 3 on the real frame, not a human: these are agreement-with-SAM 3 numbers.

    python scripts/grade_heldout.py --sweep /scratch/.../outputs/heldout_v35b/quad2_01
"""
import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image

NOGO = [3, 5, 10, 11, 12, 13]        # traversability_v14_walkway score 0 (as in consistency_measures.py)
MOVERS = [12, 13]                    # person, vehicle: they move between frames, left out of every score


def acc(a, b, m):
    return float((a[m] == b[m]).mean()) if m.sum() >= 50 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep", required=True, type=Path); a = ap.parse_args()
    d = a.sweep
    Q = list(csv.DictReader(open(d / "queries.csv")))
    P = np.load(d / "semantic_labels.npz")["labels"].astype(np.int64)
    G = np.load(d / "gt_labels.npz")["labels"].astype(np.int64)
    Hn = np.load(d / "hint_labels.npz")["labels"].astype(np.int64)
    A = np.load(d / "alpha.npz")["alpha"].astype(np.float32)
    nogo = np.zeros(32, bool); nogo[NOGO] = True
    N, H, W = P.shape
    box = np.zeros((H, W), bool); box[int(0.62 * H):, int(0.25 * W):int(0.75 * W)] = True
    rows = []
    for i, q in enumerate(Q):
        g, p, h = G[i], P[i], Hn[i]
        valid = (g > 0) & ~np.isin(g, MOVERS)
        seen = A[i] > 0.5
        tg, tp, th = nogo[np.clip(g, 0, 31)], nogo[np.clip(p, 0, 31)], nogo[np.clip(h, 0, 31)]
        gen = np.asarray(Image.open(d / "frames" / f"q_{i:04d}.png").convert("RGB"), np.float32) / 255
        real = np.asarray(Image.open(d / "gt" / f"q_{i:04d}.png").convert("RGB"), np.float32) / 255
        l1 = np.abs(gen - real).mean(-1)
        rows.append(dict(q, acc_class=acc(p, g, valid), acc_trav=acc(tp, tg, valid),
                         acc_trav_seen=acc(tp, tg, valid & seen), acc_trav_unseen=acc(tp, tg, valid & ~seen),
                         acc_trav_ahead=acc(tp, tg, valid & box), hint_trav_seen=acc(th, tg, valid & seen & (h > 0)),
                         rgb_l1=float(l1.mean()), rgb_l1_unseen=float(l1[~seen].mean()) if (~seen).sum() >= 50 else np.nan,
                         frac_unseen=float((~seen).mean())))
    cols = list(rows[0])
    with open(d / "graded.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    r = {k: np.array([float(x[k]) for x in rows]) for k in ("gap", "coverage", "acc_trav", "acc_trav_ahead", "hint_trav_seen")}
    side = np.array([x["side"] for x in rows])
    for sd, g in sorted({(a, b) for a, b in zip(side, r["gap"])}, key=lambda x: (["in", "behind", "ahead"].index(x[0]), x[1])):
        m = (side == sd) & (r["gap"] == g)
        print(f"  {sd:6s} gap {g:2.0f}: n {m.sum():3d}  coverage {np.nanmean(r['coverage'][m]):.2f}  walkable acc {np.nanmean(r['acc_trav'][m]):.3f}"
              f"  ahead {np.nanmean(r['acc_trav_ahead'][m]):.3f}  hint(seen) {np.nanmean(r['hint_trav_seen'][m]):.3f}")
    print("wrote", d / "graded.csv")


if __name__ == "__main__":
    main()
