"""Grade a held-out sweep (scripts/heldout_sweep.py): generated labels and image against the REAL frame.
CPU only (numpy + PIL), runs on the login node next to the data. Writes <sweep>/graded.csv, one row per query:
  acc_class        generated class == SAM 3 on the real frame, per pixel, over the WHOLE image except pixels SAM 3
                   left unlabelled, sky, and person/vehicle
  acc_trav         the same for walkable vs not (what the reward reads)
  chance_trav      what always guessing the frame's bigger side (walkable or not) would score on the same pixels
  acc_trav_ground / acc_class_ground   only pixels SAM 3 calls ground (trail..stairs): is the floor labelled right
  acc_trav_seen / acc_trav_unseen   split by the window's opacity (alpha > 0.5 = some kept frame saw it)
  acc_trav_ahead   walkable vs not in the ground-ahead box (bottom 38 %, middle half: where the footprint lands)
  hint_trav_seen   the rasterized hint's walkable-vs-not accuracy on seen pixels (sanity: should be high)
  rgb_l1 / rgb_l1_unseen   mean |generated - real| per channel, 0-1
  frac_unseen      share of pixels with alpha <= 0.5
  all_trav / all_class            WHOLE IMAGE, every pixel (2026-11-25, the supplement's numbers): traversable vs not
                   with non-traversable = s(c)=0 (the paper's O), and exact class
  s3_all_trav / s3_all_class      the same for SAM 3 on the generated image (with --sam3)
  --sam3 <npz>     (optional) SAM 3 run on the GENERATED frames (sam3_precompute_labels.py on frames/, raw prompt
                   indices, remapped to v14 here): the "segment the generated image" baseline, same truth:
    s3_acc_trav      SAM 3 on generated vs SAM 3 on real, walkable vs not; pixels SAM 3 left unlabelled on the generated
                     frame count as WRONG (as in the paper's 76.6 vs 57.4 comparison)
    s3_acc_trav_lab  the same over only the pixels it did label;   s3_labelled  share of scored pixels it labelled
Ground truth is SAM 3 on the real frame, not a human: these are agreement-with-SAM 3 numbers.

    python scripts/grade_heldout.py --sweep /scratch/.../outputs/heldout_v35b/quad2_01
"""
import argparse, csv, importlib.util
from pathlib import Path
import numpy as np
from PIL import Image

NOGO = [3, 5, 10, 11, 12, 13]        # traversability_v14_walkway score 0 (as in consistency_measures.py)
MOVERS = [12, 13]                    # person, vehicle: they move between frames, left out of every score
SKY = 1                              # 2026-11-25: left out too -- it is neither walkable nor no-go, and counting it as
                                     # "walkable" (not in NOGO) made every view agree on the sky for free
GROUND = [2, 3, 4, 5, 6, 7, 8, 9]    # trail grass rough water sidewalk road pavement stairs
# 2026-11-25 (Joana): WHOLE-IMAGE scores, no pixel left out. Non-traversable = the paper's set O = every class with
# s(c) = 0 in traversability_v14_walkway.yaml (supplement Table 2): void, sky, grass, water, obstacle, vegetation,
# person, vehicle. Traversable: trail, rough, sidewalk, road, pavement, stairs.
S_ZERO = [0, 1, 3, 5, 10, 11, 12, 13]


def acc(a, b, m):
    return float((a[m] == b[m]).mean()) if m.sum() >= 50 else np.nan


def s3_scores(s3, g, valid, nogo):
    lab = valid & (s3 > 0) & (s3 != SKY)
    ok = nogo[np.clip(s3, 0, 31)] == nogo[np.clip(g, 0, 31)]
    n = valid.sum()
    return dict(s3_acc_trav=float((ok & lab).sum() / n) if n >= 50 else np.nan,
                s3_acc_trav_lab=acc(nogo[np.clip(s3, 0, 31)], nogo[np.clip(g, 0, 31)], lab),
                s3_labelled=float(lab.sum() / n) if n >= 50 else np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--sam3", type=Path, default=None)
    ap.add_argument("--neoverse", type=Path, default=Path("/scratch/m000204-pm07b/joana/NeoVerse")); a = ap.parse_args()
    S3 = None
    if a.sam3 is not None and a.sam3.exists():
        spec = importlib.util.spec_from_file_location("ct", a.neoverse / "diffsynth/utils/class_taxonomy.py")
        ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
        z = np.load(a.sam3); lut = ct.remap_array_from_names([str(n) for n in z["class_names"]])
        S3 = lut[np.clip(z["labels"].astype(np.int64), 0, len(lut) - 1)].astype(np.int64)
        print(f"SAM 3 on generated frames: {S3.shape} from {a.sam3}")
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
        valid = (g > 0) & (g != SKY) & ~np.isin(g, MOVERS)
        gnd = valid & np.isin(g, GROUND)
        seen = A[i] > 0.5
        tg, tp, th = nogo[np.clip(g, 0, 31)], nogo[np.clip(p, 0, 31)], nogo[np.clip(h, 0, 31)]
        gen = np.asarray(Image.open(d / "frames" / f"q_{i:04d}.png").convert("RGB"), np.float32) / 255
        real = np.asarray(Image.open(d / "gt" / f"q_{i:04d}.png").convert("RGB"), np.float32) / 255
        l1 = np.abs(gen - real).mean(-1)
        o = np.zeros(32, bool); o[S_ZERO] = True
        og, op = o[np.clip(g, 0, 31)], o[np.clip(p, 0, 31)]
        wa = float(og.mean())
        whole = dict(all_trav=float((og == op).mean()), all_class=float((g == p).mean()), all_chance=max(wa, 1 - wa))
        if S3 is not None:
            o3 = o[np.clip(S3[i], 0, 31)]
            whole.update(s3_all_trav=float((og == o3).mean()), s3_all_class=float((g == S3[i]).mean()))
        ws = float(tg[valid].mean()) if valid.sum() else np.nan          # share of no-go pixels in the real frame
        rows.append(dict(q, acc_class=acc(p, g, valid), acc_trav=acc(tp, tg, valid),
                         chance_trav=max(ws, 1 - ws) if ws == ws else np.nan,      # always guessing the bigger side
                         acc_trav_ground=acc(tp, tg, gnd), acc_class_ground=acc(p, g, gnd),
                         acc_trav_seen=acc(tp, tg, valid & seen), acc_trav_unseen=acc(tp, tg, valid & ~seen),
                         acc_trav_ahead=acc(tp, tg, valid & box), hint_trav_seen=acc(th, tg, valid & seen & (h > 0)),
                         rgb_l1=float(l1.mean()), rgb_l1_unseen=float(l1[~seen].mean()) if (~seen).sum() >= 50 else np.nan,
                         frac_unseen=float((~seen).mean()),
                         **whole, **({} if S3 is None else s3_scores(S3[i], g, valid, nogo))))
    cols = list(rows[0])
    with open(d / "graded.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    r = {k: np.array([float(x[k]) for x in rows]) for k in ("gap", "coverage", "acc_trav", "acc_trav_ahead", "hint_trav_seen")}
    side = np.array([x["side"] for x in rows])
    for sd, g in sorted({(a, b) for a, b in zip(side, r["gap"])}, key=lambda x: (["in", "behind", "ahead"].index(x[0]), x[1])):
        m = (side == sd) & (r["gap"] == g)
        print(f"  {sd:6s} gap {g:2.0f}: n {m.sum():3d}  coverage {np.nanmean(r['coverage'][m]):.2f}  walkable acc {np.nanmean(r['acc_trav'][m]):.3f}"
              f"  ahead {np.nanmean(r['acc_trav_ahead'][m]):.3f}  hint(seen) {np.nanmean(r['hint_trav_seen'][m]):.3f}")
    cov = r["coverage"]; acc_ = r["acc_trav"]
    print("  by coverage:   " + "  ".join(f"{lo:.1f}-{lo + 0.1:.1f}: {np.nanmean(acc_[(cov >= lo) & (cov < lo + 0.1)]):.2f} (n {((cov >= lo) & (cov < lo + 0.1)).sum()})"
                                      for lo in np.arange(0, 1.0, 0.1) if ((cov >= lo) & (cov < lo + 0.1)).sum()))
    if "turn_deg" in rows[0]:
        tu = np.array([float(x["turn_deg"]) for x in rows]); la = np.array([float(x["lat_m"]) for x in rows])
        for nm, v, edges in (("turned (deg)", tu, [0, 10, 20, 45, 90, 181]), ("sideways (m)", la, [0, 0.25, 0.5, 1, 2, 99])):
            print(f"  by {nm}: " + "  ".join(f"{a}-{b}: acc {np.nanmean(acc_[(v >= a) & (v < b)]):.2f} cov {np.nanmean(cov[(v >= a) & (v < b)]):.2f} (n {((v >= a) & (v < b)).sum()})"
                                           for a, b in zip(edges[:-1], edges[1:]) if ((v >= a) & (v < b)).sum()))
    print("wrote", d / "graded.csv")


if __name__ == "__main__":
    main()
