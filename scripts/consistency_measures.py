"""Consistency measures on the seeded offset sweep (2026-11-24, threshold question for the rebuttal).
Runs on the login node (numpy only, no GPU) next to the data; writes one small csv per scene.

Input  <sweep_root>/<scene>/: queries.csv, semantic_labels.npz (seed 0), seed_labels.npz (seeds 1..S-1),
       cameras.npz (K, w2c in the nav frame, ground at z = 0), grade/queries_graded.csv (September measures)
       SAM 3 on the seed-0 frames: <neoverse>/outputs/sam3_labels/offset_frames_<sweep>_<scene>.npz
Output <sweep_root>/<scene>/consistency.csv, one row per view:
  seed_dis_class   share of pixels whose class differs between seed 0 and another seed (mean over seeds)
  seed_dis_trav    the same for walkable-vs-not only (what the reward reads)
  nb_ours          neighbour consistency, our labels: ground points seen in this view AND the previous view of the
                   same series (sideways k-1 m, or the previous turn step), same label in both; NaN for the first view
  nb_sam3          the same with SAM 3's labels, on points SAM 3 labelled in both views
  nb_points        how many shared ground points the neighbour measure used
  mix_ours/mix_sam3  scene-content consistency over the WHOLE image (no point matching): share of each class in this
                   view vs the previous view of the series, 1 - total variation distance (1 = same class mix);
                   SAM 3 version over SAM 3's labelled pixels only
  all_ours/all_sam3  the same against EVERY earlier view of the series (drift, not only flicker): each earlier view
                   sharing >= MIN_SHARED ground points counts, weighted by its shared points; all_points = their sum
  + coverage and the September columns (hint_pix, agree_pix, align) copied from grade/queries_graded.csv

Ground points: pixels whose label (in the earlier view) is a ground class, cast onto the plane z = 0, kept within
MAX_RANGE m of the camera, projected into the other view. Occlusion is not modelled, so the range is kept short.

    python scripts/consistency_measures.py --sweep_root /scratch/.../outputs/offsets_xfar_seeds_v35b
"""
from __future__ import annotations
import argparse, csv, importlib.util, sys
from pathlib import Path
import numpy as np

GROUND = [2, 3, 4, 5, 6, 7, 8, 9]          # v14: trail, grass, rough, water, sidewalk, road, pavement, stairs
NOGO = [3, 5, 10, 11, 12, 13]              # score 0 in traversability_v14_walkway (lawn, water, obstacle, vegetation, person, vehicle)
MAX_RANGE = 6.0
STRIDE = 4
MIN_SHARED = 100


def taxonomy(neoverse: Path):
    spec = importlib.util.spec_from_file_location("ct", neoverse / "diffsynth/utils/class_taxonomy.py")
    ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
    return ct


def ground_points(lab, K, w2c, H, W):
    """World points (z = 0) of the ground-labelled pixels of one view, with their labels."""
    v, u = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
    v, u = v.ravel(), u.ravel()
    keep = np.isin(lab[v, u], GROUND)
    v, u = v[keep], u[keep]
    c2w = np.linalg.inv(w2c.astype(np.float64))
    d = (c2w[:3, :3] @ (np.linalg.inv(K.astype(np.float64)) @ np.stack([u + 0.5, v + 0.5, np.ones_like(u, float)])))
    C = c2w[:3, 3]
    ok = d[2] < -1e-6
    t = np.where(ok, -C[2] / np.where(ok, d[2], -1.0), np.nan)
    P = C[:, None] + d * t
    rng = np.linalg.norm(P[:2] - C[:2, None], axis=0)
    ok &= np.isfinite(t) & (rng <= MAX_RANGE)
    return P[:, ok], v[ok], u[ok]


def project(P, K, w2c, H, W):
    Pc = w2c[:3, :3].astype(np.float64) @ P + w2c[:3, 3:4].astype(np.float64)
    z = Pc[2]
    uv = (K.astype(np.float64) @ Pc)[:2] / np.where(z > 1e-6, z, np.nan)
    ub, vb = np.floor(uv[0]).astype(int), np.floor(uv[1]).astype(int)
    ok = (z > 1e-6) & (ub >= 0) & (ub < W) & (vb >= 0) & (vb < H)
    return ok, vb, ub


def class_mix(lab, valid=False, n=14):
    v = lab[lab > 0] if valid else lab.ravel()
    if v.size == 0:
        return None
    return np.bincount(np.clip(v, 0, n - 1), minlength=n) / v.size


def mix_consistency(labA, labB, valid=False):
    a, b = class_mix(labA, valid), class_mix(labB, valid)
    return np.nan if a is None or b is None else float(1.0 - 0.5 * np.abs(a - b).sum())


def neighbour(labA, labB, KA, wA, KB, wB, H, W, valid=None):
    P, va, ua = ground_points(labA, KA, wA, H, W)
    if P.shape[1] == 0:
        return np.nan, 0
    ok, vb, ub = project(P, KB, wB, H, W)
    a = labA[va[ok], ua[ok]]; b = labB[vb[ok], ub[ok]]
    if valid is not None:
        m = (a > 0) & (b > 0); a, b = a[m], b[m]
    return (float((a == b).mean()) if len(a) else np.nan), int(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", required=True, type=Path)
    ap.add_argument("--neoverse", default="/scratch/m000204-pm07b/joana/NeoVerse", type=Path)
    args = ap.parse_args()
    ct = taxonomy(args.neoverse)
    nogo = np.zeros(32, bool); nogo[NOGO] = True
    for sd in sorted(p for p in args.sweep_root.iterdir() if (p / "cameras.npz").exists()):
        Q = list(csv.DictReader(open(sd / "queries.csv")))
        G = {r["idx"]: r for r in csv.DictReader(open(sd / "grade/queries_graded.csv"))} if (sd / "grade/queries_graded.csv").exists() else {}
        L0 = np.load(sd / "semantic_labels.npz")["labels"].astype(np.int64)
        LS = np.load(sd / "seed_labels.npz")["labels"].astype(np.int64) if (sd / "seed_labels.npz").exists() else None
        cam = np.load(sd / "cameras.npz"); K, Wc = cam["K"], cam["w2c"]
        sam_f = args.neoverse / "outputs/sam3_labels" / f"offset_frames_{args.sweep_root.name}_{sd.name}.npz"
        S3 = None
        if sam_f.exists():
            s = np.load(sam_f)
            lut = ct.remap_array_from_names([str(n) for n in s["class_names"]])
            S3 = lut[np.clip(s["labels"].astype(np.int64), 0, len(lut) - 1)].astype(np.int64)
        else:
            print(f"  {sd.name}: no SAM 3 labels at {sam_f} -- nb_sam3 left empty")
        N, H, W = L0.shape
        prev = {}                                   # (base, kind) -> previous query index in that series
        earlier = {}                                # (base, kind) -> all earlier query indices in that series
        order = sorted(range(N), key=lambda i: (int(Q[i]["base_frame"]), Q[i]["kind"], float(Q[i]["offset"])))
        rows = {}
        for i in order:
            b, kind, o = int(Q[i]["base_frame"]), Q[i]["kind"], float(Q[i]["offset"])
            r = {"idx": i, "base_frame": b, "kind": kind, "offset": o, "coverage": float(Q[i]["coverage"])}
            if LS is not None:
                r["seed_dis_class"] = float(np.mean([(L0[i] != LS[s, i]).mean() for s in range(LS.shape[0])]))
                r["seed_dis_trav"] = float(np.mean([(nogo[np.clip(L0[i], 0, 31)] != nogo[np.clip(LS[s, i], 0, 31)]).mean()
                                                    for s in range(LS.shape[0])]))
            # the series: sideways views in offset order; turned views start from the 0 m sideways view
            key = (b, kind)
            j = prev.get(key)
            if kind == "yaw" and j is None:
                j = prev.get((b, "lat0"))
            if j is not None:
                r["mix_ours"] = mix_consistency(L0[j], L0[i])
                if S3 is not None:
                    r["mix_sam3"] = mix_consistency(S3[j], S3[i], valid=True)
                r["nb_ours"], r["nb_points"] = neighbour(L0[j], L0[i], K[j], Wc[j], K[i], Wc[i], H, W)
                if S3 is not None:
                    r["nb_sam3"], _ = neighbour(S3[j], S3[i], K[j], Wc[j], K[i], Wc[i], H, W, valid=True)
            # drift: against every earlier view of the series (turned views include the 0 m view)
            E = list(earlier.get(key, [])) + ([] if kind == "lat" else earlier.get((b, "lat0"), []))
            for tag, LAB, valid in (("ours", L0, None), ("sam3", S3, True)):
                if LAB is None or not E:
                    continue
                vals, wts = [], []
                for j2 in E:
                    v_, n_ = neighbour(LAB[j2], LAB[i], K[j2], Wc[j2], K[i], Wc[i], H, W, valid=valid)
                    if n_ >= MIN_SHARED and np.isfinite(v_):
                        vals.append(v_); wts.append(n_)
                if vals:
                    r[f"all_{tag}"] = float(np.average(vals, weights=wts))
                    if tag == "ours":
                        r["all_points"] = int(sum(wts))
            prev[key] = i
            earlier.setdefault(key, []).append(i)
            if kind == "lat" and o == 0:
                prev[(b, "lat0")] = i
                earlier[(b, "lat0")] = [i]
            g = G.get(str(i), {})
            for c in ("hint_pix", "agree_pix", "align"):
                r[c] = float(g[c]) if c in g and g[c] not in ("", "nan") else np.nan
            rows[i] = r
        cols = ["idx", "base_frame", "kind", "offset", "coverage", "seed_dis_class", "seed_dis_trav",
                "nb_ours", "nb_sam3", "nb_points", "mix_ours", "mix_sam3", "all_ours", "all_sam3", "all_points", "hint_pix", "agree_pix", "align"]
        with open(sd / "consistency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for i in sorted(rows):
                w.writerow({c: rows[i].get(c, "") for c in cols})
        print(f"{sd.name}: {len(rows)} views -> {sd / 'consistency.csv'}", flush=True)


if __name__ == "__main__":
    main()
