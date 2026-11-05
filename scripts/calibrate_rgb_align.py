#!/usr/bin/env python3
"""What similarity does the generated frame have to the Gaussian raster it was
conditioned on, ON the recorded trajectory vs off it? That number is RGBALIGNTAU.

Reads an offset sweep (outputs/offsets_*_v35b/<scene>/: frames/q_*.png = generated,
raster/q_*.png = the raster, alpha.npz, queries.csv) and prints the similarity
distribution per offset, using the SAME rgb_alignment() the reward charges on.
The threshold should sit just below the on-trajectory (offset 0) 10th percentile,
so faithful frames are never charged and drift is (2026-09-25).

    /users/jmizrahi/.conda/envs/neoverse/bin/python scripts/calibrate_rgb_align.py \\
        /scratch/m000204-pm07b/joana/outputs/offsets_xfar_v35b
"""
import ast, csv, os, sys, textwrap, pathlib, collections
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "src" / "env" / "scene_env.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "rgb_alignment")
ns = {"np": np}; exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns); rgb_alignment = ns["rgb_alignment"]


def main():
    import cv2
    sweep = sys.argv[1]
    res = collections.defaultdict(list)
    for scene in sorted(os.listdir(sweep)):
        d = os.path.join(sweep, scene)
        if not os.path.isfile(os.path.join(d, "queries.csv")):
            continue
        al = np.load(os.path.join(d, "alpha.npz")); A = al[list(al.keys())[0]]
        for r in csv.DictReader(open(os.path.join(d, "queries.csv"))):
            i = int(r["idx"])
            fg, fr = os.path.join(d, "frames", f"q_{i:04d}.png"), os.path.join(d, "raster", f"q_{i:04d}.png")
            if not (os.path.exists(fg) and os.path.exists(fr)):
                continue
            g = cv2.imread(fg)[:, :, ::-1]; h = cv2.imread(fr)[:, :, ::-1]
            if h.shape != g.shape:
                h = cv2.resize(h, (g.shape[1], g.shape[0]))
            a = A[i] if A.ndim == 3 else A[i, ..., 0]
            if a.shape != g.shape[:2]:
                a = cv2.resize(a.astype(np.float32), (g.shape[1], g.shape[0]))
            res[(r["kind"], float(r["offset"]))].append((rgb_alignment(g, h, a), float(r["coverage"])))
    print(f"{'kind':<5}{'offset':>7}{'n':>5}{'cov':>6}{'sim p10':>9}{'median':>8}{'p90':>8}{'no ref':>8}")
    for (k, o), v in sorted(res.items()):
        s = [x for x, _ in v if x is not None]; c = np.mean([y for _, y in v])
        if s:
            print(f"{k:<5}{o:>7.1f}{len(v):>5}{c:>6.2f}{np.percentile(s, 10):>9.3f}{np.median(s):>8.3f}{np.percentile(s, 90):>8.3f}{len(v)-len(s):>8}")
        else:
            print(f"{k:<5}{o:>7.1f}{len(v):>5}{c:>6.2f}{'-':>9}{'-':>8}{'-':>8}{len(v)-len(s):>8}")
    on = [x for (k, o), v in res.items() if o == 0.0 for x, _ in v if x is not None]
    if on:
        print(f"\non-trajectory (offset 0): p10 {np.percentile(on, 10):.3f}  median {np.median(on):.3f}"
              f"  -> RGBALIGNTAU just below the p10, e.g. {np.percentile(on, 10) - 0.03:.2f}")


if __name__ == "__main__":
    main()
