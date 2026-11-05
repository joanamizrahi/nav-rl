"""Top-down terrain map + goal/spawn audit, straight from the Gaussian cloud.

Why not a render: rendering the world from 25 m up asks the splats for a view
no camera ever took (they are one-sided), so the picture is mush regardless of
how good the scene is. The CLOUD, though, knows exactly where every labeled
gaussian sits. Project the ground-level gaussians to XY, take the majority
class per cell, and you get an honest floor map — then drop the REAL sampled
spawns and goals on top of it.

Answers three questions with numbers, not vibes:
  1. What terrain do sampled GOALS land on? (walkable / grass / obstacle /
     unobserved-void)  -> is the edge-stopping curriculum actually present?
  2. What terrain do sampled SPAWNS land on? -> is the spawn filter working
     after jitter?
  3. What terrain does the RECORDED PATH sit on? -> THE label sanity check:
     the robot demonstrably walked there, so anything our labels call
     non-traversable along that path is a LABELING error, not terrain.

Usage (login node, CPU):
    python scripts/topdown_audit.py --scene gnd_AUw240 \
        --cloud /scratch/.../outputs/scene_clouds/gnd_AUw240_cloud.npz \
        --out /scratch/.../outputs/topdown_audit
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env.real_calibrated import NavCalibration

# v14 taxonomy inlined (name, RGB) — importing diffsynth.utils.class_taxonomy
# drags in the whole diffusion stack (modelscope), which the login-node base
# env lacks; this audit is pure numpy/matplotlib by design.
V14 = [
    ("void",       (  0,   0,   0)),
    ("sky",        (200, 225, 245)),
    ("trail",      (150, 100,  55)),
    ("grass",      ( 75, 190,  80)),
    ("rough",      ( 95,  65,  35)),
    ("water",      ( 50, 120, 200)),
    ("sidewalk",   (210, 210, 210)),
    ("road",       ( 70,  70,  85)),
    ("pavement",   (235, 205, 150)),
    ("stairs",     (220, 140,  80)),
    ("obstacle",   (185,  55,  50)),
    ("vegetation", (170, 200,  55)),
    ("person",     (205,  70, 145)),
    ("vehicle",    (110, 130, 220)),
]
V14_NAMES = [n for n, _ in V14]

# Fallback if --trav can't be read: the original J-spec strict table.
STRICT = {6: 1.0, 8: 1.0, 7: 0.15, 9: 0.3}


def load_trav(path):
    """Read a traversability yaml (id -> {name, score}) so the audit reports
    the SAME scores training uses. Falls back to STRICT on any problem."""
    try:
        import yaml
        d = yaml.safe_load(open(path))
        t = {int(k): float(v["score"]) for k, v in d.items()}
        print(f"[trav] {path}: "
              + ", ".join(f"{V14_NAMES[k]}={s:g}" for k, s in sorted(t.items())
                          if s > 0), flush=True)
        return t
    except Exception as e:
        print(f"[trav] could not read {path} ({e}); using strict fallback",
              flush=True)
        return STRICT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--poses_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/poses")
    ap.add_argument("--labels_dir",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14")
    ap.add_argument("--cone_deg", type=float, default=50.0)
    ap.add_argument("--dist_range", default="5,10")
    ap.add_argument("--spawn_yaw_jitter", type=float, default=20.0)
    ap.add_argument("--spawn_lat_jitter", type=float, default=0.4)
    ap.add_argument("--spawn_classes", default="6,8")
    ap.add_argument("--spawn_min", type=int, default=10)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--cell_m", type=float, default=0.25)
    ap.add_argument("--ground_lo", type=float, default=-0.35)
    ap.add_argument("--ground_hi", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trav",
                    default=str(REPO_ROOT / "config/traversability_v14_walkway.yaml"),
                    help="traversability yaml whose scores the report uses "
                         "(default: the walkway table the J-arms now train on)")
    ap.add_argument("--rgb", action="store_true",
                    help="paint the map with the gaussians' own COLORS "
                         "(mean per cell) instead of semantic class — the "
                         "label-independent view of the same ground")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo_d, hi_d = (float(v) for v in args.dist_range.split(","))
    allowed = set(int(c) for c in args.spawn_classes.split(","))
    trav = load_trav(args.trav)
    rng = np.random.default_rng(args.seed)

    cal = NavCalibration.from_npz(f"{args.poses_dir}/{args.scene}_poses.npz")
    path = np.asarray(cal.positions, dtype=float)[:, :2]

    # Recorded-heading vs path-tangent, at EVERY pose (2026-09-01). The
    # single-frame version of this check (drive_preview's [spin] line) reads
    # a corner as a broken scene: at a turn the tangent swings ~90 deg over
    # two frames while the camera lags. Systematic mis-registration shows up
    # as a large MEDIAN; a corner shows up as a small median with a big p90.
    angs = []
    for i in range(2, len(path) - 2):
        fw = path[i + 1] - path[i - 1]
        if np.linalg.norm(fw) < 1e-6:
            continue
        rp = cal.robot_pose_nav(i)[:2, 0]
        angs.append(np.degrees(np.arctan2(rp[0] * fw[1] - rp[1] * fw[0],
                                          float(np.dot(rp, fw)))))
    if angs:
        a = np.abs(np.array(angs))
        print(f"[heading] recorded-vs-tangent over {len(a)} poses: "
              f"median {np.median(a):.1f}deg, p90 {np.percentile(a, 90):.1f}deg, "
              f"{100.0 * (a > 30).mean():.0f}% of poses >30deg "
              f"({'OK' if np.median(a) < 15 else 'SUSPECT'})", flush=True)

    z = np.load(args.cloud)
    pts, lab = z["points"], z["labels"].astype(int)
    ground = (pts[:, 2] > args.ground_lo) & (pts[:, 2] < args.ground_hi)
    gp, gl = pts[ground][:, :2], lab[ground]
    gc = z["colors"][ground].astype(np.float64) if "colors" in z else None
    print(f"[cloud] {len(pts)} gaussians, {ground.sum()} at ground level "
          f"(z in [{args.ground_lo},{args.ground_hi}] m)", flush=True)

    # ---- majority-class grid (the floor map) ----
    margin = 12.0
    lo = np.minimum(gp.min(0), path.min(0)) - margin
    hi = np.maximum(gp.max(0), path.max(0)) + margin
    nx, ny = (np.ceil((hi - lo) / args.cell_m).astype(int) + 1)
    votes = np.zeros((nx * ny, len(V14)), dtype=np.int32)
    ix = np.clip(((gp[:, 0] - lo[0]) / args.cell_m).astype(int), 0, nx - 1)
    iy = np.clip(((gp[:, 1] - lo[1]) / args.cell_m).astype(int), 0, ny - 1)
    np.add.at(votes, (ix * ny + iy, np.clip(gl, 0, len(V14) - 1)), 1)
    total = votes.sum(1)
    cls = np.where(total > 0, votes.argmax(1), -1).reshape(nx, ny)

    def terrain_at(xy):
        i = int(np.clip((xy[0] - lo[0]) / args.cell_m, 0, nx - 1))
        j = int(np.clip((xy[1] - lo[1]) / args.cell_m, 0, ny - 1))
        win = cls[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].ravel()
        win = win[win >= 0]
        if len(win) == 0:
            return -1
        vals, cnt = np.unique(win, return_counts=True)
        return int(vals[cnt.argmax()])

    # ---- sample episodes exactly like J-50 training does ----
    labels = np.load(f"{args.labels_dir}/{args.scene}.npz")["labels"]
    H, W = labels.shape[-2:]
    patch = labels[:, int(H * 0.8):, int(W * 0.35):int(W * 0.65)]
    ok = np.array([np.bincount(p.ravel(), minlength=len(V14)).argmax() in allowed
                   for p in patch])
    cand = [f for f in range(args.spawn_min, min(len(path) - 6, len(ok)))
            if ok[f]]
    print(f"[spawn filter] {int(ok.sum())}/{len(ok)} frames spawnable, "
          f"{len(cand)} in range", flush=True)
    if not cand:
        cand = list(range(args.spawn_min, len(path) - 6))
        print("[spawn filter] WARNING: none valid, using unfiltered", flush=True)

    spawns, goals, headings = [], [], []
    for _ in range(args.samples):
        f = int(cand[rng.integers(0, len(cand))])
        p = path[f].copy()
        fwd = path[min(f + 1, len(path) - 1)] - path[max(f - 1, 0)]
        base = float(np.arctan2(fwd[1], fwd[0]))
        yaw = base + np.deg2rad(rng.uniform(-1, 1) * args.spawn_yaw_jitter)
        p = p + rng.uniform(-1, 1) * args.spawn_lat_jitter * np.array(
            [-np.sin(base), np.cos(base)])
        th = base + np.deg2rad(rng.uniform(-1, 1) * args.cone_deg / 2.0)
        d = rng.uniform(lo_d, hi_d)
        spawns.append(p)
        headings.append(yaw)
        goals.append(p + d * np.array([np.cos(th), np.sin(th)]))
    spawns, goals = np.array(spawns), np.array(goals)

    def report(name, xys):
        ids = np.array([terrain_at(q) for q in xys])
        print(f"\n=== {name} (n={len(ids)}) ===", flush=True)
        walk = 0
        for c in np.unique(ids):
            n = int((ids == c).sum())
            nm = "UNOBSERVED-VOID" if c < 0 else V14_NAMES[c]
            sc = 0.0 if c < 0 else trav.get(int(c), 0.0)
            if sc >= 0.5:
                walk += n
            print(f"  {nm:>16}: {n:4d}  ({100.0*n/len(ids):5.1f}%)  "
                  f"trav={sc:.2f}", flush=True)
        print(f"  --> traversable share: {100.0*walk/len(ids):.1f}%", flush=True)
        return ids

    goal_ids = report("SAMPLED GOALS", goals)
    report("SAMPLED SPAWNS (after jitter)", spawns)
    report("RECORDED PATH (robot WALKED here - anything non-traversable "
           "here is a LABEL ERROR)", path)

    # ---- picture ----
    img = np.zeros((nx, ny, 3), dtype=np.uint8)
    img[:] = (25, 25, 30)
    if args.rgb and gc is not None:
        csum = np.zeros((nx * ny, 3))
        np.add.at(csum, ix * ny + iy, gc)
        with np.errstate(invalid="ignore"):
            mean = (csum / np.maximum(total, 1)[:, None]).reshape(nx, ny, 3)
        occ = (total > 0).reshape(nx, ny)
        img[occ] = np.clip(mean[occ], 0, 255).astype(np.uint8)
    else:
        for c in range(len(V14)):
            img[cls == c] = V14[c][1]
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower",
              extent=[lo[0], lo[0] + nx * args.cell_m,
                      lo[1], lo[1] + ny * args.cell_m])
    ax.plot(path[:, 0], path[:, 1], "-", color="deepskyblue", lw=2.5,
            label="recorded path")
    ax.quiver(spawns[:, 0], spawns[:, 1],
              np.cos(headings), np.sin(headings), color="orange",
              scale=40, width=0.003, label="spawns (jittered)")
    void = goal_ids < 0
    ax.plot(goals[~void, 0], goals[~void, 1], "+", color="lime", ms=7,
            mew=1.6, label="goals (on cloud)")
    if void.any():
        ax.plot(goals[void, 0], goals[void, 1], "x", color="red", ms=7,
                mew=1.6, label="goals (UNOBSERVED)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"{args.scene}  cone {args.cone_deg:.0f}deg  "
                 f"goals {lo_d:.0f}-{hi_d:.0f} m  "
                 f"jitter {args.spawn_yaw_jitter:.0f}deg/"
                 f"{args.spawn_lat_jitter:.1f} m  (gaussian ground labels)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"TOPDOWN_{args.scene}{'_rgb' if args.rgb else ''}.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    print(f"\n==> {f}", flush=True)


if __name__ == "__main__":
    main()
