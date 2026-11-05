"""Top-down goal-cone preview per scene (J-certification tool, 2026-08-31).

For each scene: recorded path + a tangent-centered goal wedge (cone angle,
dmin-dmax ring) drawn at every Nth pose — shows exactly where J-training can
place goals, scene by scene. CPU-only (poses npz + matplotlib), login-safe.

Usage:
    python scripts/cone_preview.py \
        --scenes gnd_AUw240 gnd_AUw360 sitex_w135 sitex_w180 \
                 gtown2c1_w150 gtown2c1_w210 \
        --cone_deg 80 --out /scratch/.../outputs/cone_previews
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env.real_calibrated import NavCalibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--poses_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/poses")
    ap.add_argument("--cone_deg", type=float, default=80.0)
    ap.add_argument("--dist_range", default="5,10")
    ap.add_argument("--every", type=int, default=10,
                    help="draw a wedge every N poses")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dmin, dmax = (float(v) for v in args.dist_range.split(","))
    half = np.deg2rad(args.cone_deg) / 2.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        cal = NavCalibration.from_npz(f"{args.poses_dir}/{scene}_poses.npz")
        pos = np.asarray(cal.positions)[:, :2]
        fig, ax = plt.subplots(figsize=(9, 9))
        for i in range(2, len(pos) - 2, args.every):
            fwd = pos[min(i + 1, len(pos) - 1)] - pos[max(i - 1, 0)]
            n = np.linalg.norm(fwd)
            if n < 1e-6:
                continue
            base = float(np.arctan2(fwd[1], fwd[0]))
            th = np.linspace(base - half, base + half, 24)
            ring = np.concatenate([
                pos[i] + dmin * np.stack([np.cos(th), np.sin(th)], 1),
                pos[i] + dmax * np.stack([np.cos(th[::-1]),
                                          np.sin(th[::-1])], 1)])
            ax.fill(ring[:, 0], ring[:, 1], alpha=0.10, color="tab:green")
            ax.plot(pos[i, 0], pos[i, 1], ".", color="tab:orange", ms=5)
        ax.plot(pos[:, 0], pos[:, 1], "k-", lw=1.8, label="recorded path")
        ax.plot(pos[0, 0], pos[0, 1], "ks", ms=8, label="pose 0")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(f"{scene}   cone {args.cone_deg:.0f}deg   "
                     f"goals {dmin:.0f}-{dmax:.0f}m   "
                     f"wedge every {args.every} poses")
        f = out / f"CONE{args.cone_deg:.0f}_{scene}.png"
        fig.savefig(f, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"==> {f}", flush=True)


if __name__ == "__main__":
    main()
