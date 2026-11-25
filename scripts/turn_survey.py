"""Where do the recordings turn? (2026-11-25, held-out test, Joana: use the walk's own turns as sideways ground truth)
For every frame b and a window of --window recorded frames that leaves b out (behind: ends --gap frames before b;
ahead: starts --gap frames after b), how far does b look away from EVERY window frame (the smallest heading
difference = the part of b's view no window frame faced), and how far does b sit sideways of the nearest window frame
(perpendicular to that frame's heading)? numpy only; runs on the login node.

    python scripts/turn_survey.py --poses_dir /scratch/.../outputs/poses quad2_01 packard1_11 ...
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", required=True); ap.add_argument("scenes", nargs="+")
    ap.add_argument("--window", type=int, default=21); ap.add_argument("--gaps", default="1,5,10,20,30")
    a = ap.parse_args()
    for s in a.scenes:
        d = np.load(f"{a.poses_dir}/{s}_poses.npz")
        P, Hd = d["positions"][:, :2].astype(float), d["headings"][:, :2].astype(float)
        yaw = np.degrees(np.arctan2(Hd[:, 1], Hd[:, 0])); n = len(P)
        total = abs(((yaw[-1] - yaw[0] + 180) % 360) - 180)
        line = [f"{s:14s} {n} frames, walk {np.linalg.norm(np.diff(P, axis=0), axis=1).sum():5.1f} m, heading change start->end {total:3.0f} deg"]
        for side in ("behind", "ahead"):
            for g in [int(x) for x in a.gaps.split(",")]:
                turn, lat = [], []
                for b in range(n):
                    lo, hi = (b - g - a.window + 1, b - g) if side == "behind" else (b + g, b + g + a.window - 1)
                    lo, hi = max(lo, 0), min(hi, n - 1)
                    if hi - lo + 1 < 8:
                        continue
                    W = np.arange(lo, hi + 1)
                    dy = np.abs(((yaw[b] - yaw[W] + 180) % 360) - 180)
                    turn.append(dy.min())
                    f = W[np.argmin(np.linalg.norm(P[W] - P[b], axis=1))]
                    h = Hd[f] / max(np.linalg.norm(Hd[f]), 1e-9)
                    v = P[b] - P[f]; lat.append(abs(h[0] * v[1] - h[1] * v[0]))
                t, l = np.array(turn), np.array(lat)
                line.append(f"   {side:6s} gap {g}: n {len(t):2d}  turned >=20 deg {(t >= 20).sum():2d}, >=45 {(t >= 45).sum():2d}, >=90 {(t >= 90).sum():2d} (max {t.max():3.0f})"
                            f"   sideways >=0.5 m {(l >= 0.5).sum():2d}, >=1 m {(l >= 1).sum():2d} (max {l.max():.1f} m)")
        print("\n".join(line))


if __name__ == "__main__":
    main()
