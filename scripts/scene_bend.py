#!/usr/bin/env python3
"""Where is the bend, and does a spawn range actually sit before it?

A corner eval is only a corner test if the BEND lies between the spawn and the
goal. The spawn range 12-22 with goal 48 was copied off an old folder name, never
measured (2026-09-23, Joana: "might not get past the bend either").

    python scripts/scene_bend.py quad2_00 --goal 48 --spawn 12 22
"""
import argparse, math, sys
import numpy as np

POSES = "/scratch/m000204-pm07b/joana/outputs/poses/{}_poses.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--goal", type=int, required=True)
    ap.add_argument("--spawn", type=int, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--win", type=int, default=4, help="frames over which a turn is measured")
    a = ap.parse_args()

    try:
        z = np.load(POSES.format(a.scene))
    except Exception as e:
        sys.exit(f"could not load poses for {a.scene}: {e}")
    key = next((k for k in ("positions", "xyz", "t", "trans") if k in z), None)
    P = np.asarray(z[key] if key else z[list(z.keys())[0]])
    if P.ndim == 3:                      # [N,4,4] transforms
        P = P[:, :3, 3]
    P = P[:, :2]
    N = len(P)

    # heading of the walk, smoothed over --win frames
    hdg = np.full(N, np.nan)
    for i in range(N - a.win):
        d = P[i + a.win] - P[i]
        if np.hypot(*d) > 1e-6:
            hdg[i] = math.atan2(d[1], d[0])
    turn = np.full(N, 0.0)               # signed heading change per frame
    for i in range(1, N):
        if not (np.isnan(hdg[i]) or np.isnan(hdg[i - 1])):
            turn[i] = (hdg[i] - hdg[i - 1] + math.pi) % (2 * math.pi) - math.pi

    print(f"{a.scene}: {N} frames, path {np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)):.1f} m")
    # cumulative turn between consecutive 5-frame blocks, to find where it bends
    print(f"\n{'frames':<12}{'turn':>8}{'cum':>8}{'m from start':>14}")
    cum = 0.0
    for lo in range(0, N - 1, 5):
        hi = min(lo + 5, N - 1)
        t = math.degrees(turn[lo:hi].sum()); cum += t
        seg = np.sum(np.linalg.norm(np.diff(P[:hi + 1], axis=0), axis=1))
        mark = ""
        if abs(t) > 15: mark = "  <== BEND"
        print(f"{lo:>3}-{hi:<8}{t:>7.0f}d{cum:>7.0f}d{seg:>13.1f}{mark}")

    if a.spawn:
        lo, hi = a.spawn
        g = min(a.goal, N - 1)
        bend_in = math.degrees(abs(turn[hi:g]).sum())
        bend_before = math.degrees(abs(turn[lo:hi]).sum())
        d_lo = np.linalg.norm(P[g] - P[lo]); d_hi = np.linalg.norm(P[g] - P[hi])
        print(f"\nspawn {lo}-{hi}, goal {a.goal}:")
        print(f"  straight-line distance to the goal: {d_hi:.1f} m (latest spawn) "
              f"to {d_lo:.1f} m (earliest)")
        print(f"  total turning BETWEEN the spawn range and the goal: {bend_in:.0f} deg")
        print(f"  turning inside the spawn range itself:              {bend_before:.0f} deg")
        if bend_in < 30:
            print("  VERDICT: not a corner test -- the walk barely turns after the spawn.")
        elif bend_in < 60:
            print("  VERDICT: a gentle curve, not a corner.")
        else:
            print("  VERDICT: a real corner -- the robot must turn to reach the goal.")


if __name__ == "__main__":
    main()
