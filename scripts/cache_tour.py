"""Zigzag tour through a training cache: record exactly what it serves.

Why: policies train on cached diffused observations. The eval numbers say the
cache works; this shows WHAT the policy actually sees over the whole scene —
dreams, seams, hallucinated people/cars — by weaving lane-to-lane along the
trail with the same lookup machinery the env uses (CachedDiffusedBackend).
Panels per frame: FPV rgb | raw semantics (model belief) | gated semantics
(what the reward reads), with lookup telemetry and a top-down inset.

Usage (Marlowe GPU, via scripts/slurm/cache_tour.sh):
    python scripts/cache_tour.py --scene rugd_trail_00 \
        --obs_cache /scratch/.../outputs/ribbon_cache \
        --clips_dir ... --poses_dir ... --labels_dir ... \
        --out /scratch/.../outputs/cache_tour/CACHETOUR_rugd_trail_00.mp4
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

from src.env.real_calibrated import CalibratedBackendConfig
from src.env.cached_backend import CachedDiffusedBackend
from src.eval.palette import CLASS_COLORS_V14_255, CLASS_COLORS_255


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    P = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    P[0, 0], P[0, 1], P[1, 0], P[1, 1] = c, -s, s, c
    P[:2, 3] = (x, y)
    return P


def _colorize(lab, H, W, tag):
    import cv2
    # v14 runs carry ids 0-13; legacy raster runs carry 30-class ids.
    pal = CLASS_COLORS_V14_255 if int(np.max(lab)) < 14 else CLASS_COLORS_255
    col = pal[np.clip(lab, 0, len(pal) - 1)]
    if col.shape[:2] != (H, W):
        col = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
    col = np.ascontiguousarray(col)
    cv2.putText(col, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--obs_cache", required=True,
                    help="cache root(s), comma-separated for hybrid")
    ap.add_argument("--model_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--amplitude", type=float, default=1.2,
                    help="max lateral offset (m); cache covers +-1.5")
    ap.add_argument("--cycles", type=float, default=3.0,
                    help="full left-right weaves over the whole trail")
    ap.add_argument("--no_alpha_gate", action="store_true")
    ap.add_argument("--sweep_sticky", type=float, default=0.0,
                    help="sticky-sweep penalty (m): served view stays in one "
                         "dream until a sweep is genuinely closer; try 0.3")
    ap.add_argument("--black_invented", action="store_true",
                    help="honest-obs mode: invented pixels served as black")
    ap.add_argument("--frame_range", default=None,
                    help='"a,b": restrict the tour to path frames a..b '
                         '(pilot caches that only cover a segment)')
    ap.add_argument("--wiggle_deg", type=float, default=0.0,
                    help="sinusoidal heading offset amplitude (deg) on top "
                         "of the motion tangent — simulates the policy's "
                         "turn actions (try 30)")
    ap.add_argument("--wiggle_cycles", type=float, default=8.0)
    args = ap.parse_args()

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    world = CachedDiffusedBackend(cfg, args.obs_cache,
                                  alpha_gate=not args.no_alpha_gate,
                                  sweep_switch_penalty_m=args.sweep_sticky)
    world.obs_black_invented = args.black_invented
    world.load_scene(args.scene)
    cal = world._calib[args.scene]

    # Zigzag: follow the recorded trail while weaving laterally. Heading
    # follows the ZIGZAG's own motion, so the camera also sweeps off-path
    # yaws — the tour covers translation AND rotation lookups.
    p = cal.positions[:, :2]
    if args.frame_range:
        f0, f1 = (float(v) for v in args.frame_range.split(","))
    else:
        f0, f1 = 0.0, len(p) - 1.0
    ts = np.linspace(f0, f1, args.frames)
    center = np.stack([np.interp(ts, np.arange(len(p)), p[:, i])
                       for i in range(2)], axis=1)
    d = np.gradient(center, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    lat = args.amplitude * np.sin(2 * np.pi * args.cycles * ts / (len(p) - 1))
    tour = center + lat[:, None] * n
    tv = np.gradient(tour, axis=0)
    yaw = np.arctan2(tv[:, 1], tv[:, 0])
    if args.wiggle_deg:
        u = np.linspace(0.0, 1.0, args.frames)
        yaw = yaw + np.deg2rad(args.wiggle_deg) * np.sin(
            2 * np.pi * args.wiggle_cycles * u)

    import cv2
    lo, hi = p.min(0) - 2.0, p.max(0) + 2.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))

    frames_out = []
    for i in range(args.frames):
        rgb, _, _ = world.render(_pose(tour[i, 0], tour[i, 1], float(yaw[i])))
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        fpv = np.ascontiguousarray(rgb)
        H, W = fpv.shape[:2]

        lk = getattr(world, "_last_lookup", None)
        txt = (f"f{i:03d} lat={lat[i]:+.2f}m"
               + (f" cache={lk[0]*100:.0f}cm/{lk[1]:.0f}deg" if lk else ""))
        cv2.rectangle(fpv, (0, 0), (W, 18), (0, 0, 0), -1)
        cv2.putText(fpv, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # top-down inset, bottom-right: trail gray, tour-so-far white, now red
        m = 110

        def to_px(q):
            return (int(W - m - 6 + (q[0] - lo[0]) / span * m),
                    int(H - 6 - (q[1] - lo[1]) / span * m))
        cv2.rectangle(fpv, (W - m - 10, H - m - 10), (W - 2, H - 2),
                      (0, 0, 0), -1)
        cv2.polylines(fpv, [np.array([to_px(q) for q in p[::2]])], False,
                      (160, 160, 160), 1)
        if i > 1:
            cv2.polylines(fpv, [np.array([to_px(q) for q in tour[:i]])],
                          False, (255, 255, 255), 1)
        cv2.circle(fpv, to_px(tour[i]), 3, (0, 0, 255), -1)
        hd = np.array([np.cos(yaw[i]), np.sin(yaw[i])])
        cv2.arrowedLine(fpv, to_px(tour[i]), to_px(tour[i] + hd * 1.2),
                        (0, 165, 255), 2, tipLength=0.35)

        panels = [fpv]
        raw = getattr(world, "_last_semantic_raw", None)
        used = getattr(world, "_last_semantic_image", None)
        if raw is not None and raw is not used:
            panels.append(_colorize(raw, H, W, "sem RAW (model belief)"))
        if used is not None:
            tag = "sem REWARD (gated)" if len(panels) > 1 else "sem REWARD"
            panels.append(_colorize(used, H, W, tag))
        frames_out.append(np.concatenate(panels, axis=1))

    import imageio.v3 as iio
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(frames_out), fps=10, codec="libx264",
                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {args.out} ({len(frames_out)} frames)")


if __name__ == "__main__":
    main()
