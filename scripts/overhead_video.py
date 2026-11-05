"""Side-by-side: the episode video (what the policy saw) next to an overhead
map panel drawn in step with it -- the robot's path so far, the CRASH BOX at
the pose the frame was rendered from, the goal, and for lawn goals the verge
with its radius. Login node only: it reads metrics.json and the scene cloud,
no world model. Joana, 2026-09-06: "an overhead view mp4 consistent with the
video would be super useful".

    python scripts/overhead_video.py --eval_dir <eval dir> --scene gnd_AUw360 \
        --collahead 0.6 [--episodes 3,8] [--verge_dist 1.5]

Writes <eval dir>/overhead/episode_<k>_side.mp4. Frame t of the episode video
is the view BEFORE action t, i.e. rendered at traj[t]; the box is drawn there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid, footprint_samples  # noqa: E402
from src.eval.traversability import load_traversability  # noqa: E402

BODY_L, BODY_W = 0.6, 0.3
COL = {"GOAL": (60, 180, 60), "CRASH": (40, 40, 220), "HALTED": (0, 140, 255), "TIMEOUT": (200, 120, 0), "INCOHERENT": (160, 60, 160)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--collahead", type=float, default=0.6)
    ap.add_argument("--inflate", type=float, default=0.1)
    ap.add_argument("--inflate_classes", default="10,11,13")
    ap.add_argument("--verge_dist", type=float, default=1.5)
    ap.add_argument("--episodes", default="", help="comma list; empty = all")
    ap.add_argument("--window_m", type=float, default=12.0, help="side of the overhead window around the robot")
    ap.add_argument("--px", type=int, default=336, help="overhead panel height in pixels (matches the video height)")
    args = ap.parse_args()
    import cv2

    E = Path(args.eval_dir)
    m = json.load(open(E / "metrics.json"))
    scores = load_traversability(args.trav); nontrav = scores <= 0.1
    c = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
    g = build_label_grid(c["points"], c["labels"].astype(int), nontrav, res=0.1, inflate_m=args.inflate,
                         inflate_classes=tuple(int(v) for v in args.inflate_classes.split(",") if v.strip()), walk_xy=walk)
    L = g.labels; known = L >= 0; nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
    # base map image: white walkable, black non-traversable, grey void; row 0 = y0 (flip for display)
    base = np.full(L.shape + (3,), 128, np.uint8); base[known & ~nt] = 255; base[nt] = 0
    # walkway strip for the verge (walkable cells within 3 m of the walk), as the env does
    wy, wx = np.nonzero(known & ~nt)
    cells = np.c_[g.x0 + (wx + 0.5) * g.res, g.y0 + (wy + 0.5) * g.res].astype(np.float32)
    keep = np.zeros(len(cells), bool)
    for i0 in range(0, len(cells), 4096):
        blk = cells[i0:i0 + 4096]; keep[i0:i0 + 4096] = ((blk[:, None, :] - walk[None, :, :]) ** 2).sum(-1).min(axis=1) <= 9.0
    strip = cells[keep] if keep.any() else walk

    def to_px(xy, cx, cy, half, px):
        # world -> panel pixel (panel is a px x px crop centred on (cx, cy), y up)
        u = (xy[0] - (cx - half)) / (2 * half) * px
        v = px - (xy[1] - (cy - half)) / (2 * half) * px
        return int(round(u)), int(round(v))

    def panel(tr, k, e):
        px = args.px; half = args.window_m / 2.0
        cx, cy = float(tr[k, 0]), float(tr[k, 1])
        # crop the base map around the robot (nearest-cell sampling)
        xs = cx - half + (np.arange(px) + 0.5) / px * 2 * half
        ys = cy + half - (np.arange(px) + 0.5) / px * 2 * half
        ix = np.clip(((xs - g.x0) / g.res).astype(int), 0, L.shape[1] - 1)
        iy = np.clip(((ys - g.y0) / g.res).astype(int), 0, L.shape[0] - 1)
        img = base[iy[:, None], ix[None, :]].copy()
        outside = (xs[None, :] < g.x0) | (xs[None, :] > g.x0 + L.shape[1] * g.res) | (ys[:, None] < g.y0) | (ys[:, None] > g.y0 + L.shape[0] * g.res)
        img[outside] = 128
        col = COL.get(e["outcome"], (0, 0, 0))
        # recorded walk
        pts = [to_px(w, cx, cy, half, px) for w in walk]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 90, 230), 1, cv2.LINE_AA)
        # path so far
        if k > 0:
            pp = [to_px(tr[i, :2], cx, cy, half, px) for i in range(k + 1)]
            cv2.polylines(img, [np.array(pp, np.int32)], False, col, 2, cv2.LINE_AA)
        # goal + verge
        gx, gy = e["goal_xy"]
        gpx = to_px((gx, gy), cx, cy, half, px)
        cv2.drawMarker(img, gpx, col, cv2.MARKER_STAR, 14, 2)
        if e.get("goal_traversable") is False:
            vp = strip[int(np.argmin(np.linalg.norm(strip - np.array([gx, gy], np.float32)[None, :], axis=1)))]
            vpx = to_px(vp, cx, cy, half, px)
            cv2.rectangle(img, (vpx[0] - 5, vpx[1] - 5), (vpx[0] + 5, vpx[1] + 5), col, 1)
            cv2.circle(img, vpx, int(args.verge_dist / (2 * half) * px), col, 1, cv2.LINE_AA)
        # crash box at the reward pose of this frame
        yaw = float(tr[k, 2]); h = np.array([np.cos(yaw), np.sin(yaw)]); n = np.array([-h[1], h[0]])
        ctr = tr[k, :2] + args.collahead * h
        corners = [ctr + a * h + b * n for a, b in ((BODY_L / 2, BODY_W / 2), (BODY_L / 2, -BODY_W / 2), (-BODY_L / 2, -BODY_W / 2), (-BODY_L / 2, BODY_W / 2))]
        fp = footprint_samples(ctr, h, BODY_L, BODY_W, g.res / 2.0)
        cl = g.lookup(fp).astype(int); cl = np.where(cl < 0, 0, cl)
        frac = float((nontrav[cl] & (cl != 0)).mean())
        bc = (40, 40, 220) if frac >= 0.35 else (0, 200, 0)
        cv2.polylines(img, [np.array([to_px(cc, cx, cy, half, px) for cc in corners], np.int32)], True, bc, 2, cv2.LINE_AA)
        # robot
        rp = to_px(tr[k, :2], cx, cy, half, px); tip = to_px(tr[k, :2] + 0.8 * h, cx, cy, half, px)
        cv2.circle(img, rp, 4, col, -1); cv2.line(img, rp, tip, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"t={k} box {frac:.2f} {'CRASH' if frac >= 0.35 else ''}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"ep {e['episode']} {e['outcome']}", (6, px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    out = E / "overhead"; out.mkdir(exist_ok=True)
    want = set(int(v) for v in args.episodes.split(",") if v.strip())
    for e in m["episodes"]:
        if want and e["episode"] not in want:
            continue
        vid = E / f"episode_{e['episode']}.mp4"
        if not vid.exists():
            continue
        tr = np.asarray(e["traj"], float)
        cap = cv2.VideoCapture(str(vid)); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            continue
        H = frames[0].shape[0]; args.px = H
        n = min(len(frames), len(tr))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        wpath = out / f"episode_{e['episode']}_side.mp4"
        w = cv2.VideoWriter(str(wpath), fourcc, 4.0, (frames[0].shape[1] + H, H))
        if not w.isOpened():
            w = cv2.VideoWriter(str(wpath), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (frames[0].shape[1] + H, H))
        for k in range(n):
            side = np.concatenate([frames[k], panel(tr, k, e)], axis=1)
            w.write(side)
        w.release()
        # QuickTime shows mp4v as a green screen: re-encode to H.264 yuv420p with
        # the ffmpeg on the path or the one bundled in the neoverse env.
        import shutil, subprocess, os
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        if ffmpeg:
            tmp = str(wpath) + ".tmp.mp4"
            os.replace(str(wpath), tmp)
            r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "18", str(wpath)])
            if r.returncode == 0 and wpath.exists():
                os.remove(tmp)
            else:
                os.replace(tmp, str(wpath))
                print("    (ffmpeg re-encode failed; kept the mp4v file)")
        print(f"ep {e['episode']:2d} {e['outcome']:<10} {n} frames -> {wpath.name}")


if __name__ == "__main__":
    main()
