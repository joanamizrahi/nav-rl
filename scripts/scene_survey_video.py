"""Per-scene survey video for CHOOSING scenes by eye (Joana, 2026-09-07):
the ORIGINAL video with the SAM3 labels overlaid (the labels the map is built
from), beside the overhead traversability map with the walk, the current
pose and the frame number. No GPU, no generator: cv2 + numpy on the login
node, ~10 s per scene.

    python scripts/scene_survey_video.py --scenes gnd_GTc3d210 ... --out_dir <dir>

Writes <out_dir>/<scene>_survey.mp4 (H.264 via ffmpeg when available).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid  # noqa: E402
from src.eval.palette import display_palette, CLASS_NAMES_V14  # noqa: E402


def load_scores(trav: str) -> np.ndarray:
    if trav == "none":
        s = np.ones(14); s[[0, 1, 3, 5, 10, 11, 12, 13]] = 0.0; s[7] = 0.85
        return s
    from src.eval.traversability import load_traversability
    return load_traversability(trav)


def find_clip(scene, dirs):
    for d in dirs:
        p = Path(d) / f"{scene}.mp4"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--labels_dir", default="/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14")
    ap.add_argument("--clips_dirs", default="/scratch/m000204-pm07b/joana/data/gnd_clips,/scratch/m000204-pm07b/joana/data/rugd_clips,/scratch/m000204-pm07b/joana/data/gnd_survey")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sem_palette", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.45, help="label overlay opacity")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--map_px", type=int, default=336, help="overhead panel size (px), square")
    ap.add_argument("--window_m", type=float, default=16.0, help="overhead window side around the robot")
    args = ap.parse_args()
    import cv2
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pal = display_palette(args.sem_palette)                      # [14,3] RGB
    pal_bgr = pal[:, ::-1].astype(np.uint8)
    scores = load_scores(args.trav); nontrav = scores <= 0.1
    clip_dirs = [d for d in args.clips_dirs.split(",") if d]
    for sc in args.scenes:
        clip = find_clip(sc, clip_dirs)
        if clip is None:
            print(f"{sc}: no clip"); continue
        lab_p = Path(args.labels_dir) / f"{sc}.npz"
        labels = np.load(lab_p)["labels"] if lab_p.exists() else None
        cloud_p = Path(args.clouds_dir) / f"{sc}_cloud.npz"
        g = walk = None
        if cloud_p.exists():
            c = np.load(cloud_p)
            walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
            g = build_label_grid(c["points"], c["labels"].astype(int), nontrav, res=0.1, inflate_m=0.1,
                                 inflate_classes=(10, 11, 13), walk_xy=walk)
            L = g.labels; known = L >= 0; nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
            base = np.full(L.shape + (3,), 128, np.uint8); base[known & ~nt] = 255; base[nt] = 0
        cap = cv2.VideoCapture(str(clip)); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            print(f"{sc}: empty clip"); continue
        H, W = frames[0].shape[:2]
        px = args.map_px; half = args.window_m / 2.0
        def panel(k):
            if g is None:
                p = np.full((px, px, 3), 40, np.uint8)
                cv2.putText(p, "NO CLOUD YET", (20, px // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                return p
            i = min(k, len(walk) - 1); cx, cy = float(walk[i, 0]), float(walk[i, 1])
            xs = cx - half + (np.arange(px) + 0.5) / px * 2 * half
            ys = cy + half - (np.arange(px) + 0.5) / px * 2 * half
            ix = np.clip(((xs - g.x0) / g.res).astype(int), 0, L.shape[1] - 1)
            iy = np.clip(((ys - g.y0) / g.res).astype(int), 0, L.shape[0] - 1)
            img = base[iy[:, None], ix[None, :]].copy()
            outside = (xs[None, :] < g.x0) | (xs[None, :] > g.x0 + L.shape[1] * g.res) | (ys[:, None] < g.y0) | (ys[:, None] > g.y0 + L.shape[0] * g.res)
            img[outside] = 128
            def to_px(xy):
                return int((xy[0] - (cx - half)) / (2 * half) * px), int(px - (xy[1] - (cy - half)) / (2 * half) * px)
            pts = [to_px(w) for w in walk]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 90, 230), 2, cv2.LINE_AA)
            for j in range(0, len(walk), 10):
                u, v = to_px(walk[j]); cv2.putText(img, str(j), (u + 4, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, str(j), (u + 4, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            dv = walk[min(i + 1, len(walk) - 1)] - walk[max(i - 1, 0)]; n = np.linalg.norm(dv) + 1e-9; hd = dv / n
            rp = to_px(walk[i]); tip = to_px(walk[i] + 1.5 * hd)
            cv2.circle(img, rp, 6, (0, 0, 255), -1); cv2.line(img, rp, tip, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"frame {k}   white walkable / black non-walkable / grey unknown", (6, px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"frame {k}   white walkable / black non-walkable / grey unknown", (6, px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            return img
        # legend strip under the video: class colours + names
        leg = np.full((26, W, 3), 30, np.uint8); x = 6
        for cid, nm in enumerate(CLASS_NAMES_V14):
            cv2.rectangle(leg, (x, 6), (x + 14, 20), tuple(int(v) for v in pal_bgr[cid]), -1)
            cv2.putText(leg, nm[:7], (x + 17, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            x += 17 + 7 * min(len(nm), 7) + 10
        rows = []
        for k, f in enumerate(frames):
            if labels is not None and k < len(labels):
                lab = labels[k]
                if lab.shape != (H, W):
                    lab = cv2.resize(lab.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
                col = pal_bgr[np.clip(lab.astype(int), 0, 13)]
                over = cv2.addWeighted(f, 1.0 - args.alpha, col, args.alpha, 0)
            else:
                over = f.copy()
            cv2.putText(over, f"{sc}  frame {k}  (original video + SAM3 labels)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(over, f"{sc}  frame {k}  (original video + SAM3 labels)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            left = np.concatenate([over, leg], axis=0)
            pm = panel(k)
            pm = cv2.resize(pm, (left.shape[0], left.shape[0]), interpolation=cv2.INTER_NEAREST)
            rows.append(np.concatenate([left, pm], axis=1))
        Hh, Ww = rows[0].shape[:2]
        wpath = out / f"{sc}_survey.mp4"
        w = cv2.VideoWriter(str(wpath), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (Ww, Hh))
        for r in rows:
            w.write(r)
        w.release()
        import shutil, subprocess, os
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg; ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        if ffmpeg:
            tmp = str(wpath) + ".tmp.mp4"; os.replace(str(wpath), tmp)
            r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(wpath)])
            if r.returncode == 0 and wpath.exists():
                os.remove(tmp)
            else:
                os.replace(tmp, str(wpath))
        print(f"{sc}: {len(rows)} frames, labels {'yes' if labels is not None else 'NO'}, map {'yes' if g is not None else 'NO'} -> {wpath.name}", flush=True)
    print(f"==> {out}")


if __name__ == "__main__":
    main()
