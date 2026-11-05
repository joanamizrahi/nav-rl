"""One picture per scene: the overhead map (white walkable, black non-walkable,
grey unknown) with the recorded walk and its frame numbers, next to a contact
sheet of the ORIGINAL video frames at those frame numbers. For picking, by
eye, the scenes and stretches that contain corners, narrow passages and open
areas (Joana, 2026-09-07: "a clean folder with overhead views and the og
video"). cv2 + numpy only; run with the neoverse python on the login node.

    python scripts/scene_sheet.py --scenes gnd_AUw360 gnd_AUd150 --out_dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid  # noqa: E402


def load_scores(trav: str) -> np.ndarray:
    if trav == "none":
        s = np.ones(14); s[[0, 1, 3, 5, 10, 11, 12, 13]] = 0.0; s[7] = 0.85
        return s
    from src.eval.traversability import load_traversability
    return load_traversability(trav)


def find_clip(scene: str, dirs) -> "Path | None":
    for d in dirs:
        p = Path(d) / f"{scene}.mp4"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--clips_dirs", default="/scratch/m000204-pm07b/joana/data/gnd_clips,/scratch/m000204-pm07b/joana/data/rugd_clips,/scratch/m000204-pm07b/joana/data/gnd_survey")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--every", type=int, default=10, help="label / show every Nth walk frame")
    ap.add_argument("--px_per_m", type=float, default=20.0)
    ap.add_argument("--tile_w", type=int, default=280)
    args = ap.parse_args()
    import cv2
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.trav); nontrav = scores <= 0.1
    clip_dirs = [d for d in args.clips_dirs.split(",") if d]
    for sc in args.scenes:
        p = Path(args.clouds_dir) / f"{sc}_cloud.npz"
        if not p.exists():
            # no cloud yet: still show the video so the scene can be judged by eye
            clip = find_clip(sc, clip_dirs)
            tiles = []
            if clip is not None:
                cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(0, n, args.every):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
                    if not ok:
                        continue
                    fr = cv2.resize(fr, (args.tile_w, int(args.tile_w * fr.shape[0] / fr.shape[1])))
                    cv2.putText(fr, f"frame {i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(fr, f"frame {i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(fr)
                cap.release()
            if not tiles:
                print(f"{sc}: no cloud and no clip"); continue
            cols = 3; th, tw = tiles[0].shape[:2]; rows = int(np.ceil(len(tiles) / cols))
            sheet = np.zeros((rows * th + 30, cols * tw, 3), np.uint8)
            cv2.putText(sheet, f"{sc}: NO CLOUD YET (video only)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            for k, t in enumerate(tiles):
                r, cc = divmod(k, cols); sheet[30 + r * th:30 + (r + 1) * th, cc * tw:(cc + 1) * tw] = t
            cv2.imwrite(str(out / f"{sc}_sheet.png"), sheet)
            print(f"{sc}: no cloud, {len(tiles)} video frames -> {sc}_sheet.png", flush=True)
            continue
        c = np.load(p)
        walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
        g = build_label_grid(c["points"], c["labels"].astype(int), nontrav, res=0.1, inflate_m=0.1,
                             inflate_classes=(10, 11, 13), walk_xy=walk)
        L = g.labels; known = L >= 0; nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
        # overhead image: y up -> flip rows; scale to px_per_m
        base = np.full(L.shape + (3,), 128, np.uint8); base[known & ~nt] = 255; base[nt] = 0
        img = base[::-1].copy()                       # row 0 = top = max y
        s = args.px_per_m * g.res
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        H = img.shape[0]
        def to_px(xy):
            return int((xy[0] - g.x0) * args.px_per_m), int(H - (xy[1] - g.y0) * args.px_per_m)
        pts = [to_px(w) for w in walk]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 90, 230), 2, cv2.LINE_AA)
        for i in range(0, len(walk), args.every):
            u, v = to_px(walk[i])
            cv2.circle(img, (u, v), 5, (0, 90, 230), -1)
            cv2.putText(img, str(i), (u + 6, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(i), (u + 6, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # start arrow
        u0, v0 = to_px(walk[0]); u1, v1 = to_px(walk[min(3, len(walk) - 1)])
        cv2.arrowedLine(img, (u0, v0), (u1, v1), (0, 200, 0), 3, tipLength=0.5)
        cv2.putText(img, f"{sc}  walk {float(np.linalg.norm(np.diff(walk, axis=0), axis=1).sum()):.0f} m, {len(walk)} frames  (white walkable, black non-walkable, grey unknown)",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        # contact sheet from the original video at the labelled frames
        clip = find_clip(sc, clip_dirs)
        tiles = []
        if clip is not None:
            cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = list(range(0, min(n, len(walk)), args.every))
            for i in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
                if not ok:
                    continue
                fr = cv2.resize(fr, (args.tile_w, int(args.tile_w * fr.shape[0] / fr.shape[1])))
                cv2.putText(fr, f"frame {i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(fr, f"frame {i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(fr)
            cap.release()
        if tiles:
            cols = 3; th, tw = tiles[0].shape[:2]
            rows = int(np.ceil(len(tiles) / cols))
            sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
            for k, t in enumerate(tiles):
                r, cc = divmod(k, cols); sheet[r * th:(r + 1) * th, cc * tw:(cc + 1) * tw] = t
        else:
            sheet = np.zeros((img.shape[0], 3 * args.tile_w, 3), np.uint8)
            cv2.putText(sheet, f"no clip found for {sc}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # side by side, padded to the same height
        Hm = max(img.shape[0], sheet.shape[0])
        def pad(a):
            o = np.full((Hm, a.shape[1], 3), 40, np.uint8); o[:a.shape[0], :a.shape[1]] = a; return o
        panel = np.concatenate([pad(img), pad(sheet)], axis=1)
        cv2.imwrite(str(out / f"{sc}_sheet.png"), panel)
        print(f"{sc}: overhead {img.shape[1]}x{img.shape[0]} px, {len(tiles)} video frames -> {sc}_sheet.png", flush=True)
    print(f"==> {out}")


if __name__ == "__main__":
    main()
