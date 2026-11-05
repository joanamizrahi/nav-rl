"""Overhead picture of eval trajectories on the scene map: white walkable, black
non-walkable, grey unknown, recorded walk in orange, each episode's path in
its own colour with S (spawn) and a star at the goal. Sighted and blind evals
of the SAME seed are drawn side by side. Runs on the login node (cv2)."""
import sys, json, numpy as np
sys.path.insert(0, "/scratch/m000204-pm07b/joana/nav-rl")
import cv2
from src.eval.reward_map import build_label_grid
from src.eval.traversability import load_traversability
scene, out = sys.argv[1], sys.argv[2]; runs = [a.split("=", 1) for a in sys.argv[3:]]   # label=metrics.json
c = np.load(f"/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds/{scene}_cloud.npz")
walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
scores = load_traversability("/scratch/m000204-pm07b/joana/nav-rl/config/traversability_v14_walkway.yaml"); nontrav = scores <= 0.1
g = build_label_grid(c["points"], c["labels"].astype(int), nontrav, res=0.1, inflate_m=0.1, inflate_classes=(10, 11, 13), walk_xy=walk)
L = g.labels; known = L >= 0; nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
import os; ppm = float(os.environ.get("PPM", "25")); ZOOM = os.environ.get("ZOOM", "0") == "1"
base = np.full(L.shape + (3,), 128, np.uint8); base[known & ~nt] = 255; base[nt] = 0
img0 = cv2.resize(base[::-1].copy(), None, fx=ppm * g.res, fy=ppm * g.res, interpolation=cv2.INTER_NEAREST); H = img0.shape[0]
def px(xy): return int((xy[0] - g.x0) * ppm), int(H - (xy[1] - g.y0) * ppm)
panels = []
cols = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (0, 120, 255), (128, 0, 128), (0, 255, 0), (255, 128, 0), (60, 60, 60)]
for label, path in runs:
    img = img0.copy(); cv2.polylines(img, [np.array([px(w) for w in walk], np.int32)], False, (0, 90, 230), 2, cv2.LINE_AA)
    # FRAMETICKS=N (2026-09-21, Joana: "select good scenes"): number the walk every N recorded frames
    # so spawn and goal frames can be chosen from the picture
    _ft = int(os.environ.get("FRAMETICKS", "0") or 0)
    if _ft > 0:
        for _i in range(0, len(walk), _ft):
            _p = px(walk[_i]); cv2.circle(img, _p, 3, (0, 60, 160), -1)
            cv2.putText(img, str(_i), (_p[0] + 4, _p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(_i), (_p[0] + 4, _p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    eps = json.load(open(path))["episodes"]
    import os as _os
    _ec = _os.path.join(_os.path.dirname(path), "env_config.json")
    goal_radius = float(json.load(open(_ec)).get("goal_radius", 1.0)) if _os.path.exists(_ec) else 1.0
    for i, e in enumerate(eps):
        col = cols[i % len(cols)]; t = np.array(e["traj"])[:, :2]
        cv2.polylines(img, [np.array([px(p) for p in t], np.int32)], False, col, 2, cv2.LINE_AA)
        s = px(t[0]); cv2.circle(img, s, 5, col, -1); cv2.putText(img, str(i), (s[0] + 4, s[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        gp = px(e["goal_xy"]); cv2.drawMarker(img, gp, col, cv2.MARKER_STAR, 16, 2)
        # dotted straight line spawn -> goal (the naive route), so a detour is visible
        _n = max(2, int(np.hypot(gp[0] - s[0], gp[1] - s[1]) / 6))
        for _k in range(0, _n, 2):
            _a = (int(s[0] + (gp[0] - s[0]) * _k / _n), int(s[1] + (gp[1] - s[1]) * _k / _n))
            _b = (int(s[0] + (gp[0] - s[0]) * (_k + 1) / _n), int(s[1] + (gp[1] - s[1]) * (_k + 1) / _n))
            cv2.line(img, _a, _b, col, 1, cv2.LINE_AA)
        cv2.circle(img, gp, int(goal_radius * ppm), col, 1, cv2.LINE_AA)
        cv2.putText(img, e.get("outcome", "?")[:1], (gp[0] + 6, gp[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    panels.append((img, (f"{label.upper()}  {sum(1 for e in eps if e.get('outcome') == 'GOAL')}/{len(eps)} GOAL  mean {np.mean([e['steps'] for e in eps]):.0f} steps  goal radius {goal_radius:.1f} m")
                   if eps else f"{scene}  map + recorded walk, frame numbers every {_ft or 0}"))
# crop all panels to the bounding box of walk + trajectories + goals (same crop for both), with margin
pts = [] if ZOOM else [px(w) for w in walk]
for _, path in runs:
    for e in json.load(open(path))["episodes"]:
        pts += [px(p) for p in np.array(e["traj"])[:, :2]] + [px(e["goal_xy"])]
xs, ys = zip(*pts); m = int((2.5 if ZOOM else 4) * ppm)
x0, x1 = max(0, min(xs) - m), min(img0.shape[1], max(xs) + m); y0, y1 = max(0, min(ys) - m), min(H, max(ys) + m)
done = []
for img, title in panels:
    crop = img[y0:y1, x0:x1]
    bar = np.full((44, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    done.append(np.concatenate([bar, crop], axis=0))
foot = np.full((36, sum(d.shape[1] for d in done) + 8 * (len(done) - 1), 3), 255, np.uint8)
cv2.putText(foot, f"{scene}: white walkable, black non-walkable, grey unknown; orange = recorded walk; dot+number = spawn, star = goal, circle = goal radius, letter = outcome (G goal, C crash, T timeout)", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
sep = np.full((done[0].shape[0], 8, 3), 255, np.uint8)
row = done[0]
for d in done[1:]: row = np.concatenate([row, sep, d], axis=1)
cv2.imwrite(out, np.concatenate([row, foot], axis=0)); print("wrote", out, row.shape)
