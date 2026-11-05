#!/usr/bin/env python3
"""Is this scene an OBSTACLE corner, and is the spawn window on walkable ground?

scene_bend.py answers neither question: it reads only the camera poses. So it
called gnd_AUw360 a corner because the WALK turns 151 deg, when nothing forces
the ROBOT to turn, and it passed gnd_AUw170's spawn window 20-25, which sits on
grass -- every episode of every arm crashed at step 1 (2026-09-24, Joana: "it
starts on grass", and "that other gnd scene is not an obstacle corner so it
doesn't really show anything").

This reads the same label grid the REWARD scores against and calls the same
classify_pair the env calls to label a goal, so "corner" here means what it
means in training: the straight line from spawn to goal is BLOCKED and the
robot has to travel detour_min x further around something.

Frame positions come from the cloud's own traj_positions (with the env's y-flip),
not from the poses file, so frame i is in the same frame as the grid by
construction.

    python scripts/scene_corner_scan.py gnd_AUw170 gnd_AUw360 gnd_AU_180
    python scripts/scene_corner_scan.py quad2_00 --goal 55 --spawn 5 12
    python scripts/scene_corner_scan.py gnd_AUw170 --band 5 8 --top 8
"""
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLOUDS = "/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds"
# The table TRAINING and EVAL score with (configs/arms/_base_campus.env: TRAV=...walkway).
# In it grass is 0.0 -- a collision -- while the plain v14 table gives grass 0.75. The
# first overheads used v14 and drew every lawn as walkable, so the "grass corner" was
# measured with its grass switched off (2026-09-25, Joana: "grass does cause collision though?").
TRAV = "config/traversability_v14_walkway.yaml"


def person_clusters(clouds_dir, scene, classes=(12,), min_pts=30, cell=0.5):
    """Where the people are, in world xy.

    People captured mid-walk are frozen into the cloud where they stood, so a
    pedestrian test is findable the same way a corner is: find the person points,
    then pick a spawn and goal that put one between the robot and its target
    (2026-09-24, Joana: "we had found a good spawn and goal ... where there was a
    person"). They are excluded from the label grid by map_ignore_classes, so they
    have to be read from the cloud directly rather than off the grid.
    """
    from pathlib import Path
    d = np.load(Path(clouds_dir) / f"{scene}_cloud.npz")
    pts, labs = d["points"], d["labels"].astype(int)
    hi = int(labs.max()) if len(labs) else 0
    for c in classes:
        if c > hi:
            print(f"        WARNING: class {c} requested but {scene}'s labels only reach {hi}. "
                  f"This cloud uses the 14-class taxonomy where person=12.")
    sel = np.isin(labs, list(classes)) & (pts[:, 2] > 0.15) & (pts[:, 2] < 2.0)
    P = pts[sel][:, :2]
    if not len(P):
        return []
    key = np.round(P / cell).astype(np.int64)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    out = []
    for k in np.where(cnt >= min_pts)[0]:
        out.append((P[inv == k].mean(0), int(cnt[k])))
    # merge cells that touch, so one person is one cluster
    merged = []
    for c, n in sorted(out, key=lambda x: -x[1]):
        for i, (mc, mn) in enumerate(merged):
            if np.linalg.norm(mc - c) < 1.2:
                merged[i] = ((mc * mn + c * n) / (mn + n), mn + n)
                break
        else:
            merged.append((c, n))
    return merged


def on_the_line(clusters, spawn_xy, goal_xy, within_m=1.5):
    """Person clusters lying within `within_m` of the spawn->goal segment, and how
    far along it they sit. A person BEHIND the spawn or past the goal is not a test."""
    s = np.asarray(spawn_xy, float)[:2]; g = np.asarray(goal_xy, float)[:2]
    v = g - s; L = float(np.linalg.norm(v))
    if L < 1e-6:
        return []
    hits = []
    for c, n in clusters:
        t = float(np.dot(c - s, v) / (L * L))
        if not (0.05 <= t <= 0.95):
            continue
        perp = float(np.linalg.norm((s + t * v) - c))
        if perp <= within_m:
            hits.append({"along_m": t * L, "offset_m": perp, "points": n})
    return sorted(hits, key=lambda h: h["along_m"])


def explain_line(grid, maps, spawn_xy, goal_xy):
    """What is the straight line actually made of?

    classify_pair can only say walkable/blocked. It cannot say whether the cells it
    walked over came from POINTS or were invented: build_label_grid fills enclosed
    void regions up to fill_max_area_m2 (10 m2 = 1000 cells) with their rim's
    majority label, and paints a 0.4 m corridor along the recorded walk walkable.
    The inside of a bend is precisely what the camera never sees, so a real corner
    can be filled into a shortcut and read as "open" (2026-09-24, Joana: "the quad2
    g55 is a corner, I can see it"). n_points == 0 under a non-void label means no
    point voted for that cell -- the map made it up."""
    res = grid.res
    s = np.asarray(spawn_xy, float)[:2]; g = np.asarray(goal_xy, float)[:2]
    n = max(2, int(np.ceil(float(np.linalg.norm(g - s)) / (res / 2))))
    line = s[None, :] + (g - s)[None, :] * np.linspace(0, 1, n)[:, None]
    tot = ok = invented = void = 0
    for q in line:
        i = int((q[1] - grid.y0) / res); j = int((q[0] - grid.x0) / res)
        if not (0 <= i < grid.labels.shape[0] and 0 <= j < grid.labels.shape[1]):
            continue
        tot += 1
        if maps["body_ok"][i, j]:
            ok += 1
        if grid.labels[i, j] < 0:
            void += 1
        elif grid.n_points[i, j] == 0:
            invented += 1
    return {"samples": tot, "walkable": ok, "invented": invented, "void": void}


def clip_frames(clips_dir, scene, idxs, clip_name=None):
    """The camera views at these frame indices, from the scene's clip.

    A top-down map answers "is there something in the way"; it does not answer
    "what does the robot see there", which is what actually has to be judged when
    choosing an eval (2026-09-24, Joana: "with the corresponding views at the
    frames"). Returns {idx: BGR image} for whatever could be read."""
    import cv2
    from pathlib import Path
    # The clip is not always named after the cloud: the gtown scenes reconstruct as
    # gnd_G2c1d330 but their clip is gtown2c1_w330 (2026-09-24). --clip names it.
    stem = clip_name or scene
    cands = sorted(Path(clips_dir).glob(f"{stem}*.mp4")) + sorted(Path(clips_dir).glob(f"{stem}/*.mp4"))
    if not cands:
        print(f"        (no clip matching {stem!r} under {clips_dir}; map only. "
              f"If the clip is named differently from the scene, pass --clip)")
        return {}
    cap = cv2.VideoCapture(str(cands[0]))
    want = sorted(set(int(i) for i in idxs))
    got, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = fr
        i += 1
        if want and i > max(want):
            break
    cap.release()
    missing = [i for i in want if i not in got]
    if missing:
        print(f"        (clip {cands[0].name} has {i} frames; missing {missing})")
    return got


def plot_pair(grid, maps, walk, spawn_xy, goal_xy, out_dir, name, views=None, labels=None, paths=None, geometry=True):
    """Draw the map the scan is reasoning about: walkable, wall, void, and -- in
    blue -- the cells that carry a label with NO points behind them, i.e. the ones
    build_label_grid invented by filling. Plus the walk, spawn, goal, the straight
    line and the geodesic path. A ratio is not evidence; this is."""
    import cv2
    from pathlib import Path
    from src.eval.goal_cases import snap, bfs_path
    L, res = grid.labels, grid.res
    H, W = L.shape
    img = np.zeros((H, W, 3), np.uint8)
    img[...] = (70, 70, 70)                                  # void: grey
    img[maps["free"]] = (232, 232, 228)                      # walkable: near-white
    img[(L >= 0) & ~maps["free"]] = (60, 60, 190)            # non-traversable: red (BGR)
    invented = (L >= 0) & (grid.n_points == 0)
    img[invented] = (200, 150, 60)                           # invented by the fill: blue
    # crop to the walk plus a margin, same as the survey: the cloud reaches tens of
    # metres past the path and the first overheads were 85% empty grey (2026-09-24)
    MARG = 6.0
    i0 = max(0, int((walk[:, 1].min() - grid.y0 - MARG) / res))
    i1 = min(H, int((walk[:, 1].max() - grid.y0 + MARG) / res))
    j0 = max(0, int((walk[:, 0].min() - grid.x0 - MARG) / res))
    j1 = min(W, int((walk[:, 0].max() - grid.x0 + MARG) / res))
    img = img[i0:i1, j0:j1]
    H, W = img.shape[:2]
    sc = max(2, int(round(1100 / max(1, W))))
    img = cv2.resize(img, (W * sc, H * sc), interpolation=cv2.INTER_NEAREST)
    def px(xy):
        return (int(((xy[0] - grid.x0) / res - j0) * sc), int(((xy[1] - grid.y0) / res - i0) * sc))
    for k in range(len(walk) - 1):
        cv2.line(img, px(walk[k]), px(walk[k + 1]), (110, 110, 110), 1, cv2.LINE_AA)
    s_px, g_px = px(spawn_xy), px(goal_xy)
    if geometry:
        cv2.line(img, s_px, g_px, (40, 160, 40), 2, cv2.LINE_AA)     # straight line: green
    def cell(xy):
        return (int((xy[1] - grid.y0) / res), int((xy[0] - grid.x0) / res))
    a_ij, b_ij = snap(maps["body_ok"], cell(spawn_xy)), snap(maps["body_ok"], cell(goal_xy))
    if geometry and a_ij is not None and b_ij is not None:
        m = int(6.0 / res)
        HF, WF = L.shape
        box = (max(0, min(a_ij[0], b_ij[0]) - m), min(HF, max(a_ij[0], b_ij[0]) + m),
               max(0, min(a_ij[1], b_ij[1]) - m), min(WF, max(a_ij[1], b_ij[1]) + m))
        path = bfs_path(maps["body_ok"], a_ij, b_ij, box)
        if path:
            for k in range(len(path) - 1):
                cv2.line(img, ((path[k][1] - j0) * sc, (path[k][0] - i0) * sc),
                         ((path[k + 1][1] - j0) * sc, (path[k + 1][0] - i0) * sc), (200, 80, 200), 2, cv2.LINE_AA)
    # eval episodes on the SAME map (2026-09-27): [(outcome, [(x, y), ...]), ...]. The old
    # overheads floated on a blank page; here the lawn, walls and void are under the paths.
    OUTCOME_BGR = {"GOAL": (60, 140, 30), "CRASH": (40, 40, 210), "TIMEOUT": (30, 140, 220), "INCOHERENT": (170, 80, 120)}
    for outcome, xy in (paths or []):
        col = OUTCOME_BGR.get(outcome, (90, 90, 90))
        for k in range(len(xy) - 1):
            cv2.line(img, px(xy[k]), px(xy[k + 1]), col, 2, cv2.LINE_AA)
        if xy:
            cv2.circle(img, px(xy[0]), 3 * sc // 2, col, -1)
    if not paths:
        cv2.circle(img, s_px, 5 * sc // 2, (30, 30, 30), -1)
    cv2.circle(img, g_px, int(1.0 / res * sc), (30, 140, 30), 2)
    img = cv2.flip(img, 0)                                    # +y up, as the paths plot
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    legend = ("white=walkable  red=wall  grey=void  BLUE=label with NO points (invented by fill)"
              + ("   paths: green=GOAL red=CRASH orange=TIMEOUT purple=INCOHERENT  ring=goal 1 m" if paths
                 else "   green=straight  magenta=path  black=spawn"))
    cv2.putText(bar, legend, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    stack = [bar, img]
    if views:
        # the camera views at the spawn and goal frames, scaled to one height and
        # laid side by side under the map
        HV = 260
        tiles = []
        for k, (idx, fr) in enumerate(views):
            if fr is None:
                continue
            h, w = fr.shape[:2]
            t = cv2.resize(fr, (int(w * HV / h), HV))
            cap = np.full((22, t.shape[1], 3), 255, np.uint8)
            txt = (labels or {}).get(idx, f"frame {idx}")
            cv2.putText(cap, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (20, 20, 20), 1, cv2.LINE_AA)
            tiles.append(np.vstack([cap, t]))
        if tiles:
            strip = np.hstack(tiles)
            W2 = img.shape[1]
            if strip.shape[1] > W2:
                sc2 = W2 / strip.shape[1]
                strip = cv2.resize(strip, (W2, int(strip.shape[0] * sc2)))
            elif strip.shape[1] < W2:
                pad = np.full((strip.shape[0], W2 - strip.shape[1], 3), 255, np.uint8)
                strip = np.hstack([strip, pad])
            stack.append(strip)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / f"{name}.png"
    cv2.imwrite(str(f), np.vstack(stack))
    print(f"        wrote {f}")


def survey(grid, maps, walk, people, out_dir, scene, every=5, jitter=0.4):
    """One top-down picture of the whole scene, with the walk numbered by frame and
    the people marked, so a spawn and a goal can be CHOSEN by eye instead of taken
    from a ranking (2026-09-24, Joana: "i'd like to try and find spawns and goals
    with a top down view"). Frames that cannot hold a spawn once jittered are drawn
    hollow, so a number you pick is already known to be legal."""
    import cv2
    from pathlib import Path
    L, res = grid.labels, grid.res
    H, W = L.shape
    img = np.zeros((H, W, 3), np.uint8)
    img[...] = (70, 70, 70)
    img[maps["free"]] = (235, 235, 232)
    img[(L >= 0) & ~maps["free"]] = (60, 60, 190)
    img[(L >= 0) & (grid.n_points == 0)] = (205, 160, 70)
    # crop to the WALK plus a margin: the cloud reaches tens of metres past the path
    # and without this the part anyone needs is squeezed into a corner of the image
    # (2026-09-24, first survey was 85% empty grey).
    MARG = 8.0
    i0 = max(0, int((walk[:, 1].min() - grid.y0 - MARG) / res))
    i1 = min(H, int((walk[:, 1].max() - grid.y0 + MARG) / res))
    j0 = max(0, int((walk[:, 0].min() - grid.x0 - MARG) / res))
    j1 = min(W, int((walk[:, 0].max() - grid.x0 + MARG) / res))
    img = img[i0:i1, j0:j1]
    sc = max(2, int(round(1100 / max(1, img.shape[1]))))
    img = cv2.resize(img, (img.shape[1] * sc, img.shape[0] * sc), interpolation=cv2.INTER_NEAREST)
    px = lambda xy: (int(((xy[0] - grid.x0) / res - j0) * sc),
                     int(((xy[1] - grid.y0) / res - i0) * sc))
    for k in range(len(walk) - 1):
        cv2.line(img, px(walk[k]), px(walk[k + 1]), (120, 120, 120), 2, cv2.LINE_AA)
    for c, n in people:
        cv2.circle(img, px(c), int(0.6 / res * sc), (200, 40, 200), 3)
        cv2.circle(img, px(c), 4, (200, 40, 200), -1)
    for k in range(0, len(walk), every):
        ok = frame_ok(maps, grid, walk[k], jitter)[0]
        p_ = px(walk[k])
        cv2.circle(img, p_, 7, (30, 30, 30), -1 if ok else 2)
    img = cv2.flip(img, 0)                       # +y up
    # labels go on AFTER the flip so the text is not upside down
    for k in range(0, len(walk), every):
        x, y = px(walk[k]); y = img.shape[0] - y
        cv2.putText(img, str(k), (x + 9, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, str(k), (x + 9, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (20, 20, 20), 1, cv2.LINE_AA)
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"{scene}   filled dot = spawnable with {jitter:g} m jitter, hollow = not"
                "   MAGENTA = person   white=walkable  red=wall  grey=void  blue=invented",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / f"SURVEY_{scene}_path.png"
    cv2.imwrite(str(f), np.vstack([bar, img]))
    print(f"  wrote {f}")


def load_scene(scene, clouds_dir, trav_path, collision_threshold=0.1):
    from src.eval.traversability import load_traversability
    from src.eval.reward_map import build_label_grid
    from src.eval.goal_cases import scene_case_maps

    p = Path(clouds_dir) / f"{scene}_cloud.npz"
    if not p.exists():
        raise SystemExit(f"no cloud at {p}")
    d = np.load(p)
    pts, labs = d["points"], d["labels"].astype(int)
    scores = load_traversability(Path(trav_path) if trav_path else None)
    non_trav = scores <= collision_threshold
    if "traj_positions" not in d:
        raise SystemExit(f"{scene}: cloud has no traj_positions, cannot place frames")
    walk = (np.asarray(d["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
    # exactly the call SceneEnv makes (scene_env.py:1438), with its config defaults
    grid = build_label_grid(pts, labs, non_trav, res=0.1, inflate_m=0.1, fill_m=0.3,
                            fill_max_area_m2=10.0, ignore_classes=(),
                            walk_xy=walk, walk_halfwidth_m=0.4, inflate_classes=())
    return grid, scene_case_maps(grid, non_trav), walk


BODY_HALF = 0.15          # src/eval/goal_cases.py BODY_W / 2


def frame_ok(maps, grid, xy, jitter=0.0):
    """Clearance at this frame, and whether a spawn there survives the jitter.

    body_ok alone is not enough. EVAL jitters the spawn laterally (spawn_lat_jitter,
    0.4 m), so a frame with 0.41 m of clearance passes body_ok and still lands the
    robot on the boundary once jittered -- which is why every gnd_AUw170 episode of
    both arms died at step 1 while every spawn frame read walkable (2026-09-24).
    Require clearance >= body half-width + the jitter the eval will actually apply."""
    i = int((xy[1] - grid.y0) / grid.res); j = int((xy[0] - grid.x0) / grid.res)
    if not (0 <= i < grid.labels.shape[0] and 0 <= j < grid.labels.shape[1]):
        return False, 0.0
    d = float(maps["dist"][i, j])
    return bool(maps["body_ok"][i, j]) and d >= BODY_HALF + float(jitter), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--clouds_dir", default=CLOUDS)
    ap.add_argument("--trav", default=str(REPO / TRAV))
    ap.add_argument("--goal", type=int, help="score this goal frame only")
    ap.add_argument("--spawn", type=int, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--band", type=float, nargs=2, default=(5.0, 8.0),
                    metavar=("LO", "HI"), help="target spawn-goal distance, m")
    ap.add_argument("--detour_min", type=float, default=1.15,
                    help="the env's goal_case_detour_min: above this the straight line is blocked")
    ap.add_argument("--top", type=int, default=6, help="how many candidate windows to print")
    ap.add_argument("--survey", metavar="DIR",
                    help="write SURVEY_<scene>_path.png: the whole scene top-down with the walk "
                         "numbered by frame and the people marked, for picking spawns and goals by eye")
    ap.add_argument("--every", type=int, default=5, help="label every Nth frame in the survey")
    ap.add_argument("--person", action="store_true",
                    help="report people between the spawn and the goal (class 29), so a "
                         "pedestrian test can be defined the same way a corner test is")
    ap.add_argument("--person_class", default="12",
                    help="comma list of person-like class ids. 12=person, 13=vehicle in the "
                         "14-class navigation taxonomy these clouds use (config/traversability_v14.yaml). "
                         "NOT 29 -- that is the 30-class Go2W table in src/eval/reward.py and no cloud "
                         "contains it, which is why the first run found zero people everywhere.")
    ap.add_argument("--person_m", type=float, default=1.5,
                    help="how close to the straight line a person must be to count")
    ap.add_argument("--clip", metavar="NAME",
                    help="clip file stem when it differs from the scene name, e.g. "
                         "--clip gtown2c1_w330 for scene gnd_G2c1d330")
    ap.add_argument("--clips", metavar="DIR",
                    help="clip directory (e.g. /scratch/.../data/gnd_clips). With --plot, the "
                         "camera views at the spawn and goal frames are drawn under the map.")
    ap.add_argument("--plot", metavar="DIR",
                    help="write a PNG of the label grid with the walk, spawn, goal, straight "
                         "line and geodesic path drawn on it -- so a claim about the map can "
                         "be looked at instead of argued from a ratio")
    ap.add_argument("--overlay", nargs="*", metavar="EVAL",
                    help="eval output dirs (or metrics.json files) whose episodes are drawn on the "
                         "label map of the scene given as the positional argument; one PNG per eval "
                         "in --plot, named after the eval dir. Colour = outcome. (2026-09-27)")
    ap.add_argument("--jitter", type=float, default=0.4,
                    help="the lateral spawn jitter eval will apply (SPAWNJLAT). A spawn must "
                         "clear body/2 + this, or the jitter puts it on the boundary.")
    a = ap.parse_args()

    from src.eval.goal_cases import classify_pair

    for scene in a.scenes:
        print(f"\n{'='*78}\n{scene}")
        try:
            grid, maps, walk = load_scene(scene, a.clouds_dir, a.trav)
        except SystemExit as e:
            print(f"  SKIP: {e}"); continue
        N = len(walk)

        if a.survey:
            _pc = person_clusters(a.clouds_dir, scene,
                                  tuple(int(v) for v in a.person_class.split(",") if v.strip()))
            survey(grid, maps, walk, _pc, a.survey, scene, a.every, a.jitter)

        ok = np.array([frame_ok(maps, grid, walk[i], a.jitter)[0] for i in range(N)])
        clr = np.array([frame_ok(maps, grid, walk[i], a.jitter)[1] for i in range(N)])
        bare = np.array([frame_ok(maps, grid, walk[i], 0.0)[0] for i in range(N)])
        bad = np.where(~ok)[0]
        print(f"  {N} frames, {ok.sum()} spawnable with {a.jitter:g} m jitter "
              f"({100*ok.mean():.0f}%); {bare.sum()} would pass the body check alone")
        if len(bad):
            runs, s = [], bad[0]
            for k in range(1, len(bad) + 1):
                if k == len(bad) or bad[k] != bad[k-1] + 1:
                    runs.append((s, bad[k-1])); s = bad[k] if k < len(bad) else None
            print(f"  NOT spawnable: " + ", ".join(f"{x}-{y}" if x != y else str(x) for x, y in runs))

        if a.overlay:
            import json as _json
            for ev in a.overlay:
                mp = Path(ev) / "metrics.json" if Path(ev).is_dir() else Path(ev)
                if not mp.exists():
                    print(f"  SKIP overlay: {mp} missing"); continue
                m = _json.load(open(mp))
                eps = [e for e in m.get("video_episodes", []) if len(e.get("traj") or []) > 1]
                scored = {int(e.get("episode", i)): e.get("outcome") for i, e in enumerate(m.get("episodes", []))}
                if not eps:
                    print(f"  SKIP overlay: no trajectories in {mp}"); continue
                paths = [(scored.get(int(e.get("episode", i)), e.get("outcome")), [(float(t[0]), float(t[1])) for t in e["traj"]])
                         for i, e in enumerate(eps)]
                g = eps[0].get("goal_xy") or m["episodes"][0].get("goal_xy")
                plot_pair(grid, maps, walk, np.array(paths[0][1][0]), np.array(g[:2]), a.plot or ".",
                          f"OVERLAY_{mp.parent.name}", paths=paths, geometry=False)
            continue

        if a.goal is not None and a.spawn:
            lo, hi = a.spawn; g = min(a.goal, N - 1)
            # WALK length spawn->goal: cumulative distance along the recorded poses.
            # This is pose-only and owes nothing to the label grid, which on quad2_00
            # is mostly void and cannot be trusted for geometry (2026-09-24, looked at
            # the plot). It is also what the robot must actually drive on a corner,
            # so it is the right TESTDIST -- the straight line is not.
            seg = np.linalg.norm(np.diff(walk, axis=0), axis=1)
            print(f"\n  goal {g}, spawn {lo}-{hi}:")
            print(f"  {'spawn':>6}{'walk to goal':>14}{'straight':>10}   <- walk length is pose-only; use it for TESTDIST")
            for sp in range(lo, min(hi, N - 1) + 1):
                w = float(seg[sp:g].sum()) if g > sp else 0.0
                st = float(np.linalg.norm(walk[g] - walk[sp]))
                print(f"  {sp:>6}{w:>14.2f}{st:>10.2f}")
            wmax = max(float(seg[sp:g].sum()) for sp in range(lo, min(hi, N - 1) + 1) if g > sp)
            print(f"  -> TESTDIST={wmax:.1f}  (longest walk in the spawn range; "
                  f"budget {round(40 + 16 * wmax)} actions)")
            print()
            print(f"  {'spawn':>6}{'ok':>5}{'clear':>8}{'straight':>10}{'detour':>9}{'width':>8}  case")
            PC = []
            if a.person:
                PC = person_clusters(a.clouds_dir, scene,
                                     tuple(int(v) for v in a.person_class.split(",") if v.strip()))
                print(f"  {len(PC)} person cluster(s) in the scene"
                      + (f": {', '.join(f'{n} pts' for _, n in PC)}" if PC else ""))
            print(f"  {'':>6}{'':>5}{'':>8}{'':>10}{'':>9}{'':>8}         straight line: cells from POINTS vs INVENTED by the map")
            for s in range(lo, min(hi, N - 1) + 1):
                o, c = frame_ok(maps, grid, walk[s], a.jitter)
                r = classify_pair(grid, maps, walk[s], walk[g], detour_min=a.detour_min)
                ex = explain_line(grid, maps, walk[s], walk[g])
                if a.plot:
                    vs = []
                    if a.clips:
                        fr = clip_frames(a.clips, scene, [s, g], a.clip)
                        vs = [(s, fr.get(s)), (g, fr.get(g))]
                    plot_pair(grid, maps, walk, walk[s], walk[g], a.plot,
                              f"{scene}_spawn{s}_goal{g}", views=vs,
                              labels={s: f"SPAWN  frame {s}", g: f"GOAL  frame {g}"})
                print(f"  {s:>6}{'y' if o else 'JIT' if c >= BODY_HALF else 'NO':>5}{c:>8.2f}{r.get('straight_m',0):>10.2f}"
                      f"{r.get('detour',float('nan')):>9.2f}{r.get('min_width_m',0):>8.2f}  {r['cls']:<8}"
                      f"  {ex['walkable']}/{ex['samples']} walkable, "
                      f"{100*ex['invented']/max(1,ex['samples']):.0f}% INVENTED (no points), "
                      f"{100*ex['void']/max(1,ex['samples']):.0f}% void")
                if a.person:
                    hits = on_the_line(PC, walk[s], walk[g], a.person_m)
                    if hits:
                        for h in hits:
                            print(f"         PERSON {h['along_m']:.1f} m along the line, "
                                  f"{h['offset_m']:.2f} m off it ({h['points']} pts)")
                    else:
                        print(f"         no person within {a.person_m:g} m of this line")
            continue

        if a.person:
            PC = person_clusters(a.clouds_dir, scene,
                                 tuple(int(v) for v in a.person_class.split(",") if v.strip()))
            print(f"  {len(PC)} person cluster(s) in the scene")
            if not PC:
                print("  no people here: this scene cannot serve as a pedestrian test")
                continue
            best = []
            for sp in range(N - 1):
                if not ok[sp]:
                    continue
                for g in range(sp + 3, N):
                    d = float(np.linalg.norm(walk[g] - walk[sp]))
                    if not (a.band[0] <= d <= a.band[1]):
                        continue
                    hits = on_the_line(PC, walk[sp], walk[g], a.person_m)
                    if hits:
                        best.append((min(h["offset_m"] for h in hits), len(hits), sp, g, d, hits))
            if not best:
                print(f"  people exist but none sit within {a.person_m:g} m of any spawn/goal line "
                      f"in the {a.band[0]}-{a.band[1]} m band")
                continue
            best.sort()
            print(f"  {len(best)} spawn/goal pairs put a person in the way")
            print(f"\n  {'spawn':>6}{'goal':>6}{'dist':>8}{'people':>8}{'nearest offset':>16}{'along':>8}")
            seen = set()
            for offs, nh, sp, g, d, hits in best[:a.top]:
                if (sp // 2, g // 2) in seen:
                    continue
                seen.add((sp // 2, g // 2))
                print(f"  {sp:>6}{g:>6}{d:>8.1f}{nh:>8}{offs:>16.2f}{hits[0]['along_m']:>8.1f}")
            offs, nh, sp, g, d, hits = best[0]
            seg = np.linalg.norm(np.diff(walk, axis=0), axis=1)
            wl = float(seg[sp:g].sum())
            print(f"\n  eval_scenes.env line:")
            print(f"  {scene}_ped  GOAL_FRAME={g:<4} SPAWN_MIN={sp},SPAWN_MAX={sp} TESTDIST={wl:.1f} "
                  f"SCENE={scene}  # person {hits[0]['along_m']:.1f} m along, {offs:.2f} m off the line")
            continue

        # sweep: every (spawn, goal) pair inside the distance band, ranked by detour
        cands = []
        for s in range(N - 1):
            if not ok[s]:
                continue
            for g in range(s + 3, N):
                d = float(np.linalg.norm(walk[g] - walk[s]))
                if not (a.band[0] <= d <= a.band[1]):
                    continue
                r = classify_pair(grid, maps, walk[s], walk[g], detour_min=a.detour_min)
                if r["cls"] in ("offmap", "blocked"):
                    continue
                cands.append((s, g, d, r))
        if not cands:
            print(f"  no spawn/goal pair between {a.band[0]}-{a.band[1]} m"); continue

        corners = [c for c in cands if c[3]["detour"] >= a.detour_min]
        print(f"  {len(cands)} pairs in band, {len(corners)} are OBSTACLE corners "
              f"(detour >= {a.detour_min})")
        if not corners:
            best = max(cands, key=lambda c: c[3]["detour"])
            print(f"  VERDICT: no obstacle corner here. Best detour is {best[3]['detour']:.2f} "
                  f"(spawn {best[0]} -> goal {best[1]}), i.e. the straight line is walkable.")
            print(f"  This scene tests open navigation, not turning around something.")
            continue

        # group by goal frame, keep the widest contiguous spawn window per goal
        bygoal = {}
        for s, g, d, r in corners:
            bygoal.setdefault(g, []).append((s, d, r))
        rows = []
        for g, lst in bygoal.items():
            ss = sorted(x[0] for x in lst)
            run, best = [ss[0]], (ss[0], ss[0])
            for k in range(1, len(ss)):
                if ss[k] == ss[k-1] + 1:
                    run.append(ss[k])
                else:
                    if len(run) > best[1] - best[0] + 1: best = (run[0], run[-1])
                    run = [ss[k]]
            if len(run) > best[1] - best[0] + 1: best = (run[0], run[-1])
            sel = [x for x in lst if best[0] <= x[0] <= best[1]]
            rows.append((g, best, np.mean([x[2]["detour"] for x in sel]),
                         min(x[2]["min_width_m"] for x in sel),
                         min(x[1] for x in sel), max(x[1] for x in sel), len(sel)))
        rows.sort(key=lambda r: (-r[6], -r[2]))
        print(f"\n  {'goal':>5}{'spawn':>10}{'n':>4}{'detour':>9}{'min width':>11}{'distance':>16}")
        for g, (lo, hi), det, w, dmin, dmax, n in rows[:a.top]:
            print(f"  {g:>5}{f'{lo}-{hi}':>10}{n:>4}{det:>9.2f}{w:>11.2f}{f'{dmin:.1f}-{dmax:.1f} m':>16}")
        g, (lo, hi), det, w, dmin, dmax, n = rows[0]
        # TESTDIST must be the distance the robot DRIVES, i.e. the path around the
        # bend, not the straight line: quad2_00_far is 8.1 m straight and 13.6 m
        # along the walkway, and the straight figure under-budgeted it by a third
        # (2026-09-24, Joana: "did we add more steps than the training allows?").
        path_m = dmax * det
        print(f"\n  eval_scenes.env line   (TESTDIST is the PATH, {dmax:.1f} m straight x {det:.2f} detour):")
        print(f"  {scene:<12} GOAL_FRAME={g:<4} SPAWN_MIN={lo},SPAWN_MAX={hi} TESTDIST={path_m:.1f}  "
              f"# {dmin:.1f}-{dmax:.1f} m straight, detour {det:.2f}x, corridor {w:.1f} m: obstacle corner")


if __name__ == "__main__":
    main()
