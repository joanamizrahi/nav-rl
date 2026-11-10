"""Scripted good vs bad paths, scored by the TRAINING reward (2026-11-09, Jing:
"show that our cost is lower for a good trajectory than for a bad one, in
simulation, qualitative and quantitative").

The env is built exactly as for a policy eval (eval_policy.py adopts the
checkpoint's env_config.json and hard-checks every value), so every reward this
driver collects is the reward the paper policy was trained on. Only the actions
are scripted: no policy is loaded.

For each scene, start/goal PAIRS are chosen on the env's own fused label grid
(`_label_grids`, the map the proximity term reads): the start is a recorded
spawn frame, the goal a recorded pose 4-8 m ahead along the walk, and a pair is
kept only if the straight line between them crosses no-go ground while a
walkable path exists. Each pair is then driven four ways:

  good      A* on walkable cells (no-go inflated by the robot half-width),
            with a soft cost near no-go so it keeps a margin
  straight  the straight line start -> goal: cuts the lawn where the walk bends
            (the terrain and contact terms should price it)
  obstacle  the good path to its middle, then straight into the nearest
            obstacle/vegetation cell and 1 m through it (contact / crash should
            price it; no return leg, so it cannot loop around the cell)
  offwalk   a detour 7 m off the walk into unreconstructed space, then the
            goal (the coverage gate should price it)
  beyond    1 m along the walk, then veer 25-40 deg off it for 14 m (the side and
            angle crossing the least no-go; none if every line crosses > 0.35 m): each step is farther from where the
            camera went, so coverage falls step by step until the gate ends it

The controller is the map expert's: aim at a lookahead point on the polyline,
turn at most yaw_step_rad per step, never a still step (a still step is a halt
to the env).
"""
from __future__ import annotations

import heapq

import numpy as np

KINDS = ("good", "straight", "obstacle", "beyond")   # offwalk still available by name
OBSTACLE_CLASSES = (10, 11)          # v14 obstacle, vegetation


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class PathExpert:
    def __init__(self, env, scenes=None, n_pairs: int = 3, kinds=KINDS,
                 lookahead_m: float = 0.75, inflate_m: float = 0.6,
                 margin_cost: float = 8.0, offwalk_m: float = 7.0,
                 ahead_range=(4.0, 8.0), min_cross_m: float = 0.5, seed: int = 0):
        self.env = env
        self.u = env.unwrapped
        self.scenes = list(scenes) if scenes else list(self.u.scene_ids)
        self.n_pairs = int(n_pairs)
        self.kinds = tuple(kinds)
        self.lookahead = float(lookahead_m)
        self.inflate = float(inflate_m)
        self.margin_cost = float(margin_cost)
        self.offwalk = float(offwalk_m)
        self.ahead = tuple(float(v) for v in ahead_range)
        self.min_cross = float(min_cross_m)
        self.rng = np.random.default_rng(seed)
        self.plan = []          # list of dicts: scene, pair, kind, start_frame, goal_xy, path [K,2]
        self.cur = None
        self._wp_i = 0
        self.policy = None      # eval_policy's summary reads model.policy.log_std; a scripted driver has none

    # ------------------------------------------------------------ map access
    def _grid(self, scene):
        """Load the scene into the env (reset with that scene) to get its label grid."""
        g = getattr(self.u, "_label_grids", {}).get(scene)
        if g is None:
            self.env.reset(options={"scene": scene})
            g = getattr(self.u, "_label_grids", {}).get(scene)
        if g is None:
            raise SystemExit(f"[path] no label grid for {scene}: the env must run with map_diagnostics "
                             f"(the paper config does) for the planner to use the reward's map")
        walk = getattr(self.u, "_walk_xy", {}).get(scene)
        return g, (None if walk is None else np.asarray(walk, dtype=float)[:, :2])

    def _masks(self, g):
        from scipy import ndimage
        lab = g.labels.astype(int)
        known = lab >= 0
        nontrav = known & self.u._non_trav[np.clip(lab, 0, len(self.u._non_trav) - 1)]
        free = known & ~nontrav
        k = max(1, int(round(self.inflate / g.res)))
        blocked = ndimage.binary_dilation(nontrav, iterations=k)
        free_inf = free & ~blocked
        dist = ndimage.distance_transform_edt(~nontrav) * g.res        # metres to nearest no-go
        return free, free_inf, nontrav, dist

    def _cell(self, g, xy):
        return (int(np.floor((xy[1] - g.y0) / g.res)), int(np.floor((xy[0] - g.x0) / g.res)))

    def _xy(self, g, rc):
        return np.array([g.x0 + (rc[1] + 0.5) * g.res, g.y0 + (rc[0] + 0.5) * g.res])

    def _astar(self, g, free, dist, a_xy, b_xy):
        H, W = free.shape
        s, t = self._cell(g, a_xy), self._cell(g, b_xy)
        if not (0 <= s[0] < H and 0 <= s[1] < W and 0 <= t[0] < H and 0 <= t[1] < W):
            return None
        free = free.copy()
        free[s] = free[t] = True                      # start/goal may sit on the inflated margin
        pen = self.margin_cost * np.exp(-dist / 1.0)      # prefer the middle of the walkway (2026-11-09: a 0.25 m margin skimmed the lawn edge and the box memory charged it)
        nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        openq = [(0.0, s)]
        gcost = {s: 0.0}
        came = {}
        while openq:
            _, c = heapq.heappop(openq)
            if c == t:
                path = [c]
                while c in came:
                    c = came[c]
                    path.append(c)
                return np.array([self._xy(g, rc) for rc in path[::-1]])
            for dr, dc, w in nb:
                n = (c[0] + dr, c[1] + dc)
                if not (0 <= n[0] < H and 0 <= n[1] < W) or not free[n]:
                    continue
                ng = gcost[c] + w * (1.0 + pen[n])
                if ng < gcost.get(n, np.inf):
                    gcost[n] = ng
                    came[n] = c
                    h = np.hypot(n[0] - t[0], n[1] - t[1])
                    heapq.heappush(openq, (ng + h, n))
        return None

    def _nogo_len(self, g, nontrav, a_xy, b_xy, step=0.05):
        n = max(2, int(np.linalg.norm(b_xy - a_xy) / step))
        pts = a_xy[None, :] + np.linspace(0, 1, n)[:, None] * (b_xy - a_xy)[None, :]
        rc = np.array([self._cell(g, p) for p in pts])
        ok = (rc[:, 0] >= 0) & (rc[:, 1] >= 0) & (rc[:, 0] < nontrav.shape[0]) & (rc[:, 1] < nontrav.shape[1])
        hit = np.zeros(len(pts), bool)
        hit[ok] = nontrav[rc[ok, 0], rc[ok, 1]]
        return float(hit.sum() * step)

    # ------------------------------------------------------------ planning
    def build_plan(self):
        cfg = self.u.world_backend.cfg
        spawn_lists = {}
        for tok in str(getattr(cfg, "spawn_frames_by_scene", "") or "").split(";"):
            if ":" in tok:
                k, v = tok.split(":", 1)
                spawn_lists[k.strip()] = [int(x) for x in v.split(",") if x.strip()]
        for scene in self.scenes:
            g, walk = self._grid(scene)
            free, free_inf, nontrav, dist = self._masks(g)
            cal = self.u.world_backend._calib[scene]
            frames = spawn_lists.get(scene) or list(range(5, len(cal.positions) - 6))
            cands = []
            for f in frames:
                p0 = np.asarray(cal.robot_pose_nav(f)[:2, 3], dtype=float)
                acc, j = 0.0, f
                while j + 1 < len(walk) and acc < self.ahead[1]:
                    acc += float(np.linalg.norm(walk[j + 1] - walk[j]))
                    j += 1
                    if acc < self.ahead[0]:
                        continue
                    goal = walk[j]
                    cross = self._nogo_len(g, nontrav, p0, goal)
                    if cross >= self.min_cross:
                        cands.append((cross, f, j, goal))
            cands.sort(key=lambda c: -c[0])
            chosen, used = [], []
            for cross, f, j, goal in cands:
                if any(abs(f - u) < 8 for u in used):
                    continue
                p0 = np.asarray(cal.robot_pose_nav(f)[:2, 3], dtype=float)
                good = self._astar(g, free_inf, dist, p0, goal)
                if good is None:          # walkway narrower than the margin: relax to 0.3 m, still clear of the edge cells
                    from scipy import ndimage
                    _k = max(1, int(round(0.3 / g.res)))
                    good = self._astar(g, free & ~ndimage.binary_dilation(nontrav, iterations=_k), dist, p0, goal)
                if good is None:
                    continue
                chosen.append((f, goal, good, cross))
                used.append(f)
                if len(chosen) >= self.n_pairs:
                    break
            print(f"[path] {scene}: {len(cands)} candidate pairs cross no-go, kept {len(chosen)}", flush=True)
            for pi, (f, goal, good, cross) in enumerate(chosen):
                p0 = good[0]
                paths = {"good": good, "straight": np.array([p0, goal])}
                # obstacle: nearest obstacle/vegetation cell to the good path's middle, within 4 m
                mid = good[len(good) // 2]
                lab = g.labels.astype(int)
                ob = np.argwhere(np.isin(lab, OBSTACLE_CLASSES))
                if len(ob):
                    oxy = np.array([self._xy(g, rc) for rc in ob])
                    d = np.linalg.norm(oxy - mid[None, :], axis=1)
                    if d.min() <= 4.0:
                        o = oxy[int(np.argmin(d))]
                        u = (o - mid) / max(np.linalg.norm(o - mid), 1e-6)
                        paths["obstacle"] = np.vstack([good[:len(good) // 2], o, o + 1.0 * u])
                # offwalk: 7 m perpendicular to the walk direction at the middle, toward unknown space
                tang = good[min(len(good) - 1, len(good) // 2 + 3)] - good[max(0, len(good) // 2 - 3)]
                tang = tang / max(np.linalg.norm(tang), 1e-6)
                nrm = np.array([-tang[1], tang[0]])
                best = None
                for sgn in (1.0, -1.0):
                    q = mid + sgn * self.offwalk * nrm
                    rc = self._cell(g, q)
                    void = 1.0
                    if 0 <= rc[0] < lab.shape[0] and 0 <= rc[1] < lab.shape[1]:
                        r0, c0 = rc
                        win = lab[max(0, r0 - 10):r0 + 10, max(0, c0 - 10):c0 + 10]
                        void = float((win < 0).mean()) if win.size else 1.0
                    if best is None or void > best[0]:
                        best = (void, q)
                paths["offwalk"] = np.array([p0, best[1], goal])
                # beyond: set off along the walk's heading at the start, then veer 25-40 deg to one side and keep
                # going 14 m: every step is farther from where the camera went, so coverage falls gradually.
                # The side and angle are the ones whose line crosses the least no-go ground on the map
                # (a contact crash would end it before coverage does); ties go to the line that leaves the map sooner.
                fw = walk[min(len(walk) - 1, int(f) + 4)] - walk[int(f)]
                fw = fw / max(np.linalg.norm(fw), 1e-6)
                bestb = None
                for ang in (20, -20, 25, -25, 30, -30, 35, -35, 40, -40, 45, -45):
                    a = np.deg2rad(ang); d_ = np.array([np.cos(a) * fw[0] - np.sin(a) * fw[1], np.sin(a) * fw[0] + np.cos(a) * fw[1]])
                    q0 = p0 + 1.0 * fw; q1 = q0 + 14.0 * d_
                    ng = self._nogo_len(g, nontrav, q0, q1)
                    pts = q0[None, :] + np.linspace(0, 1, 60)[:, None] * (q1 - q0)[None, :]
                    rcs = [self._cell(g, pt) for pt in pts]
                    known = np.array([0 <= r < lab.shape[0] and 0 <= c < lab.shape[1] and lab[r, c] >= 0 for r, c in rcs])
                    key = (round(ng, 1), float(known.mean()))
                    if bestb is None or key < bestb[0]:
                        bestb = (key, np.array([p0, q0, q1]))
                if bestb[0][0] <= 0.35:         # no clean line: a contact crash would hide the coverage gate, so skip
                    paths["beyond"] = bestb[1]
                for kind in self.kinds:
                    if kind in paths:
                        self.plan.append(dict(scene=scene, pair=pi, kind=kind, start_frame=int(f),
                                              goal_xy=[float(goal[0]), float(goal[1])], cross_m=float(cross),
                                              path=paths[kind].tolist()))
        print(f"[path] plan: {len(self.plan)} episodes "
              "(" + ", ".join("%s %d" % (k, sum(p["kind"] == k for p in self.plan)) for k in self.kinds) + ")", flush=True)
        return self.plan

    # ------------------------------------------------------------ episodes
    def options(self, i):
        p = self.plan[i]
        return {"scene": p["scene"], "start_frame": p["start_frame"], "goal_xy": p["goal_xy"]}

    def begin(self, i):
        self.cur = self.plan[i]
        self.path = np.asarray(self.cur["path"], dtype=float)
        self._wp_i = 0

    # SB3 signature, like MapExpert
    def predict(self, obs, deterministic: bool = True):
        return self.act(), None

    def act(self) -> np.ndarray:
        u, cfg = self.u, self.u.cfg
        pose = u._robot_pose_world
        p = np.asarray(pose[:2, 3], dtype=float)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        path = self.path
        # aim point: project the robot onto the polyline (segments from the last one reached on), then walk
        # `lookahead` metres further along it. (Fix 2026-11-09: the old nearest-VERTEX rule aimed back at the
        # start vertex once the robot was > lookahead past it, so sparse 2-3 vertex paths looped.)
        best = (np.inf, self._wp_i, 0.0)
        for j in range(self._wp_i, len(path) - 1):
            a_, b_ = path[j], path[j + 1]; ab = b_ - a_; L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-12 else float(np.clip((p - a_) @ ab / L2, 0.0, 1.0))
            dd = float(np.linalg.norm(a_ + t * ab - p))
            if dd < best[0] - 1e-9:
                best = (dd, j, t)
        _, j, t = best
        self._wp_i = j
        q = path[j] + t * (path[j + 1] - path[j]) if len(path) > 1 else path[0]
        rem, wp = self.lookahead, path[-1]
        while j < len(path) - 1:
            seg_len = float(np.linalg.norm(path[j + 1] - q))
            if seg_len >= rem:
                wp = q + (path[j + 1] - q) * (rem / max(seg_len, 1e-9)); break
            rem -= seg_len; j += 1; q = path[j]
        dyaw = _wrap(float(np.arctan2(wp[1] - p[1], wp[0] - p[0])) - yaw)
        a_yaw = float(np.clip(dyaw / float(cfg.yaw_step_rad), -1.0, 1.0))
        thr = 1.0 if abs(dyaw) < 0.35 else (0.6 if abs(dyaw) < 0.9 else 0.4)
        floor = float(cfg.halt_throttle_eps) + 0.1
        d_goal = float(np.linalg.norm(path[-1] - p))
        thr = max(min(thr, max(d_goal / float(cfg.step_size_m), floor)), floor)
        a = [thr, a_yaw] + ([-1.0] if cfg.stop_action else [])
        return np.asarray(a, dtype=np.float32)
