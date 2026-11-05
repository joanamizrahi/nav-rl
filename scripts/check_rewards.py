"""Reward mechanism check: GATE / VOID / IMGVOID, measured not argued.

Renders the SAME live-diffusion views training uses, at poses drawn from the
J-spec distribution, and prints for each step what every reward mechanism
actually computes — over a SWEEP of alpha-gate thresholds on identical renders.

The renders happen ONCE (ungated). Every threshold is then applied in numpy to
the same labels + alpha, so the tau column is the only thing that varies: no
diffusion-noise difference between rows, and tau=0.0 IS the ungated baseline.

Answers, with numbers:
  * ALPHA        the support distribution as a histogram, whole-image AND
                 inside the footprint (the region the reward actually reads),
                 so the 0.5 gate threshold stops being an inherited guess
  * SWEEP        for each tau: img void, footprint void, collision, crash-level
                 steps, and every reward term — the whole trade curve at once
  * GEOVOID      the IMGVOIDTERM proper: share of the image with NO rasterizer
                 support (alpha <= tau). Pure geometry, measured on the RENDER,
                 no diffusion in it. Printed next to mean alpha (the `cov`
                 number the spin certificates use) — different statistics that
                 nobody had compared
  * SEMVOID      the DIFFUSED semantics' own class-0 share, i.e. where the world
                 model itself claims ignorance. Separate number, never conflated
                 with GEOVOID
  * VOID         footprint void share, and proof it is EXCLUDED from
                 collision_frac when void_cost > 0
  * REWARD       every term at the real x0.01 scale, so we can see what
                 dominates and whether phantom terrain is driving crashes

Usage (GPU node):
    python scripts/check_rewards.py --scene gnd_AUw360 \
        --trav_path config/traversability_v14_walkway.yaml \
        --sweep 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8

Obstacle probe (drive at a KNOWN real obstacle instead of a sampled goal, to
test whether the gate also deletes obstacles that ARE observed):
    ... --goal_xy 1.57,-6.06 --spawn_frame 40
"""
from __future__ import annotations
import os as _os
_S = _os.environ.get("SCRATCH_ROOT", "/scratch/m000204-pm07b/joana")  # 2026-11-05: Cycle 7 scratch; override with SCRATCH_ROOT

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
from src.env.vec_live_env import BatchedLiveDiffusedBackend
from src.eval.traversability import load_traversability
from src.eval.reward_2d import (
    RewardWeights, compute_reward, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
    _footprint_corners_world, _project_points, _fill_polygon,
)

V14_NAMES = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk",
             "road", "pavement", "stairs", "obstacle", "vegetation", "person",
             "vehicle"]


def pose_at(xy, heading):
    c, s = np.cos(heading), np.sin(heading)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = (c, s, 0.0)
    m[:3, 1] = (-s, c, 0.0)
    m[:3, 3] = (xy[0], xy[1], 0.0)
    return m


def episode_poses(cal, rng, n_steps, cone_deg, dist_range, yaw_jit, lat_jit,
                  spawn_min=10, goal_xy=None, spawn_frame=None,
                  walk="straight", step_size_m=0.3, yaw_step_rad=0.5,
                  yaw_span_deg=90.0, support=None,
                  wander_fwd_min=0.0):
    """One J-spec episode: jittered spawn on the recorded path, goal in the
    tangent cone, then a walk from there.

    walk="straight": drive dead at the goal. This is the BEST case — the
      perfectly-trained policy that never leaves the cone. It samples the
      corridor we already believe is coherent, so it systematically MISSES the
      regime where the world model actually breaks down (Joana, 2026-09-01:
      "the policy still wanders off on its own... going too far away into
      completely incoherent/hallucinated zone").

    walk="wander": random actions through the REAL kinematics (yaw first, then
      translate along the new heading — mirrors SceneEnv._advance_pose), which
      is what an untrained policy does. This is the regime the gate has to
      survive.

    goal_xy overrides the sampled goal (obstacle probe): the walk then heads at
    a fixed world point, so the same real obstacle is approached every episode.
    """
    path = np.asarray(cal.positions, dtype=float)[:, :2]
    if spawn_frame is not None:
        f = int(np.clip(spawn_frame, 1, len(path) - 2))
    else:
        f = int(rng.integers(spawn_min, max(len(path) - 6, spawn_min + 1)))
    fw = path[min(f + 1, len(path) - 1)] - path[max(f - 1, 0)]
    base = float(np.arctan2(fw[1], fw[0]))
    p = path[f] + rng.uniform(-1, 1) * lat_jit * np.array(
        [-np.sin(base), np.cos(base)])
    yaw = base + np.deg2rad(rng.uniform(-1, 1) * yaw_jit)
    if goal_xy is not None:
        goal = np.asarray(goal_xy, dtype=float)
    else:
        th = base + np.deg2rad(rng.uniform(-1, 1) * cone_deg / 2.0)
        goal = p + rng.uniform(*dist_range) * np.array([np.cos(th), np.sin(th)])
        # Same support check training applies, so the survey walks toward the
        # goals training would actually draw (her ask: the comparison is only
        # meaningful if the goal distribution matches). `support` is
        # (ground_xy, radius, min_count); None = raw draws.
        if support is not None:
            gpts, srad, sneed = support
            for _t in range(12):
                if int((np.linalg.norm(gpts - goal, axis=1) <= srad).sum()) >= sneed:
                    break
                th = base + np.deg2rad(rng.uniform(-1, 1) * cone_deg / 2.0)
                goal = p + rng.uniform(*dist_range) * np.array(
                    [np.cos(th), np.sin(th)])

    poses = [pose_at(p, yaw)]
    if walk == "path":
        # Follow the RECORDED trajectory with its recorded heading -- the walk
        # that was actually filmed, so the render is as honest as this scene
        # ever gets. `straight` drives at a randomly sampled cone goal instead,
        # which crosses whatever terrain lies between and says more about the
        # goal sampler than about the scene (2026-09-02).
        poses = []
        for k in range(max(2, n_steps)):
            i = int(min(f + k, len(path) - 2))
            fwk = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
            poses.append(pose_at(path[i],
                                 float(np.arctan2(fwk[1], fwk[0]))))
    elif walk == "yaw":
        # YAW LADDER (2026-09-02). Two evals came back with
        # ground_share {'none': 1.0} -- the reward footprint never projected
        # into the image for 20 episodes -- while scripted walks on the same
        # scenes projected fine. The difference was the driver: B pans hard
        # left-right and never translates. So: hold the position and sweep the
        # HEADING, and report at which turn angle the footprint leaves the
        # frame. A box 1.5 m dead ahead of a camera pointing the same way
        # should never leave it, so if it does, the footprint heading and the
        # camera yaw disagree -- and the reward reads terrain the robot is not
        # facing.
        poses = [pose_at(p, yaw + np.deg2rad(d))
                 for d in np.linspace(-yaw_span_deg, yaw_span_deg,
                                      max(2, n_steps))]
    elif walk == "wander":
        cur_p, cur_yaw = p.astype(float).copy(), float(yaw)
        for _ in range(1, n_steps):
            a_fwd = rng.uniform(wander_fwd_min, 1.0)
            a_yaw = rng.uniform(-1.0, 1.0)
            cur_yaw += a_yaw * yaw_step_rad          # yaw first ...
            cur_p = cur_p + a_fwd * step_size_m * np.array(
                [np.cos(cur_yaw), np.sin(cur_yaw)])   # ... then translate
            poses.append(pose_at(cur_p, cur_yaw))
    else:
        u = (goal - p) / max(np.linalg.norm(goal - p), 1e-6)
        for k in range(1, n_steps):
            poses.append(pose_at(p + u * step_size_m * k,
                                 float(np.arctan2(u[1], u[0]))))
    return poses, goal


def render_replay(args):
    """REPLAY MODE (2026-09-06, Joana): re-render the RECORDED poses of chosen
    eval episodes and project the reward MAP into both the rasterised
    reconstruction and the diffused frame the policy saw. Answers 'is the
    render half a metre off from the reconstruction, or are the map's labels
    off?': if the projected grass edge sits on visible pavement in the RASTER
    too, the labels are off; if only in the diffused frame, the generator
    moved the edge. Writes REPLAY_<scene>_ep<k>_s<step>.png = diffused+map |
    raster+map | alpha, with the near (magenta) and far (yellow) boxes."""
    import json, cv2
    from src.eval.reward_map import build_label_grid, footprint_samples
    from src.eval.reward_2d import _project_points, _footprint_corners_world
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only", sem_palette_version=args.sem_palette, static_scene=bool(getattr(args, "static_scene", False)), static_movers=str(getattr(args, "static_movers", "") or ""),
        model_path=args.model_path, reconstructor_path=args.reconstructor_path,
        H=args.height, W=args.width)
    # fusion window knobs (2026-09-16): the same window as training, else the survey
    # renders the full fused scene while the policy trains on 21 frames
    if int(getattr(args, 'render_window', 0)) > 0:
        cfg.render_window = int(args.render_window)
    if int(getattr(args, 'coverage_window', 0)) > 0:
        cfg.coverage_window = int(args.coverage_window)
    world = BatchedLiveDiffusedBackend(cfg, checkpoint=args.live_ckpt, alpha_gate=False)
    world.num_inference_steps = args.num_steps
    world.load_scene(args.scene)
    trav = load_traversability(Path(args.trav_path) if args.trav_path else None)
    non_trav = trav <= args.collision_threshold
    c = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
    g = build_label_grid(c["points"], c["labels"].astype(int), non_trav, res=0.1, inflate_m=0.1,
                         inflate_classes=(10, 11, 13), walk_xy=walk)
    L = g.labels; known = L >= 0; nt = known & non_trav[np.clip(L, 0, len(non_trav) - 1)]
    iy, ix = np.nonzero(known)
    cell_xy = np.c_[g.x0 + (ix + 0.5) * g.res, g.y0 + (iy + 0.5) * g.res].astype(np.float32)
    cell_nt = nt[iy, ix]
    m = json.load(open(args.replay_metrics))
    want = set(int(v) for v in str(args.replay_episodes).split(",") if v.strip())
    od = Path(args.out_dir) if getattr(args, "out_dir", None) else Path(_S + "/outputs/check_rewards")
    od.mkdir(parents=True, exist_ok=True)
    near_d = float(args.collision_look_ahead)
    print(f"=== REPLAY {args.replay_metrics} episodes {sorted(want)}", flush=True)
    for e in m["episodes"]:
        if want and e["episode"] not in want:
            continue
        tr = np.asarray(e["traj"], float)
        n = int(e["steps"])
        print(f"--- ep {e['episode']} {e['outcome']} {n} steps, goal_traversable {e.get('goal_traversable')}", flush=True)
        for k in range(min(n, len(tr))):
            pose = pose_at(tr[k, :2], float(tr[k, 2]))
            (rgb, K, w2c, lab) = world.render_batch([(0, pose)])[0]
            # GENERATED semantics (the label head's own output for this frame),
            # colorized -- this is what differs between checkpoints; the RGB
            # hole-filling is the shared base model's (Joana, 2026-09-06).
            from src.eval.palette import display_palette
            CLASS_COLORS_V14_255 = display_palette(args.sem_palette)   # same colors as the eval videos
            lab_i = np.clip(np.asarray(lab, dtype=int), 0, len(CLASS_COLORS_V14_255) - 1)
            semc = CLASS_COLORS_V14_255[lab_i].astype(np.uint8)
            alpha = getattr(world, "last_alpha", None)
            a = None if not alpha else np.asarray(alpha[0], dtype=np.float32)
            ras = getattr(world, "last_raster", None)
            ras = ras[0] if ras else np.zeros_like(rgb)
            pos = np.array([pose[0, 3], pose[1, 3], 0.0]); hd = np.asarray(pose[:3, 0], float)
            # map cells within 8 m ahead and 5 m to the side, projected into this camera
            rel = cell_xy - pos[None, :2]
            fwd = rel @ hd[:2]; lat = rel[:, 0] * (-hd[1]) + rel[:, 1] * hd[0]
            sel = (fwd > 0.2) & (fwd < 8.0) & (np.abs(lat) < 5.0)
            pts3 = np.c_[cell_xy[sel], np.zeros(sel.sum(), np.float32)]
            uv, front = _project_points(pts3, K, w2c)
            H, W = rgb.shape[:2]
            inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            dif = np.ascontiguousarray(rgb[:, :, ::-1]).copy(); rasb = np.ascontiguousarray(ras[:, :, ::-1]).copy()
            semb = np.ascontiguousarray(semc[:, :, ::-1]).copy()
            # generated label under each projected map cell: agreement counts
            _uv_in = uv[inside].astype(int); _map_nt = cell_nt[sel][inside]
            _gen_nt = non_trav[np.clip(lab_i[np.clip(_uv_in[:, 1], 0, H - 1), np.clip(_uv_in[:, 0], 0, W - 1)], 0, len(non_trav) - 1)]
            n_cells = int(len(_map_nt)); n_agree = int((_gen_nt == _map_nt).sum())
            n_gen_walk_map_grass = int((~_gen_nt & _map_nt).sum()); n_gen_grass_map_walk = int((_gen_nt & ~_map_nt).sum())
            for img in (dif, semb, rasb):
                for (u, v), bad in zip(uv[inside].astype(int), cell_nt[sel][inside]):
                    cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255) if bad else (0, 200, 0), -1)
                for dist, colr in ((near_d, (255, 0, 255)), (args.look_ahead, (0, 255, 255))):
                    cw = _footprint_corners_world(pos, hd, look_ahead_dist=dist, length=0.6, width=0.3)
                    cu, cf = _project_points(cw, K, w2c)
                    if cf.all():
                        cv2.polylines(img, [cu.astype(np.int32)], True, colr, 2, cv2.LINE_AA)
            fp = footprint_samples(pos[:2] + near_d * hd[:2] / (np.linalg.norm(hd[:2]) + 1e-9), hd[:2], 0.6, 0.3, 0.05)
            cl = g.lookup(fp).astype(int); cl = np.where(cl < 0, 0, cl)
            near_frac = float((non_trav[cl] & (cl != 0)).mean())
            cov = float(a.mean()) if a is not None else float("nan")
            apan = (np.repeat((np.clip(a, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2) if a is not None else np.zeros_like(dif))
            panel = np.concatenate([dif, semb, rasb, apan], axis=1)
            _ag = (n_agree / n_cells) if n_cells else float("nan")
            cv2.putText(panel, f"ep {e['episode']} step {k} | map near box {near_frac:.2f} | alpha {cov:.2f} | gen-vs-map agree {_ag:.2f} on {n_cells} cells, gen walkable/map grass {n_gen_walk_map_grass} | red = map non-trav, green = walkable; magenta near, yellow far box",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            for j, name in enumerate(["RGB diffused (what the policy saw) + MAP", "GENERATED semantics + MAP", "RGB raster (reconstruction) + MAP", "SUPPORT (alpha)"]):
                cv2.putText(panel, name, (j * W + 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(od / f"REPLAY_{args.scene}_ep{e['episode']}_s{k:02d}.png"), panel)
            # ---- STEP-0 HISTORY COMPARISON (Joana, 2026-09-06): the first frame
            # of every episode is conditioned on a SYNTHETIC approach (four poses
            # back-extrapolated behind the spawn). Render the same spawn under
            # each history mode and put them side by side:
            #   cold   = the synthetic straight walk-in (what training/eval do)
            #   walk   = the four RECORDED camera poses before the nearest walk frame
            #   same   = five identical poses (the old cold start)
            #   single = one frame, no history (num_frames=1), if the pipe allows
            if k == 0 and str(getattr(args, "hist_modes", "") or "").strip():
                saved_hists = {rid: list(h) for rid, h in world._hists.items()}
                saved_k = world.live_frames
                tiles_rgb, tiles_sem, names = [], [], []
                # reference tiles, independent of the history: the raster the
                # generator is conditioned on (with the map dots) and the
                # raster's own label conditioning, colorized with the same palette
                ref_rgb = rasb.copy()
                cv2.putText(ref_rgb, "RASTER (input render) + MAP", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                _sr = getattr(world, "last_sem_raster", None)
                if _sr:
                    _sri = np.clip(np.asarray(_sr[0], dtype=int), 0, len(CLASS_COLORS_V14_255) - 1)
                    ref_sem = np.ascontiguousarray(CLASS_COLORS_V14_255[_sri][:, :, ::-1]).copy()
                else:
                    ref_sem = np.zeros_like(rasb)
                for (u, v), bad in zip(uv[inside].astype(int), cell_nt[sel][inside]):
                    cv2.circle(ref_sem, (int(u), int(v)), 1, (0, 0, 255) if bad else (0, 200, 0), -1)
                cv2.putText(ref_sem, "RASTER labels (conditioning) + MAP", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                # the REAL photograph: the source video frame whose camera is
                # nearest the spawn, with its SAM3 labels (the ground truth the
                # reconstruction and the generator both descend from)
                photo = np.zeros_like(rasb); photo_lab = np.zeros_like(rasb)
                try:
                    i_ph = int(np.argmin(np.linalg.norm(src[:, :3, 3] - pr[:3, 3][None, :], axis=1)))
                    _scale = float(getattr(world._calib[world._current_scene_id], "scale", 1.0) or 1.0)
                    d_ph = float(np.linalg.norm(src[i_ph, :3, 3] - pr[:3, 3])) * _scale
                    im = scene_c["views"]["img"][0, i_ph].detach().float().cpu().numpy()   # [3,H,W] 0..1
                    im = (np.clip(np.transpose(im, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
                    if im.shape[:2] != rasb.shape[:2]:
                        im = cv2.resize(im, (rasb.shape[1], rasb.shape[0]), interpolation=cv2.INTER_AREA)
                    photo = np.ascontiguousarray(im[:, :, ::-1]).copy()
                    cv2.putText(photo, f"PHOTO walk frame {i_ph}, {d_ph:.1f} m from spawn", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    lb = scene_c["views"]["labels"][0, i_ph].detach().cpu().numpy().astype(int)
                    lb = np.clip(lb, 0, len(CLASS_COLORS_V14_255) - 1)
                    pl = CLASS_COLORS_V14_255[lb].astype(np.uint8)
                    if pl.shape[:2] != rasb.shape[:2]:
                        pl = cv2.resize(pl, (rasb.shape[1], rasb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    photo_lab = np.ascontiguousarray(pl[:, :, ::-1]).copy()
                    cv2.putText(photo_lab, "SAM3 labels of the photo", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as ex:
                    print(f"    [hist photo] not available: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
                scene_c = world._cache[world._current_scene_id]
                pr, _ = world._pose_nav_to_recon(pose)
                src = scene_c["cam2world"].detach().cpu().float().numpy()
                for mode in [m.strip() for m in str(args.hist_modes).split(",") if m.strip()]:
                    world._hists = {}; world.live_frames = saved_k; world.hist_jump_m = 0.5
                    try:
                        if mode == "walk":
                            i0 = int(np.argmin(np.linalg.norm(src[:, :3, 3] - pr[:3, 3][None, :], axis=1)))
                            idx = [max(0, i0 - d) for d in range(saved_k - 1, 0, -1)]
                            world._hists = {0: [src[t].astype(np.float32) for t in idx]}
                            world.hist_jump_m = 1e9
                        elif mode == "same":
                            world._hists = {0: [pr.astype(np.float32).copy() for _ in range(saved_k - 1)]}
                            world.hist_jump_m = 1e9
                        elif mode == "single":
                            world.live_frames = 1
                        (r2, _, _, lab2) = world.render_batch([(0, pose)])[0]
                        a2 = float(np.asarray(world.last_alpha[0], dtype=np.float32).mean()) if getattr(world, "last_alpha", None) else float("nan")
                        t_rgb = np.ascontiguousarray(r2[:, :, ::-1]).copy()
                        l2 = np.clip(np.asarray(lab2, dtype=int), 0, len(CLASS_COLORS_V14_255) - 1)
                        t_sem = np.ascontiguousarray(CLASS_COLORS_V14_255[l2][:, :, ::-1]).copy()
                        for img2 in (t_rgb, t_sem):
                            for (u, v), bad in zip(uv[inside].astype(int), cell_nt[sel][inside]):
                                cv2.circle(img2, (int(u), int(v)), 1, (0, 0, 255) if bad else (0, 200, 0), -1)
                        cv2.putText(t_rgb, f"{mode}  alpha {a2:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        tiles_rgb.append(t_rgb); tiles_sem.append(t_sem); names.append(mode)
                    except Exception as ex:
                        print(f"    [hist {mode}] FAILED: {type(ex).__name__}: {str(ex)[:160]}", flush=True)
                world._hists = saved_hists; world.live_frames = saved_k; world.hist_jump_m = 0.5
                if tiles_rgb:
                    hp = np.concatenate([np.concatenate([photo, ref_rgb] + tiles_rgb, axis=1), np.concatenate([photo_lab, ref_sem] + tiles_sem, axis=1)], axis=0)
                    cv2.putText(hp, f"ep {e['episode']} step 0 spawn: left = the render and its labels; then the first frame under each HISTORY mode (top RGB, bottom generated semantics), map dots everywhere",
                                (8, hp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(str(od / f"REPLAY_{args.scene}_ep{e['episode']}_s00_HIST.png"), hp)
                    print(f"    step  0 history modes rendered: {names}", flush=True)
            _ti = getattr(world, "last_t_idx", None); _ti = (_ti[0] if _ti else -1)
            print(f"    step {k:2d}: map near box {near_frac:.2f}  alpha {cov:.2f}  projected cells {int(inside.sum())}  "
                  f"gen-vs-map agree {_ag:.2f}  gen-walkable/map-grass {n_gen_walk_map_grass}  gen-grass/map-walkable {n_gen_grass_map_walk}  "
                  f"source frame {_ti}", flush=True)
    print(f"==> {od}", flush=True)


def sweep_coverage(args):
    """COVERAGE vs YAW OFFSET (2026-09-06): at walk frames, turn the camera
    off the walk direction by each yaw offset and read the raster alpha (mean
    over the frame, and over the bottom third = the near ground). Rasterizer
    only, the diffusion model is never loaded. Says how far off the walk the
    camera can look before the view is mostly invented -- the number that
    should set the lawn-goal cone, instead of a guess."""
    scenes = [v.strip() for v in str(args.scene).split(",") if v.strip()]   # SCENE may be a comma list
    cfg = CalibratedBackendConfig(
        scene_video_paths={sc: f"{args.clips_dir}/{sc}.mp4" for sc in scenes},
        scene_poses_paths={sc: f"{args.poses_dir}/{sc}_poses.npz" for sc in scenes},
        scene_labels_paths={sc: f"{args.labels_dir}/{sc}.npz" for sc in scenes},
        render_mode="rasterizer_only", sem_palette_version=args.sem_palette, static_scene=bool(getattr(args, "static_scene", False)), static_movers=str(getattr(args, "static_movers", "") or ""),
        model_path=args.model_path, reconstructor_path=args.reconstructor_path,
        H=args.height, W=args.width)
    # fusion window knobs (2026-09-16): the same window as training, else the survey
    # renders the full fused scene while the policy trains on 21 frames
    if int(getattr(args, 'render_window', 0)) > 0:
        cfg.render_window = int(args.render_window)
    if int(getattr(args, 'coverage_window', 0)) > 0:
        cfg.coverage_window = int(args.coverage_window)
    # --cov_pictures: ALSO run the generator at each view and save a panel
    # raster | diffused RGB | generated semantics | alpha (Joana, 2026-09-07:
    # "I'd like to see the generated output too, not only the raster").
    pics = bool(getattr(args, "cov_pictures", False))
    world = BatchedLiveDiffusedBackend(cfg, checkpoint=args.live_ckpt, alpha_gate=False, raster_obs=not pics)
    world.num_inference_steps = args.num_steps
    grand = {}
    for sc in scenes:
      args.scene = sc
      world.load_scene(sc)
      if not pics:
          world._cache = {k: v for k, v in world._cache.items() if k == sc}   # one resident scene at a time
      c = np.load(Path(args.clouds_dir) / f"{sc}_cloud.npz")
      walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1.0, -1.0, 1.0], np.float32))[:, :2]
      frames = [int(v) for v in str(args.cov_frames).split(",") if v.strip() and int(v) < len(walk) - 1]
      yaws = [float(v) for v in str(args.cov_yaws).split(",") if v.strip()]
      if pics:
          import cv2
          from src.eval.palette import display_palette
          _pal = display_palette(args.sem_palette)
          od = Path(args.out_dir) if getattr(args, "out_dir", None) else Path(_S + "/outputs/check_rewards") / f"covsweep_{args.scene}"
          od.mkdir(parents=True, exist_ok=True)
      print(f"=== COVERAGE SWEEP {args.scene}: frames {frames}, yaw offsets {yaws} deg", flush=True)
      print("frame  " + " ".join(f"{y:>+9.0f}" for y in yaws) + "   (mean alpha / near-ground alpha)", flush=True)
      per_yaw = {y: [] for y in yaws}; per_yaw_g = {y: [] for y in yaws}
      for f in frames:
          dv = walk[min(f + 1, len(walk) - 1)] - walk[f]; yaw0 = float(np.arctan2(dv[1], dv[0]))
          row = []
          for y in yaws:
              pose = pose_at(walk[f], yaw0 + np.deg2rad(y))
              world._hists = {}                       # fresh history: each view stands alone
              (rgb_g, _, _, lab_g) = world.render_batch([(0, pose)])[0]
              a = np.asarray(world.last_alpha[0], dtype=np.float32)
              if pics:
                  ras = world.last_raster[0] if getattr(world, "last_raster", None) else np.zeros_like(rgb_g)
                  li = np.clip(np.asarray(lab_g, dtype=int), 0, len(_pal) - 1)
                  semc = _pal[li].astype(np.uint8)
                  apan = np.repeat((np.clip(a, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
                  tiles = [np.ascontiguousarray(t[:, :, ::-1]).copy() for t in (ras, rgb_g, semc)] + [apan]
                  for t, nm in zip(tiles, ("RASTER", "DIFFUSED RGB", "GENERATED semantics", "alpha")):
                      cv2.putText(t, nm, (8, t.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                  pan = np.concatenate(tiles, axis=1)
                  cv2.putText(pan, f"{args.scene} frame {f} yaw {y:+.0f} deg  {'STATIC' if getattr(cfg, 'static_scene', False) else 'dynamic'} scene  alpha {float(a.mean()):.2f}",
                              (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                  cv2.imwrite(str(od / f"COV_{args.scene}_{'static' if getattr(cfg, 'static_scene', False) else 'dynamic'}_f{f:02d}_yaw{int(y):+04d}.png"), pan)
              m_all = float(a.mean()); m_gnd = float(a[int(a.shape[0] * 2 / 3):].mean())
              per_yaw[y].append(m_all); per_yaw_g[y].append(m_gnd)
              row.append(f"{m_all:.2f}/{m_gnd:.2f}")
          print(f"{f:>5}  " + " ".join(f"{r:>9}" for r in row), flush=True)
      print("MEAN   " + " ".join(f"{np.mean(per_yaw[y]):.2f}/{np.mean(per_yaw_g[y]):.2f}".rjust(9) for y in yaws), flush=True)
      grand[sc] = {y: (float(np.mean(per_yaw[y])), float(np.mean(per_yaw_g[y]))) for y in yaws}


    print("=== GRAND TABLE (near-ground alpha) ===", flush=True)
    print(f"{'scene':16s} " + " ".join(f"{y:>+6.0f}" for y in yaws), flush=True)
    for sc, row in grand.items():
        print(f"{sc:16s} " + " ".join(f"{row[y][1]:6.2f}" for y in yaws), flush=True)
    print("==> coverage sweep done", flush=True)


def render_episodes(args):
    """Render every J-spec pose ONCE, ungated. Keep labels + alpha so the gate
    can be applied afterwards at any threshold on identical pixels."""
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        sem_palette_version=args.sem_palette, static_scene=bool(getattr(args, "static_scene", False)), static_movers=str(getattr(args, "static_movers", "") or ""),
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.height, W=args.width,
    )
    # fusion window knobs (2026-09-16): the same window as training, else the survey
    # renders the full fused scene while the policy trains on 21 frames
    if int(getattr(args, 'render_window', 0)) > 0:
        cfg.render_window = int(args.render_window)
    if int(getattr(args, 'coverage_window', 0)) > 0:
        cfg.coverage_window = int(args.coverage_window)
    world = BatchedLiveDiffusedBackend(cfg, checkpoint=args.live_ckpt,
                                       alpha_gate=False)
    world.num_inference_steps = args.num_steps
    world.load_scene(args.scene)
    cal = world._calib[args.scene]
    # 2026-09-15: the blind-zone prediction must use the height the RENDER
    # camera actually sits at, which is per scene (poses npz camera_height_m,
    # the reconstruction's own height x scale), not the --camera_height
    # default. With the default the pilot printed 0.60 m for every scene of
    # both cameras and the certification would have compared nothing.
    if getattr(cal, "camera_height_m", None):
        print(f"[check_rewards] camera height for the geometry block: {float(cal.camera_height_m):.2f} m "
              f"from {args.scene}_poses.npz (--camera_height {args.camera_height:.2f} ignored)", flush=True)
        args.camera_height = float(cal.camera_height_m)

    goal_xy = None
    if args.goal_xy:
        goal_xy = [float(v) for v in args.goal_xy.split(",")]

    # Goal support, calibrated the way SceneEnv does it: the reference is the
    # ground-point density at the RECORDED path positions, and a goal must
    # reach min_frac of it. Says nothing about traversability -- a goal on
    # grass is legitimate and is the point of the curriculum.
    _support = None
    if args.goal_support_radius > 0.0:
        try:
            _c = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
            _p, _l = _c["points"], _c["labels"].astype(int)
            _g = _p[(_p[:, 2] < 0.15) & (_l >= 0)][:, :2][::4]
            _path = (np.asarray(_c["traj_positions"], dtype=float)
                     * np.array([1.0, -1.0, 1.0]))[:, :2]
            _cnt = [int((np.abs(_g - q).max(axis=1) <= args.goal_support_radius).sum())
                    for q in _path[::max(1, len(_path) // 40)]]
            _ref = float(np.median(_cnt)) if _cnt else 0.0
            _need = max(1, int(args.goal_support_min_frac * _ref))
            _support = (_g, args.goal_support_radius, _need)
            print(f"goal support: reference {_ref:.0f} pts within "
                  f"{args.goal_support_radius} m on the recorded path -> "
                  f"goals need >= {_need}", flush=True)
        except Exception as _e:
            print(f"goal support: DISABLED ({_e})", flush=True)

    rng = np.random.default_rng(args.seed)
    recs = []
    for ep in range(args.episodes):
        poses, goal = episode_poses(
            cal, rng, args.steps, args.cone_deg,
            tuple(float(v) for v in args.dist_range.split(",")),
            args.spawn_yaw_jitter, args.spawn_lat_jitter,
            goal_xy=goal_xy, spawn_frame=args.spawn_frame,
            walk=args.walk, step_size_m=args.step_size_m,
            yaw_span_deg=args.yaw_span, support=_support,
            yaw_step_rad=args.yaw_step_rad,
            wander_fwd_min=args.wander_fwd_min)
        prev = None
        for si, pose in enumerate(poses):
            (rgb, K, w2c, lab) = world.render_batch([(0, pose)])[0]
            alpha = getattr(world, "last_alpha", None)
            a = None if not alpha else np.asarray(alpha[0], dtype=np.float32).copy()
            _ras = getattr(world, "last_raster", None)
            _sr = getattr(world, "last_sem_raster", None)
            pos = np.array([pose[0, 3], pose[1, 3], 0.0], dtype=float)
            recs.append(dict(
                ep=ep, step=si, lab=np.asarray(lab).copy(), alpha=a,
                K=np.asarray(K).copy(), w2c=np.asarray(w2c).copy(),
                pos=pos, head=np.asarray(pose[:3, 0], dtype=float).copy(),
                goal=np.array([goal[0], goal[1], 0.0], dtype=float),
                prev=None if prev is None else prev.copy(),
                rgb=rgb.copy(),
                ras=(_ras[0].copy() if _ras else None),
                sras=(_sr[0].copy() if _sr else None)))
            prev = pos.copy()
        print(f"  rendered episode {ep + 1}/{args.episodes}", flush=True)
    return recs


def build_ladder(recs, od, stem, args, cv2, cols=6, tw=240, th=144):
    """The coverage ladder — every rendered pose sorted by mean alpha, best
    first, each tile showing what the model was GIVEN (raster) beside what it
    MADE (diffused), with the number on it.

    Her call, 2026-09-01: a threshold is a perceptual judgment. Ground-truth
    error curves are a paper experiment; picking the knob is a matter of
    scrolling until the pictures stop looking like the world and reading the
    number off that tile. No metric can do that better than eyes can.
    """
    order = sorted(range(len(recs)),
                   key=lambda i: -(float(recs[i]["alpha"].mean())
                                   if recs[i]["alpha"] is not None else 0.0))
    tiles = []
    for i in order:
        r = recs[i]
        dif = cv2.resize(r["rgb"], (tw, th))
        ras = (np.zeros_like(dif) if r["ras"] is None
               else cv2.resize(r["ras"], (tw, th)))
        pair = np.hstack([ras, dif])
        bar = np.zeros((24, pair.shape[1], 3), dtype=np.uint8)
        if r["alpha"] is not None:
            a = float(r["alpha"].mean())
            v = float((r["alpha"] <= args.gate_tau).mean())
            txt = f"alpha {a:.2f}   void@{args.gate_tau} {v:.2f}   ep{r['ep']}s{r['step']}"
        else:
            txt = f"ep{r['ep']}s{r['step']}"
        cv2.putText(bar, txt, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tile = np.vstack([bar, pair])
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                      (60, 60, 60), 1)
        tiles.append(tile)

    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + cols])
                      for r in range(0, len(tiles), cols)])
    head = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(head, f"{stem}  COVERAGE LADDER  (left=raster given to the "
                      f"model, right=what it generated)  sorted by mean alpha,"
                      f" best first", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([head, grid])[:, :, ::-1]
    p = od / f"LADDER_{stem}.png"
    cv2.imwrite(str(p), np.ascontiguousarray(out))
    print(f"==> coverage ladder: {p}", flush=True)


def footprint_uv(rec, look_ahead):
    """The projected footprint quad, or None if it straddles the camera."""
    corners = _footprint_corners_world(
        rec["pos"], rec["head"], look_ahead_dist=look_ahead,
        length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
    uv, in_front = _project_points(corners, rec["K"], rec["w2c"])
    return uv if in_front.all() else None


def footprint_mask(rec, look_ahead):
    """The exact pixels compute_reward scores — so alpha can be measured THERE
    and not just over the whole image."""
    uv = footprint_uv(rec, look_ahead)
    if uv is None:
        return None
    h, w = rec["lab"].shape[:2]
    return _fill_polygon(h, w, uv)


def quicktime_safe(path) -> None:
    """Re-encode an OpenCV mp4 so QuickTime will actually play it.

    OpenCV on this cluster has no H.264 encoder, so VideoWriter falls back to
    mp4v (MPEG-4 Part 2), which QuickTime renders as a GREEN SCREEN -- VLC
    plays it fine, which is how it went unnoticed. There is no system ffmpeg
    either, but the neoverse env ships one inside imageio_ffmpeg. Use it, in
    place, and keep the original if anything fails. -pix_fmt yuv420p is not
    optional: QuickTime refuses even H.264 without it.
    """
    import shutil
    import subprocess
    from pathlib import Path as _P
    src = _P(path)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    if not exe or not src.exists():
        return
    tmp = src.with_suffix(".h264.mp4")
    try:
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)],
                       check=True, timeout=300)
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(src)
            print(f"    re-encoded H.264 (QuickTime-safe): {src.name}", flush=True)
    except Exception as e:
        print(f"    [quicktime_safe] left as-is ({e})", flush=True)
        if tmp.exists():
            tmp.unlink()


def yaw_ladder(recs, non_trav, args):
    """At which TURN ANGLE does the reward stop seeing the ground?

    Position held fixed, heading swept. For each rung: does the footprint
    project in front of the camera at all, does it cover any pixels, how many,
    and what does it score. If pixel coverage collapses away from 0 deg, a
    turning policy is being rewarded blind -- which makes turning free and
    moving expensive, and is a far better explanation for a policy that pans
    in place than anything in the reward weights.
    """
    n = len(recs)
    mid = n // 2
    base = float(np.arctan2(recs[mid]["head"][1], recs[mid]["head"][0]))
    print("\n===== YAW LADDER (does the footprint survive a turn?) =====")
    print(f"  position fixed, heading swept +-{args.yaw_span:.0f}deg over "
          f"{n} rungs, footprint at {args.look_ahead} m\n")
    print(f"  {'yaw':>7}{'in front':>10}{'HAS PIXELS':>12}{'px':>8}"
          f"{'coll':>8}{'mean trav':>11}{'cov':>8}")
    lost = []
    for r in recs:
        yaw = float(np.arctan2(r["head"][1], r["head"][0]))
        d = np.degrees((yaw - base + np.pi) % (2 * np.pi) - np.pi)
        uv = footprint_uv(r, args.look_ahead)
        if uv is None:
            print(f"  {d:+7.1f}{'NO':>10}{'-':>12}{'-':>8}{'-':>8}"
                  f"{'-':>11}{'-':>8}   BEHIND CAMERA")
            lost.append(d)
            continue
        h, w = r["lab"].shape[:2]
        m = _fill_polygon(h, w, uv)
        npx = int(m.sum())
        cov = (float(r["alpha"].mean()) if r.get("alpha") is not None
               else float("nan"))
        if npx == 0:
            print(f"  {d:+7.1f}{'yes':>10}{'0%':>12}{0:>8}{'-':>8}"
                  f"{'-':>11}{cov:8.3f}   OFF FRAME -> reward is BLIND")
            lost.append(d)
            continue
        cls = r["lab"][m]
        idx = np.clip(cls, 0, len(non_trav) - 1)
        coll = float(non_trav[idx].mean())
        trav = float(1.0 - non_trav[idx].mean())
        print(f"  {d:+7.1f}{'yes':>10}{'100%':>12}{npx:8d}{coll:8.3f}"
              f"{trav:11.3f}{cov:8.3f}")
    if lost:
        print(f"\n  BLIND at {len(lost)}/{n} angles: "
              f"{', '.join(f'{v:+.0f}' for v in lost)} deg")
        print("  A footprint centred on the robot's own heading should never "
              "leave a camera\n  pointing the same way. If it does, the "
              "footprint heading and the camera yaw\n  disagree -- the reward "
              "is scoring ground the robot is not facing.")
    else:
        print("\n  Footprint projects at every angle: turning does NOT blind "
              "the reward,\n  so the off-frame evals came from position, not "
              "heading.")


def look_ahead_ladder(recs, non_trav, args):
    """Which collision look-ahead distances are actually VISIBLE in the frame?

    Splitting the lethal collision test from the graded semantic score only
    works if the near box lands in the image. The camera sits low (0.25 m by
    default, but `camera_height_m` is PER SCENE) and the ground within roughly
    the first 0.6 m falls below the bottom edge -- a box there projects to zero
    pixels, `collision_frac` reads 0, and crashes stop firing altogether. That
    failure is silent, so it gets measured on real diffused frames before any
    training run rather than argued from a comment.

    For each distance: how often the quad is in front of the camera at all, how
    often it covers any pixels, how big it is, and what it scores.
    """
    dists = [float(v) for v in args.ladder_dists.split(",")]

    # ---- the geometry that DECIDES the answer, printed before the table ----
    # For a level camera h above the footprint plane, a footprint d metres
    # ahead lands at image row v = cy + fy*(h/d). It is in the picture only
    # while v <= H-1, i.e.  d >= fy*h / (H-1-cy).  That threshold is the
    # scene's blind-zone radius and it is a pure function of the intrinsics --
    # so if it differs scene to scene, the reward going blind on 3 of 5 scenes
    # is explained without appealing to the policy at all (2026-09-02).
    K = recs[0]["K"]
    H, W = recs[0]["lab"].shape[:2]
    fy, cy = float(K[1, 1]), float(K[1, 2])
    h = float(args.camera_height)
    denom = (H - 1.0) - cy
    d_min = (fy * h / denom) if denom > 0 else float("inf")
    # Implied vertical FOV: the single number that says whether fy is
    # PLAUSIBLE for this render size. A walking camera is ~45-55 deg
    # vertically, which at H=336 means fy ~ 320-400. If fy comes back at
    # 500-700 (27-37 deg) the intrinsics were probably never rescaled from the
    # reconstructor's working resolution to the render size -- a bug we can
    # fix, rather than a camera that genuinely could not see its own feet.
    vfov = 2.0 * np.degrees(np.arctan((H / 2.0) / fy)) if fy > 0 else float("nan")
    print("\n===== CAMERA GEOMETRY =====")
    print(f"  render {W}x{H}   fy {fy:.1f}   cy {cy:.1f}   "
          f"camera height {h:.2f} m")
    print(f"  implied vertical FOV {vfov:.1f} deg"
          + ("   <-- IMPLAUSIBLY NARROW for a walking camera; suspect the "
             "intrinsics\n      were not rescaled to the render size"
             if vfov < 40.0 else "   (plausible)"))
    print(f"  cy offset from centre {cy - H / 2.0:+.1f} px"
          + ("   <-- camera is PITCHED" if abs(cy - H / 2.0) > 0.08 * H
             else "   (centred, not pitched)"))
    print(f"  horizon row (cy) {cy:.1f}, bottom row {H - 1}")
    print(f"  PREDICTED BLIND ZONE: ground closer than {d_min:.2f} m is below "
          f"the frame")
    if d_min > args.look_ahead:
        print(f"  ==> the {args.look_ahead} m shaping footprint is INSIDE the "
              f"blind zone on this scene:\n      the reward cannot see the "
              f"ground it is scoring.")
    else:
        print(f"  ==> the {args.look_ahead} m shaping footprint clears it "
              f"({args.look_ahead - d_min:.2f} m of margin).")
    print("\n===== LOOK-AHEAD LADDER (is the near collision box visible?) =====")
    print(f"  body {GO2_BODY_LENGTH:.2f} m long, so a box centred at d spans "
          f"d+-{GO2_BODY_LENGTH / 2:.2f} m")
    print(f"  {len(recs)} rendered steps, labels are the DIFFUSED semantics "
          f"(what the reward reads)\n")
    print(f"  {'centre':>7}{'near edge':>11}{'in front':>10}{'HAS PIXELS':>12}"
          f"{'med px':>9}{'coll':>8}{'>=crash':>9}")
    for la in dists:
        n_front = n_px = 0
        pix, coll = [], []
        for r in recs:
            uv = footprint_uv(r, la)
            if uv is None:
                continue
            n_front += 1
            h, w = r["lab"].shape[:2]
            m = _fill_polygon(h, w, uv)
            npx = int(m.sum())
            if npx == 0:
                continue
            n_px += 1
            pix.append(npx)
            cls = r["lab"][m]
            idx = np.clip(cls, 0, len(non_trav) - 1)
            coll.append(float((non_trav[idx] & (cls != 0)).mean()))
        n = max(len(recs), 1)
        if not pix:
            print(f"  {la:7.2f}{la - GO2_BODY_LENGTH / 2:11.2f}"
                  f"{100.0 * n_front / n:9.0f}%{0.0:11.0f}%"
                  f"{'-':>9}{'-':>8}{'-':>9}   NEVER VISIBLE")
            continue
        cr = float(np.mean(np.array(coll) >= args.crash_frac))
        print(f"  {la:7.2f}{la - GO2_BODY_LENGTH / 2:11.2f}"
              f"{100.0 * n_front / n:9.0f}%{100.0 * n_px / n:11.0f}%"
              f"{int(np.median(pix)):9d}{np.mean(coll):8.3f}{100.0 * cr:8.0f}%")
    print(f"\n  Pick the SMALLEST centre whose HAS PIXELS is ~100%. Below that "
          f"the box is in the\n  camera blind zone and collision silently "
          f"reads 0. `coll` and `>=crash` are what the\n  reward would see at "
          f"that distance on these frames.")


def score(recs, tau, trav, non_trav, weights, args):
    """Apply the gate at threshold tau and re-score every rendered step."""
    rows = []
    n_mem = int(getattr(args, "collision_box_memory", 0) or 0)
    for r in recs:
        lab = r["lab"]
        if tau > 0.0 and r["alpha"] is not None:
            lab = np.where(r["alpha"] > tau, lab, 0)
        # FRAME MEMORY (2026-09-16, Joana): the previous n_mem steps of the same
        # episode, oldest first, exactly what the env hands compute_reward when
        # --collision_box_memory > 0. The near box that sits below the camera now
        # is read from the newest stored frame that still contains it.
        fm = None
        if n_mem > 0:
            fm = [(q["lab"], q["K"], q["w2c"]) for q in recs
                  if q["ep"] == r["ep"] and 0 <= r["step"] - q["step"] <= n_mem and q["step"] < r["step"]]
            fm = fm[-n_mem:] or None
        b = compute_reward(
            semantic_image=lab, K=r["K"], w2c=r["w2c"],
            robot_position=r["pos"], robot_heading=r["head"], goal=r["goal"],
            traversability_scores=trav, non_traversable_mask=non_trav,
            previous_position=r["prev"], look_ahead_dist=args.look_ahead,
            collision_look_ahead_dist=(args.collision_look_ahead if n_mem > 0 and args.collision_look_ahead > 0 else None),
            frame_memory=fm,
            memory_aggregate=str(getattr(args, "collision_box_memory_agg", "newest") or "newest"),
            body_length=GO2_BODY_LENGTH, body_width=GO2_BODY_WIDTH,
            weights=weights)
        r["box_memory_age"] = float(getattr(b, "box_memory_age", 0.0))
        r["memory_hits"] = int(getattr(b, "memory_hits", 0))
        cls, cnt = np.unique(lab, return_counts=True)
        # TWO different "void" numbers, never to be conflated:
        #   sem_void = the DIFFUSED semantics' own class-0 share (what the world
        #              model says it doesn't know — measured at 0.000, it never
        #              says it)
        #   geo_void = the RASTERIZER's unsupported share, alpha <= tau. Pure
        #              geometry, no diffusion involved. THIS is the IMGVOIDTERM.
        sup = float((r["alpha"] > tau).mean()) if r["alpha"] is not None else float("nan")
        rows.append(dict(
            tau=tau, ep=r["ep"], step=r["step"],
            sem_void=float((r["lab"] == 0).mean()),
            geo_void=1.0 - sup,
            img_void=float((lab == 0).mean()),
            mean_alpha=float(r["alpha"].mean()) if r["alpha"] is not None else float("nan"),
            a10=float(np.percentile(r["alpha"], 10)) if r["alpha"] is not None else float("nan"),
            a50=float(np.percentile(r["alpha"], 50)) if r["alpha"] is not None else float("nan"),
            a90=float(np.percentile(r["alpha"], 90)) if r["alpha"] is not None else float("nan"),
            sup=sup,
            fp_void=float(b.void_frac),
            fp_coll=float(-b.collision / max(weights.collision, 1e-6)),
            fp_ok=float(max(0.0, 1.0 - b.void_frac
                            + b.collision / max(weights.collision, 1e-6))),
            sem=float(b.semantic), goal_t=float(b.goal),
            coll_t=float(b.collision), void_t=float(b.void),
            total=float(b.total),
            dom=V14_NAMES[int(cls[np.argmax(cnt)])]))
    return rows


def alpha_histogram(vals, nbins=10):
    h, edges = np.histogram(np.clip(vals, 0.0, 1.0), bins=nbins, range=(0.0, 1.0))
    frac = h / max(h.sum(), 1)
    lines = []
    for i in range(nbins):
        bar = "#" * int(round(frac[i] * 60))
        lines.append(f"    [{edges[i]:.1f},{edges[i+1]:.1f})  {frac[i]*100:5.1f}%  {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", default=_S + "/data/rugd_clips")
    ap.add_argument("--poses_dir", default=_S + "/outputs/poses")
    ap.add_argument("--labels_dir",
                    default=_S + "/NeoVerse/outputs/sam3_labels_v14")
    ap.add_argument("--live_ckpt",
                    default=_S + "/runs/train_semantic_v21/checkpoint-epoch-12.safetensors")
    ap.add_argument("--model_path", default=_S + "/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default=_S + "/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--trav_path", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--num_steps", type=int, default=4)
    ap.add_argument("--static_movers", type=str, default="", help="with --static_scene: classes kept per-frame, e.g. 12,13 (needs the movers reconstructor)")
    ap.add_argument("--static_scene", action="store_true",
                    help="reconstruct as a STATIC scene: every source frame's Gaussians render from any pose")
    ap.add_argument("--sem_palette", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cone_deg", type=float, default=50.0)
    ap.add_argument("--dist_range", default="5,10")
    ap.add_argument("--spawn_yaw_jitter", type=float, default=20.0)
    ap.add_argument("--spawn_lat_jitter", type=float, default=0.4)
    ap.add_argument("--semantic_weight", type=float, default=5.0)
    ap.add_argument("--goal_weight", type=float, default=10.0)
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--look_ahead", type=float, default=1.5)
    ap.add_argument("--survey_video", action="store_true",
                    help="write SURVEY_<scene>.mp4: RGB diffused | SEM diffused "
                         "| SEM raster (SAM3 splat labels), side by side. For "
                         "choosing scenes: is the RGB good, are the diffused "
                         "semantics good, and do the cloud labels that place "
                         "spawns and goals agree with them?")
    ap.add_argument("--goal_support_radius", type=float, default=0.0,
                    help="apply training's goal support check when sampling "
                         "the straight-walk goal (needs --clouds_dir)")
    ap.add_argument("--goal_support_min_frac", type=float, default=0.25)
    ap.add_argument("--clouds_dir",
                    default=_S + "/outputs/scene_clouds/clouds")
    ap.add_argument("--coh_hard_tau", type=float, default=0.1,
                    help="coverage below this terminates the episode in "
                         "training (coherence_terminate_tau)")
    ap.add_argument("--gate_tau_cov", type=float, default=0.4,
                    help="coverage threshold drawn on the survey HUD; matches "
                         "coherence_tau")
    ap.add_argument("--render_window", type=int, default=0,
                    help="fusion window (frames) for the rasters the generator is given; 0 = whole scene. "
                         "2026-09-16: must match training (RENDWIN) so the survey shows what the policy sees")
    ap.add_argument("--coverage_window", type=int, default=0,
                    help="window for the coverage the gate reads; 0 = whole scene (match COVWIN)")
    ap.add_argument("--camera_height", type=float, default=0.6,
                    help="camera height above the footprint plane, for the "
                         "blind-zone prediction. NOTE this is the same number "
                         "extract_poses uses to set the scene's METRIC SCALE "
                         "(scale = camera_height_m / h_median), so it is an "
                         "assumption, not a measurement.")
    ap.add_argument("--yaw_span", type=float, default=90.0,
                    help="with --walk yaw: sweep the heading +-this many "
                         "degrees from the path tangent, position held fixed")
    ap.add_argument("--ladder_dists", default="0.6,0.8,1.0,1.2,1.5,1.8",
                    help="collision look-ahead distances to test for visibility")
    ap.add_argument("--anchor_box_every", type=int, default=0,
                    help="2026-09-16 (Joana): every N steps, FIX the near box in the world and keep drawing that same "
                         "ground patch in the following frames (magenta, with its age) as the robot walks up to it "
                         "and it slides out of the frame -- the picture of what the frame memory reads")
    ap.add_argument("--anchor_dist", type=float, default=0.0,
                    help="distance ahead at which an anchored box is placed (0 = the collision box distance). "
                         "2.0 with --anchor_box_every 4 keeps one patch of ground in view for ~8 steps as the "
                         "robot walks onto it: the projection check")
    ap.add_argument("--path_overlay", action="store_true",
                    help="2026-09-16: draw the walk's NEXT 8 recorded poses on each frame (green = executed path) and a "
                         "fake 10-step straight plan (orange) -- verifies the rollout-video overlay without a policy")
    ap.add_argument("--collision_box_memory_agg", type=str, default="newest", choices=("newest", "mean"),
                    help="newest stored frame containing the near box, or the MEAN over all stored frames containing it")
    ap.add_argument("--collision_box_memory", type=int, default=0,
                    help="2026-09-16: score the near box from the previous N frames when the camera "
                         "cannot see it (the env's collision_box_memory); adds a MEMORY column to the survey")
    ap.add_argument("--collision_look_ahead", type=float, default=1.0,
                    help="second footprint drawn on the panels in MAGENTA: the "
                         "proposed lethal box, next to the yellow shaping box")
    ap.add_argument("--reward_scale", type=float, default=0.01)
    ap.add_argument("--crash_frac", type=float, default=0.35)
    ap.add_argument("--sweep", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
                    help="alpha-gate thresholds to score; 0 = ungated")
    ap.add_argument("--gate_tau", type=float, default=0.5,
                    help="the threshold shown in the panels and detail blocks")
    ap.add_argument("--walk", default="straight",
                    choices=["straight", "wander", "yaw", "path"],
                    help="straight = drive at the goal (best case, coherent "
                         "corridor); wander = random actions through the real "
                         "kinematics (what an untrained policy does)")
    ap.add_argument("--step_size_m", type=float, default=0.3)
    ap.add_argument("--yaw_step_rad", type=float, default=0.5)
    ap.add_argument("--wander_fwd_min", type=float, default=0.0,
                    help="lower bound on the forward action while wandering; "
                         "-1 allows backing up, 0 keeps it moving outward")
    ap.add_argument("--goal_xy", default="",
                    help="obstacle probe: fixed world goal 'x,y' for all episodes")
    ap.add_argument("--spawn_frame", type=int, default=None,
                    help="obstacle probe: fixed spawn frame on the recorded path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay_metrics", default="", help="metrics.json of an eval: re-render its recorded poses with the map projected in")
    ap.add_argument("--replay_episodes", default="", help="comma list of episode ids to replay (empty = all)")
    ap.add_argument("--cov_sweep", action="store_true", help="coverage vs yaw offset at walk frames (rasterizer only)")
    ap.add_argument("--hist_modes", default="", help="replay: also render each episode's step 0 under these history modes (cold,walk,same,single)")
    ap.add_argument("--cov_frames", default="10,20,30,40,50,60,70")
    ap.add_argument("--cov_yaws", default="0,15,30,45,60,90,-15,-30,-45,-60,-90")
    ap.add_argument("--cov_pictures", action="store_true", help="coverage sweep also runs the generator and saves raster|diffused|semantics|alpha panels")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out",
                    default=_S + "/outputs/reward_check")
    args = ap.parse_args()

    taus = [float(v) for v in args.sweep.split(",")]
    if args.gate_tau not in taus:
        taus.append(args.gate_tau)
    taus = sorted(set(taus))

    print(f"=== {args.scene} | trav {args.trav_path} | cone {args.cone_deg} | "
          f"jitter {args.spawn_yaw_jitter}deg/{args.spawn_lat_jitter}m | "
          f"reward_scale {args.reward_scale} | look_ahead {args.look_ahead}",
          flush=True)
    if args.goal_xy:
        print(f"=== OBSTACLE PROBE: fixed goal {args.goal_xy} "
              f"spawn_frame {args.spawn_frame}", flush=True)
    print(f"=== sweep taus {taus}", flush=True)

    if getattr(args, "cov_sweep", False):
        sweep_coverage(args)
        return
    if getattr(args, "replay_metrics", None):
        render_replay(args)
        return
    recs = render_episodes(args)
    n = len(recs)
    print(f"==> {n} rendered steps, gating applied post-hoc on identical pixels",
          flush=True)

    trav = load_traversability(Path(args.trav_path) if args.trav_path else None)
    non_trav = trav <= args.collision_threshold
    weights = RewardWeights(semantic=args.semantic_weight, goal=args.goal_weight,
                            collision=1.0, step_cost=0.05, void_cost=0.3,
                            terrain_as_cost=True)

    if args.walk == "yaw":
        yaw_ladder(recs, non_trav, args)
    look_ahead_ladder(recs, non_trav, args)

    # ---- the support distribution: whole image AND inside the footprint ----
    have_alpha = recs[0]["alpha"] is not None
    if have_alpha:
        all_a = np.concatenate([r["alpha"].ravel() for r in recs])
        fp_a = []
        for r in recs:
            m = footprint_mask(r, args.look_ahead)
            if m is not None and m.any():
                fp_a.append(r["alpha"][m])
        fp_a = np.concatenate(fp_a) if fp_a else np.zeros(1, dtype=np.float32)
        print("\n===== ALPHA DISTRIBUTION (whole image, all steps pooled) =====")
        print(f"  mean {all_a.mean():.3f}  p10 {np.percentile(all_a, 10):.3f}  "
              f"p50 {np.percentile(all_a, 50):.3f}  "
              f"p90 {np.percentile(all_a, 90):.3f}")
        print(alpha_histogram(all_a))
        print("\n===== ALPHA DISTRIBUTION (FOOTPRINT ONLY — what the reward reads) =====")
        print(f"  mean {fp_a.mean():.3f}  p10 {np.percentile(fp_a, 10):.3f}  "
              f"p50 {np.percentile(fp_a, 50):.3f}  "
              f"p90 {np.percentile(fp_a, 90):.3f}")
        print(alpha_histogram(fp_a))
        print("  A GAP between two humps = a principled threshold. A smooth "
              "ramp = any threshold is a policy choice, not a fact.", flush=True)

    # ---- COHERENCE: does consecutive stepping stay in the SAME world? ----
    # Joana's worse-than-phantom failure mode (2026-09-01): "it hallucinates a
    # completely different scene or something completely incoherent". A 0.3 m
    # step in a coherent world changes the image a little; a world model that
    # flips to a different scene changes it a lot. Measure that jump directly,
    # and check whether raster support predicts it.
    jump_keys = set()
    jumps = []
    for i in range(1, len(recs)):
        a, b = recs[i - 1], recs[i]
        if b["ep"] != a["ep"] or b["step"] != a["step"] + 1:
            continue
        d = float(np.abs(b["rgb"].astype(np.int16)
                         - a["rgb"].astype(np.int16)).mean() / 255.0)
        dist = float(np.linalg.norm(b["pos"][:2] - a["pos"][:2]))
        ma = float(b["alpha"].mean()) if b["alpha"] is not None else float("nan")
        jumps.append((d, dist, ma, b["ep"], b["step"]))
    if jumps:
        J = np.array([[j[0], j[1], j[2]] for j in jumps], dtype=float)
        print("\n===== COHERENCE BETWEEN CONSECUTIVE STEPS =====")
        print(f"  frame-to-frame |dRGB| (0-1):  mean {J[:, 0].mean():.4f}  "
              f"p50 {np.percentile(J[:, 0], 50):.4f}  "
              f"p90 {np.percentile(J[:, 0], 90):.4f}  max {J[:, 0].max():.4f}")
        print(f"  distance moved per step (m):  mean {J[:, 1].mean():.3f}  "
              f"max {J[:, 1].max():.3f}")
        if not np.isnan(J[:, 2]).all():
            order = np.argsort(J[:, 2])
            k = max(len(order) // 3, 1)
            lo, hi = order[:k], order[-k:]
            print(f"  LOW-support third  (alpha {J[lo, 2].mean():.3f}):  "
                  f"jump {J[lo, 0].mean():.4f}")
            print(f"  HIGH-support third (alpha {J[hi, 2].mean():.3f}):  "
                  f"jump {J[hi, 0].mean():.4f}")
            print("  Low-support third jumping MORE = coverage predicts "
                  "coherence (Joana's hypothesis).")
        worst = sorted(jumps, key=lambda j: -j[0])[:4]
        print("  worst jumps (panels saved):  " + ", ".join(
            f"ep{e}s{s} d={d:.3f} alpha={m:.2f}" for d, _, m, e, s in worst))
        jump_keys = {(e, s) for _, _, _, e, s in worst}

    # ---- does the diffusion degrade where the rasterizer has no support? ----
    # Joana's hypothesis (2026-09-01): world-model coherence tracks how much of
    # the view the UNDIFFUSED raster already covers. Operational version: bin
    # every pixel by its alpha and ask what the model paints there, and whether
    # it still agrees with the splat labels it was conditioned on.
    if have_alpha and recs[0]["sras"] is not None:
        nb = 10
        tot = np.zeros(nb); obs = np.zeros(nb); wlk = np.zeros(nb)
        agr = np.zeros(nb); agrd = np.zeros(nb)
        for r in recs:
            b = np.clip((r["alpha"] * nb).astype(int), 0, nb - 1).ravel()
            lab = np.clip(r["lab"], 0, len(trav) - 1)
            isobs = (non_trav[lab] & (lab != 0)).ravel().astype(float)
            iswalk = (trav[lab] > 0.5).ravel().astype(float)
            tot += np.bincount(b, minlength=nb)
            obs += np.bincount(b, weights=isobs, minlength=nb)
            wlk += np.bincount(b, weights=iswalk, minlength=nb)
            sr = np.asarray(r["sras"]).astype(int)
            v = (sr > 0).ravel()
            if v.any():
                agrd += np.bincount(b[v], minlength=nb)
                agr += np.bincount(b[v], weights=(lab.ravel()[v] == sr.ravel()[v]
                                                 ).astype(float), minlength=nb)
        print("\n===== DIFFUSION vs RASTER SUPPORT (all pixels, all steps) =====")
        print("  alpha bin   share    P(non-trav)  P(walkable)  agrees w/ splat lbl")
        for i in range(nb):
            if tot[i] == 0:
                continue
            ag = f"{agr[i]/agrd[i]:.3f}" if agrd[i] > 0 else "   -  "
            print(f"   [{i/nb:.1f},{(i+1)/nb:.1f})   {tot[i]/tot.sum():.3f}    "
                  f"{obs[i]/tot[i]:.3f}        {wlk[i]/tot[i]:.3f}        {ag}")
        print("  If P(non-trav) climbs sharply as alpha falls, the model INVENTS\n"
              "  obstacles exactly where it has no geometry — that is the phantom\n"
              "  mechanism, quantified. If it stays flat, low alpha is not the\n"
              "  problem and the gate is aimed at the wrong thing.", flush=True)

    # ---- the sweep ----
    per_tau = {t: score(recs, t, trav, non_trav, weights, args) for t in taus}

    # are the crash steps the low-support steps?
    u0 = per_tau[0.0]
    ca = np.array([r["mean_alpha"] for r in u0 if r["fp_coll"] >= args.crash_frac])
    na = np.array([r["mean_alpha"] for r in u0 if r["fp_coll"] < args.crash_frac])
    if ca.size and na.size:
        print(f"\n  mean alpha on CRASH steps    {ca.mean():.3f}  (n={ca.size})")
        print(f"  mean alpha on non-crash steps {na.mean():.3f}  (n={na.size})",
              flush=True)

    s = args.reward_scale
    print(f"\n===== GATE THRESHOLD SWEEP ({n} steps, identical renders) =====")
    print("   tau   sup   GEOvoid  SEMvoid  fp_void  fp_coll  CRASH   sem       "
          "void      coll      total")
    for t in taus:
        A = {k: np.array([r[k] for r in per_tau[t]], dtype=float)
             for k in per_tau[t][0] if k != "dom"}
        crash = int((A["fp_coll"] >= args.crash_frac).sum())
        mark = "  <- ungated" if t == 0.0 else (
            "  <- current gate" if t == args.gate_tau else "")
        print(f"  {t:4.2f}  {A['sup'].mean():.3f}   {A['geo_void'].mean():.3f}    "
              f"{A['sem_void'].mean():.3f}    {A['fp_void'].mean():.3f}    "
              f"{A['fp_coll'].mean():.3f}   {crash:2d}/{n}  "
              f"{A['sem'].mean()*s:+.4f}  {A['void_t'].mean()*s:+.4f}  "
              f"{A['coll_t'].mean()*s:+.4f}  {A['total'].mean()*s:+.4f}{mark}")

    # ---- detail blocks for ungated and the chosen gate ----
    for t in [t for t in (0.0, args.gate_tau) if t in per_tau]:
        rows = per_tau[t]
        A = {k: np.array([r[k] for r in rows], dtype=float)
             for k in rows[0] if k != "dom"}
        tag = "UNGATED" if t == 0.0 else f"GATED tau={t}"
        print(f"\n===== {tag} ({n} rendered steps) =====")
        print(f"  SUPPORT      frac(alpha>{t}) {A['sup'].mean():.3f}")
        print(f"  GEOVOID      alpha<={t} share  {A['geo_void'].mean():.3f}   "
              f"(max over steps {A['geo_void'].max():.3f})   <- IMGVOIDTERM: "
              f"pure rasterizer support, no diffusion")
        print(f"  SEMVOID      diffused class-0 {A['sem_void'].mean():.3f}   "
              f"(max {A['sem_void'].max():.3f})   <- what the world model "
              f"itself calls unknown")
        print(f"  VOID (fp)    footprint void  {A['fp_void'].mean():.3f}   "
              f"(max {A['fp_void'].max():.3f})")
        print(f"  COLLISION    footprint coll  {A['fp_coll'].mean():.3f}   "
              f"steps >={args.crash_frac} (would CRASH): "
              f"{int((A['fp_coll'] >= args.crash_frac).sum())}/{n}")
        print(f"  REWARD x{s}  semantic {A['sem'].mean()*s:+.4f}   "
              f"goal {A['goal_t'].mean()*s:+.4f}   "
              f"collision {A['coll_t'].mean()*s:+.4f}   "
              f"void {A['void_t'].mean()*s:+.4f}   "
              f"total {A['total'].mean()*s:+.4f}")
        doms = {}
        for r in rows:
            doms[r["dom"]] = doms.get(r["dom"], 0) + 1
        print("  DOMINANT     (share of STEPS whose top class is) " + ", ".join(
            f"{k} {100.0 * v / n:.0f}%" for k, v in
            sorted(doms.items(), key=lambda kv: -kv[1])[:5]))

    # ---- every number, auditable ----
    import csv as _csv
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    stem = f"CHECK_{args.scene}{args.tag}"
    csv_p = od / f"{stem}.csv"
    with open(csv_p, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(list(per_tau[taus[0]][0].keys()))
        for t in taus:
            for r in per_tau[t]:
                w.writerow(list(r.values()))
    print(f"\n==> every per-step number: {csv_p}", flush=True)

    # ---- the visual: RGB | raster | labels ungated | labels gated | ... ----
    _survey = {"w": None, "n": 0, "lo_tau": 0, "lo_hard": 0,
               "cov": []}
    try:
        import cv2
        from diffsynth.utils.class_taxonomy import v14_palette
        pal = (v14_palette(args.sem_palette).numpy() * 255).astype(np.uint8)
        build_ladder(recs, od, stem, args, cv2)
        u_rows = {(r["ep"], r["step"]): r for r in per_tau[0.0]}
        g_rows = {(r["ep"], r["step"]): r for r in per_tau[args.gate_tau]}
        crash_u = {k for k, v in u_rows.items() if v["fp_coll"] >= args.crash_frac}
        crash_g = {k for k, v in g_rows.items() if v["fp_coll"] >= args.crash_frac}

        def verdict(key):
            if key in crash_u and key in crash_g:
                return "CRASH-BOTH"
            if key in crash_u:
                return "RESCUED"        # gate turned a crash into survivable void
            if key in crash_g:
                return "NEWCRASH"       # gate CREATED a crash — must be zero
            if key in jump_keys:
                return "JUMP"           # biggest frame-to-frame incoherence
            return "ok"

        # every episode-0 step (the walk-through), plus EVERY crash-level step
        # and the worst coherence jumps wherever they happened — those are the
        # frames the whole argument rests on
        sel = [r for r in recs if r["ep"] == 0
               or verdict((r["ep"], r["step"])) != "ok"]
        for r in sel:
            key = (r["ep"], r["step"])
            vd_tag = verdict(key)
            rgb, lu, au = r["rgb"], r["lab"], r["alpha"]
            lg = lu if au is None else np.where(au > args.gate_tau, lu, 0)
            pu, pg = pal[np.clip(lu, 0, 13)], pal[np.clip(lg, 0, 13)]
            ras = np.zeros_like(rgb) if r["ras"] is None else r["ras"]
            psr = (np.zeros_like(rgb) if r["sras"] is None
                   else pal[np.clip(r["sras"], 0, 13).astype(int)])
            if au is not None:
                amax = float(np.percentile(au, 99)) or 1.0
                heat = cv2.applyColorMap(
                    (np.clip(au / max(amax, 1e-6), 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            else:
                heat = np.zeros_like(rgb)

            # ---- the verdict panel: WHY this footprint scored the way it did.
            # Every pixel the reward reads, colour-coded by the two questions
            # that decide it — is the class walkable, and is there geometry
            # under it?  This is what "phantom obstacle" has to look like.
            m = footprint_mask(r, args.look_ahead)
            vd = (rgb * 0.30).astype(np.uint8)
            if m is not None and m.any():
                nt = m & non_trav[np.clip(lu, 0, len(non_trav) - 1)] & (lu != 0)
                unsup = m & (au <= args.gate_tau) if au is not None else np.zeros_like(m)
                vd[m & ~nt & ~unsup] = (0, 190, 0)        # walkable + observed
                vd[m & nt & ~unsup] = (220, 0, 0)         # obstacle + OBSERVED
                vd[m & nt & unsup] = (255, 140, 0)        # obstacle, NO support
                vd[m & ~nt & unsup] = (0, 90, 255)        # walkable, NO support

            panel = np.hstack([rgb, ras, pu, pg, psr, heat, vd])[:, :, ::-1]
            panel = np.ascontiguousarray(panel)

            uv = footprint_uv(r, args.look_ahead)
            if uv is not None:
                poly = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
                for k in range(7):
                    cv2.polylines(panel, [poly + np.array([[k * args.width, 0]])],
                                  True, (0, 255, 255), 1, cv2.LINE_AA)
            # The proposed SPLIT: yellow = graded semantic score (far, warning),
            # magenta = the lethal collision test (near, at the body). Seeing
            # both on the same frame is the only way to judge whether the near
            # box is inside the image and over the terrain it claims to judge.
            uvc = (footprint_uv(r, args.collision_look_ahead)
                   if args.collision_look_ahead > 0 else None)
            if uvc is not None:
                polyc = np.round(uvc).astype(np.int32).reshape(-1, 1, 2)
                for k in range(7):
                    cv2.polylines(panel, [polyc + np.array([[k * args.width, 0]])],
                                  True, (255, 0, 255), 1, cv2.LINE_AA)

            ru, rg = u_rows[key], g_rows[key]
            hud = (f"ep{r['ep']} step{r['step']}  [{vd_tag}]  UNGATED coll "
                   f"{ru['fp_coll']:.2f} void {ru['fp_void']:.2f} | "
                   f"tau={args.gate_tau} coll {rg['fp_coll']:.2f} void "
                   f"{rg['fp_void']:.2f} | imgvoid {rg['img_void']:.2f} "
                   f"alpha_mean {rg['mean_alpha']:.2f}")
            cv2.putText(panel, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "footprint: green=walkable+seen  RED=obstacle+SEEN"
                               "  ORANGE=obstacle+unseen(phantom)  blue=walkable"
                               "+unseen", (6 * args.width + 8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                        cv2.LINE_AA)
            for k, name in enumerate(["RGB diffused", "RGB raster (splats)",
                                      "SEM diffused ungated",
                                      f"SEM diffused gated tau={args.gate_tau}",
                                      "SEM raster (splat labels)",
                                      "SUPPORT (alpha)",
                                      "FOOTPRINT VERDICT"]):
                cv2.putText(panel, name, (k * args.width + 8,
                                          panel.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.imwrite(str(od / f"{stem}_ep{r['ep']}_s{r['step']:02d}_"
                               f"{vd_tag}.png"), panel)

            # SURVEY VIDEO (2026-09-02, her ask): a per-scene mp4 of the three
            # layers that actually matter when choosing scenes --
            #   RGB diffused  = what the POLICY sees
            #   SEM diffused  = what the REWARD reads (v26 e10, palette 4)
            #   SEM raster    = the SAM3 labels carried by the GAUSSIAN SPLATS,
            #                   which is a DIFFERENT label source and the one
            #                   the spawn filter (--spawn_classes) and goal
            #                   support actually consult.
            # Those last two disagreeing is not a bug, it is the thing to look
            # at: the reward grades the diffusion while spawns and goals are
            # placed from the cloud.
            if args.survey_video:
                # Fourth column: SUPPORT. Without it you cannot tell whether a
                # region of the diffused semantics is grounded or invented --
                # and on 2026-09-02 the gate showed MOST of the collision
                # signal sits in unobserved regions, so "are the hallucinated
                # regions good" is the question the survey exists to answer.
                # Bright = geometry backs this pixel, dark = the model made it
                # up.
                cols = [np.ascontiguousarray(rgb).copy(), np.ascontiguousarray(pu).copy(), psr, heat]
                names = ["RGB diffused (policy sees)",
                         "SEM diffused (reward reads)",
                         "SEM raster / SAM3 splats (spawns + goals)",
                         "SUPPORT alpha (dark = invented)"]
                # MEMORY column (2026-09-16, Joana): with --collision_box_memory
                # the near box is read from an EARLIER frame when the camera
                # cannot see it now. Show that frame's labels with the near box
                # drawn where the reward read it, and its age; if no stored
                # frame contained the box, show the newest stored frame and say so.
                _nmem = int(getattr(args, "collision_box_memory", 0) or 0)
                if _nmem > 0:
                    # STRIP, drawn by the SAME code as the training panel
                    # (src/eval/panels.py): every stored frame with the near box,
                    # its age, whether it contains the box, the non-walkable share
                    # and the distance from the robot to that box.
                    from src.eval.panels import memory_strip
                    _stored = [q for q in recs if q["ep"] == r["ep"] and 0 < r["step"] - q["step"] <= _nmem]
                    _cw = _footprint_corners_world(r["pos"], r["head"],
                                                   look_ahead_dist=(args.collision_look_ahead if args.collision_look_ahead > 0 else args.look_ahead),
                                                   length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
                    _fm = [(q["lab"], q["K"], q["w2c"]) for q in _stored]     # oldest first, as the env stores it
                    # 2026-11-05 (Fig. 5c): dump the strip's raw inputs so the paper panel can be
                    # drawn at full size offline: stored labels + cameras, the box corners, the
                    # robot position, and the current frame.
                    if _fm:
                        np.savez_compressed(str(od / f"{stem}_ep{r['ep']}_s{r['step']:02d}_memory.npz"),
                                            labels=np.stack([np.asarray(q["lab"]).astype(np.int8) for q in _stored]),
                                            K=np.stack([np.asarray(q["K"], np.float32) for q in _stored]),
                                            w2c=np.stack([np.asarray(q["w2c"], np.float32) for q in _stored]),
                                            steps=np.array([q["step"] for q in _stored]),
                                            corners_world=np.asarray(_cw, np.float32), robot_pos=np.asarray(r["pos"], np.float32),
                                            cur_rgb=np.asarray(rgb).astype(np.uint8), cur_labels=np.asarray(r["lab"]).astype(np.int8),
                                            cur_K=np.asarray(r["K"], np.float32), cur_w2c=np.asarray(r["w2c"], np.float32),
                                            cur_alpha=(np.asarray(au, np.float32) if au is not None else np.zeros(np.asarray(rgb).shape[:2], np.float32)))
                    mem = memory_strip(_fm, _cw, pal, non_trav, r["pos"],
                                       out_h=rgb.shape[0], out_w=rgb.shape[1])
                    if mem is None:
                        mem = np.zeros_like(rgb); _tag = "MEMORY: none yet (episode start)"
                    else:
                        _nh = int(r.get("memory_hits", 0)); _ag = str(getattr(args, "collision_box_memory_agg", "newest"))
                        _tag = (f"MEMORY {_ag}: {_nh} frame(s) contain the box" if _nh > 0
                                else ("MEMORY: box NOT in any stored frame" if r.get("box_memory_age", 0.0) < 0
                                      else "MEMORY: box seen in the current frame"))
                    cols.append(mem); names.append(_tag)
                # ANCHORED BOXES (Joana): the near box fixed in the world every N steps,
                # drawn in every later frame of the same episode while it is still in view
                _anc = int(getattr(args, "anchor_box_every", 0) or 0)
                if _anc > 0:
                    _la = (float(args.anchor_dist) if float(getattr(args, "anchor_dist", 0.0)) > 0
                           else (args.collision_look_ahead if args.collision_look_ahead > 0 else args.look_ahead))
                    for _q in recs:
                        if _q["ep"] != r["ep"] or _q["step"] > r["step"] or _q["step"] % _anc:
                            continue
                        _cwq = _footprint_corners_world(_q["pos"], _q["head"], look_ahead_dist=_la,
                                                        length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
                        _uvq, _frq = _project_points(_cwq, r["K"], r["w2c"])
                        if not _frq.all():
                            continue
                        _pq = np.round(_uvq).astype(np.int32).reshape(-1, 1, 2)
                        _ageq = r["step"] - _q["step"]
                        _colq = (255, 0, 255) if _ageq > 0 else (255, 255, 0)
                        # distance from the robot NOW to the anchored patch: it must
                        # shrink by one step each frame if the projection is right
                        _dq = float(np.linalg.norm(np.asarray(_cwq).mean(0)[:2] - np.asarray(r["pos"])[:2]))
                        for _c in (0, 1):
                            cv2.polylines(cols[_c], [_pq], True, _colq, 2, cv2.LINE_AA)
                            cv2.putText(cols[_c], f"box@{_q['step']} age {_ageq}  {_dq:.2f}m",
                                        tuple(int(v) for v in _uvq[0]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _colq, 1, cv2.LINE_AA)
                # PATH OVERLAY check: green = the next 8 recorded poses of this walk projected
                # into this frame (what the rollout video draws as the executed path), orange =
                # a fake 10-step straight plan from this pose (the planned-chunk drawing)
                if bool(getattr(args, "path_overlay", False)):
                    _nxt = [q for q in recs if q["ep"] == r["ep"] and 0 < q["step"] - r["step"] <= 8]
                    if _nxt:
                        _uvn, _frn = _project_points(np.array([q["pos"] for q in _nxt]), r["K"], r["w2c"])
                        _ptsn = [tuple(int(v) for v in np.round(_uvn[i])) for i in range(len(_nxt)) if _frn[i]]
                        for _a, _b in zip(_ptsn[:-1], _ptsn[1:]):
                            cv2.line(cols[0], _a, _b, (0, 230, 0), 2, cv2.LINE_AA)
                        for _pt in _ptsn:
                            cv2.circle(cols[0], _pt, 3, (0, 230, 0), -1, cv2.LINE_AA)
                    # the fake 10-step plan, drawn by the training panel's own code
                    from src.eval.panels import integrate_plan, draw_plan
                    _pose4 = np.eye(4); _pose4[:3, 3] = r["pos"]
                    _hd = np.asarray(r["head"], float); _hd = _hd / (np.linalg.norm(_hd) + 1e-9)
                    _pose4[:3, 0] = _hd
                    _fake = np.tile(np.array([1.0, 0.0]), 10)      # straight ahead, full speed
                    _plan = integrate_plan(_pose4, _fake, 0.25, 0.0)
                    draw_plan(cols[0], _pose4, _plan, r["K"], r["w2c"],
                              look_ahead_dist=float(args.look_ahead), label=False)
                    cv2.putText(cols[0], "green = next 8 recorded poses   orange = fake 10-step plan (same code as training)",
                                (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                trio = np.hstack(cols)[:, :, ::-1]
                trio = np.ascontiguousarray(trio)
                for k, name in enumerate(names):
                    cv2.putText(trio, name, (k * args.width + 8,
                                             trio.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    cv2.putText(trio, name, (k * args.width + 8,
                                             trio.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                # Draw the reward footprint on every column. If it is absent
                # for a frame, the reward saw NOTHING that step -- which is
                # how three of five training scenes turned out to be blind
                # (2026-09-02). A survey without it can look perfectly healthy
                # while the reward is reading nothing at all.
                if uv is not None:
                    poly = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
                    for k in range(min(4, len(cols))):
                        cv2.polylines(trio,
                                      [poly + np.array([[k * args.width, 0]])],
                                      True, (0, 255, 255), 2, cv2.LINE_AA)
                    fp_txt = f"footprint {args.look_ahead}m OK"
                    fp_col = (0, 255, 255)
                else:
                    fp_txt = f"FOOTPRINT MISSING -- reward is BLIND this frame"
                    fp_col = (0, 0, 255)
                # coverage (mean alpha) every frame, flagged against tau.
                # The coherence cost is -w*max(0, tau-coverage) and the
                # terminator fires below 0.1, so "is it always above 0.4" is
                # answerable only if the number is on screen (her ask).
                cov = (float(r["alpha"].mean())
                       if r.get("alpha") is not None else float("nan"))
                cov_ok = not (cov == cov) or cov >= args.gate_tau_cov
                if cov == cov:
                    _survey["n"] += 1
                    _survey["cov"].append(cov)
                    if cov < args.gate_tau_cov:
                        _survey["lo_tau"] += 1
                    if cov < args.coh_hard_tau:
                        _survey["lo_hard"] += 1
                cv2.putText(trio, f"{args.scene}  ep{r['ep']} step{r['step']}"
                                  f"   {fp_txt}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            fp_col, 2, cv2.LINE_AA)
                cv2.putText(trio, f"coverage {cov:.3f}"
                                  + ("" if cov_ok else
                                     f"   BELOW tau {args.gate_tau_cov}"),
                            (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if cov_ok else (0, 80, 255), 2,
                            cv2.LINE_AA)
                if _survey["w"] is None:
                    # mp4v renders as a GREEN SCREEN in QuickTime; try H.264
                    # first so the file opens on a Mac without VLC.
                    size = (trio.shape[1], trio.shape[0])
                    for tag in ("avc1", "H264", "mp4v"):
                        w = cv2.VideoWriter(
                            str(od / f"SURVEY_{args.scene}_{args.walk}.mp4"),
                            cv2.VideoWriter_fourcc(*tag), 4.0, size)
                        if w.isOpened():
                            print(f"    survey codec: {tag}", flush=True)
                            _survey["w"] = w
                            break
                        w.release()
                if _survey["w"] is not None:
                    _survey["w"].write(trio)
        if args.survey_video and _survey["w"] is not None:
            _survey["w"].release()
            quicktime_safe(od / f"SURVEY_{args.scene}_{args.walk}.mp4")
            n = _survey["n"]
            if n:
                cv_arr = np.array(_survey["cov"])
                print(f"\n===== COVERAGE OVER THE {args.walk.upper()} WALK "
                      f"({n} frames) =====")
                print(f"  mean {cv_arr.mean():.3f}  min {cv_arr.min():.3f}  "
                      f"p10 {np.percentile(cv_arr, 10):.3f}  "
                      f"max {cv_arr.max():.3f}")
                print(f"  frames below tau {args.gate_tau_cov} (graded cost "
                      f"fires): {_survey['lo_tau']}/{n} = "
                      f"{100.0 * _survey['lo_tau'] / n:.1f}%")
                print(f"  frames below {args.coh_hard_tau} (episode would "
                      f"TERMINATE): {_survey['lo_hard']}/{n} = "
                      f"{100.0 * _survey['lo_hard'] / n:.1f}%")
                if args.walk == "straight":
                    print("  A straight run at a randomly sampled goal LEAVES "
                          "the reconstructed\n  corridor on purpose -- this is "
                          "the rate the coherence terminator would\n  fire at "
                          "on exactly the behaviour the goal sampler asks for.")
            print(f"==> survey video: "
                  f"{od}/SURVEY_{args.scene}_{args.walk}.mp4", flush=True)
        print(f"==> visual frames ({len(sel)}): {od}/{stem}_ep*_s*.png", flush=True)
        print(f"    crash-level steps ungated {sorted(crash_u)}", flush=True)
        print(f"    crash-level steps gated   {sorted(crash_g)}", flush=True)
    except Exception as e:
        print(f"[viz] skipped: {e}", flush=True)

    uc = np.array([r["fp_coll"] for r in per_tau[0.0]])
    gc = np.array([r["fp_coll"] for r in per_tau[args.gate_tau]])
    print(f"\n===== GATE EFFECT at tau={args.gate_tau} (identical renders) =====")
    print(f"  crash-level steps  ungated "
          f"{int((uc >= args.crash_frac).sum())}/{n}   ->  gated "
          f"{int((gc >= args.crash_frac).sum())}/{n}")
    print(f"  mean collision     ungated {uc.mean():.3f}  ->  gated {gc.mean():.3f}")
    print(f"  mean geo  void     ungated "
          f"{np.mean([r['geo_void'] for r in per_tau[0.0]]):.3f}  ->  gated "
          f"{np.mean([r['geo_void'] for r in per_tau[args.gate_tau]]):.3f}")
    print("\n  Crash-level steps collapsing to 0 under the gate means those "
          "crashes were\n  phantom terrain in unobserved regions. It does NOT "
          "prove the gate keeps REAL\n  obstacles — for that, run the obstacle "
          "probe (--goal_xy at a known obstacle)\n  and check that crash steps "
          "SURVIVE the gate.", flush=True)


if __name__ == "__main__":
    main()
