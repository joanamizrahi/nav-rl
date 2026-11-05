"""Moving-through-the-world diffusion preview at any resolution.

Walks the recorded trajectory pose-by-pose through BatchedLiveDiffusedBackend
(the exact machinery batched live training uses) and writes a side-by-side
rgb | semantic mp4 — the quality gate Joana judges before committing a
training run to a lower resolution / fewer sampler steps.

Usage (GPU node):
    python scripts/drive_preview.py \
        --scene rugd_trail_00 --clips_dir ... --poses_dir ... --labels_dir ... \
        --height 224 --width 336 --num_steps 2 --frames 40 \
        --out /scratch/.../outputs/drive_preview
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
from src.env.vec_live_env import BatchedLiveDiffusedBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm07b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors")
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--num_steps", type=int, default=4)
    ap.add_argument("--frames", type=int, default=40,
                    help="drive length in poses along the recorded walk")
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--heading", default="tangent",
                    choices=["tangent", "recorded"],
                    help="tangent: face the direction of motion (what a "
                         "driving robot does); recorded: the walker's camera "
                         "heading (pans and drifts off-path)")
    ap.add_argument("--target_xy", default=None,
                    help="'x,y': leave the recorded path at --start and walk "
                         "STRAIGHT AT this nav-frame point in 0.25 m policy "
                         "steps, camera facing it (obstacle visibility quest)")
    ap.add_argument("--goal_frame", type=int, default=None,
                    help="mark the pose at this frame index as the GOAL: "
                         "projected dot in both panels + topdown inset + "
                         "distance in the HUD (goal-placement design tool)")
    ap.add_argument("--goal_xy", default=None,
                    help="'x,y': mark this nav-frame point as the GOAL "
                         "(ignored if --goal_frame is set)")
    ap.add_argument("--spin", action="store_true",
                    help="J-spec spawn certification: hold position at --start "
                         "and rotate a full 360 over --frames steps; HUD shows "
                         "heading + reconstruction coverage per view")
    ap.add_argument("--spin_deg", type=float, default=360.0,
                    help="with --spin: sweep only +-this/2 around the path "
                         "tangent (visualize the goal cone's view range)")
    ap.add_argument("--raster", action="store_true",
                    help="append a third panel: the RAW Gaussian-splat raster "
                         "the diffusion is conditioned on — shows where the "
                         "splats end and hallucination begins")
    ap.add_argument("--strafe", action="store_true",
                    help="lateral-jitter certification: hold --start pose and "
                         "heading, slide sideways 0 -> -strafe_m -> +strafe_m "
                         "(monotonic, video-coherent); cov per offset says how "
                         "far off-path spawns can wander before rendering lies")
    ap.add_argument("--strafe_m", type=float, default=0.5,
                    help="with --strafe: max lateral offset in meters each side")
    ap.add_argument("--goal_sample", default=None,
                    help="'cone_deg,dmin,dmax,seed': sample a goal EXACTLY "
                         "like J-training does (tangent-centered cone at "
                         "--start, uniform dist) then walk straight at it, "
                         "camera on it — a rendered J-episode. Spawn is the "
                         "recorded path pose at --start.")
    ap.add_argument("--aerial", default=None,
                    help="'height_m,pitch_deg,samples': ONE bird's-eye frame "
                         "of the scene — raster RGB | semantic gaussians from "
                         "above — overlaid with the recorded path, `samples` "
                         "REAL jittered spawn draws (orange, with heading "
                         "ticks) and their sampled goals (green) using the "
                         "J-50 training distribution. Saves OVERVIEW_*.png.")
    ap.add_argument("--sem_palette", type=int, default=1,
                    help="v14 palette version for semantic decode — must "
                         "match the live_ckpt's training palette (v21=1, "
                         "v24/v25 line=4)")
    ap.add_argument("--pano_views", action="store_true",
                    help="OPT IN to fusing pano side views into the "
                         "reconstruction (off by default so the dense-data "
                         "and pano experiments stay separate)")
    ap.add_argument("--goal_at", default=None,
                    help="'offset_deg,dist_m': place the goal at EXACTLY this "
                         "angle off the path tangent at --start (goal-offset "
                         "ladder; her ask 2026-08-31: +-10/15/20/25 to find "
                         "the true cone edge). Same episode shape as "
                         "goal_sample: spawn on recorded pose, turn, walk.")
    ap.add_argument("--spawn_yaw_deg", type=float, default=0.0,
                    help="with --goal_sample: rotate the SPAWN heading by "
                         "this many degrees — cold-start heading-jitter test "
                         "(the video history resets to 5x this jittered view, "
                         "exactly like a jittered training spawn would)")
    ap.add_argument("--spawn_lat_m", type=float, default=0.0,
                    help="with --goal_sample: offset the SPAWN laterally by "
                         "this many meters — cold-start lateral-jitter test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.height % 112 or args.width % 112:
        raise SystemExit("height/width must be multiples of 112 "
                         "(reconstructor patch 14 x VAE/DiT 16)")

    import cv2
    import torch
    from diffsynth.utils.class_taxonomy import v14_palette

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        use_pano_views=args.pano_views,
        sem_palette_version=args.sem_palette,
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.height,
        W=args.width,
    )
    world = BatchedLiveDiffusedBackend(cfg, checkpoint=args.live_ckpt)
    world.num_inference_steps = args.num_steps
    world.load_scene(args.scene)
    cal = world._calib[args.scene]

    pal = (v14_palette(args.sem_palette).numpy() * 255).astype(np.uint8)  # [14,3] RGB

    def tangent_pose(i: int) -> np.ndarray:
        pos = np.asarray(cal.positions)
        fwd = pos[min(i + 1, len(pos) - 1)] - pos[max(i - 1, 0)]
        fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        pose = np.eye(4)
        pose[:3, 0] = fwd
        pose[:3, 1] = np.cross(up, fwd)
        pose[:3, 2] = up
        pose[:3, 3] = pos[i] * np.array([1.0, 1.0, 0.0])
        return pose.astype(np.float32)

    def pose_at(position_xy: np.ndarray, fwd_xy: np.ndarray) -> np.ndarray:
        fwd = np.array([fwd_xy[0], fwd_xy[1], 0.0])
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        pose = np.eye(4)
        pose[:3, 0] = fwd
        pose[:3, 1] = np.cross(up, fwd)
        pose[:3, 2] = up
        pose[:3, 3] = np.array([position_xy[0], position_xy[1], 0.0])
        return pose.astype(np.float32)

    target = (np.array([float(v) for v in args.target_xy.split(",")])
              if args.target_xy else None)

    goal = None
    gs_seed = None
    ga_tag = None
    if args.goal_at:
        off_deg, d = (float(v) for v in args.goal_at.split(","))
        pos = np.asarray(cal.positions, dtype=float)[:, :2]
        i0 = args.start
        fwd = pos[min(i0 + 1, len(pos) - 1)] - pos[max(i0 - 1, 0)]
        base = float(np.arctan2(fwd[1], fwd[0]))
        th = base + float(np.deg2rad(off_deg))
        goal = np.array([pos[i0, 0] + d * np.cos(th),
                         pos[i0, 1] + d * np.sin(th), 0.0])
        target = goal[:2].copy()
        ga_tag = f"_ga{off_deg:g}_{d:g}"
        print(f"[goal_at] offset {off_deg:+.1f}deg dist {d:.2f}m "
              f"-> goal ({goal[0]:.2f},{goal[1]:.2f})", flush=True)
    elif args.goal_sample:
        cone_deg, dmin, dmax, s = (float(v) for v in args.goal_sample.split(","))
        gs_seed = int(s)
        rng = np.random.default_rng(gs_seed)
        pos = np.asarray(cal.positions, dtype=float)[:, :2]
        i0 = args.start
        fwd = pos[min(i0 + 1, len(pos) - 1)] - pos[max(i0 - 1, 0)]
        base = float(np.arctan2(fwd[1], fwd[0]))
        th = base + float(rng.uniform(-np.deg2rad(cone_deg) / 2.0,
                                      np.deg2rad(cone_deg) / 2.0))
        d = float(rng.uniform(dmin, dmax))
        goal = np.array([pos[i0, 0] + d * np.cos(th),
                         pos[i0, 1] + d * np.sin(th), 0.0])
        target = goal[:2].copy()
        print(f"[goal_sample] cone {cone_deg:.0f}deg dist {dmin:.0f}-{dmax:.0f}m "
              f"seed {gs_seed} -> goal ({goal[0]:.2f},{goal[1]:.2f})  "
              f"offset {np.degrees(th - base):+.1f}deg  dist {d:.2f}m",
              flush=True)
    elif args.goal_frame is not None:
        goal = np.asarray(cal.positions[args.goal_frame], dtype=float).copy()
        goal[2] = 0.0
    elif args.goal_xy:
        gx, gy = (float(v) for v in args.goal_xy.split(","))
        goal = np.array([gx, gy, 0.0])

    path_xy = np.asarray(cal.positions, dtype=float)[:, :2]

    def topdown_inset(cur_xy, side, cur_fwd=None, rec_fwd=None):
        """Mini-map: recorded path, spawn end, current pose, goal. BGR.
        cur_fwd/rec_fwd: world-frame XY unit vectors — orange arrow = where
        the render is facing, white arrow = the RECORDED camera heading at
        this pose (should lie along the path; if not, the recorded
        orientation itself is wrong and spins center on garbage)."""
        allp = (np.vstack([path_xy, goal[None, :2]])
                if goal is not None else path_xy)
        lo = allp.min(0) - 1.0
        span = max(*(allp.max(0) + 1.0 - lo))
        def px(p):
            x = int((p[0] - lo[0]) / span * (side - 12)) + 6
            y = side - 1 - (int((p[1] - lo[1]) / span * (side - 12)) + 6)
            return x, y
        img = np.full((side, side, 3), 25, np.uint8)
        for a, b in zip(path_xy[:-1], path_xy[1:]):
            cv2.line(img, px(a), px(b), (190, 190, 190), 1, cv2.LINE_AA)
        cv2.circle(img, px(path_xy[0]), 3, (255, 255, 255), -1)   # pose 0
        if goal is not None:
            cv2.drawMarker(img, px(goal[:2]), (0, 255, 0),
                           cv2.MARKER_CROSS, 9, 2)
        cx, cy = px(cur_xy)
        def arrow(fwd_xy, length, color, thick):
            n = float(np.linalg.norm(fwd_xy))
            if n < 1e-6:
                return
            fx, fy = fwd_xy[0] / n, fwd_xy[1] / n
            tip = (int(cx + fx * length), int(cy - fy * length))  # y up->down
            cv2.arrowedLine(img, (cx, cy), tip, color, thick,
                            cv2.LINE_AA, tipLength=0.35)
        if rec_fwd is not None:
            arrow(rec_fwd, side * 0.30, (255, 255, 255), 1)
        if cur_fwd is not None:
            arrow(cur_fwd, side * 0.20, (0, 140, 255), 2)
        cv2.circle(img, (cx, cy), 4, (0, 140, 255), -1)           # current
        return img

    def project_goal(K, w2c):
        """Goal -> pixel via the render's own K/w2c (OpenCV convention)."""
        K = np.asarray(K, dtype=float).reshape(3, 3)
        M = np.asarray(w2c, dtype=float)
        if M.shape == (3, 4):
            M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])
        p = M @ np.array([goal[0], goal[1], goal[2], 1.0])
        if p[2] <= 0.1:
            return None
        u = K[0, 0] * p[0] / p[2] + K[0, 2]
        v = K[1, 1] * p[1] / p[2] + K[1, 2]
        if not (0 <= u < args.width and 0 <= v < args.height):
            return None
        return int(u), int(v)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.scene}_{args.width}x{args.height}_s{args.num_steps}"
           + ("" if args.heading == "tangent" else "_rec")
           + ("" if args.start == 5 else f"_st{args.start}")
           + ("" if target is None else "_totgt")
           + (("" if not args.spin else
               ("_spin" if args.spin_deg >= 360.0 else f"_spin{args.spin_deg:.0f}")))
           + ("" if args.goal_frame is None else f"_gf{args.goal_frame}")
           + ("" if (args.goal_frame is not None or not args.goal_xy) else
              "_g" + args.goal_xy.replace(",", "_"))
           + ("" if gs_seed is None else f"_gs{gs_seed}")
           + (ga_tag or "")
           + (f"_sy{args.spawn_yaw_deg:g}" if args.spawn_yaw_deg else "")
           + (f"_sl{args.spawn_lat_m:g}" if args.spawn_lat_m else "")
           + (f"_strafe{args.strafe_m:g}" if args.strafe else "")
           + (f"_pal{args.sem_palette}" if args.sem_palette != 1 else "")
           + ("_ras" if args.raster else ""))
    if args.spin:
        # Recorded-heading sanity: angle between the recorded camera forward
        # and the path tangent at the spin pose. Large values mean the spin
        # center itself is off-path and every offset inherits that error.
        rp = cal.robot_pose_nav(args.start)[:2, 0]
        tp = tangent_pose(args.start)[:2, 0]
        dang = np.degrees(np.arctan2(rp[0] * tp[1] - rp[1] * tp[0],
                                     float(np.dot(rp, tp))))
        print(f"[spin] recorded heading vs path tangent at pose "
              f"{args.start}: {dang:+.1f}deg", flush=True)
    vw = None
    for step in range(args.frames):
        i = min(args.start + step, 80)
        if args.spin:
            i = args.start
            if args.spin_deg < 360.0:
                # MONOTONIC pan (her fix 2026-08-31 #2): the generator is a
                # VIDEO model conditioned on the previous poses — the earlier
                # diverging order 0,+5,-5,+10,... fed it a jittering camera
                # and poisoned the conditioning. Sweep smoothly instead:
                # 0 -> -S/2 -> back through 0 -> +S/2, opening on the recorded
                # pose (her spec) so the history warms up on the best view.
                # Use FRAMES = 3*m+1 (e.g. 49 @ SPINDEG=160 = 5deg/step) for
                # equal angular speed on both legs.
                n = max(args.frames - 1, 3)
                m = max(n // 3, 1)
                half = np.deg2rad(args.spin_deg) / 2.0
                if step <= m:
                    ang = -half * step / m
                else:
                    ang = -half + 2.0 * half * (step - m) / (n - m)
            else:
                ang = 2.0 * np.pi * step / max(args.frames, 1)
            # Center on the RECORDED camera heading — maximal quality by
            # construction — not the path tangent.
            base = cal.robot_pose_nav(i)
            fwd = base[:2, 0]
            c, s = np.cos(ang), np.sin(ang)
            fwd_rot = np.array([c * fwd[0] - s * fwd[1],
                                s * fwd[0] + c * fwd[1]])
            pose = pose_at(base[:2, 3], fwd_rot)
            spin_off_deg = float(np.degrees(ang))
        elif args.aerial:
            a_h, a_pitch, _ = (float(v) for v in args.aerial.split(","))
            i = args.start
            base = tangent_pose(i)
            fwd = base[:3, 0].copy()
            up = np.array([0.0, 0.0, 1.0])
            thp = np.deg2rad(max(a_pitch, 15.0))
            f2 = np.cos(thp) * fwd - np.sin(thp) * up
            u2 = np.cos(thp) * up + np.sin(thp) * fwd
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 0] = f2
            pose[:3, 2] = u2
            pose[:3, 1] = np.cross(u2, f2)
            pmid = path_xy.mean(0)
            back = a_h / np.tan(thp)
            pose[:3, 3] = np.array([pmid[0] - fwd[0] * back,
                                    pmid[1] - fwd[1] * back, a_h],
                                   dtype=np.float32)
        elif args.strafe:
            i = args.start
            n = max(args.frames - 1, 3)
            m = max(n // 3, 1)
            if step <= m:
                off = -args.strafe_m * step / m
            else:
                off = -args.strafe_m + 2.0 * args.strafe_m * (step - m) / (n - m)
            base = tangent_pose(i)
            fwd = base[:2, 0]
            lat = np.array([-fwd[1], fwd[0]])
            pose = pose_at(base[:2, 3] + off * lat, fwd)
            strafe_off_m = float(off)
        elif target is not None:
            origin = np.asarray(cal.positions[args.start])[:2]
            d = target - origin
            dist = np.linalg.norm(d)
            u = d / max(dist, 1e-6)
            if gs_seed is not None or ga_tag is not None:
                # Spawn realism (her spec 2026-08-31): open EXACTLY like a
                # J-training episode — training spawns are robot_pose_nav
                # (recorded pose, full gaussian support in view, video
                # history warms up on it) and the policy must TURN toward
                # the goal itself. Mirror that: face the tangent, rotate
                # <=10deg/frame toward the goal, then walk.
                # spawn_yaw_deg / spawn_lat_m: COLD-START jitter tests —
                # frame 0 (and thus the whole reset history) is the jittered
                # pose, the honest simulation of a jittered training spawn.
                bf = tangent_pose(args.start)[:2, 0]
                a0 = float(np.arctan2(bf[1], bf[0]))
                if args.spawn_lat_m:
                    origin = origin + args.spawn_lat_m * np.array(
                        [-np.sin(a0), np.cos(a0)])
                    d = target - origin
                    dist = np.linalg.norm(d)
                    u = d / max(dist, 1e-6)
                a0 += float(np.deg2rad(args.spawn_yaw_deg))
                a1 = float(np.arctan2(u[1], u[0]))
                dang = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
                n_turn = max(int(np.ceil(abs(np.degrees(dang)) / 10.0)), 1)
                if step <= n_turn:
                    a = a0 + dang * step / n_turn
                    pose = pose_at(origin, np.array([np.cos(a), np.sin(a)]))
                    walked_m = 0.0
                else:
                    walked_m = 0.25 * (step - n_turn)
                    cur = origin + u * min(walked_m, max(dist - 0.4, 0.0))
                    pose = pose_at(cur, u)
            else:
                walked_m = 0.25 * step
                cur = origin + u * min(walked_m, max(dist - 0.4, 0.0))
                pose = pose_at(cur, u)
        else:
            pose = (tangent_pose(i) if args.heading == "tangent"
                    else cal.robot_pose_nav(i))
        (rgb, K, w2c, lab) = world.render_batch([(0, pose)])[0]
        sem_rgb = pal[np.clip(lab, 0, 13).astype(int)]        # HxWx3 RGB
        panels = [rgb, sem_rgb]
        if args.raster:
            ras = getattr(world, "last_raster", None)
            ras = ras[0] if ras else np.zeros_like(rgb)
            if ras.shape[:2] != rgb.shape[:2]:
                ras = cv2.resize(ras, (rgb.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            panels.append(ras)
            # 4th panel: the SPLATS' OWN semantic labels (pre-diffusion) —
            # what the reconstruction knows vs what the generator invents.
            sras = getattr(world, "last_sem_raster", None)
            agree_txt = ""
            if sras:
                s0 = sras[0]
                if s0.shape[:2] != rgb.shape[:2]:
                    s0 = cv2.resize(s0.astype(np.int32),
                                    (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                panels.append(pal[np.clip(s0, 0, 13).astype(int)])
            else:
                s0 = None
                panels.append(np.zeros_like(rgb))
            # 5th panel: the SUPPORT map (alpha), + the coherence number this
            # whole thread has been arguing about: where the reconstruction
            # HAS evidence, does the generated semantics agree with it?
            alph = getattr(world, "last_alpha", None)
            if alph is not None:
                a0 = alph[0]
                if a0.shape[:2] != rgb.shape[:2]:
                    a0 = cv2.resize(a0, (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                amax = float(np.percentile(a0, 99)) or 1.0
                a8 = np.clip(a0 / max(amax, 1e-6), 0, 1)
                heat = cv2.applyColorMap((a8 * 255).astype(np.uint8),
                                         cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
                panels.append(np.ascontiguousarray(heat))
                sup = a0 > 0.5
                if s0 is not None and sup.any():
                    agree = float((lab[sup] == s0[sup]).mean())
                    agree_txt = (f"  sup {100.0 * sup.mean():.0f}%"
                                 f"  agree {100.0 * agree:.0f}%")
                else:
                    agree_txt = f"  sup {100.0 * sup.mean():.0f}%"
            else:
                panels.append(np.zeros_like(rgb))
        frame = np.hstack(panels)[:, :, ::-1]                 # to BGR
        frame = np.ascontiguousarray(frame)
        if args.raster:
            cv2.putText(frame, "SPLAT RASTER (pre-diffusion)",
                        (2 * args.width + 8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "SPLAT SEMANTICS (gaussian labels)",
                        (3 * args.width + 8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "SUPPORT (alpha): bright = observed",
                        (4 * args.width + 8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        if args.aerial:
            # Bird's-eye training-distribution atlas (her ask 2026-08-31):
            # overlay the recorded path + REAL sampled spawns/goals from the
            # J-50 distribution onto every panel, save one PNG, done.
            _, _, n_samp = (float(v) for v in args.aerial.split(","))
            world.cfg.goal_dir_360 = True
            world.cfg.goal_cone_deg = 50.0
            world.cfg.goal_dist_range = (5.0, 10.0)
            world.cfg.spawn_label_classes = (6, 8)
            world.cfg.spawn_yaw_jitter_deg = 20.0
            world.cfg.spawn_lat_jitter_m = 0.4
            world.cfg.spawn_min_frame = 10
            srng = np.random.default_rng(0)

            def apt(pt_xy, dx):
                K2 = np.asarray(K, dtype=float).reshape(3, 3)
                M = np.asarray(w2c, dtype=float)
                if M.shape == (3, 4):
                    M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])
                p = M @ np.array([pt_xy[0], pt_xy[1], 0.0, 1.0])
                if p[2] <= 0.1:
                    return None
                u = K2[0, 0] * p[0] / p[2] + K2[0, 2]
                v = K2[1, 1] * p[1] / p[2] + K2[1, 2]
                if not (0 <= u < args.width and 0 <= v < args.height):
                    return None
                return int(u) + dx, int(v)

            draws = []
            for _s in range(int(n_samp)):
                sp = world.sample_start_pose(args.scene, srng)
                # Same cone centre the env passes -- the RECORDED heading at
                # the spawn frame, not the jittered pose -- so this picture
                # shows the cone the policy actually trains against.
                g = world.sample_goal_position(
                    args.scene, srng, sp[:2, 3],
                    cone_yaw=world.last_spawn_base_yaw())
                draws.append((sp, g))
            for dx in (k * args.width for k in range(len(panels))):
                prev = None
                for p2 in path_xy:
                    uv = apt(p2, dx)
                    if uv is not None and prev is not None:
                        cv2.line(frame, prev, uv, (255, 200, 0), 2,
                                 cv2.LINE_AA)
                    prev = uv
                for sp, g in draws:
                    suv = apt(sp[:2, 3], dx)
                    tip = apt(sp[:2, 3] + sp[:2, 0] * 1.2, dx)
                    guv = apt(g[:2], dx)
                    if suv is not None and guv is not None:
                        cv2.line(frame, suv, guv, (0, 255, 0), 1, cv2.LINE_AA)
                    if suv is not None:
                        cv2.circle(frame, suv, 5, (0, 140, 255), -1)
                        if tip is not None:
                            cv2.line(frame, suv, tip, (0, 140, 255), 2,
                                     cv2.LINE_AA)
                    if guv is not None:
                        cv2.drawMarker(frame, guv, (0, 255, 0),
                                       cv2.MARKER_CROSS, 12, 2)
            png = out_dir / (f"OVERVIEW_{args.scene}_pal{args.sem_palette}"
                             ".png")
            cv2.imwrite(str(png), frame)
            print(f"==> {png}", flush=True)
            break

        # In totgt mode the walk leaves the recorded path, so "pose i" would be
        # a lie (found 2026-08-30: the off-path tree got mislocated to "pose 25")
        # — label by steps/meters walked instead.
        if args.spin and args.spin_deg < 360.0:
            where = f"rec{spin_off_deg:+.0f}deg @pose {i}"
        elif args.spin:
            where = f"heading {int(360 * step / max(args.frames, 1)):3d}deg @pose {i}"
        elif args.strafe:
            where = f"lat {strafe_off_m:+.2f}m @pose {i}"
        elif target is not None:
            where = f"step {step} ({walked_m:.1f}m walked)"
        else:
            where = f"pose {i}"
        hud = (f"{tag}  {where}  {world.last_timings['total']:.2f}s"
               + (agree_txt if args.raster else ""))
        cov = getattr(world, "last_coverage", None)
        if cov is not None and cov == cov:
            hud += f"  cov {cov * 100:.0f}%"
        if goal is not None:
            gd = float(np.linalg.norm(goal[:2] - pose[:3, 3][:2]))
            hud += f"  goal {gd:.1f}m"
            uv = project_goal(K, w2c)
            if uv is not None:
                for dx in (0, args.width):
                    cv2.circle(frame, (uv[0] + dx, uv[1]), 8, (0, 255, 0), 2)
                    cv2.drawMarker(frame, (uv[0] + dx, uv[1]), (0, 255, 0),
                                   cv2.MARKER_CROSS, 10, 2)
            else:
                hud += " (goal off-screen)"
        side = max(120, args.height // 3)
        inset = topdown_inset(pose[:3, 3][:2], side,
                              cur_fwd=pose[:2, 0],
                              rec_fwd=cal.robot_pose_nav(i)[:2, 0])
        frame[frame.shape[0] - side - 8:frame.shape[0] - 8,
              args.width - side - 8:args.width - 8] = inset
        cv2.putText(frame, hud,
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(str(out_dir / f"DRIVE_{tag}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                                 (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        cov_txt = ""
        if cov is not None and cov == cov:
            cov_txt = f"  cov {cov * 100:5.1f}%"
        # dlt = mean |dRGB| between consecutive GENERATED frames, 0-1. Printed
        # beside cov so the offset ladder can be read for both at once: does
        # the frame-difference measure of incoherence agree with the geometric
        # one, and where does each start to blow up? (2026-09-02)
        dlt_txt = ""
        _prev = globals().get("_PREV_DRIVE_RGB")
        if _prev is not None and _prev.shape == rgb.shape:
            _d = float(np.abs(rgb.astype(np.float32)
                              - _prev.astype(np.float32)).mean() / 255.0)
            dlt_txt = f"  dlt {_d:.4f}"
        globals()["_PREV_DRIVE_RGB"] = rgb.copy()
        off_txt = (f"  off {spin_off_deg:+6.1f}deg"
                   if (args.spin and args.spin_deg < 360.0) else "")
        if args.strafe:
            off_txt = f"  lat {strafe_off_m:+6.2f}m"
        print(f"[{step + 1}/{args.frames}] pose {i} "
              f"{world.last_timings['total']:.2f}s{off_txt}{cov_txt}{dlt_txt}",
              flush=True)
    if vw is not None:
        vw.release()
        print(f"==> {out_dir / f'DRIVE_{tag}.mp4'}", flush=True)


if __name__ == "__main__":
    main()
