"""Evaluate a saved policy checkpoint: success rate, steps-to-goal, videos.

Why: training saves checkpoints every 2k steps, and PPO sometimes destroys its
own best policy late in training (observed twice). The final rollout.mp4 then
shows the corpse. This tool evaluates ANY checkpoint over N episodes with the
standard protocol, so we can (a) rescue the pre-collapse peak and (b) report
honest numbers: success rate, mean steps-to-goal, collision exposure.

Usage (Marlowe GPU):
    python scripts/eval_policy.py \
        --checkpoint /scratch/.../ppo_v3_bc_trail00/checkpoints/ppo_130000_steps.zip \
        --scene rugd_trail_00 --episodes 20 \
        --clips_dir ... --poses_dir ... --labels_dir ... \
        --out_dir /scratch/.../outputs/eval_v3_peak
Outputs: metrics.json (incl. per-step [x, y, yaw, collision] trajectories for
top-down plotting via scripts/make_topdown_figure.py), eval_summary printed,
rollout videos for the first 3 episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.env.scene_env import SceneEnv, SceneEnvConfig
from src.env.real_calibrated import (
    CalibratedRealWorldBackend, CalibratedBackendConfig, GaussianLabelBackend,
)
from src.eval.reward_2d import RewardWeights


def _parse_range(text, cast, name):
    """"2,4" -> (2.0, 4.0). Refuses anything else by name instead of raising deep inside the env."""
    parts = [p for p in str(text).split(",") if p.strip() != ""]
    if len(parts) != 2:
        raise SystemExit(f"REFUSED: --{name} is {text!r}, expected two numbers like '2,4'")
    return tuple(cast(p) for p in parts)


def _resolved_budget(args) -> int:
    """_eval_step_budget, computed once and remembered on args so the metrics.json
    summary can record the budget the cell actually ran under instead of the CLI
    default (2026-09-24)."""
    n = _eval_step_budget(args)
    args.resolved_max_steps = int(n)
    return int(n)


def _eval_step_budget(args) -> int:
    """The same budget training uses: base + per_m * (top of the goal band).

    Returns args.max_steps unchanged when per_m is 0, which is every checkpoint
    trained before 2026-09-23.
    """
    per_m = float(getattr(args, "max_steps_per_m", None) or 0.0)
    if per_m <= 0.0:
        return int(args.max_steps)
    d = getattr(args, "goal_dist", None)
    # A FIXED-GOAL scene test is the same distance for every arm, but `goal_dist`
    # is the ARM's own curriculum reach, so each arm was sized differently on the
    # identical test: on the 8.1 m far corner chunk5 (reach 4 m) got 104 actions
    # and memDINO_warm (reach 7.5 m) got 160, and chunk5's 104 cannot cover 8.1 m
    # at any throttle it has ever shown. It timed out on the budget, not on skill
    # (2026-09-24, Joana: "this one seems to time out too early"). When the caller
    # states the test's own distance, size from whichever is larger.
    _td = getattr(args, "test_goal_dist", None)
    if _td:
        _td = float(_td)
        if d is None or _td > float(d):
            print(f"[eval] test distance {_td:g} m exceeds the curriculum reach "
                  f"{'unset' if d is None else format(float(d), 'g')} m; sizing the budget "
                  f"from the test so every arm gets the same budget", flush=True)
            d = _td
    if d is None:
        rng = getattr(args, "goal_dist_range", None)
        if isinstance(rng, str) and rng:
            try:
                d = max(float(v) for v in rng.split(","))
            except Exception:
                d = None
    if d is None:
        print(f"[eval] no goal distance to size the budget from; keeping max_steps={args.max_steps}", flush=True)
        return int(args.max_steps)
    base = int(getattr(args, "max_steps_base", None) or 40)
    n = int(round(base + per_m * float(d)))
    print(f"[eval] step budget {n} = {base} + {per_m:g} x {float(d):g} m "
          f"(training used the same rule); fixed default was {args.max_steps}", flush=True)
    return n


def build_env(args):
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        goal_frame=(int(args.goal_frame) if getattr(args, "goal_frame", None) is not None else 30),
        goal_xy_override=(tuple(float(v) for v in args.goal_xy.split(","))
                          if args.goal_xy else None),
        # 2026-09-02: eval could ONLY use a fixed goal_xy or the default goal
        # frame, while every J arm trains on goals sampled in a cone at a
        # random distance from the spawn. So the eval had never once
        # reproduced the goal distribution the policy learned -- with a fixed
        # goal and a wide spawn range, d_start varied uncontrolled and some
        # episodes were unreachable inside the step budget before the policy
        # acted. Pass --goal_dir_360 to sample goals exactly as training does.
        goal_dir_360=args.goal_dir_360,
        # spawn jitter lives on the BACKEND config (it is applied in
        # sample_start_pose), not on SceneEnvConfig. Passing it to the wrong
        # dataclass killed six evals in four seconds on 2026-09-03.
        spawn_yaw_jitter_deg=getattr(args, "spawn_yaw_jitter", 0.0),
        spawn_lat_jitter_m=getattr(args, "spawn_lat_jitter", 0.0),
        goal_dist_m=getattr(args, "goal_dist", None),
        goal_dist_range=(_parse_range(args.goal_dist_range, float, "goal_dist_range")
                         if args.goal_dist_range else None),
        goal_cone_deg=args.goal_cone_deg,
        goal_frame_range=(_parse_range(args.goal_frame_range, int, "goal_frame_range")
                          if args.goal_frame_range else None),
        spawn_max_frame=args.spawn_max_frame,
        spawn_min_frame=args.spawn_min_frame,
        render_mode="rasterizer_only",
        static_scene=bool(getattr(args, "static_scene", False)),
        static_movers=str(getattr(args, "static_movers", "") or ""),
        render_window=int(getattr(args, "render_window", 0) or 0),
        coverage_window=int(getattr(args, "coverage_window", 0) or 0),
        spawn_heading_from_walk=bool(getattr(args, "spawn_heading_from_walk", False)),
        spawn_frames_by_scene=_clip_spawn_frames(str(getattr(args, "spawn_frames", "") or ""),
                                                 args.spawn_min_frame, args.spawn_max_frame),
        # 2026-09-07: the backend palette was never set in evals (default v1)
        # while training passes --sem_palette 4; see the log line "active palette"
        sem_palette_version=int(getattr(args, "sem_palette", 4)),
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.render_height or args.obs_height,
        W=args.render_width or args.obs_width,
    )
    if args.live:
        # Live-trained policies (--live runs) must be evaluated on live
        # observations too — cache lookups would be an obs-distribution shift.
        from src.env.live_backend import LiveDiffusedBackend
        world = LiveDiffusedBackend(
            cfg, checkpoint=args.live_ckpt,
            alpha_gate=not args.no_alpha_gate,
            alpha_gate_tau=float(getattr(args, "alpha_gate_tau", None) or 0.5),
            raster_obs=bool(getattr(args, "raster_obs", False)))
    elif args.obs_cache:
        from src.env.cached_backend import CachedDiffusedBackend
        world = CachedDiffusedBackend(cfg, args.obs_cache,
                                      alpha_gate=not args.no_alpha_gate,
                                      sweep_switch_penalty_m=args.sweep_sticky)
    else:
        world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    env_cfg = SceneEnvConfig(
        goal_support_radius_m=args.goal_support_radius,
        goal_support_min_frac=args.goal_support_min_frac,
        collision_look_ahead_m=args.collision_look_ahead,
        collision_box_memory=int(getattr(args, "collision_box_memory", 0)),
        collision_box_memory_agg=str(getattr(args, "collision_box_memory_agg", "newest") or "newest"),
        collision_box_memory_always=bool(getattr(args, "collision_box_memory_always", False)),
        collision_at_next_pose=bool(getattr(args, "collision_at_next_pose", False)),
        look_ahead_auto=bool(getattr(args, "look_ahead_auto", False)),
        footprint_next_heading=bool(getattr(args, "footprint_next_heading", False)),
        crash_requires_motion=bool(getattr(args, "crash_requires_motion", False)),
        # GND/SCAND clips advance ~1 m per recorded frame vs RUGD's ~0.1 m, so
        # the same goal_frame is a far longer walk there — raise the budget
        # instead of moving the goal (moving it collapses the spawn range).
        # KINEMATICS MUST MATCH TRAINING (2026-09-01). These were hardcoded to
        # 0.25 / 0.3 while every J arm trains at 0.30 / 0.50, so the policy was
        # evaluated with 17% shorter steps and 40% weaker turning than the
        # action model it learned. Combined with forward_only defaulting off —
        # RW5 clamps reverse away during training — the robot could also drive
        # backwards in eval, which it never could while learning.
        # 2026-09-23: eval adopted max_steps_base/_per_m but never called
        # set_goal_dist, where the scaling lives, so it would have kept the fixed
        # 90-step budget while training grew its own. Apply the same formula here,
        # off the goal distance the curriculum actually reached.
        max_steps=_resolved_budget(args),
        step_size_m=args.step_size_m, yaw_step_rad=args.yaw_step_rad,
        reward=RewardWeights(
            semantic=(args.semantic_weight if getattr(args, "semantic_weight", None) is not None else 1.0),
            goal=getattr(args, "goal_weight", 1.5),
            collision=getattr(args, "collision_weight", 1.0),
            step_cost=getattr(args, "step_cost", 0.05),
            void_cost=getattr(args, "void_cost", 0.3),
            terrain_as_cost=bool(getattr(args, "terrain_as_cost", True))),
        # read from args so the adoption above actually takes effect
        look_ahead_dist=getattr(args, "look_ahead_dist", 1.5),
        goal_radius=(args.goal_radius if getattr(args, "goal_radius", None) is not None else 0.75),
        collision_threshold=getattr(args, "collision_threshold", 0.1),
        spin_cost=(getattr(args, "spin_cost", None) if getattr(args, "spin_cost", None) is not None else 0.05),
        backward_cost=getattr(args, "backward_cost", 0.0),
        action_smooth_cost=(getattr(args, "action_smooth_cost", None) if getattr(args, "action_smooth_cost", None) is not None else 0.0),
        goal_bonus=getattr(args, "goal_bonus", 50.0),
        timeout_penalty=getattr(args, "timeout_penalty", 0.0),
        proximity_weight=getattr(args, "proximity_weight", 0.0),
        proximity_margin=getattr(args, "proximity_margin", 1.0),
        proximity_delta=bool(getattr(args, "proximity_delta", False)),
        proximity_ground_weight=float(getattr(args, "proximity_ground_weight", 0.0)),
        proximity_ground_margin=float(getattr(args, "proximity_ground_margin", 1.2)),
        proximity_ground_classes=str(getattr(args, "proximity_ground_classes", "3,4,5") or "3,4,5"),
        timeout_distance_scaled=bool(
            getattr(args, "timeout_distance_scaled", False)),
        clouds_dir=getattr(args, "clouds_dir", None),
        reward_scale=(getattr(args, "reward_scale", None) if getattr(args, "reward_scale", None) is not None else 1.0),
        coherence_cost_weight=getattr(args, "coherence_cost_weight", 0.0),
        coherence_tau=getattr(args, "coherence_tau", 0.4),
        coherence_terminate_tau=getattr(args, "coherence_terminate_tau", 0.0),
        coherence_terminate_penalty=getattr(
            args, "coherence_terminate_penalty", 100.0),
        void_terminate_frac=getattr(args, "void_terminate_frac", 0.0),
        void_terminate_penalty=getattr(args, "void_terminate_penalty", 100.0),
        halt_terminate_steps=getattr(args, "halt_terminate_steps", 0),
        halt_throttle_eps=getattr(args, "halt_throttle_eps", 0.05),
        halt_penalty_scale=getattr(args, "halt_penalty_scale", 1.0),
        refusal_bonus=float(getattr(args, "refusal_bonus", 0.0) or 0.0),
        refusal_dist_m=float(getattr(args, "refusal_dist_m", 2.0) or 2.0),
        refusal_verge_m=float(getattr(args, "refusal_verge_m", 1.5) or 1.5),
        halt_wrong_penalty=float(getattr(args, "halt_wrong_penalty", 0.0) or 0.0),
        nontrav_goal_unreachable=bool(getattr(args, "nontrav_goal_unreachable", False)),
        goal_requires_stop=bool(getattr(args, "goal_requires_stop", False)),
        stop_action=bool(getattr(args, "stop_action", False)),
        lawn_progress_to_verge=bool(getattr(args, "lawn_progress_to_verge", False)),
        terrain_speed_scaled=bool(getattr(args, "terrain_speed_scaled", False)),
        reward_source=getattr(args, "reward_source", "generated"),
        map_res_m=float(getattr(args, "map_res_m", 0.1)),
        map_fallback_void_frac=float(getattr(args, "map_fallback_void_frac", 0.5)),
        map_fallback_min_alpha=float(getattr(args, "map_fallback_min_alpha", 0.4)),
        map_inflate_m=float(getattr(args, "map_inflate_m", 0.1)),
        map_inflate_classes=str(getattr(args, "map_inflate_classes", "") or ""),
        map_fill_m=float(getattr(args, "map_fill_m", 0.3)),
        map_fill_max_area_m2=float(getattr(args, "map_fill_max_area_m2", 10.0)),
        goal_traversable_mix=float(getattr(args, "goal_traversable_mix", 0.0)),
        spawn_support_tries=int(getattr(args, "spawn_support_tries", 0)),
        goal_mix_map_draw=bool(getattr(args, "goal_mix_map_draw", False)),
        goal_nontrav_classes=str(getattr(args, "goal_nontrav_classes", "3,4,5") or "3,4,5"),
        goal_nontrav_edge_m=float(getattr(args, "goal_nontrav_edge_m", 0.0)),
        goal_nontrav_tries=int(getattr(args, "goal_nontrav_tries", 0)),
        goal_nontrav_cone_deg=float(getattr(args, "goal_nontrav_cone_deg", 0.0)),
        goal_case_mix=str(getattr(args, "goal_case_mix", "") or ""),
        # default is None so adoption can tell "not given" from "given as 1"
        max_steps_base=int(getattr(args, "max_steps_base", None) or 40),
        max_steps_per_m=float(getattr(args, "max_steps_per_m", None) or 0.0),
        obs_frame_stack=int(getattr(args, "obs_frame_stack", None) or 1),
        obs_frame_stride=int(getattr(args, "obs_frame_stride", None) or 1),
        goal_case_tries=int(getattr(args, "goal_case_tries", 24)),
        goal_center_clear_m=float(getattr(args, "goal_center_clear_m", 0.0) or 0.0),
        mirror_prob=float(getattr(args, "mirror_prob", 0.0) or 0.0),
        map_walk_halfwidth_m=float(getattr(args, "map_walk_halfwidth_m", 0.4)),
        map_ignore_classes=str(getattr(args, "map_ignore_classes", "")),
        label_remap=str(getattr(args, "label_remap", "") or ""),
        random_spawn=True,
        trav_path=args.trav_path,
        collision_terminate_frac=args.collision_terminate_frac,
        phantom_veto=bool(getattr(args, "phantom_veto", False)),
        collision_terminate_penalty=args.collision_terminate_penalty,
        action_chunk=args.action_chunk,
        footprint_along_motion=args.footprint_along_motion,
        forward_only=args.forward_only,
        failure_snap_dir=str(args.out_dir / "failures"),
        memory_snap_every=int(getattr(args, "memory_snap_every", 0)),
        obs_out_hw=((args.obs_height, args.obs_width)
                    if (args.render_height or args.render_width) else None),
    )
    return SceneEnv(world_backend=world, semantic_backend=sem,
                    scene_ids=[args.scene], cfg=env_cfg)


# Every reward term SceneEnv puts on `info`. Summed per episode so the table
# shows WHY a return is what it is -- a -1000 return from one crash and a -1000
# return from a hundred bad steps are different diagnoses.
EVAL_COMPONENTS = ("semantic", "goal", "collision", "step", "spin", "backward",
                   "smooth", "timeout", "crash", "proximity", "proximity_ground", "goal_bonus", "speed_refund", "refusal_bonus",
                   "coherence", "coherence_crash")


def _pose_xyyaw(env) -> list:
    P = env.unwrapped._robot_pose_world
    return [round(float(P[0, 3]), 3), round(float(P[1, 3]), 3),
            round(float(np.arctan2(P[1, 0], P[0, 0])), 3)]



def _clip_spawn_frames(spec: str, lo, hi) -> str:
    """2026-09-19 (Joana: "spawns are not where we decided"): the training run's
    spawn-frame list is adopted from env_config.json and, being an explicit
    list, it overrode SPAWN_MIN/SPAWN_MAX -- corner evals spawned anywhere in
    5..60 and half the goals ended up behind the robot. Keep the adopted list
    (unflagged frames) but intersect it with the requested range."""
    if not spec:
        return spec
    lo = int(lo or 0); hi = int(hi) if hi is not None else 10 ** 9
    out = []
    for part in spec.split(";"):
        if ":" not in part:
            continue
        sc, fr = part.split(":", 1)
        keep = [f for f in fr.split(",") if f.strip() and lo <= int(f) <= hi]
        if keep:
            out.append(f"{sc}:{','.join(keep)}")
    kept = ";".join(out)
    print(f"[eval] spawn frames clipped to [{lo}, {hi}]: {kept[:120]}{'...' if len(kept) > 120 else ''}", flush=True)
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--goal_frame", type=int, default=None,
                    help="pin the goal to this recorded frame (fixed goal, every episode); when given, the "
                         "training goal sampling (goal_frame_range / goal_dist_range) is NOT adopted")
    ap.add_argument("--max_steps", type=int, default=60,
                    help="episode step budget (60 = training default; raise "
                         "for scenes recorded with large per-frame motion)")
    ap.add_argument("--goal_xy", default=None,
                    help='designed obstacle test: pin the goal to "x,y" (nav-'
                         'frame meters, read off the top-down figure axes) — '
                         'e.g. just past a tree so the straight line crosses it')
    # 2026-09-03: these were readable ONLY through env_config.json adoption
    # (`getattr(args, ..., default)`), so when that file is missing there was no
    # way to make eval terminate the way training does. Every eval that day ran
    # with crash termination OFF (frac 0) and coherence termination OFF, which
    # is why five runs reported ZERO crashes against a 72% crash rate in
    # training, and why goal counts came back at 20/20. Exposed as real flags so
    # a run can be matched to its training by hand.
    # Reward weights. These do NOT change what a frozen policy does -- only the
    # `return=` it reports -- but a return that is not on training's scale is a
    # number nobody can compare to anything, so expose them all. Defaults here
    # are the OLD eval defaults, deliberately: passing nothing reproduces
    # previous evals exactly.
    ap.add_argument("--goal_weight", type=float, default=1.5,
                    help="training's RW5 value is 10")
    ap.add_argument("--collision_weight", type=float, default=1.0)
    ap.add_argument("--goal_bonus", type=float, default=50.0,
                    help="training's RW5 value is 1000")
    ap.add_argument("--timeout_penalty", type=float, default=0.0,
                    help="training's RW5 value is 100. At 0 a TIMEOUT costs "
                         "NOTHING, so --timeout_distance_scaled is also inert.")
    ap.add_argument("--timeout_distance_scaled", action="store_true",
                    help="scale the timeout by remaining/initial distance, as "
                         "training does with TIMEOUTDIST=1")
    ap.add_argument("--proximity_weight", type=float, default=0.0)
    ap.add_argument("--proximity_margin", type=float, default=1.0)
    ap.add_argument("--proximity_delta", action="store_true")
    ap.add_argument("--proximity_ground_weight", type=float, default=0.0)
    ap.add_argument("--proximity_ground_margin", type=float, default=1.2)
    ap.add_argument("--proximity_ground_classes", type=str, default="3,4,5")
    ap.add_argument("--void_cost", type=float, default=0.3)
    ap.add_argument("--step_cost", type=float, default=0.05)
    ap.add_argument("--coherence_cost_weight", type=float, default=0.0,
                    help="graded cost below --coherence_tau. Training uses 10.")
    ap.add_argument("--coherence_tau", type=float, default=0.4)
    ap.add_argument("--coherence_terminate_tau", type=float, default=0.0,
                    help="END the episode below this coverage. Training uses "
                         "0.1 (0.05 on the COHTERM arm). 0 = never terminate.")
    ap.add_argument("--coherence_terminate_penalty", type=float, default=100.0)
    # Spawn jitter. Training spawns with +-20 deg of heading and +-0.4 m of
    # lateral offset; eval had no flag at all, so it spawned exactly on the
    # recorded pose -- an easier and different distribution (2026-09-03).
    ap.add_argument("--spawn_yaw_jitter", type=float, default=0.0)
    ap.add_argument("--spawn_lat_jitter", type=float, default=0.0)
    ap.add_argument("--void_terminate_frac", type=float, default=0.0,
                    help="END the episode when this fraction of the footprint "
                         "is void. The whole-frame coherence terminal does NOT "
                         "catch an unsupported ground patch under a "
                         "well-rendered view. 0 = off, as in training.")
    ap.add_argument("--void_terminate_penalty", type=float, default=100.0)
    ap.add_argument("--halt_terminate_steps", type=int, default=0,
                    help="must match training, or a policy trained to halt is "
                         "scored as if it merely timed out")
    ap.add_argument("--halt_throttle_eps", type=float, default=0.05)
    ap.add_argument("--refusal_bonus", type=float, default=0.0)
    ap.add_argument("--refusal_dist_m", type=float, default=2.0)
    ap.add_argument("--refusal_verge_m", type=float, default=1.5)
    ap.add_argument("--stop_action", action="store_true",
                    help="third action output: > 0 halts now (cold start only, changes the action space)")
    ap.add_argument("--lawn_progress_to_verge", action="store_true",
                    help="on lawn-goal episodes the progress term measures distance to the verge, not the goal")
    ap.add_argument("--goal_requires_stop", action="store_true",
                    help="a goal counts as reached only after the robot is still for the halt steps inside its radius")
    ap.add_argument("--nontrav_goal_unreachable", action="store_true",
                    help="entering the radius of a non-traversable goal ends nothing and pays nothing; only a halt can earn there")
    ap.add_argument("--halt_wrong_penalty", type=float, default=0.0,
                    help="flat penalty for HALTING on a traversable goal (mirror of --refusal_bonus)")
    ap.add_argument("--halt_penalty_scale", type=float, default=1.0)
    ap.add_argument("--terrain_speed_scaled", action="store_true")
    ap.add_argument("--reward_source", default="generated",
                    choices=("generated", "map", "map_then_generated", "split"))
    ap.add_argument("--map_res_m", type=float, default=0.1)
    ap.add_argument("--map_fallback_void_frac", type=float, default=0.5)
    ap.add_argument("--map_fallback_min_alpha", type=float, default=0.4)
    ap.add_argument("--map_inflate_m", type=float, default=0.1)
    ap.add_argument("--map_inflate_classes", type=str, default="",
                    help="comma list of class ids that inflate; empty = all non-traversable")
    ap.add_argument("--map_fill_m", type=float, default=0.3)
    ap.add_argument("--map_fill_max_area_m2", type=float, default=10.0)
    ap.add_argument("--goal_mix_map_draw", action="store_true",
                    help="draw the non-traversable share of the goal mix straight from map cells (grass etc.) in the window and cone")
    ap.add_argument("--goal_nontrav_classes", type=str, default="3,4,5")
    ap.add_argument("--goal_nontrav_cone_deg", type=float, default=0.0,
                    help="cone of the map-direct lawn draw (0 = goal cone); adopted from env_config.json when present")
    ap.add_argument("--goal_nontrav_tries", type=int, default=0,
                    help="tries of the map-direct lawn draw (0 = 12); adopted from env_config.json when present")
    ap.add_argument("--goal_nontrav_edge_m", type=float, default=0.0,
                    help="map-direct lawn goals must have walkable ground within this distance (0 = anywhere on the lawn)")
    ap.add_argument("--spawn_support_tries", type=int, default=0,
                    help="redraw a spawn whose crash box is already at the crash threshold on the map, up to N times (0 = off)")
    ap.add_argument("--goal_traversable_mix", type=float, default=0.0,
                    help="P(goal on traversable ground by the map); 0 = the sampler's natural mix")
    ap.add_argument("--map_walk_halfwidth_m", type=float, default=0.4)
    ap.add_argument("--map_ignore_classes", default="")
    ap.add_argument("--label_remap", type=str, default="", help="adopted from env_config.json when present")
    ap.add_argument("--goal_center_clear_m", type=float, default=0.0, help="adopted from env_config.json when present")
    ap.add_argument("--mirror_prob", type=float, default=0.0,
                    help="mirrored-episode probability; NOT adopted from training -- evaluate in the real world unless you pass it explicitly")
    ap.add_argument("--goal_radius", type=float, default=None,
                    help="arrival radius; training's FINAL value, not its start")
    ap.add_argument("--semantic_weight", type=float, default=None)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--action_smooth_cost", type=float, default=None)
    ap.add_argument("--spin_cost", type=float, default=None)
    ap.add_argument("--static_scene", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--static_movers", type=str, default="", help="adopted from env_config.json when present")
    ap.add_argument("--render_window", type=int, default=0, help="adopted from env_config.json when present")
    ap.add_argument("--coverage_window", type=int, default=0, help="adopted from env_config.json when present")
    ap.add_argument("--spawn_heading_from_walk", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--spawn_frames", type=str, default="", help="adopted from env_config.json when present")
    ap.add_argument("--goal_case_mix", type=str, default="", help="adopted from env_config.json when present")
    ap.add_argument("--max_steps_base", type=int, default=None, help="adopted from env_config.json when present")
    ap.add_argument("--max_steps_per_m", type=float, default=None, help="adopted from env_config.json when present")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions (mean + N(0,std)) as training does, instead of the deterministic mean")
    ap.add_argument("--test_goal_dist", type=float, default=None,
                    help="the distance THIS test actually spans (m). A fixed-goal scene is "
                         "the same length for every arm, so the budget must not come from "
                         "the arm's own curriculum reach. Measured with scene_corner_scan.py.")
    ap.add_argument("--obs_frame_stack", type=int, default=None, help="adopted from env_config.json when present")
    ap.add_argument("--obs_frame_stride", type=int, default=None, help="adopted from env_config.json when present")
    ap.add_argument("--goal_dist", type=float, default=None,
                    help="target spawn-to-goal distance; adopted from env_config.json. Without it the\n                         sampler falls back to a random recorded frame, which is NOT how training drew goals")
    ap.add_argument("--goal_case_tries", type=int, default=24)
    ap.add_argument("--sem_palette", type=int, default=4,
                    help="colour table for the video semantic panels. MUST "
                         "match the semantics model (v26 = 4, v21 = 1) or the "
                         "panels are coloured with the wrong classes.")
    ap.add_argument("--eval_seed", type=int, default=7,
                    help="episode e uses seed eval_seed*10000+e, so two "
                         "policies evaluated at the same value see IDENTICAL "
                         "spawns and goals and can be compared pair by pair")
    ap.add_argument("--goal_support_min_frac", type=float, default=0.25,
                    help="adopted from env_config.json when present")
    ap.add_argument("--goal_support_radius", type=float, default=0.0,
                    help="reject goals with no cloud support within this "
                         "radius, as training does. Adopted from "
                         "env_config.json when present.")
    ap.add_argument("--no_adopt", action="store_true",
                    help="do NOT adopt env values from the checkpoint's env_config (mechanism checks with --expert)")
    ap.add_argument("--force_env_keys", type=str, default="",
                    help="comma list of env_config keys the CLI overrides instead of adopting from training")
    ap.add_argument("--collision_box_memory", type=int, default=0)
    ap.add_argument("--collision_box_memory_agg", type=str, default="newest", choices=("newest", "mean"),
                    help="adopted from env_config.json when present (memory arm trains with mean; 2026-09-20)")
    ap.add_argument("--collision_box_memory_always", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--collision_at_next_pose", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--look_ahead_auto", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--footprint_next_heading", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--crash_requires_motion", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--path_scenes", type=str, default="", help="--expert path: comma list of scenes (default: all training scenes)")
    ap.add_argument("--path_pairs", type=int, default=3, help="--expert path: start/goal pairs per scene")
    ap.add_argument("--path_kinds", type=str, default="good,straight,obstacle,beyond", help="--expert path: which paths to drive")
    ap.add_argument("--path_plan_only", action="store_true", help="--expert path: write path_plan.json and the label grids, then stop (look before driving)")
    ap.add_argument("--path_frames", action="store_true", help="--expert path: save the generated frame, labels and opacity of every step")
    ap.add_argument("--expert", choices=["none", "map", "path"], default="none",
                    help="map: drive with the scripted map-reading expert instead of the "
                         "checkpoint's policy and record (frame, goal, action) demos; the "
                         "checkpoint still supplies the env config and resolution")
    ap.add_argument("--demos_out", type=Path, default=None,
                    help="npz for --expert demos (default <out_dir>/demos.npz); keys obs/goal/act/episode, "
                         "the format bc_pretrain reads")
    ap.add_argument("--keep_failed_demos", action="store_true",
                    help="also keep episodes whose stop the reward did not accept")
    ap.add_argument("--raster_obs", action="store_true",
                    help="policy sees the splat raster (adopted from env_config.json when present)")
    ap.add_argument("--collision_look_ahead", type=float, default=0.0,
                    help="0 = collision judged on the same box as shaping")
    ap.add_argument("--goal_dir_360", action="store_true",
                    help="sample goals the way training does (cone at a random "
                         "distance from the spawn) instead of a fixed goal")
    ap.add_argument("--goal_dist_range", default="",
                    help="e.g. 5,10 — must match training's --goal_dist_range")
    ap.add_argument("--goal_cone_deg", type=float, default=360.0)
    ap.add_argument("--goal_frame_range", default="",
                    help="e.g. 15,70 — must match training")
    ap.add_argument("--spawn_min_frame", type=int, default=0,
                    help="lower bound on the spawn frame. Without it a GOAL_XY "
                         "test spawns anywhere from the start of the path, so "
                         "most episodes are 10-17 m from the goal and cannot "
                         "reach it inside max_steps — the run measures reach, "
                         "not the terrain behaviour it was launched for.")
    ap.add_argument("--spawn_max_frame", type=int, default=None,
                    help="match the training rung: 3 = full traverses (rung 5)")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--obs_cache", default=None,
                    help="ribbon-cache root; evaluate on cached diffused obs "
                         "(must match what the checkpoint trained on)")
    ap.add_argument("--no_alpha_gate", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="serve LIVE per-step diffusion observations (evals of "
                         "--live-trained policies; ~1.4 s/step -> raise --time)")
    ap.add_argument("--action_chunk", type=int, default=1,
                    help="match the policy's training chunk size (trajectory arm)")
    ap.add_argument("--footprint_along_motion", action="store_true",
                    help="match the policy's training rule: footprint follows "
                         "the commanded motion direction")
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds",
                    help="proximity needs the scene clouds; without it that "
                         "term is silently zero in eval while active in training")
    ap.add_argument("--step_size_m", type=float, default=0.3,
                    help="metres per unit forward action — MUST match the "
                         "checkpoint's training value (J arms: 0.3)")
    ap.add_argument("--yaw_step_rad", type=float, default=0.5,
                    help="radians per unit yaw action — MUST match training "
                         "(J arms: 0.5)")
    ap.add_argument("--forward_only", action="store_true",
                    help="match the policy's training rule: negative velocity "
                         "clamps to 0")
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm07b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors")
    ap.add_argument("--render_height", type=int, default=None,
                    help="render-high/observe-small: match the checkpoint's "
                         "training RENDERH (world at this res, obs downsized)")
    ap.add_argument("--render_width", type=int, default=None)
    ap.add_argument("--phantom_veto", action="store_true", help="adopted from env_config.json when present")
    ap.add_argument("--collision_terminate_frac", type=float, default=0.0,
                    help="match the policy's training rule; >0 ends the episode "
                         "on a real collision (and it does NOT count as success)")
    ap.add_argument("--collision_terminate_penalty", type=float, default=20.0)
    ap.add_argument("--sweep_sticky", type=float, default=0.0,
                    help="sticky-sweep lookup penalty in meters (0 = off, "
                         "matches training; see CachedDiffusedBackend)")
    ap.add_argument("--trav_path", default=None,
                    help="traversability yaml (v14 table for cached runs)")
    ap.add_argument("--obs_height", type=int, default=336,
                    help="render/obs height — MUST match the checkpoint's "
                         "training resolution (policy CNN is size-locked)")
    ap.add_argument("--obs_width", type=int, default=560)
    ap.add_argument("--memory_snap_every", type=int, default=0,
                    help="write failures/memory_*.png (the stored frames with the same ground box) every N steps; 0 = off (2026-09-27)")
    ap.add_argument("--blind", action="store_true",
                    help="zero the rgb observation (goal vector untouched): "
                         "does the policy actually use the world model's "
                         "frames, or is it odometry-only? Videos still show "
                         "the real frames — only the policy goes blind.")
    args = ap.parse_args()
    # TRAINING AND EVAL MUST MATCH. train_ppo_real writes env_config.json beside
    # the checkpoints; adopt it, and shout about anything that disagrees rather
    # than quietly running a policy outside the action model it learned.
    try:
        import json as _json
        _run = Path(args.checkpoint).resolve().parents[1]
        _ec = _run / "env_config.json"
        if getattr(args, "no_adopt", False):
            # mechanism checks (2026-09-07): the env comes from the CLI only; the
            # checkpoint supplies resolution and, with --expert, nothing else
            print(f"[eval] adoption SKIPPED (--no_adopt): env built from the CLI, not from {_ec}", flush=True)
        elif _ec.exists():
            _tr = _json.loads(_ec.read_text())
            # 2026-09-23: env_config records goal_dist from ARGS -- the CEILING the
            # curriculum was allowed to climb to -- while curriculum_state.json beside
            # it records where the policy actually got. Adopting the ceiling tested
            # arms a full band beyond anything they trained on (goals at 9.0-9.4 m for
            # policies whose band topped out at 7.5 m) and every one of those episodes
            # timed out. The state file is the truth; prefer it.
            _cs = _run / "curriculum_state.json"
            if _cs.exists():
                try:
                    _cur = _json.loads(_cs.read_text())
                    _reached = float(_cur.get("goal_dist"))
                    _ceiling = float(_tr.get("goal_dist") or _reached)
                    if _reached < _ceiling:
                        print(f"[eval] curriculum REACHED {_reached:g} m of a {_ceiling:g} m ceiling; "
                              f"testing at {_reached:g} m (curriculum_state.json)", flush=True)
                    _tr["goal_dist"] = _reached
                    if _cur.get("goal_dist_range"):
                        _tr["goal_dist_range"] = ",".join(str(float(v)) for v in _cur["goal_dist_range"])
                except Exception as _e:
                    print(f"[eval] could not read {_cs} ({_e}); using the configured ceiling", flush=True)
            # TRAVERSABILITY TABLE (2026-09-20): a whole day of evals scored walkway-trained
            # arms with the launcher's default v14 table (grass 0.75, vegetation 0.2 = walkable):
            # A 60k walked 18 steps on a lawn without a crash. The table is adopted from
            # training unless the CLI names one; a CLI table that differs from training is
            # refused unless --force_env_keys names trav_path.
            _tt = _tr.get("trav_path")
            _fk = set(v.strip() for v in str(getattr(args, "force_env_keys", "") or "").split(",") if v.strip())
            if args.trav_path is None and _tt:
                args.trav_path = _tt
                print(f"[eval] traversability table adopted from training: {_tt}", flush=True)
            elif args.trav_path is not None and _tt and str(args.trav_path) != str(_tt) and "trav_path" not in _fk:
                raise SystemExit(f"REFUSED: --trav_path {args.trav_path} differs from the training table {_tt}; "
                                 f"drop TRAV to adopt it, or add trav_path to --force_env_keys to score a different task on purpose")
            try:
                from src.eval.traversability import load_traversability as _lt
                _sc = _lt(str(args.trav_path))
                print(f"[eval] traversability table: {args.trav_path}  grass={float(_sc[3]):.2f} vegetation={float(_sc[11]):.2f} "
                      f"(non-walkable = score <= {getattr(args, 'collision_threshold', 0.1)})", flush=True)
            except Exception as _e:
                print(f"[eval] traversability table: {args.trav_path} (could not print scores: {_e})", flush=True)
            # collision_terminate_* belong here too: training ENDS the
            # episode at >=0.35 footprint non-traversable, eval defaulted to 0
            # and let the policy keep walking through terrain that would have
            # killed it — every episode then reports TIMEOUT and the crash
            # behaviour is invisible (2026-09-02).
            # Sampling first: rebuild the training goal/spawn distribution
            # unless the caller deliberately pinned a goal with --goal_xy.
            # 2026-09-21 night: --goal_frame is a DESIGNED test too. Until now the training
            # goal_frame_range / goal_dist_range were adopted over it, and every "goal48" /
            # "goal72" / pedestrian "goal58" eval drew a fresh short goal per episode (7 distinct
            # goals, 1-7 m) -- the corner and pedestrian columns never tested the pinned goal.
            if getattr(args, "goal_frame", None) is not None and not args.goal_xy:
                print(f"[eval] FIXED GOAL: frame {args.goal_frame} every episode; training goal sampling "
                      f"(frames={_tr.get('goal_frame_range')} range={_tr.get('goal_dist_range')}) NOT adopted", flush=True)
            if not args.goal_xy and getattr(args, "goal_frame", None) is None:
                _gr = _tr.get("goal_dist_range")
                _fr = _tr.get("goal_frame_range")
                if _tr.get("goal_dir_360"):
                    args.goal_dir_360 = True
                # train_ppo_real records these straight from argparse, so they are the STRING
                # "2,4" for runs launched with GOALRANGE and a list for older ones. Joining a
                # string over its characters produced "2,,,4" and every eval of this week's arms
                # died in build_env (2026-09-22).
                if _gr and not args.goal_dist_range:
                    args.goal_dist_range = _gr if isinstance(_gr, str) else ",".join(str(v) for v in _gr)
                if _tr.get("goal_cone_deg") and args.goal_cone_deg >= 360.0:
                    args.goal_cone_deg = float(_tr["goal_cone_deg"])
                if _fr and not args.goal_frame_range:
                    args.goal_frame_range = _fr if isinstance(_fr, str) else ",".join(str(v) for v in _fr)
                print(f"[eval] goal sampling from training: dir360="
                      f"{args.goal_dir_360} range={args.goal_dist_range} "
                      f"cone={args.goal_cone_deg} frames={args.goal_frame_range}",
                      flush=True)
            _force = set(v.strip() for v in str(getattr(args, "force_env_keys", "") or "").split(",") if v.strip())
            # env_config keys whose eval argparse dest has a different name.
            # 2026-09-04: collision_look_ahead_m was 'adopted' onto an attribute
            # build_env never reads, so every eval of a near-box arm silently
            # ran with the FAR box (T2 pair 467271/467272). The hard check
            # after build_env below makes that class of bug fatal.
            _dest = {"collision_look_ahead_m": "collision_look_ahead",
                     "goal_support_radius_m": "goal_support_radius"}
            args._adopted = {}
            for _k in ("step_size_m", "yaw_step_rad", "forward_only",
                       "look_ahead_dist", "goal_radius", "collision_threshold",
                       "collision_terminate_frac", "collision_terminate_penalty", "phantom_veto",
                       "action_chunk",
                       # 2026-09-02: training rejects goals with no
                       # reconstruction under them (14.5% of draws) and can
                       # judge collision on its own closer footprint. Eval did
                       # neither, so it was scoring a goal distribution
                       # training never sees.
                       "goal_support_radius_m", "collision_look_ahead_m", "collision_box_memory", "collision_box_memory_agg", "collision_box_memory_always",
                       "collision_at_next_pose", "look_ahead_auto", "footprint_next_heading", "crash_requires_motion",
                       # 2026-09-06: raster-observation arms; the policy must be
                       # shown the raster again or the eval is an obs-shift test
                       "raster_obs", "static_scene", "static_movers", "render_window", "coverage_window", "sem_palette",
                       "reward_source", "terrain_speed_scaled", "timeout_distance_scaled", "goal_dist", "spawn_heading_from_walk", "spawn_frames", "goal_case_mix", # 2026-09-24: eval planted the robot exactly on the recorded frame and accepted
                       # whatever was under it, while TRAINING jitters every spawn (20 deg / 0.4 m) and
                       # REDRAWS a spawn that starts in collision. Two consequences: episodes from the
                       # same spawn frame were byte-identical, so 16 episodes collapsed to 5 distinct
                       # situations; and on gnd the robot spawned on grass and crashed at step 1 in
                       # every episode. Adopting these makes eval sample the way training does.
                       "spawn_yaw_jitter", "spawn_lat_jitter", "spawn_support_tries",
                       "max_steps_base", "max_steps_per_m", "obs_frame_stack", "obs_frame_stride", "label_remap", "goal_center_clear_m",
                       "goal_nontrav_edge_m", "goal_nontrav_tries", "goal_nontrav_cone_deg", "goal_nontrav_classes", "goal_mix_map_draw", "refusal_bonus", "refusal_dist_m", "refusal_verge_m", "halt_wrong_penalty", "nontrav_goal_unreachable", "goal_requires_stop", "stop_action", "lawn_progress_to_verge",
                       # 2026-09-03: the ALPHA GATE. Training runs ungated;
                       # eval defaulted to gated, which turns low-coverage
                       # pixels into void -- and void leaves the collision
                       # fraction when void_cost > 0, so gated evals do not
                       # crash where training would. Every eval that day
                       # under-crashed for this reason.
                       "no_alpha_gate", "alpha_gate_tau",
                       # max_steps was NOT adopted: eval ran 60-step episodes
                       # against arms trained at 90, cutting every rollout a
                       # third short -- so "it never stopped at the boundary"
                       # could just mean the clock ran out (2026-09-02).
                       "max_steps", "goal_support_min_frac",
                       # reward, so a reported return means the same thing in
                       # both places (her ask 2026-09-02: "make eval like
                       # training please")
                       "semantic_weight", "goal_weight", "collision_weight",
                       "step_cost", "void_cost", "terrain_as_cost",
                       "spin_cost", "backward_cost", "action_smooth_cost",
                       "goal_bonus", "timeout_penalty", "proximity_weight",
                       "proximity_margin", "proximity_delta", "proximity_ground_weight", "proximity_ground_margin", "proximity_ground_classes", "reward_scale",
                       "coherence_cost_weight", "coherence_tau",
                       "coherence_terminate_tau"):
                if _k not in _tr:
                    continue
                _have = getattr(args, _k, None)
                if _k in _force:
                    # deliberate override, e.g. a near box + frame memory on a
                    # checkpoint trained with the far box (2026-09-04): the
                    # eval then measures the BOX, not the policy's training
                    print(f"[eval] FORCED {_k}: keeping eval {_have} over training {_tr[_k]}", flush=True)
                    continue
                if _have is not None and _have != _tr[_k]:
                    print(f"[eval] MISMATCH {_k}: eval {_have} -> training "
                          f"{_tr[_k]} (adopting training)", flush=True)
                setattr(args, _k, _tr[_k])
                if _k in _dest:
                    setattr(args, _dest[_k], _tr[_k])
                args._adopted[_k] = _tr[_k]
            print(f"[eval] adopted training env from {_ec}", flush=True)
        else:
            print(f"[eval] WARNING: no env_config.json at {_ec} — this run "
                  f"predates it, so kinematics come from the CLI and may not "
                  f"match what the policy trained with", flush=True)
        # Read OUTSIDE the env_config block. Nesting it there meant a run with no
        # env_config.json -- which on 2026-09-03 was EVERY run -- also lost its
        # curriculum range, the one number eval most needs.
        # env_config.json records the range the run STARTED at. With a
        # distance curriculum the policy finishes somewhere else, and how
        # far it got is not predictable from the config -- notches are
        # EARNED. curriculum_state.json is rewritten every rollout with the
        # live range, so it is the only honest answer to "what was this
        # policy actually training on when we stopped it".
        _cs = _run / "curriculum_state.json"
        if _cs.exists():
            try:
                _st = json.loads(_cs.read_text())
                _fin = _st.get("goal_dist_range")
                if _fin:
                    _fin_s = ",".join(str(v) for v in _fin)
                    if args.goal_dist_range != _fin_s:
                        print(f"[eval] curriculum FINISHED at "
                              f"{_fin_s} m (env_config says "
                              f"{args.goal_dist_range}, its START). "
                              f"Evaluating at the finished range.",
                              flush=True)
                    args.goal_dist_range = _fin_s
            except Exception as _e2:
                print(f"[eval] could not read {_cs}: {_e2}", flush=True)
        else:
            print(f"[eval] no curriculum_state.json — either no distance "
                  f"curriculum, or a run from before it was recorded; "
                  f"using the env_config range {args.goal_dist_range}",
                  flush=True)
    except Exception as _e:
        print(f"[eval] could not read training env config: {_e}", flush=True)

    print(f"[eval] kinematics: step {args.step_size_m} m, yaw "
          f"{args.yaw_step_rad} rad, forward_only={args.forward_only} "
          f"— these MUST match the training run", flush=True)

    # The policy CNN is size-locked, so an eval at the wrong resolution dies
    # with "Observation spaces do not match" — after queueing, waiting for a
    # node, and loading the 14B pipeline. The checkpoint already KNOWS its
    # resolution, so stop making a human retype it from the run name.
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        _d, _, _ = load_from_zip_file(args.checkpoint, device="cpu",
                                      print_system_info=False)
        _rgb = _d["observation_space"]["rgb"]            # channels-first
        _h, _w = int(_rgb.shape[1]), int(_rgb.shape[2])
        if (_h, _w) != (args.obs_height, args.obs_width):
            print(f"[eval] obs resolution from checkpoint: {_w}x{_h} "
                  f"(overriding {args.obs_width}x{args.obs_height})", flush=True)
        args.obs_height, args.obs_width = _h, _w
    except Exception as e:
        print(f"[eval] could not read obs resolution from checkpoint ({e}); "
              f"using {args.obs_width}x{args.obs_height}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inner_env = build_env(args)
    # HARD CHECK (2026-09-21 night): a pinned --goal_frame must reach the sampler as a FIXED goal.
    if getattr(args, 'goal_frame', None) is not None and not args.goal_xy:
        _bc = getattr(getattr(inner_env.unwrapped, 'world_backend', None), 'cfg', None)
        if _bc is None or getattr(_bc, 'goal_frame_range', None) is not None or int(getattr(_bc, 'goal_frame', -1)) != int(args.goal_frame):
            raise SystemExit(f"REFUSED: --goal_frame {args.goal_frame} but the backend has goal_frame={getattr(_bc, 'goal_frame', None)} "
                             f"goal_frame_range={getattr(_bc, 'goal_frame_range', None)}: the pinned goal would be ignored")
        print(f"[eval] FIXED GOAL confirmed in the backend: frame {args.goal_frame}, goal_frame_range=None", flush=True)
    # HARD CHECK: every value adopted from the training env_config must be
    # what the built env actually carries. An adopted key that build_env
    # does not read would otherwise pass silently (the 09-04 far-box evals).
    _cfg = getattr(inner_env.unwrapped, "cfg", None)
    _bad = []
    # max_steps is DELIBERATELY recomputed when the budget scales with the goal band
    # (40 + 16 x band). The adopted value is training's starting 90, and comparing the
    # two makes this check refuse its own intended override. 2026-09-23.
    _budget_scaled = float(getattr(args, "max_steps_per_m", None) or 0.0) > 0.0
    for _k, _v in (getattr(args, "_adopted", {}) or {}).items():
        if _k == "max_steps" and _budget_scaled:
            continue
        _have = getattr(_cfg, _k, None) if _cfg is not None else None
        if _have is None:
            continue
        try:
            _same = abs(float(_have) - float(_v)) < 1e-6
        except (TypeError, ValueError):
            _same = (_have == _v)
        if not _same:
            _bad.append(f"{_k}: env has {_have!r}, training had {_v!r}")
    if _bad:
        print("[eval] REFUSED: adopted training values did not reach the env:\n  "
              + "\n  ".join(_bad), flush=True)
        raise SystemExit(3)
    # raster_obs lives on the world backend, not on the env cfg: check it there.
    if "raster_obs" in (getattr(args, "_adopted", {}) or {}):
        _wb = getattr(inner_env.unwrapped, "world_backend", None)
        _wr = getattr(_wb, "raster_obs", None)
        if _wr is None or bool(_wr) != bool(args._adopted["raster_obs"]):
            print(f"[eval] REFUSED: training had raster_obs={args._adopted['raster_obs']} "
                  f"but the built backend has {_wr!r} (is this a LIVE=1 eval?)", flush=True)
            raise SystemExit(3)
    if "sem_palette" in (getattr(args, "_adopted", {}) or {}):
        _wb = getattr(inner_env.unwrapped, "world_backend", None)
        _wp = getattr(getattr(_wb, "cfg", None), "sem_palette_version", None)
        if _wp is None or int(_wp) != int(args._adopted["sem_palette"]):
            print(f"[eval] REFUSED: training used palette v{args._adopted['sem_palette']} "
                  f"but the built backend has v{_wp!r} (the generator's conditioning would be miscolored)", flush=True)
            raise SystemExit(3)
    if "static_scene" in (getattr(args, "_adopted", {}) or {}):
        _wb = getattr(inner_env.unwrapped, "world_backend", None)
        _ws = getattr(getattr(_wb, "cfg", None), "static_scene", None)
        if _ws is None or bool(_ws) != bool(args._adopted["static_scene"]):
            print(f"[eval] REFUSED: training had static_scene={args._adopted['static_scene']} "
                  f"but the built backend has {_ws!r}", flush=True)
            raise SystemExit(3)
    if "static_movers" in (getattr(args, "_adopted", {}) or {}):
        _wb = getattr(inner_env.unwrapped, "world_backend", None)
        _wm = str(getattr(getattr(_wb, "cfg", None), "static_movers", "") or "")
        if _wm != str(args._adopted["static_movers"] or ""):
            print(f"[eval] REFUSED: training had static_movers={args._adopted['static_movers']!r} "
                  f"but the built backend has {_wm!r}", flush=True)
            raise SystemExit(3)
    if getattr(args, "_adopted", None):
        print(f"[eval] verified {len(args._adopted)} adopted env values against the built env", flush=True)
    if args.blind:
        import gymnasium as gym

        class BlindObs(gym.ObservationWrapper):
            def observation(self, obs):
                obs = dict(obs)
                obs["rgb"] = np.zeros_like(obs["rgb"])
                return obs
        inner_env = BlindObs(inner_env)
    env = Monitor(inner_env)
    path_steps = []
    if args.expert == "path":
        # Scripted good/bad paths scored by the training reward (2026-11-09): no policy is
        # loaded; the env is the checkpoint's, adopted and hard-checked above.
        from src.env.path_expert import PathExpert
        model = PathExpert(env, scenes=[v for v in args.path_scenes.split(",") if v.strip()] or None,
                           n_pairs=args.path_pairs, kinds=tuple(k for k in args.path_kinds.split(",") if k.strip()),
                           seed=args.eval_seed)
        model.build_plan()
        args.episodes = len(model.plan)
        args.videos = 0
        (args.out_dir / "path_plan.json").write_text(json.dumps(model.plan))
        for _sc in model.scenes:      # the fused label grid the planner and the proximity term read, for the figure
            _g = getattr(env.unwrapped, "_label_grids", {}).get(_sc)
            if _g is not None:
                np.savez_compressed(args.out_dir / f"path_grid_{_sc}.npz", labels=_g.labels, x0=_g.x0, y0=_g.y0, res=_g.res,
                                    non_trav=np.asarray(env.unwrapped._non_trav),
                                    walk=np.asarray(getattr(env.unwrapped, "_walk_xy", {}).get(_sc), dtype=np.float32))
        print(f"[expert] scripted paths: {args.episodes} episodes (no policy loaded; env from {args.checkpoint})", flush=True)
        if args.path_plan_only:
            print(f"[expert] plan only: wrote {args.out_dir}/path_plan.json and path_grid_*.npz", flush=True)
            return
    elif args.expert == "map":
        # Scripted privileged driver instead of a policy: records demos and
        # doubles as an end-to-end check of the reward's stop rules.
        from src.env.map_expert import MapExpert
        model = MapExpert(env)
        print(f"[expert] map-driven scripted expert (no policy loaded; env from {args.checkpoint})", flush=True)
    else:
        model = PPO.load(args.checkpoint, env=env, device="cuda")
        print(f"loaded {args.checkpoint}", flush=True)
        # IMAGE CHECK (2026-09-22): report whether this policy reacts to its image; refuse a
        # post-fix checkpoint (env_config image_norm_fix) that is being evaluated with the double /255
        try:
            from src.policy.encoders import image_sensitivity
            _is = image_sensitivity(model)
            print(f"[eval] image check: normalize_images={_is['normalize_images']}  |action(random image) - action(black)| = {_is['delta']:.4f}"
                  f"  ({'uses the image' if _is['delta'] > 1e-3 else 'BLIND: image has no effect'})", flush=True)
            if bool((_tr if '_tr' in dir() else {}).get("image_norm_fix", False)) and _is["normalize_images"]:
                raise SystemExit("REFUSED: checkpoint trained with --image_norm_fix but loaded with normalize_images=True")
        except SystemExit:
            raise
        except Exception as _e:
            print(f"[eval] image check skipped: {_e}", flush=True)
    demo_obs, demo_goal, demo_act, demo_ep = [], [], [], []
    demo_kept = 0

    # Reuse the HUD rollout recorder from the training script.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from train_ppo_real import save_rollout_video

    results, video_records = [], []
    agree_rows = []   # per step: alpha, gen frac, map frac, gen class, map class, map void
    for ep in range(args.episodes):
        if ep < args.videos:
            rec = save_rollout_video(model, env,
                                     args.out_dir / f"episode_{ep}.mp4",
                                     seed=args.eval_seed * 10000 + ep,
                                     sem_palette=args.sem_palette)
            if rec:
                rec["video"] = f"episode_{ep}.mp4"
                video_records.append(rec)
            # save_rollout_video runs its own episode; count it via a fresh one below
        # Same seed as the video rollout above, so the video IS this episode,
        # and so a second policy evaluated at the same --eval_seed sees
        # identical spawns and goals. Unseeded, blind and sighted runs drew
        # different goals and could only be compared in aggregate.
        if args.expert == "path":
            obs, _ = env.reset(seed=args.eval_seed * 10000 + ep, options=model.options(ep))
            model.begin(ep)
            _pp = model.plan[ep]
            print(f"[path] ep {ep}: {_pp['scene']} pair {_pp['pair']} {_pp['kind']} "
                  f"start frame {_pp['start_frame']} goal {_pp['goal_xy']} (straight line crosses {_pp['cross_m']:.1f} m no-go)", flush=True)
            _ep_frames = {"rgb": [], "labels": [], "alpha": [], "pose": [], "K": [], "w2c": []}
        else:
            obs, _ = env.reset(seed=args.eval_seed * 10000 + ep)
        # The reward of step k is charged at the pose BEFORE action k; traj
        # records poses AFTER each action. Keep the spawn so step 1 is drawable.
        spawn_pose = _pose_xyyaw(env)
        goal = env.unwrapped._goal_world
        # 5 fields, matching every later row: [x, y, yaw, collision_frac,
        # dominant_class]. It used to be 4 here and 5 below, which made `traj`
        # ragged and np.array() on it raise (2026-09-03).
        traj = [_pose_xyyaw(env) + [0.0, -1]]
        done, steps, collided, total_r = False, 0, 0, 0.0
        ground_counts: dict = {}
        # 2026-09-02: a bare success=False conflated "crashed at step 8",
        # "walked to the boundary and held" and "never moved" -- three
        # completely different policies with identical summaries. Everything
        # below exists to separate them.
        d_start = float(getattr(env.unwrapped, "_initial_goal_dist", 0.0) or 0.0)
        comps = {k: 0.0 for k in EVAL_COMPONENTS}
        cov_sum, cov_n, trespass, min_dist = 0.0, 0, 0, float("inf")
        goal_trav = float("nan")
        at_verge, passed_goal = False, False
        phantom = missed = 0
        last_phantom = 0.0
        bm_hit, bm_miss = 0, 0
        ep_obs, ep_goal, ep_act = [], [], []
        while not done:
            # --stochastic samples the action exactly as PPO does during rollout collection
            # (mean + N(0, std)); the default is the deterministic mean. Training curves are
            # of the SAMPLED policy and its std never annealed from 1.0, so the two can differ
            # a lot (2026-09-25). This is the switch that measures how much.
            action, _ = model.predict(obs, deterministic=not bool(getattr(args, "stochastic", False)))
            if args.expert != "none":
                ep_obs.append(np.asarray(obs["rgb"], dtype=np.uint8).copy())
                ep_goal.append(np.asarray(obs["goal"], dtype=np.float32).copy())
                ep_act.append(np.asarray(action, dtype=np.float32).copy())
            if args.expert == "path" and args.path_frames:
                _pre_rgb = None if env.unwrapped._last_rgb is None else np.asarray(env.unwrapped._last_rgb, dtype=np.uint8).copy()
                _pre_pose = _pose_xyyaw(env)
            obs, r, term, trunc, info = env.step(action)
            total_r += float(r)
            if args.expert == "path":
                _u = env.unwrapped
                _row = {"ep": ep, "scene": model.cur["scene"], "pair": model.cur["pair"], "kind": model.cur["kind"],
                        "step": steps, "reward": float(r), "pose": _pose_xyyaw(env),
                        "coverage": float(info.get("coverage", float("nan"))),
                        "dominant_class": int(info.get("dominant_class_id", -1)),
                        "terms": {k: float(info.get(k, 0.0)) for k in EVAL_COMPONENTS}}
                path_steps.append(_row)
                if args.path_frames:
                    # the frame the reward of THIS step was computed on: the RGB before the move, and the
                    # labels + camera the env stored in its memory buffer for that frame (stored after the reward)
                    _fm = getattr(_u, "_frame_memory", None) or []
                    _sem, _K, _w2c = (_fm[-1] if _fm else (None, None, None))
                    _ep_frames["rgb"].append(_pre_rgb if _pre_rgb is not None else np.zeros((1, 1, 3), np.uint8))
                    _ep_frames["labels"].append(np.asarray(_sem, dtype=np.int16) if _sem is not None else np.zeros((1, 1), np.int16))
                    _ep_frames["K"].append(np.asarray(_K, dtype=np.float32) if _K is not None else np.eye(3, dtype=np.float32))
                    _ep_frames["w2c"].append(np.asarray(_w2c, dtype=np.float32) if _w2c is not None else np.eye(4, dtype=np.float32))
                    _a = getattr(_u, "_last_alpha", None)
                    _ep_frames["alpha"].append(np.asarray(_a, dtype=np.float16) if _a is not None else np.zeros(1, np.float16))
                    _ep_frames["pose"].append(_pre_pose)
            steps += 1
            for k in EVAL_COMPONENTS:
                v = float(info.get(k, 0.0))
                if v == v:
                    comps[k] += v
            _c = float(info.get("coverage", float("nan")))
            if _c == _c:
                cov_sum += _c
                cov_n += 1
            elif steps == 1 and ep == 0 and (float(getattr(args, "coherence_cost_weight", 0.0) or 0.0) > 0.0
                                             or float(getattr(args, "coherence_terminate_tau", 0.0) or 0.0) > 0.0):
                # 2026-09-04: every eval before today ran with the coherence
                # terms silently INERT because this path reported no alpha.
                # An eval that asks for coherence and gets no coverage is not
                # the eval that was requested: stop, do not print numbers.
                raise SystemExit("REFUSED: coherence requested but the backend reports no view alpha "
                                 "(coverage is NaN on the first step). The coherence cost and terminal "
                                 "would be inert and this eval would not match training.")
            min_dist = min(min_dist, float(info.get("dist_to_goal", min_dist)))
            # trespass = the footprint's dominant class is grass. This is the
            # number that says whether "did not reach the goal" means restraint
            # or failure, when the goal itself sits on grass.
            if int(info.get("dominant_class_id", -1)) == 3:
                trespass += 1
            # generator vs map reading of the footprint (logged when a cloud exists)
            phantom += int(float(info.get("phantom", 0.0)) > 0)
            bm_hit += int(float(info.get("box_memory_hit", 0.0)) > 0)
            bm_miss += int(float(info.get("box_memory_miss", 0.0)) > 0)
            missed += int(float(info.get("missed", 0.0)) > 0)
            last_phantom = float(info.get("phantom", 0.0))
            goal_trav = float(info.get("goal_traversable", float("nan")))
            _v = info.get("halt_at_verge", float("nan")); at_verge = (float(_v) > 0) if _v == _v else at_verge
            _pt = info.get("passed_through_goal", float("nan")); passed_goal = (float(_pt) > 0) if _pt == _pt else passed_goal
            if "map_collision_frac" in info:
                agree_rows.append((float(info.get("coverage", float("nan"))),
                                   float(info["gen_collision_frac"]), float(info["map_collision_frac"]),
                                   int(info.get("gen_dominant_class_id", -1)), int(info.get("map_dominant_class_id", -1)),
                                   float(info.get("map_void_frac", float("nan")))))
            # collision magnitude = footprint fraction on non-traversable
            # classes (eval reward weight is 1.0, so |term| = the fraction).
            frac = round(float(max(0.0, -info.get("collision", 0.0))), 3)
            hit = int(frac > 0.01)
            collided += hit
            # terrain occupancy: which class was under the footprint this step
            # (the sidewalk-preference metric, 2026-08-28)
            gc = int(info.get("dominant_class_id", -1))
            ground_counts[gc] = ground_counts.get(gc, 0) + 1
            traj.append(_pose_xyyaw(env) + [frac, gc])
            done = term or trunc
        d_final = float(info.get("dist_to_goal", -1))
        if args.expert == "path" and args.path_frames and _ep_frames["rgb"]:
            np.savez_compressed(args.out_dir / f"path_ep{ep:03d}_{model.cur['scene']}_p{model.cur['pair']}_{model.cur['kind']}.npz",
                                rgb=np.stack(_ep_frames["rgb"]), alpha=np.stack(_ep_frames["alpha"]),
                                labels=np.stack(_ep_frames["labels"]), K=np.stack(_ep_frames["K"]), w2c=np.stack(_ep_frames["w2c"]), pose=np.asarray(_ep_frames["pose"], dtype=np.float32),
                                path=np.asarray(model.cur["path"], dtype=np.float32), goal=np.asarray(model.cur["goal_xy"], dtype=np.float32))
        if term:
            outcome = "GOAL"
        elif float(info.get("crash", 0.0)) != 0.0:
            outcome = "CRASH"
        elif float(info.get("coherence_crash", 0.0)) != 0.0:
            outcome = "INCOHERENT"
        elif float(info.get("halted", 0.0)) != 0.0:
            # Stopped deliberately, somewhere safe, having closed distance.
            # Distinct from TIMEOUT on purpose: "walked to the boundary and
            # correctly refused an unreachable goal" and "wandered until the
            # clock ran out" were previously the same outcome, so the behaviour
            # this project is about had no metric.
            outcome = "HALTED"
        else:
            outcome = "TIMEOUT"
        results.append({"episode": ep, "success": bool(term),
                        "outcome": outcome,
                        "steps": steps, "return": round(total_r, 2),
                        "collision_steps": collided,
                        "d_start": round(d_start, 2),
                        "closed_frac": (round(1.0 - d_final / d_start, 3)
                                        if d_start > 1e-6 else None),
                        "min_dist": round(min_dist, 2),
                        "trespass_steps": trespass,
                        "phantom_steps": phantom, "missed_steps": missed,
                        "box_memory_hit": bm_hit, "box_memory_miss": bm_miss,
                        "goal_traversable": (None if goal_trav != goal_trav else (True if goal_trav >= 0.75 else (False if goal_trav <= 0.25 else "edge"))),
                        "halt_at_verge": bool(at_verge), "passed_through_goal": bool(passed_goal),
                        "goal_case": str(getattr(env.unwrapped, "_goal_case", "unknown")),
                        "goal_detour": float(getattr(env.unwrapped, "_goal_detour", float("nan"))),
                        "goal_min_width_m": float(getattr(env.unwrapped, "_goal_min_width", float("nan"))),
                        # the crash that ended this episode was one the map did not see
                        "crash_was_phantom": bool(outcome == "CRASH" and last_phantom > 0),
                        "mean_coverage": (round(cov_sum / cov_n, 3)
                                          if cov_n else None),
                        "reward_components": {k: round(v, 2)
                                              for k, v in comps.items()},
                        "final_dist": round(d_final, 2),
                        "goal_xy": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
                        "ground_class_counts": {str(k): v for k, v
                                                in sorted(ground_counts.items())},
                        "traj": traj, "spawn": spawn_pose})
        print(f"ep {ep:2d}: {outcome:<10} steps={steps:3d} "
              f"d {d_start:5.1f} -> {d_final:5.1f} m  closest {min_dist:5.1f}  "
              f"grass {trespass:3d}  return={total_r:+.1f}", flush=True)
        if args.expert != "none" and ep_act:
            # a demo is worth cloning when the expert's stop was ACCEPTED by
            # the reward: GOAL on a walkable goal, HALTED at the verge on lawn
            _good = (outcome == "GOAL") or (outcome == "HALTED" and at_verge)
            if _good or args.keep_failed_demos:
                demo_obs.extend(ep_obs); demo_goal.extend(ep_goal); demo_act.extend(ep_act)
                demo_ep.extend([ep] * len(ep_act)); demo_kept += 1
            print(f"    [expert] {'kept' if (_good or args.keep_failed_demos) else 'dropped'} "
                  f"{len(ep_act)} steps (outcome {outcome}, at_verge={at_verge}, "
                  f"goal_traversable={goal_trav})", flush=True)

    succ = [r for r in results if r["success"]]
    # OUTCOMES BY GOAL CASE (2026-09-07): the corner number is the paper's number
    _cases = {}
    for r in results:
        _cases.setdefault(r.get("goal_case", "unknown"), []).append(r)
    print("OUTCOMES BY GOAL CASE (open / corner / narrow from the map's straight-line rule; the rule is not perfect, judge with the sheets):", flush=True)
    for _k, _rs in sorted(_cases.items()):
        _n = len(_rs); _s = sum(1 for r in _rs if r["success"]); _c = sum(1 for r in _rs if r["outcome"] == "CRASH")
        _dt = [r["goal_detour"] for r in _rs if r.get("goal_detour", float("nan")) == r.get("goal_detour", float("nan")) and np.isfinite(r.get("goal_detour", float("nan")))]
        print(f"  {_k:14s} n={_n:3d}  GOAL {_s:3d} ({100.0 * _s / max(_n, 1):4.0f}%)  CRASH {_c:3d}  median detour {np.median(_dt) if _dt else float('nan'):.2f}", flush=True)
    if args.expert == "path":
        with open(args.out_dir / "path_steps.jsonl", "w") as _fh:
            for _row in path_steps:
                _fh.write(json.dumps(_row) + "\n")
        print(f"[path] wrote {len(path_steps)} step rows -> {args.out_dir / 'path_steps.jsonl'}", flush=True)
    # aggregate terrain occupancy across all episodes (share of ALL steps)
    agg: dict = {}
    for rr in results:
        for k, v in rr.get("ground_class_counts", {}).items():
            agg[k] = agg.get(k, 0) + v
    n_all = max(1, sum(agg.values()))
    class_names = {
        "0": "void", "1": "sky", "2": "trail", "3": "grass", "4": "rough",
        "5": "water", "6": "sidewalk", "7": "road", "8": "pavement",
        "9": "stairs", "10": "obstacle", "11": "vegetation", "12": "person",
        "13": "vehicle", "-1": "none"}
    ground_share = {class_names.get(k, k): round(v / n_all, 3)
                    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])}
    outcomes: dict = {}
    for rr in results:
        outcomes[rr["outcome"]] = outcomes.get(rr["outcome"], 0) + 1
    closed = [r["closed_frac"] for r in results if r["closed_frac"] is not None]
    covs = [r["mean_coverage"] for r in results if r["mean_coverage"] is not None]
    comp_mean = {k: round(float(np.mean([r["reward_components"][k]
                                         for r in results])), 2)
                 for k in EVAL_COMPONENTS}
    summary = {
        "checkpoint": str(args.checkpoint),
        "scene": args.scene,
        "episodes": args.episodes,
        # The BUDGET this cell ran under, and what sized it. Without these a reader
        # can only reverse-engineer the budget from a truncation step count -- which
        # is how chunk5's 104 and memDINO_warm's 160 on the same 8.1 m course went
        # unnoticed, and left both arms' curriculum reach a matter of inference
        # (2026-09-24, Joana: "is it also the case for the mem dino warm curriculum?").
        "max_steps": int(getattr(args, "resolved_max_steps", 0) or getattr(args, "max_steps", 0)),
        "curriculum_goal_dist_m": (round(float(args.goal_dist), 2)
                                   if getattr(args, "goal_dist", None) is not None else None),
        "test_goal_dist_m": (round(float(args.test_goal_dist), 2)
                             if getattr(args, "test_goal_dist", None) else None),
        "action_chunk": int(getattr(args, "action_chunk", 1) or 1),
        "deterministic": not bool(getattr(args, "stochastic", False)),
        "policy_std": (float(__import__("torch").exp(model.policy.log_std).mean().item())
                       if hasattr(model.policy, "log_std") else None),
        "outcomes": outcomes,
        "success_rate": round(len(succ) / args.episodes, 3),
        "mean_d_start": round(float(np.mean([r["d_start"] for r in results])), 2),
        "mean_d_final": round(float(np.mean([r["final_dist"] for r in results])), 2),
        "mean_closed_frac": round(float(np.mean(closed)), 3) if closed else None,
        "mean_min_dist": round(float(np.mean([r["min_dist"] for r in results])), 2),
        "trespass_rate": round(float(np.mean(
            [1.0 if r["trespass_steps"] > 0 else 0.0 for r in results])), 3),
        "mean_trespass_steps": round(float(np.mean(
            [r["trespass_steps"] for r in results])), 2),
        "mean_coverage": round(float(np.mean(covs)), 3) if covs else None,
        "mean_steps_to_goal": round(float(np.mean([r["steps"] for r in succ])), 1) if succ else None,
        "mean_return": round(float(np.mean([r["return"] for r in results])), 2),
        "mean_collision_steps": round(float(np.mean([r["collision_steps"] for r in results])), 2),
        "mean_phantom_steps": round(float(np.mean([r["phantom_steps"] for r in results])), 2),
        # near-box frame memory: steps read from an older frame vs steps that fell back to the far box
        "box_memory": "hit %d / miss %d steps" % (sum(r.get("box_memory_hit", 0) for r in results), sum(r.get("box_memory_miss", 0) for r in results)),
        "mean_missed_steps": round(float(np.mean([r["missed_steps"] for r in results])), 2),
        "crashes_that_were_phantoms": "%d/%d" % (sum(r["crash_was_phantom"] for r in results), sum(r["outcome"] == "CRASH" for r in results)),
        "reward_components_mean": comp_mean,
        "ground_share": ground_share,
    }
    # ---- refusal metric: what the policy did, by what ground the goal sat on (Joana, 2026-09-04) ----
    by_ground = {}
    for r in results:
        k = {True: "goal on traversable ground", False: "goal on NON-traversable ground", None: "goal ground unknown", "edge": "goal on the EDGE (25-75% of the disc walkable)"}[r.get("goal_traversable")]
        by_ground.setdefault(k, {})
        by_ground[k][r["outcome"]] = by_ground[k].get(r["outcome"], 0) + 1
    summary["outcomes_by_goal_ground"] = by_ground
    print("\n=== OUTCOMES BY GOAL GROUND (map) ===")
    for k, v in by_ground.items():
        n = sum(v.values())
        print("  %-34s n=%2d  " % (k, n) + "  ".join("%s %d" % (o, c) for o, c in sorted(v.items(), key=lambda kv: -kv[1])))
    _nt = [r for r in results if r.get("goal_traversable") is False]
    _nt_halt = [r for r in _nt if r["outcome"] == "HALTED"]
    _verge = sum(1 for r in _nt_halt if r.get("halt_at_verge"))
    _pt = sum(1 for r in results if r.get("passed_through_goal"))
    summary["refusals_at_verge"] = "%d/%d halts on non-traversable goals were at the verge (of %d such goals)" % (_verge, len(_nt_halt), len(_nt))
    summary["passed_through_goal"] = _pt
    print("  correct refusals = HALTED on non-traversable; freezes = HALTED on traversable; trespass arrivals = GOAL on non-traversable")
    print("  " + summary["refusals_at_verge"] + "; episodes that crossed a goal's radius without stopping: %d" % _pt)
    # ---- generator vs map, per step, as a function of alpha (Joana, 2026-09-04) ----
    if agree_rows:
        A = np.array(agree_rows, dtype=float)
        thr = max(float(getattr(args, "collision_terminate_frac", 0.35) or 0.35), 1e-9)
        cov, fg, fm, cg, cm, mv = A[:, 0], A[:, 1], A[:, 2], A[:, 3].astype(int), A[:, 4].astype(int), A[:, 5]
        inframe = np.isfinite(fg)          # image crash box below the frame -> no traversability comparison
        def stats(sel):
            n = int(sel.sum())
            if n == 0:
                return None
            sel_t = sel & inframe
            gen_bad, map_bad = fg[sel_t] >= thr, fm[sel_t] >= thr
            mvs = mv[sel]
            return {"steps": n,
                    "semantic_agree": round(float((cg[sel] == cm[sel]).mean()), 3),
                    "steps_with_image_box_in_frame": int(sel_t.sum()),
                    "traversability_agree": (round(float((gen_bad == map_bad).mean()), 3) if sel_t.any() else None),
                    "phantom": (round(float((gen_bad & ~map_bad).mean()), 3) if sel_t.any() else None),
                    "missed": (round(float((~gen_bad & map_bad).mean()), 3) if sel_t.any() else None),
                    "mean_abs_frac_gap": (round(float(np.abs(fg[sel_t] - fm[sel_t]).mean()), 3) if sel_t.any() else None),
                    "map_void_share": (round(float(np.nanmean(mvs)), 3) if np.isfinite(mvs).any() else None)}
        by_alpha = {}
        for lo, hi in ((0.0, 0.1), (0.1, 0.4), (0.4, 0.7), (0.7, 1.01)):
            st = stats((cov >= lo) & (cov < hi))
            if st:
                by_alpha["%.1f-%.1f" % (lo, min(hi, 1.0))] = st
        st_all = stats(np.ones(len(A), dtype=bool))
        dis = cg != cm
        conf = {}
        for a_, b_ in zip(cg[dis], cm[dis]):
            k = "gen %s / map %s" % (class_names.get(str(int(a_)), str(int(a_))), class_names.get(str(int(b_)), str(int(b_))))
            conf[k] = conf.get(k, 0) + 1
        top = sorted(conf.items(), key=lambda kv: -kv[1])[:8]
        summary["footprint_agreement"] = {"all": st_all, "by_alpha": by_alpha, "top_confusions": top}
        print("\n=== FOOTPRINT: generator vs map, per step, by alpha ===")
        print("  %9s %6s %10s %11s %8s %7s %6s %9s" % ("alpha", "steps", "sem agree", "trav agree", "phantom", "missed", "|gap|", "map void"))
        for name, st in list(by_alpha.items()) + [("all", st_all)]:
            f = lambda v, k=100: (k * v) if v is not None else float("nan")
            print("  %9s %6d %9.1f%% %10.1f%% %7.1f%% %6.1f%% %6.2f %8.1f%%   (image box in frame on %d steps)" % (
                name, st["steps"], f(st["semantic_agree"]), f(st["traversability_agree"]),
                f(st["phantom"]), f(st["missed"]), f(st["mean_abs_frac_gap"], 1), f(st["map_void_share"]),
                st["steps_with_image_box_in_frame"]))
        if top:
            print("  where they disagree, generator / map: " + ", ".join("%s x%d" % (k, v) for k, v in top))
    if args.expert != "none":
        _dp = args.demos_out or (args.out_dir / "demos.npz")
        if demo_act:
            np.savez_compressed(_dp, obs=np.stack(demo_obs), goal=np.stack(demo_goal),
                                act=np.stack(demo_act), episode=np.asarray(demo_ep, dtype=np.int32))
            print(f"[expert] demos: {len(demo_act)} steps from {demo_kept}/{len(results)} episodes "
                  f"-> {_dp}  (act dim {np.stack(demo_act).shape[1]}, obs {np.stack(demo_obs).shape[1:]})", flush=True)
        else:
            print(f"[expert] NO demos kept: the expert's stops were never accepted by the reward "
                  f"({len(results)} episodes) -- read the outcomes above before blaming BC", flush=True)
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "episodes": results,
                   "video_episodes": video_records}, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
