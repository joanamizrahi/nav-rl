"""SceneEnv — Gym environment wrapping the NeoVerse world model as an RL simulator.

Interface (Gymnasium API):
  reset() -> (observation, info)
  step(action) -> (observation, reward, terminated, truncated, info)

Observation:
  Dict({
    "rgb":  Box(0, 255, (H, W, 3), uint8)   -- the current camera view
    "goal": Box(-inf, inf, (3,), float32)   -- (dx, dy, dyaw) to goal, robot frame
  })

Action:
  Box(-1, 1, (2,), float32) = (v_forward, omega_yaw), each in [-1, 1].
  Scaled inside step() by config.step_size and config.yaw_step_rad.

This file is deliberately thin. The two things that actually do work live in
their own modules and are injected here:
  - `world_backend`  — turns (scene, pose) into an RGB image + camera intrinsics/extrinsics.
                       Real backend calls NeoVerse; the mock backend returns cached images.
  - `semantic_backend` — turns an RGB image into a per-pixel class-id map.
                         Default = SAM3-on-RGB. Later = the fine-tuned diffusion output.

Nothing in this file is NeoVerse-specific, which is what lets us develop the
env, the reward, and the PPO loop entirely on the Mac before cluster access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # gym is a soft dep at import time so this file can be read anywhere
    gym = None
    spaces = None

from .reward import SemanticBackend                             # keep the Protocol only

from ..eval.reward_2d import (
    RewardWeights, RewardBreakdown, compute_reward, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)
from ..eval.traversability import load_traversability, NUM_CLASSES


# ---------------------------------------------------------------------------
# World-model backend abstraction
# ---------------------------------------------------------------------------

class WorldBackend(Protocol):
    """Anything that can render an RGB view at a requested pose."""
    H: int
    W: int

    def load_scene(self, scene_id: str) -> None:
        """Prepare a scene (may build a pose cache, load Gaussians, etc.)."""
        ...

    def render(self, pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render at the given robot pose in world frame.

        Args:
            pose_world: (4, 4) camera-to-world matrix.
        Returns:
            rgb:  (H, W, 3) uint8
            K:    (3, 3)    float32 — camera intrinsics for this render
            w2c:  (4, 4)    float32 — world->camera for this render
        """
        ...

    def start_pose(self, scene_id: str) -> np.ndarray:
        """Where the robot starts in this scene, as a (4, 4) pose."""
        ...

    def goal_position(self, scene_id: str) -> np.ndarray:
        """Where the robot should reach. (3,) world coords."""
        ...


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

_ENV_SEQ = 0    # per-process env counter, tags crash snapshots


@dataclass
class SceneEnvConfig:
    max_steps: int = 100
    # 2026-09-23: max_steps was pinned at 90 when goals were 2-4 m. The curriculum
    # now runs to 10 m, and at the cruise these policies actually command (~0.07
    # m/step measured) 90 steps buys ~6.3 m -- so from ~7 m upward the goal is
    # unreachable no matter what the policy does, the goal bonus stops arriving,
    # and the cheapest remaining behaviour is to slow down. Which is exactly what
    # A_dino did: 1.92 m of travel in 90 steps regardless of goal distance.
    # With per_m > 0 the budget grows with the band: steps = base + per_m * d.
    # 16 steps/m keeps a 1.4x margin over the measured 0.070 m/step cruise, which is
    # what a corner DETOUR needs; 1.15x only covers a straight run.
    max_steps_base: int = 40
    max_steps_per_m: float = 0.0        # 0 disables; 16.0 is the calibrated value
    step_size_m: float = 0.3            # meters per unit forward action
    yaw_step_rad: float = 0.5           # radians per unit yaw action
    reward: RewardWeights = field(default_factory=RewardWeights)
    look_ahead_dist: float = 1.5        # meters; passed to the reward
    # 2026-09-02: separate the LETHAL test from the GRADED one. 0.0 = keep the
    # old single-box behaviour (collision judged at look_ahead_dist), so the
    # control arm is bit-identical to every run before today. Set ~0.8-1.0 to
    # move only crash/collision in to the body while the graded semantic score
    # keeps its 1.5 m of warning. Watch reward/collision_off_frame: if the near
    # box misses the frame it falls back and the split is not in effect.
    collision_look_ahead_m: float = 0.0
    # 2026-09-07 (new scope): judge the CRASH on the body footprint at the pose
    # the action leads to (yaw then step), on the MAP. "Where the policy would
    # bring us in the next step" (Joana). A stop cannot crash; a turn changes
    # the footprint. Replaces the fixed near box when on.
    collision_at_next_pose: bool = False
    # 2026-09-07: place the graded (visible) box at the nearest ground the
    # camera can see, per scene: camera_height * fy / (H - cy) + pad, clamped.
    # The scenes' focal lengths differ (bottom of image = 1.0-2.0 m ahead).
    look_ahead_auto: bool = False
    # 2026-09-07: orient the visible box by the heading AFTER the commanded
    # turn, so the box is "the ground this action takes us toward"; and only
    # let that box CRASH when the robot is actually moving (a stop facing
    # grass is not a collision). Both for the GENERATED-label crash design.
    footprint_next_heading: bool = False
    crash_requires_motion: bool = False
    crash_motion_eps: float = 0.15
    look_ahead_auto_pad_m: float = 0.3
    look_ahead_auto_min_m: float = 1.0
    look_ahead_auto_max_m: float = 2.5
    # Near-box memory (Joana, 2026-09-04): keep the last N generated frames
    # (labels + camera) and read the near box from the newest stored frame that
    # contains it whole, since the near box is below the camera in the current
    # view. 0 = off (far-box fallback, the pre-09-04 behaviour). 12 covers
    # throttle >= ~0.3 at 0.3 m/step (the box needs ~0.9 m of travel to enter
    # an old view). Image path only; the map reads the near box directly.
    collision_box_memory: int = 0
    collision_box_memory_agg: str = "newest"   # newest | mean (2026-09-16: average the box over all stored frames that contain it)
    collision_box_memory_always: bool = False  # 2026-09-20: average the current frame WITH the stored frames even when the box is visible now
    # 2026-09-02, measured: 14.5% of sampled goals have NO reconstruction under
    # them at all (scripts/goal_audit.py, 2000 episodes x 6 scenes). The goal
    # sampler is pure geometry -- spawn + d*(cos th, sin th) -- and never
    # consults the cloud, so the robot is regularly sent to a place that does
    # not exist in the world model: it cannot see the goal, cannot reach it,
    # and eats the timeout. Those episodes are pure noise in the return.
    # > 0 = resample until the goal has ground points within this radius.
    goal_support_radius_m: float = 0.0
    # 2026-09-16 (Joana): goals must not land on mulch beds. Walkable for GOAL
    # sampling = traversability score >= this; 0 = the crash threshold as before
    # (trail at 0.35 would count). 0.5 keeps goals on sidewalk / pavement / road.
    goal_min_score: float = 0.0
    # 2026-09-17 (Joana): path goals sit exactly on the walked line; this moves
    # each goal sideways by up to this many metres, then checks the goal disc on
    # the map (walkable share >= goal_lat_jitter_min_share) and retries, so the
    # goal keeps the variety of the cone without leaving walkable ground.
    goal_lat_jitter_m: float = 0.0
    goal_lat_jitter_min_share: float = 0.6
    goal_lat_jitter_tries: int = 8
    goal_support_tries: int = 12
    # "Enough reconstruction" cannot be "at least one point" -- a single stray
    # gaussian passes that and the goal is still in a void. Calibrate against
    # the scene itself: count ground points within the radius at every RECORDED
    # path position (places we know are well reconstructed, because someone
    # walked them and filmed it), take the median, and require the goal to
    # reach this fraction of it. Self-scaling across scenes of different
    # density. 0 = fall back to the any-point test.
    goal_support_min_frac: float = 0.25
    # 2026-09-04 (Joana): the goal MIX. Probability that an episode's goal
    # sits on traversable ground by the map (edge goals count as
    # traversable: the robot can stand on the walkable part). 0 = off, the
    # sampler's natural mix (a lawn scene at 8 m is ~all grass, a corridor
    # scene ~none). The sampler only; the policy never sees it.
    goal_traversable_mix: float = 0.0
    goal_mix_tries: int = 12
    # Tries of the map-direct LAWN draw (0 = goal_mix_tries). 12 silently
    # delivered 15% lawn goals instead of 25% on the six-scene set and 5% on
    # AUw360 alone (469883/469884, 2026-09-06): most random lawn cells fail
    # the arrival-disc tests, and a failed draw becomes a walkable goal.
    goal_nontrav_tries: int = 0
    # Cone of the map-direct LAWN draw in degrees (0 = the backend's goal
    # cone). Funnel 2026-09-06: with the 50-degree cone most walk frames
    # have lawn cells in the 4-8 m window but NONE inside the cone (lawn is
    # beside the walk, not ahead), so the draw fails and the episode gets a
    # walkable goal: 15% lawn goals instead of 25%, 5% on AUw360.
    goal_nontrav_cone_deg: float = 0.0
    # 2026-09-07 (new scope): label every episode's goal by the straight-line
    # rule (open / corner / narrow / blocked) on the map, and optionally DRAW
    # goals to a target mix, e.g. "corner:0.5,narrow:0.25,open:0.25":
    # redraw (window + cone) until the drawn goal's class matches the wanted
    # one, at most goal_case_tries times, keeping the last walkable draw.
    # Every draw must be walkable (disc share >= goal_case_walkable_min).
    # MULTIPLE IMAGES (2026-09-22, Jing's third arm): the observation is the last K rendered
    # views concatenated on the channel axis, oldest first. 1 = the current frame only.
    obs_frame_stack: int = 1
    # 2026-09-22 (Joana): three consecutive steps span only ~0.5 m. A stride keeps the same
    # encoder cost but reaches further back: stack 3 stride 3 covers 1.5 m of travel.
    obs_frame_stride: int = 1
    goal_case_mix: str = ""
    goal_case_tries: int = 24
    goal_case_walkable_min: float = 0.5
    # 2026-09-07 (Joana: edge goals with a smaller radius): reject a goal when
    # ANY non-walkable map cell lies within this distance of its CENTRE, so a
    # goal cannot sit against a wall or half on grass. 0 = off (every arm so
    # far). Set it to the arrival radius before any radius reduction.
    goal_center_clear_m: float = 0.0
    # MIRROR AUGMENTATION (2026-09-08). The five training scenes carry 158
    # right-turn corner goals against 52 left-turn ones (three of five have
    # essentially no left turns), so a policy learns "at a corner, turn
    # right". With probability mirror_prob an episode is presented mirrored:
    # the image is flipped left-right and the goal's lateral offset and
    # bearing are negated, and the policy's yaw command is negated on the way
    # back in. The world, the map and the reward are untouched -- only what
    # the policy sees and commands is reflected -- so a right turn in the
    # world is a left turn for the policy exactly half the time.
    mirror_prob: float = 0.0
    goal_case_narrow_m: float = 1.0
    goal_case_detour_min: float = 1.15
    # Map-direct draw for the NON-traversable share of the mix (2026-09-04
    # night). Rejection sampling from the walk sampler found no lawn goal in
    # 12 draws on short windows and on scenes without lawn beside the walk,
    # so gen 3 trained at 87-96% traversable and never saw enough refusable
    # goals to learn a halt. With this on, a non-traversable goal is drawn
    # from map cells of goal_nontrav_classes inside the same distance window
    # and cone, then support-checked like any goal.
    goal_mix_map_draw: bool = False
    goal_nontrav_classes: str = "3,4,5"   # grass, rough, water: ground you should refuse, not walls
    # A lawn goal must lie within this distance of the RECORDED WALK, so the
    # robot can come within refusal_dist_m of it along pavement (09-05: any
    # walkable cell was not enough -- the far side of a lawn is walkable too).
    # Without it the map-direct draw picks cells deep inside lawns (trend
    # read 2026-09-05): those goals can only end in a crash, and a refusal
    # bonus 2 m from them is unreachable. 0 = no constraint.
    # 0 = off (2026-09-05 evening): with the verge as the stopping place a lawn
    # goal anywhere in the window has a reachable verge; the window, cone and
    # known-share tests bound it.
    goal_nontrav_edge_m: float = 0.0
    # minimum reconstructed share of a lawn goal's arrival disc; 0.5 = the share
    # test's own floor (Joana 09-05: undecided whether void-edge lawn goals are wanted)
    goal_nontrav_known_min: float = 0.5
    # Spawn support: redraw a spawn whose crash box is already at/above the
    # crash threshold on the MAP, up to this many times (0 = off). Such a
    # spawn ends at step 1 whatever the policy does and teaches nothing:
    # far box 26% of spawns on gnd_AU_180, near box 9% (spawn_doom.py,
    # 2026-09-04). The whole spawn is redrawn (frame + jitter), so survivors
    # stay jittered. Needs a map (clouds_dir).
    spawn_support_tries: int = 0
    goal_radius: float = 0.5            # meters; within this counts as "reached goal"
    collision_threshold: float = 0.1    # class score at/below which counts as collision
    spin_cost: float = 0.0              # shaping v2: penalty * |yaw action| (taxes circling)
    backward_cost: float = 0.0          # penalty * max(0, -v): backing to the goal is
                                        # reward-rational (goals spawn behind, footprint
                                        # projects ahead) but camera-blind on the robot
    goal_bonus: float = 0.0             # v4: one-time reward on reaching the goal — must
                                        # exceed what an episode could farm (~+35 in v3)
    trav_path: "str | None" = None      # traversability yaml override (v14 table for cached runs)
    random_spawn: bool = False          # shaping v2: spawn along the real trajectory if the
                                        # backend offers sample_start_pose(scene_id, rng)
    failure_snap_dir: "str | None" = None  # save a figure at each collision
                                        # (obs + semantics + reward numbers)
    failure_snap_max: int = 200         # cap so long runs don't fill the disk
    # 2026-09-27: write the memory strip every N steps while the box is in stored frames
    # (not only at a crash), for the figure that shows one ground patch seen from
    # successive frames. 0 = off. Same file format as the crash strip, prefix memory_.
    memory_snap_every: int = 0
    failure_snap_min_frac: float = 0.2  # only snapshot substantial overlaps —
                                        # tiny footprint grazes are risk, not collision
    phantom_veto: bool = False               # 2026-09-19: no crash on ground the fused map knows to be walkable
    collision_terminate_frac: float = 0.0    # >0: footprint overlap at/above this
                                        # ENDS the episode (no goal bonus) — see step().
    collision_terminate_penalty: float = 20.0  # subtracted on crash
    # Void termination: >0 ends the episode when this fraction of the footprint
    # has NO gaussian support (alpha-gated to class 0). Guards the blind-spot
    # exploit that alpha-gating would otherwise open. 0 = off.
    void_terminate_frac: float = 0.0
    void_terminate_penalty: float = 100.0     # ~1/10 of a crash at RW5 scale
    # IMAGE-coverage version (her spec 2026-09-01): the footprint measure above
    # asks "is the ground under me observed"; this asks "does the WHOLE VIEW
    # have enough gaussian support for the world model to paint something
    # coherent". With the alpha gate on, (semantic_image == 0).mean() IS
    # 1 - alpha coverage — the same `cov` number the spin certificates print.
    # Her reasoning: stepping onto locally-unsupported ground is fine as long
    # as the view around it is well supported. 0 = off.
    image_void_terminate_frac: float = 0.0
    # ---- COHERENCE (2026-09-01, her design) --------------------------------
    # `coverage` = MEAN ALPHA over the frame: how much of the diffusion's
    # conditioning input has real gaussian support behind it. Pure rasterizer
    # geometry, no diffusion in it, and NOT the same statistic as the void
    # fraction the gate produces.
    #
    # Her claim, which the panels support: coherence is a WHOLE-FRAME property.
    # A frame is either the world or it is not; there is no half-trustworthy
    # frame. So the decision belongs at frame level, not per pixel.
    #
    # Two thresholds, two jobs:
    #   coherence_tau (0.4, read off the ladder by eye)  -> GRADED COST. The
    #     render is degrading; pull the policy back, do not kill it. Measured:
    #     the wander regime averages 0.397, the corridor 0.645 — terminating
    #     at 0.4 would end most exploratory episodes and teach path-hugging,
    #     which is the failure the spawn jitter exists to prevent.
    #   coherence_terminate_tau (~0.1)                   -> TERMINATE. Frames
    #     with essentially no geometry anywhere (GEOvoid = 1.000 was observed).
    #     Nothing there can be scored and nothing can be learned from it.
    coherence_cost_weight: float = 0.0      # 0 = off (every run before today)
    coherence_tau: float = 0.4
    coherence_terminate_tau: float = 0.0    # 0 = off
    coherence_terminate_penalty: float = 100.0
    disagree_cost_weight: float = 0.0        # 2026-09-25: per-step cost when generated labels and the map disagree under the footprint (0 = off)
    disagree_frac: "float | None" = None     # footprint fraction one source must read as non-traversable (and the other not) to count as a disagreement; None = COLLTERM
    rgb_align_cost_weight: float = 0.0       # 2026-09-25: per-step cost when the generated RGB drifts from the Gaussian raster where alpha is high (0 = off)
    rgb_align_tau: float = 0.8               # similarity below this is charged: w * (tau - sim)
    rgb_align_min_alpha: float = 0.5         # pixels compared only where the raster alpha is at least this
    rgb_align_min_frac: float = 0.2          # ...and only if that many pixels qualify; otherwise no reference, no charge
    # Proximity cost (2026-08-24, after Run A): charge for being NEAR obstacles,
    # measured against the STATIC reconstructed geometry (scene cloud from
    # dump_scene_cloud.py) — the diffusion cannot dream this away, unlike the
    # semantic footprint (Run A scored 0.0 semantic collisions on the tree test
    # at 0.10 m median true clearance: live renders diverge at grazing range).
    proximity_weight: float = 0.0       # 0 = off (every run before this date)
    proximity_margin: float = 1.0       # meters; cost ramps linearly inside this
    # v2 (2026-08-25): charge only SOLID classes by default. v1 included
    # vegetation (11) — trail corridors are walled with it, so the charge was
    # always-on everywhere: a flat tax the policy paid instead of a gradient
    # it followed (measured: solid clearance 0.06 m vs Run C's 0.45 m).
    proximity_classes: tuple = (10, 13)  # obstacle, vehicle
    # Potential-shaped proximity (2026-08-27, advisor reward spec): charge
    # weight*(d_{t-1}-d_t) while within proximity_margin of an obstacle —
    # approaching costs, retreating refunds, standing is free.
    proximity_delta: bool = False
    # GROUND proximity (2026-09-06, Joana): the same potential-shaped term
    # against the map's GRASS points (ground-height points of these classes).
    # Approaching grass inside the margin costs weight per metre, retreating
    # refunds it, standing is free; the potential is clipped at the margin so
    # only the part of a move inside the margin counts. With the verge
    # progress term pulling toward the edge, a weight above the goal weight
    # makes the two potentials balance about one margin short of the grass:
    # a standoff inside the verge radius where still steps come for free.
    proximity_ground_weight: float = 0.0    # 0 = off
    proximity_ground_margin: float = 1.2    # metres from the body centre
    proximity_ground_classes: str = "3,4,5"
    # Timeout penalty (2026-08-27, advisor reward spec): subtracted when the
    # episode hits max_steps without reaching the goal. Makes freezing/stalling
    # strictly worse than trying — the missing counterweight to termination
    # penalties (the -20 crash penalty froze the policy when standing was free).
    # HALTED-SAFELY terminal (2026-09-03, Joana). A goal on grass can only end
    # in a timeout, which is the SAME outcome as freezing or wandering off --
    # so "walked to the boundary and correctly stopped" has no name, no
    # terminal, and no metric, and the robot keeps paying the per-step terrain
    # cost for every step it waits (up to -5/step, because the footprint 1.5 m
    # ahead of a stopped robot is still looking at the grass).
    #
    # Fires when the policy has stopped (throttle < eps for N consecutive
    # steps), is stopped somewhere SAFE (collision below the crash threshold),
    # and has actually made progress (d < d_start). It pays EXACTLY the
    # distance-scaled timeout it would have received anyway, so it cannot be
    # farmed: the only thing it removes is the drip.
    #
    # The progress condition is load-bearing. Without it, halting at spawn pays
    # the same -100 as timing out but skips ~85 steps of living cost and
    # terrain cost, which is a reward-sanctioned freeze.
    halt_terminate_steps: int = 0      # 0 = off
    halt_throttle_eps: float = 0.05
    # 2026-09-03: a HALTED episode pays the distance-scaled timeout times
    # this. 1.0 = same as timing out (the tie that made stopping no better
    # than gambling on the goal). Below 1, refusing is cheaper than waiting
    # out the clock and far cheaper than crashing.
    halt_penalty_scale: float = 1.0
    # Refusal bonus (2026-09-05): a HALT on a NON-traversable goal (map disc
    # <= 25% walkable) within refusal_dist_m of it earns this instead of the
    # halt price. Gen 3b at 10 h: no arm told lawn goals from hard goals,
    # because a correct refusal earned nothing (it only dodged the crash)
    # while a wrong halt cost 0.3 x timeout -> halts became the give-up.
    # 0 = off. Wrong halts keep paying halt_penalty_scale x timeout.
    refusal_bonus: float = 0.0
    refusal_dist_m: float = 2.0
    # radius around the VERGE point (walk point nearest a lawn goal) within
    # which a halt counts; tight, so the robot must arrive at the verge for
    # deep goals (Joana: 2.5 m around a walk point is a 5 m stretch of walk).
    # 1.5 (09-05 evening): the halt needs the 0.3-0.9 m box clear, so a robot
    # facing the lawn stops ~0.9 m short of the edge cell; 1.0 left 10 cm.
    refusal_verge_m: float = 1.5
    # Mirror of the bonus: a halt on a TRAVERSABLE goal pays this flat penalty
    # (on top of the halt price). Without it a wrong halt 2 m from a pavement
    # goal costs -0.3 (scaled) against -10 for trying at a 50% crash rate, so
    # a crash-prone policy stops near every goal (Joana's objection, 09-05).
    halt_wrong_penalty: float = 0.0
    # A NON-traversable goal cannot be 'reached' (Joana, 2026-09-05): with the
    # radius curriculum starting at 1.0 m and lawn goals within 1.5 m of the
    # walk, the robot collected the goal bonus from the pavement -- rewarded
    # for walking up to lawn goals, the opposite of refusing. With this on,
    # entering the radius of a lawn goal ends nothing and pays nothing (it is
    # counted as reach_on_nontrav); only a halt can earn on such a goal.
    nontrav_goal_unreachable: bool = False
    # Reaching a goal requires STOPPING inside its radius (Joana, 2026-09-05:
    # 'if the episode ends how does the robot know to stay at the goal?').
    # Entering the radius while moving ends nothing; the goal bonus is paid
    # when the robot has been still (throttle < halt_throttle_eps) for
    # halt_terminate_steps (3 if halting is off) inside the radius. Every
    # successful episode then practises the stop that refusal also needs.
    goal_requires_stop: bool = False
    # STOP ACTION (2026-09-06): a third action output; > 0 = halt now (still
    # subject to the box being clear and distance closed). A fresh Gaussian
    # policy fires it ~half the time, so the bonus and penalty shape hundreds
    # of halts in the first hour instead of waiting for three still steps to
    # happen by chance (8 h of bonus arms: halt_at_verge 0 everywhere).
    # Changes the action space -> cold start only.
    stop_action: bool = False
    # LAWN PROGRESS TO THE VERGE (2026-09-06): on a lawn-goal episode the
    # goal-progress term measures distance to the verge nearest the goal, not
    # to the goal, so the reward pulls the robot to where a halt pays instead
    # of onto the grass. The policy still sees the true goal vector.
    lawn_progress_to_verge: bool = False
    # REFUSAL DWELL (Joana, 2026-09-06): on a lawn-goal episode, being inside
    # the verge zone for this many CONSECUTIVE STILL steps (stop action counts)
    # ends the episode as a refusal at the verge. -1 = off (the generic halt
    # rule decides). 0 = arrival alone pays; a curriculum raises it 0 -> 1 ->
    # 2 -> 3, each stage earned, so the event is discoverable at every stage.
    refusal_dwell_steps: int = -1
    # 2026-09-03: charge the per-step terrain cost (semantic + collision)
    # in proportion to |throttle|. Driving onto bad ground costs; standing
    # still facing it does not. Implemented as a REFUND on top of the
    # unscaled terms so the crash / halt tests, which read
    # breakdown.collision, are untouched and the logged components stay
    # comparable across arms.
    terrain_speed_scaled: bool = False
    timeout_penalty: float = 0.0
    # PROPORTIONAL TIMEOUT (2026-09-02, her call: "proportional timeout seems
    # genius"). A flat timeout penalty charges the same for stopping 1 m short
    # as for stopping 10 m short, so there is NO gradient toward "as close as
    # the terrain legally allows" — and when the goal sits on grass, stopping
    # short is the CORRECT behaviour being punished at full price.
    #
    # Scaling it by remaining/initial distance makes the reward's optimum
    # "approach as close as legally possible and hold", without anyone having
    # to define where the boundary is — which is unanswerable anyway, since
    # any definition rests on cloud labels that are ~17% wrong.
    #
    # Note the goal term already telescopes to goal_weight*(d_start - d_final),
    # so only the ENDING distance matters; this fixes the counterweight that
    # was flat. It also prices drifting back out after approaching (warm went
    # 1.14 m -> 1.89 m and paid nothing for it).
    timeout_distance_scaled: bool = False
    # uniform reward multiplier applied at the very end of step() — tames the
    # critic's value targets under +-1000-scale terminals; 1.0 = off.
    reward_scale: float = 1.0
    # Render-high / observe-small (2026-08-30, her design after the w270
    # resolution ladder: 336-render is mush on campus scenes, 560 is coherent).
    # When set to (h, w): the WORLD keeps the backend's native render size for
    # reward/footprint/labels, and only the policy-facing obs["rgb"] is
    # downsampled to (h, w). None = obs at render size (every run before this).
    obs_out_hw: "tuple | None" = None
    # Anti-odometry lever (2026-08-31): Gaussian noise (std in meters) added to
    # the goal-vector observation's xy each step. A perfect goal vector makes
    # vision optional for the dominant reward terms — a noisy compass forces
    # visual navigation. 0 = off (every run before this). Reward/termination
    # still use the TRUE goal; only the policy's observation is corrupted.
    goal_noise_std: float = 0.0
    clouds_dir: "str | None" = None     # dir holding <scene>_cloud.npz files
    # 2026-09-03: where the reward reads its labels. "generated" = the
    # co-generated semantic image (every run before this date); "map" = a
    # top-down label grid built from the scene cloud (src/eval/reward_map.py),
    # deterministic and view-consistent, no phantom obstacles;
    # "map_then_generated" = the map where it has support, the image where
    # the footprint is mostly void.
    reward_source: str = "generated"
    # 2026-09-04: compute the MAP reading of the footprint on every arm that
    # has a cloud, even when the reward reads the image, and log where the
    # two disagree: diag/phantom = generator says crash, map says clear;
    # diag/missed = map says crash, generator says clear; diag/label_agree.
    map_diagnostics: bool = True
    map_res_m: float = 0.1
    # map_then_generated: fall back to the generated labels only when the
    # map has nothing under the footprint (void share >= this) AND the view
    # is still well supported (mean alpha >= this), i.e. the generator is
    # still grounded. Off the reconstruction with low alpha the map's
    # 'unknown' stands and the coherence machinery does its job.
    map_fallback_void_frac: float = 0.5
    map_fallback_min_alpha: float = 0.4
    map_inflate_m: float = 0.1        # grow non-traversable cells; 0.1 = a one-cell wall still trips 0.35, verges cost nothing (measured 2026-09-04)
    map_inflate_classes: str = ""     # comma list of class ids that inflate; "" = every non-traversable class. "10,11,13" = walls/vegetation/vehicles only, grass keeps its true edge
    map_fill_m: float = 0.3           # fill void holes up to this radius from their neighbours
    map_fill_max_area_m2: float = 10.0  # fill ENCLOSED void regions up to this area entirely
    map_walk_halfwidth_m: float = 0.4   # the recorded walk is walkable, this far each side
    map_ignore_classes: str = ""      # e.g. "12,13" to keep person/vehicle from voting; default: they
    # 2026-09-07: remap GENERATED label ids before the reward reads them,
    # "src:dst,src:dst". Measured on 126 static sweep panels: v26 e10 paints
    # class 12 "person" (magenta, road-purple's palette neighbour) over
    # 0.2-0.5 of the crash box on static renders of pedestrian-free scenes --
    # phantom crashes. "12:0" makes it void (small void cost, never certified
    # walkable); "12:7" makes it road. Temporary until the v30 label head.
    label_remap: str = ""
    #                                   count, since the generator renders them too and the walk corridor
    #                                   already clears the ones standing on the recorded path
    # Trajectory output (plan-B arm, 2026-08-25): the policy emits k action
    # pairs per decision and only observes again after all k execute. Rewards
    # still accrue per sub-step, so the world stays action-conditioned; only
    # the POLICY's decision rate changes. 1 = per-action (default, all runs
    # before this date).
    action_chunk: int = 1
    # Action smoothness (2026-08-27): penalty * mean|a_t - a_{t-1}|. Targets
    # the bang-bang habit (PPO saturates clipped Gaussians at the bounds and
    # flip-flops the turn sign -> minimum-turn-circle capture loops); charges
    # CHANGES, so sustained gentle turns stay free. 0 = off (all runs before).
    action_smooth_cost: float = 0.0
    # Batched live training (2026-08-27): when True the env NEVER renders
    # itself — after each step/reset the owning vectorized env batches all
    # robots' poses into one diffusion call and injects the frames back via
    # inject_render(). Observations must be rebuilt (env._obs()) after
    # injection; the obs returned by step()/reset() are stale placeholders.
    defer_render: bool = False
    # Forward-only motion (2026-08-27): negative velocity actions clamp to 0
    # (stand-and-turn). Kills the entire backward-exploit class BY CONSTRUCTION
    # instead of pricing it; clamping at the env keeps the action space shape,
    # so warm-starts from older checkpoints still load.
    forward_only: bool = False
    # Motion-direction footprint (2026-08-26): score the ground THIS step's
    # action moves onto — forward actions score ahead (unchanged), backward
    # actions point the footprint behind the camera, where there are no
    # labels; compute_reward's no-info branch then prices the step as
    # worst-case terrain under terrain_as_cost. Motion onto unseen ground is
    # never cheaper than the worst visible ground, so the camera-blind
    # reversing exploit (prox-v2 100% backward, chunk-5 all-backward arcs)
    # stops paying. False = heading-pinned footprint (every run before).
    footprint_along_motion: bool = False


# ---------------------------------------------------------------------------
# The env itself
# ---------------------------------------------------------------------------

def rgb_alignment(rgb, raster, alpha, min_alpha=0.5, min_frac=0.2):
    """Similarity in [0, 1] between the generated frame and the Gaussian raster it
    was conditioned on, over the pixels the reconstruction actually covers
    (alpha >= min_alpha). 1 - mean |rgb - raster| / 255 on those pixels. Returns
    None when fewer than min_frac of the pixels qualify -- then there is nothing
    to be faithful to and the generator is SUPPOSED to be inventing. Pure numpy
    so it can be tested without an env (2026-09-25)."""
    if rgb is None or raster is None or alpha is None:
        return None
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    g = np.asarray(rgb)[..., :3].astype(np.float32)
    r = np.asarray(raster)[..., :3].astype(np.float32)
    if g.shape != r.shape or a.shape != g.shape[:2]:
        return None
    m = a >= float(min_alpha)
    if float(m.mean()) < float(min_frac):
        return None
    return float(1.0 - np.abs(g[m] - r[m]).mean() / 255.0)


class SceneEnv(gym.Env if gym is not None else object):
    """RL environment where the world model is our simulator."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        world_backend: WorldBackend,
        semantic_backend: SemanticBackend,
        scene_ids: list[str],
        cfg: SceneEnvConfig = SceneEnvConfig(),
    ):
        if gym is None:
            raise ImportError("gymnasium is required to instantiate SceneEnv")
        super().__init__()
        self.world_backend = world_backend
        self.semantic_backend = semantic_backend
        self.scene_ids = list(scene_ids)
        self.cfg = cfg

        # Pre-load traversability table + collision mask (rewards use these each step).
        from pathlib import Path as _Path
        self._trav_scores = load_traversability(
            _Path(cfg.trav_path) if cfg.trav_path else None)         # (NUM_CLASSES,) float32
        self._non_trav = self._trav_scores <= cfg.collision_threshold

        H, W = world_backend.H, world_backend.W
        if cfg.obs_out_hw is not None:
            H, W = int(cfg.obs_out_hw[0]), int(cfg.obs_out_hw[1])
        self.observation_space = spaces.Dict({
            "rgb":  spaces.Box(low=0, high=255, shape=(H, W, 3 * max(1, int(getattr(cfg, "obs_frame_stack", 1)))), dtype=np.uint8),
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * max(1, cfg.action_chunk) + (1 if cfg.stop_action else 0),), dtype=np.float32)
        if cfg.stop_action and cfg.action_chunk > 1:
            raise ValueError("stop_action is not supported with action chunks")

        # Populated on reset()
        self._scene_id: Optional[str] = None
        self._robot_pose_world: Optional[np.ndarray] = None
        self._goal_world: Optional[np.ndarray] = None
        self._prev_position: Optional[np.ndarray] = None
        self._steps: int = 0

        # Cache the latest render so step() can compute the reward without a fresh render.
        # (We render at the START of each step because that's what the policy sees;
        # after action, the NEXT render is done for the NEXT observation.)
        self._last_rgb: Optional[np.ndarray] = None
        self._last_K: Optional[np.ndarray] = None
        self._last_w2c: Optional[np.ndarray] = None
        self._failure_snaps = 0
        self._memory_strips = 0
        # x4/x8 runs put N envs behind one failure_snap_dir, each with its own
        # counter — so robot 0 and robot 1 both wrote collision_0000_... and
        # silently overwrote each other. Every env gets a tag (2026-09-01).
        global _ENV_SEQ
        _ENV_SEQ += 1
        self._env_tag = _ENV_SEQ

    def _save_memory_crash_strip(self, breakdown, prefix: str = "crash_memory") -> None:
        """One picture per memory-decided crash: [current view | stored frame age 1 | age 2 ...],
        each stored frame colorized with the near box drawn and its non-walkable share; the
        header carries the mean the crash rule compared against the threshold."""
        from pathlib import Path as _Path
        try:
            import cv2   # 2026-09-28: was missing; every strip was "skipped: name 'cv2' is not defined"
            from ..eval.reward_2d import _footprint_corners_world, _project_points, GO2_BODY_LENGTH, GO2_BODY_WIDTH
            from ..eval.palette import display_palette
            out = _Path(self.cfg.failure_snap_dir); out.mkdir(parents=True, exist_ok=True)
            pal = display_palette(int(getattr(self.cfg, "sem_palette_version", 6) or 6))
            fm = getattr(self, "_frame_memory", None) or []
            rgb = self._last_rgb
            H, W = rgb.shape[:2]
            panels = []
            cur = rgb.copy()
            pose = self._robot_pose_world
            hd = getattr(self, "_last_fp_heading", None)
            if hd is None:
                hd = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
            _la = getattr(self.cfg, "collision_look_ahead_m", None) or self.cfg.look_ahead_dist
            corners = _footprint_corners_world(pose[:3, 3], hd, look_ahead_dist=float(_la),
                                               length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
            uv, in_front = _project_points(corners, self._last_K, self._last_w2c)
            if in_front.all():
                cv2.polylines(cur, [uv.astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(cur, "NOW (box below the camera)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(cur)
            shares = []
            if prefix == "memory":
                # figure mode: every stored frame, newest first, on the GENERATED RGB, the same
                # ground box projected into each; an age whose box falls outside the image says so
                fr = getattr(self, "_frame_memory_rgb", None) or []
                hits = {int(a): float(f) for a, f, _ in breakdown.box_memory_hits}
                for age in range(1, len(fm) + 1):
                    idx = len(fm) - age
                    m_sem, mK, mw2c = fm[idx][0], fm[idx][1], fm[idx][2]
                    img = fr[idx] if idx < len(fr) and fr[idx] is not None else None
                    if img is None:
                        img = pal[np.clip(np.asarray(m_sem), 0, len(pal) - 1)].astype(np.uint8)
                    img = np.ascontiguousarray(img.copy())
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H))
                    puv, pfront = _project_points(corners, mK, mw2c)
                    inside = bool(pfront.all()) and bool((puv[:, 0] >= 0).all() and (puv[:, 0] < W).all() and (puv[:, 1] >= 0).all() and (puv[:, 1] < H).all())
                    if pfront.all():
                        cv2.polylines(img, [puv.astype(np.int32)], True, (255, 0, 255) if inside else (160, 160, 160), 2, cv2.LINE_AA)
                    txt = f"age {age} ({age * float(self.cfg.step_size_m):.2f} m back)  " + (f"non-walkable {hits[age]:.2f}" if age in hits else "box out of view")
                    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(img)
                    if age in hits:
                        shares.append(hits[age])
                if not shares:
                    shares = [0.0]
            for age, frac, buv in ([] if prefix == "memory" else breakdown.box_memory_hits):
                idx = len(fm) - int(age)
                if idx < 0 or idx >= len(fm):
                    continue
                m_sem = fm[idx][0]
                col = pal[np.clip(np.asarray(m_sem), 0, len(pal) - 1)].astype(np.uint8)
                if col.shape[:2] != (H, W):
                    sy, sx = H / col.shape[0], W / col.shape[1]
                    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
                    buv = np.asarray(buv, np.float32) * np.array([sx, sy], np.float32)
                col = np.ascontiguousarray(col)
                cv2.polylines(col, [np.asarray(buv).astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(col, f"age {int(age)}  non-walkable {frac:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(col, f"age {int(age)}  non-walkable {frac:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(col); shares.append(float(frac))
            if not shares:
                return
            panel = np.concatenate(panels, axis=1)
            bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
            agg = str(getattr(self.cfg, "collision_box_memory_agg", "newest"))
            used = float(np.mean(shares)) if agg == "mean" else shares[0]
            cv2.putText(bar, f"{'MEMORY CRASH' if prefix == 'crash_memory' else 'MEMORY READ'}  {self._scene_id}  step {self._steps}  {len(shares)} stored frame(s) contain the box  "
                             f"{agg} share {used:.2f} >= threshold {self.cfg.collision_terminate_frac:.2f}",
                        (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
            img = np.concatenate([bar, panel], axis=0)
            cv2.imwrite(str(out / f"{prefix}_e{self._env_tag:02d}_{self._memory_strips:04d}_{self._scene_id}_step{self._steps:03d}.png"), img[:, :, ::-1])
            self._memory_strips += 1
        except Exception as _e:  # a picture must never end a training step
            print(f"[memory strip] skipped: {_e}", flush=True)

    def _save_failure_snapshot(self, breakdown, semantic_image) -> None:
        """One PNG per collision: [obs RGB | v14-colorized semantics], with the
        reward numbers burned in. Files: <dir>/collision_<n>_<scene>_step<k>.png"""
        import cv2
        from pathlib import Path as _Path
        from ..eval.palette import display_palette
        out = _Path(self.cfg.failure_snap_dir)
        out.mkdir(parents=True, exist_ok=True)
        rgb = self._last_rgb
        pal = display_palette(int(getattr(self.cfg, "sem_palette_version", 6) or 6))
        col = pal[np.clip(semantic_image, 0, len(pal) - 1)]
        if col.shape[:2] != rgb.shape[:2]:
            col = cv2.resize(col, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        # draw the exact footprint quad the reward scored (her ask 2026-08-19)
        try:
            from ..eval.reward_2d import (
                _footprint_corners_world, _project_points,
                GO2_BODY_LENGTH, GO2_BODY_WIDTH,
            )
            pose = self._robot_pose_world
            hd = getattr(self, "_last_fp_heading", None)
            if hd is None:
                hd = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
            corners = _footprint_corners_world(
                pose[:3, 3], hd, look_ahead_dist=self.cfg.look_ahead_dist,
                length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
            uv, in_front = _project_points(corners, self._last_K, self._last_w2c)
            if in_front.all():
                rgb = rgb.copy()
                cv2.polylines(rgb, [uv.astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.polylines(col, [uv.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass
        panel = np.concatenate([rgb, col], axis=1)
        bar = np.zeros((44, panel.shape[1], 3), dtype=np.uint8)
        x, y = self._robot_pose_world[0, 3], self._robot_pose_world[1, 3]
        cv2.putText(bar, f"COLLISION  {self._scene_id}  step {self._steps}  "
                         f"pose=({x:+.2f},{y:+.2f})", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"collision={breakdown.collision:+.2f}  "
                         f"dominant_class={breakdown.dominant_class_id}  "
                         f"mean_score={breakdown.mean_class_score:.2f}  "
                         f"footprint_px={breakdown.n_footprint_pixels}  "
                         f"trav_px={breakdown.n_traversable_pixels}", (6, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.concatenate([bar, panel], axis=0)
        cv2.imwrite(str(out / f"collision_e{self._env_tag:02d}_"
                              f"{self._failure_snaps:04d}_"
                              f"{self._scene_id}_step{self._steps:03d}.png"),
                    img[:, :, ::-1])

        # Machine-readable sidecar (2026-09-01). The pose used to live only in
        # the burned-in text, so 1304 crash snapshots could be looked at but
        # never REPLAYED — we could not re-render a crash to ask whether the
        # obstacle was real. One CSV line per crash fixes that for good.
        try:
            hd_w = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
            csv_p = out / "crash_poses.csv"
            if not csv_p.exists():
                csv_p.write_text(
                    "env,n,scene,step,x,y,z,hx,hy,goal_x,goal_y,collision,"
                    "void_frac,dominant,footprint_px,trav_px\n")
            with open(csv_p, "a") as fh:
                fh.write(
                    f"{self._env_tag},{self._failure_snaps},"
                    f"{self._scene_id},{self._steps},"
                    f"{pose[0, 3]:.4f},{pose[1, 3]:.4f},{pose[2, 3]:.4f},"
                    f"{hd_w[0]:.4f},{hd_w[1]:.4f},"
                    f"{self._goal_world[0]:.4f},{self._goal_world[1]:.4f},"
                    f"{breakdown.collision:.4f},{breakdown.void_frac:.4f},"
                    f"{breakdown.dominant_class_id},"
                    f"{breakdown.n_footprint_pixels},"
                    f"{breakdown.n_traversable_pixels}\n")
        except Exception:
            pass

        self._failure_snaps += 1

    def _disc_known_share(self, goal) -> float:
        """Share of the arrival disc that the map has reconstructed at all."""
        _g = getattr(self, "_label_grids", {}).get(self._scene_id)
        if _g is None:
            return float("nan")
        _r = max(float(self.cfg.goal_radius), 0.3)
        _a = np.arange(-_r, _r + 1e-6, _g.res / 2.0)
        _X, _Y = np.meshgrid(_a, _a, indexing="ij")
        _pts = np.c_[_X.ravel(), _Y.ravel()]
        _pts = _pts[np.linalg.norm(_pts, axis=1) <= _r] + np.asarray(goal)[:2][None, :]
        return float((_g.lookup(_pts) >= 0).mean())

    def _refusal_point(self, goal):
        """Where a refusal of THIS goal should happen: the point of the recorded
        walk nearest the goal -- the verge the robot reaches by walking toward
        the goal on pavement (Joana, 2026-09-05: 'if lawn goals are far there
        is no way of knowing if the robot stopped at the right place'). None
        when the scene carries no walk."""
        w = getattr(self, "_walk_xy", {}).get(self._scene_id)
        if w is None or len(w) == 0:
            return None
        g = getattr(self, "_label_grids", {}).get(self._scene_id)
        goal2 = np.asarray(goal, dtype=np.float32)[:2]
        if g is not None:
            # Joana (09-05): the verge is the sidewalk EDGE nearest the goal, not
            # the centre line -- the nearest walkable cell, restricted to the
            # walkway strip (within 3 m of the recorded walk) so it is reachable.
            if not hasattr(self, "_strip_cells"):
                self._strip_cells = {}
            if self._scene_id not in self._strip_cells:
                known = g.labels >= 0
                walkable = known & ~self._non_trav[np.clip(g.labels, 0, len(self._non_trav) - 1)]
                wy, wx = np.nonzero(walkable)
                cells = np.c_[g.x0 + (wx + 0.5) * g.res, g.y0 + (wy + 0.5) * g.res].astype(np.float32)
                keep = np.zeros(len(cells), dtype=bool)
                for i0 in range(0, len(cells), 4096):
                    blk = cells[i0:i0 + 4096]
                    keep[i0:i0 + 4096] = ((blk[:, None, :] - w[None, :, :]) ** 2).sum(-1).min(axis=1) <= 9.0
                self._strip_cells[self._scene_id] = cells[keep]
            strip = self._strip_cells[self._scene_id]
            if len(strip):
                d = np.linalg.norm(strip - goal2[None, :], axis=1)
                return strip[int(np.argmin(d))].copy()
        d = np.linalg.norm(w - goal2[None, :], axis=1)
        return w[int(np.argmin(d))].copy()

    def _goal_supported(self, goal) -> bool:
        """The goal-support rule of _draw_supported_goal, as a predicate."""
        r = self.cfg.goal_support_radius_m
        if r <= 0.0:
            return True
        gp = getattr(self, "_ground_pts", {}).get(self._scene_id)
        if gp is None or not len(gp):
            return True
        ref = getattr(self, "_support_ref", {}).get(self._scene_id, 0.0)
        need = max(1, int(self.cfg.goal_support_min_frac * ref))
        d2 = np.abs(gp - np.asarray(goal)[:2]).max(axis=1)
        near = gp[d2 <= r]
        return bool(len(near) >= need
                    and float(np.linalg.norm(near - np.asarray(goal)[:2], axis=1).min()) <= r)

    def _draw_map_goal(self, _yaw):
        """A NON-traversable goal drawn straight from the map: a random cell of
        goal_nontrav_classes inside the backend's distance window and cone
        (same rules as the walk sampler), whose arrival disc is <= 25%
        walkable and which passes goal support. None if the scene has no
        such cell in range -- then the episode is honestly traversable."""
        g = getattr(self, "_label_grids", {}).get(self._scene_id)
        if g is None:
            return None
        cls = tuple(int(v) for v in str(self.cfg.goal_nontrav_classes).split(",") if v.strip())
        if not hasattr(self, "_map_goal_cells"):
            self._map_goal_cells = {}
        key = (self._scene_id, cls)
        if key not in self._map_goal_cells:
            iy, ix = np.nonzero(np.isin(g.labels, list(cls)))
            self._map_goal_cells[key] = np.c_[g.x0 + (ix + 0.5) * g.res,
                                              g.y0 + (iy + 0.5) * g.res].astype(np.float32)
        cells = self._map_goal_cells[key]
        if len(cells) == 0:
            return None
        edge = float(getattr(self.cfg, "goal_nontrav_edge_m", 0.0) or 0.0)
        if edge > 0.0:
            ekey = (self._scene_id, cls, edge)
            if ekey not in self._map_goal_cells:
                # keep only lawn cells within `edge` of the RECORDED WALK (Joana,
                # 09-05 16:00): a cell near walkable ground on the far side of a
                # lawn is unreachable along pavement, so no ending would pay;
                # beside the walk the robot can always come within the refusal
                # radius on pavement. Falls back to any walkable cell if the
                # cloud carries no walk.
                wxy = getattr(self, "_walk_xy", {}).get(self._scene_id)
                if wxy is None or len(wxy) == 0:
                    known = g.labels >= 0
                    walk = known & ~self._non_trav[np.clip(g.labels, 0, len(self._non_trav) - 1)]
                    wy, wx = np.nonzero(walk)
                    wxy = np.c_[g.x0 + (wx + 0.5) * g.res, g.y0 + (wy + 0.5) * g.res].astype(np.float32)
                keep = np.zeros(len(cells), dtype=bool)
                if len(wxy):
                    for i0 in range(0, len(cells), 2048):
                        blk = cells[i0:i0 + 2048]
                        d2 = ((blk[:, None, :] - wxy[None, :, :]) ** 2).sum(-1)
                        keep[i0:i0 + 2048] = d2.min(axis=1) <= edge * edge
                self._map_goal_cells[ekey] = cells[keep]
            cells = self._map_goal_cells[ekey]
            if len(cells) == 0:
                return None
        bcfg = self.world_backend.cfg
        lo_d, hi_d = getattr(bcfg, "goal_dist_range", None) or (5.0, 10.0)
        spawn = np.asarray(self._robot_pose_world[:2, 3], dtype=np.float32)
        d = np.linalg.norm(cells - spawn[None, :], axis=1)
        ok = (d >= float(lo_d)) & (d <= float(hi_d))
        cone = float(getattr(bcfg, "goal_cone_deg", 360.0))
        if float(getattr(self.cfg, "goal_nontrav_cone_deg", 0.0) or 0.0) > 0.0:
            cone = float(self.cfg.goal_nontrav_cone_deg)
        if cone < 360.0 and _yaw is not None:
            ang = np.arctan2(cells[:, 1] - spawn[1], cells[:, 0] - spawn[0])
            dth = (ang - float(_yaw) + np.pi) % (2.0 * np.pi) - np.pi
            ok &= np.abs(dth) <= np.deg2rad(cone) / 2.0
        idx = np.nonzero(ok)[0]
        if not hasattr(self, "_lawn_draw_stats"):
            self._lawn_draw_stats = {}
        st = self._lawn_draw_stats.setdefault(self._scene_id, {"calls": 0, "no_cells": 0, "disc_fail": 0, "ok": 0})
        st["calls"] += 1
        if len(idx) == 0:
            st["no_cells"] += 1
            self._lawn_draw_report(st)
            return None
        half = g.res / 2.0
        _tries = int(self.cfg.goal_nontrav_tries) if int(self.cfg.goal_nontrav_tries) > 0 else int(self.cfg.goal_mix_tries)
        for _ in range(max(1, _tries)):
            c = cells[idx[int(self.np_random.integers(0, len(idx)))]]
            goal = np.array([c[0] + self.np_random.uniform(-half, half),
                             c[1] + self.np_random.uniform(-half, half), 0.0], dtype=np.float32)
            wf = self._goal_walkable_share(goal)
            # No goal-support rule here (2026-09-05 funnel): it demands the
            # walk's point density under the goal, which lawns never have, and
            # it killed every lawn goal on AUw360/AUd210. A map cell is known
            # ground by construction, and the share test already refuses a
            # disc that is less than half reconstructed.
            if wf == wf and wf <= 0.25 and self._disc_known_share(goal) >= float(self.cfg.goal_nontrav_known_min):
                st["ok"] += 1
                self._lawn_draw_report(st)
                return goal
        st["disc_fail"] += 1
        self._lawn_draw_report(st)
        return None

    def _lawn_draw_report(self, st: dict) -> None:
        # why lawn goals go missing, per scene, at 20 / 200 / 2000 calls
        if st["calls"] in (20, 200, 2000):
            print(f"[goal mix] lawn draw on {self._scene_id}: {st['calls']} calls, "
                  f"{st['ok']} ok, {st['no_cells']} no lawn cell in window+cone, "
                  f"{st['disc_fail']} all tries failed the disc tests", flush=True)

    def _draw_supported_goal(self, _yaw) -> np.ndarray:
        """One goal draw from the backend's sampler, re-drawn up to
        goal_support_tries times until the reconstruction has points under
        it. Keeps the LAST draw if every try fails, so a scene whose goals are
        mostly off-cloud degrades gracefully. Says nothing about
        traversability -- that is the goal MIX's job."""
        goal = self.world_backend.sample_goal_position(
            self._scene_id, self.np_random, self._robot_pose_world[:2, 3], cone_yaw=_yaw).copy()
        r = self.cfg.goal_support_radius_m
        if r > 0.0:
            gp = getattr(self, "_ground_pts", {}).get(self._scene_id)
            if gp is not None and len(gp):
                ref = getattr(self, "_support_ref", {}).get(self._scene_id, 0.0)
                need = max(1, int(self.cfg.goal_support_min_frac * ref))
                for _try in range(max(1, self.cfg.goal_support_tries)):
                    d2 = np.abs(gp - goal[:2]).max(axis=1)
                    near = gp[d2 <= r]
                    if len(near) >= need and float(np.linalg.norm(near - goal[:2], axis=1).min()) <= r:
                        break
                    goal = self.world_backend.sample_goal_position(
                        self._scene_id, self.np_random, self._robot_pose_world[:2, 3], cone_yaw=_yaw).copy()
        # LATERAL JITTER with a map check (2026-09-17): offset perpendicular to
        # the walk's direction at the goal; accept when the disc is walkable
        # enough on the map; otherwise try again; fall back to the on-path goal.
        _jl = float(getattr(self.cfg, "goal_lat_jitter_m", 0.0) or 0.0)
        if _jl > 0.0 and getattr(self, "_label_grids", {}).get(self._scene_id) is not None:
            base = goal.copy()
            _cal = getattr(self.world_backend, "_calib", {}).get(self._scene_id)
            if _cal is not None and len(_cal.positions) > 2:
                _pos = np.asarray(_cal.positions)[:, :2]
                _i = int(np.argmin(np.linalg.norm(_pos - base[:2], axis=1)))
                _a, _b = max(_i - 1, 0), min(_i + 1, len(_pos) - 1)
                _dir = _pos[_b] - _pos[_a]
            else:
                _dir = np.array([np.cos(_yaw), np.sin(_yaw)])
            _n = np.linalg.norm(_dir); _dir = _dir / _n if _n > 1e-6 else np.array([1.0, 0.0])
            _perp = np.array([-_dir[1], _dir[0]])
            _min = float(getattr(self.cfg, "goal_lat_jitter_min_share", 0.6))
            for _try in range(max(1, int(getattr(self.cfg, "goal_lat_jitter_tries", 8)))):
                cand = base.copy()
                cand[:2] = base[:2] + _perp * float(self.np_random.uniform(-_jl, _jl))
                wf = self._goal_walkable_share(cand)
                if wf == wf and wf >= _min:
                    goal = cand
                    break
            self._goal_jitter_kept = getattr(self, "_goal_jitter_kept", 0) + int(goal is not base)
        return goal

    def _goal_walkable_share(self, goal_world, radius: "float | None" = None) -> float:
        """Walkable share of the arrival disc on the map; nan if no map or
        less than half the disc is reconstructed. Used by the goal mix at
        sampling time and by the refusal metric. `radius` overrides the
        arrival radius (the centre-clearance check uses a smaller disc)."""
        _g = getattr(self, "_label_grids", {}).get(self._scene_id)
        if _g is None:
            return float("nan")
        _r = max(float(self.cfg.goal_radius if radius is None else radius), 0.3)
        _a = np.arange(-_r, _r + 1e-6, _g.res / 2.0)
        _X, _Y = np.meshgrid(_a, _a, indexing="ij")
        _pts = np.c_[_X.ravel(), _Y.ravel()]
        _pts = _pts[np.linalg.norm(_pts, axis=1) <= _r] + np.asarray(goal_world)[:2][None, :]
        _lab = _g.lookup(_pts).astype(int)
        _known = _lab >= 0
        if _known.mean() < 0.5:
            return float("nan")
        _cl = np.clip(_lab[_known], 0, len(self._non_trav) - 1)
        if float(getattr(self.cfg, "goal_min_score", 0.0)) > 0.0:
            _nt = self._trav_scores[_cl] < float(self.cfg.goal_min_score)
        else:
            _nt = self._non_trav[_cl]
        return float(1.0 - _nt.mean())

    # ---------------- gym API ----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._last_action = None
        self._prev_obstacle_dist = None
        self._prev_grass_dist = None
        # Choose a scene (round-robin for now; can be random later).
        self._rgb_hist, self._rgb_hist_key = None, None      # frame stack restarts every episode
        self._mirrored = bool(float(getattr(self.cfg, "mirror_prob", 0.0)) > 0.0
                              and self.np_random.random() < float(self.cfg.mirror_prob))
        idx = self.np_random.integers(0, len(self.scene_ids))
        self._scene_id = self.scene_ids[idx]
        # options (2026-11-09, scripted-path reward check): a chosen scene, a recorded
        # start frame and a goal point. Only WHERE the episode starts and ends is set;
        # the reward, rendering and termination are untouched.
        _opt = dict(options or {})
        if _opt.get("scene") is not None:
            self._scene_id = str(_opt["scene"])
        if _opt:
            self._mirrored = False
        self.world_backend.load_scene(self._scene_id)
        self._load_obstacle_points(self._scene_id)
        if bool(self.cfg.look_ahead_auto):
            # nearest visible ground for THIS scene's camera: h * fy / (H - cy)
            try:
                _sc = self.world_backend._cache[self._scene_id]
                _K = _sc["K"]; _K = _K[0] if getattr(_K, "ndim", 2) == 3 else _K
                _K = _K.detach().cpu().float().numpy() if hasattr(_K, "detach") else np.asarray(_K, float)
                _h = float(self.world_backend._calib[self._scene_id].camera_height_m)
                _H = float(self.world_backend.H)
                _d = _h * float(_K[1, 1]) / max(_H - float(_K[1, 2]), 1.0)
                _la = float(np.clip(_d + self.cfg.look_ahead_auto_pad_m, self.cfg.look_ahead_auto_min_m, self.cfg.look_ahead_auto_max_m))
                if not hasattr(self, "_look_ahead_auto_seen"):
                    self._look_ahead_auto_seen = {}
                if self._scene_id not in self._look_ahead_auto_seen:
                    self._look_ahead_auto_seen[self._scene_id] = _la
                    print(f"[SceneEnv] look-ahead AUTO on {self._scene_id}: image bottom at {_d:.2f} m -> box centre {_la:.2f} m", flush=True)
                self.cfg.look_ahead_dist = _la
            except Exception as _e:
                print(f"[SceneEnv] look-ahead AUTO failed on {self._scene_id}: {type(_e).__name__}: {_e} (keeping {self.cfg.look_ahead_dist})", flush=True)

        if self.cfg.random_spawn and hasattr(self.world_backend, "sample_start_pose"):
            self._robot_pose_world = self.world_backend.sample_start_pose(
                self._scene_id, self.np_random).copy()
            _tries = int(self.cfg.spawn_support_tries)
            if _tries > 0:
                _thr = (float(self.cfg.collision_terminate_frac)
                        if self.cfg.collision_terminate_frac > 0 else 0.35)
                self._spawn_total = getattr(self, "_spawn_total", 0) + 1
                self._spawn_redraws = getattr(self, "_spawn_redraws", 0)
                self._spawn_exhausted = getattr(self, "_spawn_exhausted", 0)
                for _t in range(_tries + 1):
                    _f = self._spawn_doomed(self._robot_pose_world)
                    if _f is None or _f < _thr:
                        break
                    if _t == _tries:
                        self._spawn_exhausted += 1
                        break
                    self._spawn_redraws += 1
                    self._robot_pose_world = self.world_backend.sample_start_pose(
                        self._scene_id, self.np_random).copy()
                if self._spawn_total in (50, 500, 5000, 50000):
                    print(f"[spawn support] {self._spawn_redraws} redraws over "
                          f"{self._spawn_total} spawns "
                          f"({100.0 * self._spawn_redraws / self._spawn_total:.1f}% of draws doomed), "
                          f"{self._spawn_exhausted} episodes kept a doomed spawn after "
                          f"{_tries} tries", flush=True)
        else:
            self._robot_pose_world = self.world_backend.start_pose(self._scene_id).copy()
        if hasattr(self.world_backend, "sample_goal_position"):
            # per-episode goal when the backend has goal_frame_range set;
            # falls back to the fixed goal otherwise
            # Centre the goal cone on the RECORDED heading at the spawn
            # frame -- not the jittered pose, and not a tangent re-derived
            # from whichever frame is nearest the jittered spawn. Falls back
            # to the actual pose yaw only if the backend cannot report it.
            _yaw = None
            if hasattr(self.world_backend, "last_spawn_base_yaw"):
                _yaw = self.world_backend.last_spawn_base_yaw()
            if _yaw is None:
                _yaw = float(np.arctan2(self._robot_pose_world[1, 0],
                                        self._robot_pose_world[0, 0]))
            self._goal_world = self._draw_supported_goal(_yaw)
            # ---- the goal MIX (2026-09-04): choose the KIND of goal first ----
            p_mix = float(self.cfg.goal_traversable_mix)
            if 0.0 < p_mix < 1.0 and getattr(self, "_label_grids", {}).get(self._scene_id) is not None:
                want_trav = bool(self.np_random.random() < p_mix)
                got = None
                if not want_trav and bool(self.cfg.goal_mix_map_draw):
                    _mg = self._draw_map_goal(_yaw)
                    if _mg is not None:
                        self._goal_world = _mg          # the loop below confirms its share
                        self._map_goal_draws = getattr(self, "_map_goal_draws", 0) + 1
                        if self._map_goal_draws in (1, 100, 1000):
                            print(f"[goal mix] map-direct non-traversable goal #{self._map_goal_draws} "
                                  f"on {self._scene_id} at {float(np.linalg.norm(_mg[:2] - self._robot_pose_world[:2, 3])):.1f} m", flush=True)
                for _try in range(max(1, int(self.cfg.goal_mix_tries))):
                    wf = self._goal_walkable_share(self._goal_world)
                    if wf == wf:                       # known ground under the disc
                        is_trav = wf > 0.25            # edge goals count as reachable
                        if is_trav == want_trav:
                            got = is_trav
                            break
                    self._goal_world = self._draw_supported_goal(_yaw)
                if not hasattr(self, "_mix_misses"):
                    self._mix_misses = {}
                if got is None:
                    k = (self._scene_id, want_trav)
                    self._mix_misses[k] = self._mix_misses.get(k, 0) + 1
                    if self._mix_misses[k] in (1, 50, 500):
                        print(f"[goal mix] {self._scene_id}: wanted a "
                              f"{'traversable' if want_trav else 'NON-traversable'} goal, none in "
                              f"{self.cfg.goal_mix_tries} draws (x{self._mix_misses[k]}); kept the last draw",
                              flush=True)
        else:
            self._goal_world = self.world_backend.goal_position(self._scene_id).copy()
        # ---- GOAL CASE (2026-09-07): label the goal, optionally draw to a mix ----
        self._goal_case = "unknown"; self._goal_detour = float("nan"); self._goal_min_width = float("nan")
        _g = getattr(self, "_label_grids", {}).get(self._scene_id)
        if _g is not None and getattr(self, "_goal_world", None) is not None:
            from src.eval.goal_cases import scene_case_maps, classify_pair
            if not hasattr(self, "_case_maps"):
                self._case_maps = {}
            if self._scene_id not in self._case_maps:
                self._case_maps[self._scene_id] = scene_case_maps(_g, self._non_trav)
            _maps = self._case_maps[self._scene_id]
            _sp = np.asarray(self._robot_pose_world[:2, 3], float)
            _mix = str(self.cfg.goal_case_mix or "").strip()
            def _cls_of(goal):
                r = classify_pair(_g, _maps, _sp, np.asarray(goal, float)[:2],
                                  narrow_m=float(self.cfg.goal_case_narrow_m), detour_min=float(self.cfg.goal_case_detour_min))
                return r
            if _mix:
                _probs = {}
                for tok in _mix.split(","):
                    k, v = tok.split(":"); _probs[k.strip()] = float(v)
                _keys = list(_probs); _pv = np.array([_probs[k] for k in _keys], float); _pv /= _pv.sum()
                _want = _keys[int(self.np_random.choice(len(_keys), p=_pv))]
                _yaw0 = float(np.arctan2(self._robot_pose_world[1, 0], self._robot_pose_world[0, 0]))
                _base = getattr(self.world_backend, "last_spawn_base_yaw", None)
                _yaw0 = float(_base()) if callable(_base) and _base() is not None else _yaw0
                _best = None; _best_r = None
                for _t in range(max(1, int(self.cfg.goal_case_tries))):
                    _cand = self._goal_world if _t == 0 else self._draw_supported_goal(_yaw0)
                    _wf = self._goal_walkable_share(_cand)
                    if not (_wf == _wf and _wf >= float(self.cfg.goal_case_walkable_min)):
                        continue
                    _cl = float(getattr(self.cfg, "goal_center_clear_m", 0.0) or 0.0)
                    if _cl > 0.0:
                        _wc = self._goal_walkable_share(_cand, radius=_cl)
                        if not (_wc == _wc and _wc >= 0.999):
                            continue
                    _r = _cls_of(_cand)
                    if _r["cls"] in ("blocked", "offmap"):
                        continue
                    if _best is None:
                        _best, _best_r = _cand, _r
                    _hit = (_r["cls"] == _want) or (_want == "corner" and _r["cls"] == "corner+narrow")                            or (_want == "narrow" and _r["cls"] == "corner+narrow")
                    if _hit:
                        _best, _best_r = _cand, _r
                        break
                if _best is not None:
                    self._goal_world = np.asarray(_best, dtype=np.float32)
                    _r = _best_r
                    if not hasattr(self, "_case_stats"):
                        self._case_stats = {"want": {}, "got": {}, "n": 0}
                    self._case_stats["n"] += 1
                    self._case_stats["want"][_want] = self._case_stats["want"].get(_want, 0) + 1
                    self._case_stats["got"][_r["cls"]] = self._case_stats["got"].get(_r["cls"], 0) + 1
                    if self._case_stats["n"] in (50, 500, 2000):
                        print(f"[goal case] {self._case_stats['n']} episodes: wanted {self._case_stats['want']} got {self._case_stats['got']}", flush=True)
                else:
                    _r = _cls_of(self._goal_world)
            else:
                _r = _cls_of(self._goal_world)
            self._goal_case = str(_r["cls"]); self._goal_detour = float(_r.get("detour", float("nan"))); self._goal_min_width = float(_r.get("min_width_m", float("nan")))
            # the reward goal / distances must follow a redrawn goal
            self._initial_goal_dist = float(np.linalg.norm(self._robot_pose_world[:3, 3] - self._goal_world))
        if _opt.get("start_frame") is not None:
            _cal = self.world_backend._calib[self._scene_id]
            self._robot_pose_world = _cal.robot_pose_nav(
                int(_opt["start_frame"]),
                heading_from_walk=bool(getattr(self.cfg, "spawn_heading_from_walk", False))).copy()
        if _opt.get("goal_xy") is not None:
            _gz = float(np.asarray(self._goal_world)[2]) if getattr(self, "_goal_world", None) is not None \
                else float(self._robot_pose_world[2, 3])
            self._goal_world = np.array([float(_opt["goal_xy"][0]), float(_opt["goal_xy"][1]), _gz], dtype=np.float32)
        self._prev_position = None
        # Stale from the previous episode otherwise: the rollout video drew the
        # spawn frame's footprint with the LAST episode's final heading (09-04).
        self._last_fp_heading = None
        self._frame_memory = []
        self._frame_memory_rgb = []
        self._pos_trail = []
        self._last_decision = None
        self._decision_pose = None
        self._entered_nontrav_goal = False
        self._halt_at_verge = False
        self._still_run = 0
        self._in_radius_moving = False
        # a reset teleports the robot, so the first frame difference of an
        # episode is meaningless -- drop it rather than log a false jump
        self._rgb_delta = float("nan")
        self._steps = 0
        self._halt_run = 0
        # Episode path, for the wandb top-down panel (2026-09-25, Joana: "can we
        # visualize actual rollouts on wandb?"). The previous episode is kept as
        # _last_ep_* so a rollout-end snapshot can draw a COMPLETE episode even when
        # the current one has just started.
        if getattr(self, "_ep_xy", None):
            self._last_ep_xy = list(self._ep_xy)
            self._last_ep_goal = getattr(self, "_ep_goal", None)
            self._last_ep_scene = getattr(self, "_ep_scene", None)
        self._ep_xy = [tuple(float(v) for v in self._robot_pose_world[:2, 3])]
        self._ep_goal = tuple(float(v) for v in np.asarray(self._goal_world)[:2])
        self._ep_scene = getattr(self, "_scene_id", None)
        self._initial_goal_dist = float(np.linalg.norm(
            self._robot_pose_world[:3, 3] - self._goal_world))
        # 2026-09-04 (Joana): is THIS goal on traversable ground, by the map?
        # 1 = yes, 0 = no, nan = no map or nothing reconstructed there. The
        # refusal metric: halting on a non-traversable goal is correct,
        # halting on a traversable one is the freeze, arriving on a
        # non-traversable one is trespass.
        self._goal_traversable = float("nan")
        self._goal_walkable_frac = float("nan")
        _walk = self._goal_walkable_share(self._goal_world)
        if _walk == _walk:
            # Joana 2026-09-04: a goal on grass with part of its disc on the
            # walkway is REACHABLE (the robot stands on the walkable part).
            # Three-way: 1 = traversable (>= 75% walkable), 0 = non-traversable
            # (<= 25%), 0.5 = edge goal. The refusal flags use the extremes.
            self._goal_walkable_frac = _walk
            self._goal_traversable = 1.0 if _walk >= 0.75 else (0.0 if _walk <= 0.25 else 0.5)
        self._reward_goal = self._goal_world
        if bool(self.cfg.lawn_progress_to_verge) and float(getattr(self, "_goal_traversable", float("nan"))) == 0.0:
            _vp = self._refusal_point(self._goal_world)
            if _vp is not None:
                self._reward_goal = np.array([_vp[0], _vp[1], float(self._goal_world[2])], dtype=np.float32)

        if self.cfg.defer_render:
            self._needs_render = True
            if self._last_rgb is None:      # first-ever reset: placeholder obs
                H, W = self.observation_space["rgb"].shape[:2]
                self._last_rgb = np.zeros((H, W, 3), dtype=np.uint8)
                self._last_K = np.eye(3, dtype=np.float32)
                self._last_w2c = np.eye(4, dtype=np.float32)
        else:
            # a new episode never inherits the previous episode's frames
            _rh = getattr(self.world_backend, "reset_history", None)
            if callable(_rh):
                _rh()
            self._render_current()
        return self._obs(), {"scene_id": self._scene_id}

    def set_goal_radius(self, r: float) -> None:
        """Curriculum hook: training callback anneals the capture radius
        (e.g. 1.0 m -> 0.5 m) via vec_env.env_method("set_goal_radius", r)."""
        self.cfg.goal_radius = float(r)

    def set_refusal_dwell(self, n: int) -> None:
        """Curriculum hook: required still steps inside the verge zone."""
        self.cfg.refusal_dwell_steps = int(n)

    def set_halt_wrong_penalty(self, p: float) -> None:
        """Ramp hook (2026-09-06): the wrong-halt penalty starts at 0 so early
        halts are cheap and the verge bonus can be discovered, then rises to
        the configured value so wrong halts are priced out."""
        self.cfg.halt_wrong_penalty = float(p)

    def set_refusal_verge(self, r: float) -> None:
        """Curriculum hook (2026-09-05): the verge radius is earned down like
        the goal radius -- wide while the policy has never refused at the
        verge, tight once it does."""
        self.cfg.refusal_verge_m = float(r)

    def set_halt_enabled(self, on: bool) -> None:
        """Curriculum hook (2026-09-05): halting can be UNAVAILABLE, not just
        expensive. W cold'' froze (halt_wrong 0.56) with the halt priced at
        the full timeout, because what a 15%-success policy avoids by halting
        is the crash. Until the policy reaches goals, three still steps are
        just three still steps."""
        if not hasattr(self, "_halt_steps_cfg"):
            self._halt_steps_cfg = int(self.cfg.halt_terminate_steps)
        self.cfg.halt_terminate_steps = int(self._halt_steps_cfg) if on else 0

    def set_halt_penalty_scale(self, s: float) -> None:
        """Curriculum hook (2026-09-04): the halt is EARNED. It pays the full
        timeout while the policy cannot reach goals and notches down toward
        the configured scale as recent wins pass the threshold. A cold policy
        under a cheap reachable halt and a 2500 crash froze at 0% goals /
        70% halts within 8k steps (465704/465706/465709)."""
        self.cfg.halt_penalty_scale = float(s)

    _goal_dist_lo0: "float | None" = None

    def goal_cone_probe(self, n: int = 500, seed: int = 0) -> dict:
        """Is the goal ever BEHIND the robot? Sample n spawn/goal pairs per
        scene exactly as reset() does -- same backend, same live config, same
        scenes -- but WITHOUT rendering, so this costs milliseconds and can run
        in the startup banner of every job.

        Standalone tests can pass while the real thing is broken: the goal cone
        was centred on a path tangent re-derived from whichever frame was
        nearest the JITTERED spawn, which on a walk that doubles back can be a
        frame on the other leg pointing the opposite way. 2.25% of episodes
        across the six AU scenes put the goal more than 90 deg behind a
        forward-only robot, which can then only time out. Fixed by centring on
        the recorded heading at the spawn frame; this probe is what proves the
        fix is live in THIS run rather than in a test file.

        Returns {scene: (mean_deg, max_deg, behind_pct)}.
        """
        rng = np.random.default_rng(seed)
        wb = self.world_backend
        out = {}
        self._probe_corner = {}     # 2026-09-21: corner-episode share + bend at the goal, per scene
        for sid in self.scene_ids:
            errs = []; bends = []; corner_n = 0
            cal = getattr(wb, "_calib", {}).get(sid)
            for _ in range(int(n)):
                pose = wb.sample_start_pose(sid, rng)
                _corner = bool(getattr(wb, "_corner_episode", False))
                yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
                cone = (wb.last_spawn_base_yaw()
                        if hasattr(wb, "last_spawn_base_yaw") else None)
                g = wb.sample_goal_position(sid, rng, pose[:2, 3],
                                            cone_yaw=cone)
                d = g[:2] - pose[:2, 3]
                e = float(np.arctan2(d[1], d[0])) - yaw
                errs.append(abs(np.degrees((e + np.pi) % (2 * np.pi) - np.pi)))
                if _corner and cal is not None:
                    # bend = walk direction at the goal frame vs at the spawn frame
                    P = np.asarray(cal.positions)[:, :2]
                    fs = int(np.argmin(np.linalg.norm(P - pose[:2, 3], axis=1)))
                    fg = int(np.argmin(np.linalg.norm(P - g[:2], axis=1)))
                    d0, d1 = cal.walk_direction(fs), cal.walk_direction(fg)
                    if d0 is not None and d1 is not None:
                        c = float(np.clip(np.dot(np.asarray(d0[:2]), np.asarray(d1[:2])), -1.0, 1.0))
                        bends.append(float(np.degrees(np.arccos(c)))); corner_n += 1
                        if len(bends) <= 3:
                            self._probe_corner.setdefault(sid + "_examples", []).append((fs, fg, round(bends[-1])))
            e = np.asarray(errs)
            out[sid] = (float(e.mean()), float(e.max()),
                        float(100.0 * (e > 90).mean()))
            self._probe_corner[sid] = (100.0 * corner_n / max(int(n), 1), float(np.mean(bends)) if bends else float("nan"))
        return out

    def goal_corner_probe_result(self) -> dict:
        return dict(getattr(self, "_probe_corner", {}))

    def set_goal_turn_mix(self, m: float) -> float:
        """Corner-rule hook (2026-09-22, Joana: "corner rule is good to ramp up"). A corner goal
        needs goal_turn_beyond_m of walk before the bend AND after it, so it is geometrically
        impossible until the band is roughly 2 x that, and pinning the mix at 1.0 from step zero
        made every episode on a bend scene a forced detour with no easy episodes anywhere: the
        warm arm's mean throttle decayed from +0.92 to +0.16 in 30k steps under it. The curriculum
        raises this as the distance band grows, so the first detour arrives once the policy drives."""
        cfg = getattr(self.world_backend, "cfg", None)
        if cfg is not None:
            cfg.goal_turn_mix = float(m)
        return float(m)

    def set_goal_dist(self, d: float) -> "tuple | None":
        """Distance-curriculum hook (2026-08-29, Joana: E must bootstrap like
        B did): goals START close (~3 m) and GROW as the policy earns wins.
        Backends sample goals, so the knob lives on the backend cfg.

        2026-09-02: this only ever wrote `goal_dist_m`, which
        `sample_goal_position` reads in the RECORDED-FRAME branch. Every arm
        since 458724 passes `--goal_dir_360`, and that branch returns from
        `goal_dist_range` before `goal_dist_m` is ever consulted -- so the
        curriculum has been silently inert on every run for weeks, including
        the ones whose names carry `gds3`. It moves the UPPER bound now: the
        near end of the range stays put (close goals must remain in the mix,
        they are the easy wins that bootstrap learning) and the far end grows
        as the policy earns them.
        """
        cfg = getattr(self.world_backend, "cfg", None)
        if cfg is None:
            return None
        cfg.goal_dist_m = float(d)
        # 2026-09-17: the range is kept for BOTH goal modes now -- the
        # recorded-path branch draws its target distance inside it too.
        if True:
            lo, _hi = getattr(cfg, "goal_dist_range", None) or (3.0, float(d))
            lo = float(lo)
            # Remember where the near end STARTED -- it is a floor the window
            # can never slide below, and reading it off the live range would
            # let it ratchet.
            if self._goal_dist_lo0 is None:
                self._goal_dist_lo0 = lo
            win = getattr(cfg, "goal_dist_window_m", None)
            if win:
                # Sliding window: retire the trivially close goals as the far
                # end grows, instead of pinning the near end forever. At
                # width 5 the range walks (2,3) -> (3,8) rather than (2,8), so
                # the last band the policy has already mastered stops
                # consuming rollout steps.
                lo = max(self._goal_dist_lo0, float(d) - float(win))
            cfg.goal_dist_range = (lo, max(lo + 0.5, float(d)))
        # the episode budget rides the same clock as the band
        _pm = float(getattr(self.cfg, "max_steps_per_m", 0.0) or 0.0)
        if _pm > 0.0:
            _hi = float(getattr(cfg, "goal_dist_range", (0.0, d))[1])
            self.cfg.max_steps = int(round(float(getattr(self.cfg, "max_steps_base", 40)) + _pm * _hi))
        # Returned so the callback can log the ACTUAL range on wandb. Logging
        # only `d` hid the near end, which the sliding window moves.
        return getattr(cfg, "goal_dist_range", None)

    def _remap_labels(self, labels):
        """Apply cfg.label_remap ("12:0,13:0") to a label image via a LUT;
        identity when unset. Counts remapped pixels for the diagnostics."""
        spec = str(getattr(self.cfg, "label_remap", "") or "").strip()
        if not spec or labels is None:
            return labels
        lut = getattr(self, "_label_lut", None)
        if lut is None:
            lut = np.arange(256, dtype=np.int32)
            for tok in spec.split(","):
                a, b = tok.split(":"); lut[int(a)] = int(b)
            self._label_lut = lut
            print(f"[SceneEnv] label remap active on the reward labels: {spec}", flush=True)
        lab = np.asarray(labels).astype(np.int32)
        out = lut[np.clip(lab, 0, 255)]
        self._label_remapped_frac = float((out != lab).mean())
        return out.astype(labels.dtype) if hasattr(labels, "dtype") else out

    def inject_render(self, rgb: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                      labels: "np.ndarray | None" = None,
                      coverage: "float | None" = None,
                      raster: "np.ndarray | None" = None,
                      alpha: "np.ndarray | None" = None) -> None:
        """Batched-live path: the vec-env pushes this robot's frame in after
        the shared batched diffusion call. `labels` feeds the injected
        semantic backend (reward source for the NEXT step); `coverage` is the
        mean alpha of that frame, which only the backend can see."""
        # Frame-to-frame |dRGB|: Joana's measure of world-model coherence
        # (2026-09-02). Alpha coverage is a proxy -- it says how much geometry
        # backs the view -- but the failure she actually cares about is the
        # model CUTTING to a different scene, which is exactly a large frame
        # difference. It is also deployment-safe: real camera frames are
        # temporally continuous, so a term built on this can never fire on the
        # robot, whereas a coverage term teaches "avoid low-alpha headings",
        # which is a property of this reconstruction and not of the world.
        # Logged as a DIAGNOSTIC first -- it also grows with fast turning, so
        # its distribution has to be looked at before it becomes a cost.
        if self._last_rgb is not None and rgb is not None \
                and self._last_rgb.shape == rgb.shape:
            self._rgb_delta = float(
                np.abs(rgb.astype(np.float32)
                       - self._last_rgb.astype(np.float32)).mean() / 255.0)
        else:
            self._rgb_delta = float("nan")
        self._last_rgb, self._last_K, self._last_w2c = rgb, K, w2c
        if labels is not None:
            self._injected_labels = labels
        if coverage is not None:
            self._last_coverage = float(coverage)
        # RGB alignment (2026-09-25 meeting, Jing): the raw Gaussian render the
        # generator was conditioned on, and its per-pixel alpha, so the reward can
        # ask whether the generated frame still resembles the reconstruction where
        # the reconstruction exists.
        self._last_raster = raster
        self._last_alpha = alpha
        self._needs_render = False

    def _spawn_doomed(self, pose: np.ndarray):
        """Non-traversable share of THIS arm's crash box on the map at `pose`
        (void excluded, as the reward). None when the scene has no map."""
        g = getattr(self, "_label_grids", {}).get(self._scene_id)
        if g is None:
            return None
        from src.eval.reward_map import footprint_samples
        pos = np.asarray(pose[:2, 3], dtype=float)
        hd = np.asarray(pose[:2, 0], dtype=float)
        hd = hd / (np.linalg.norm(hd) + 1e-9)
        ahead = (float(self.cfg.collision_look_ahead_m) if self.cfg.collision_look_ahead_m > 0.0
                 else float(self.cfg.look_ahead_dist))
        fp = footprint_samples(pos + ahead * hd, hd, GO2_BODY_LENGTH, GO2_BODY_WIDTH, g.res / 2.0)
        cl = g.lookup(fp).astype(int)
        cl = np.where(cl < 0, 0, cl)
        return float((self._non_trav[cl] & (cl != 0)).mean())

    def _load_obstacle_points(self, scene_id: str) -> None:
        """Body-height obstacle points (xy) from the scene cloud, cached per
        scene. Serves the proximity cost; silently absent -> cost is 0."""
        if not hasattr(self, "_obstacle_pts"):
            self._obstacle_pts = {}
        if not hasattr(self, "_ground_pts"):
            self._ground_pts = {}
        if not hasattr(self, "_support_ref"):
            self._support_ref = {}
        if self.cfg.clouds_dir is None or scene_id in self._obstacle_pts:
            return
        from pathlib import Path
        p = Path(self.cfg.clouds_dir) / f"{scene_id}_cloud.npz"
        if not p.exists():
            self._obstacle_pts[scene_id] = None
            if self.cfg.proximity_weight > 0:
                print(f"[SceneEnv] WARNING: proximity cost ON but no cloud at {p}")
            return
        d = np.load(p)
        pts, labs = d["points"], d["labels"].astype(int)
        # Robot-height band only (0.15-1.2 m): canopy/arches overhang the
        # walkway without blocking a knee-high robot — charging for them
        # taxed walking under trees (audit finding on gnd_AU_60).
        ob = pts[np.isin(labs, list(self.cfg.proximity_classes))
                 & (pts[:, 2] > 0.15) & (pts[:, 2] < 1.2)][:, :2]
        self._obstacle_pts[scene_id] = ob[::4].astype(np.float32) if len(ob) else None
        # GROUND points, separately: `_obstacle_pts` is an obstacle-class slice
        # in a height band and says nothing about whether a patch of ground was
        # reconstructed at all. Goal support needs the latter.
        gr = pts[(pts[:, 2] < 0.15) & (labs >= 0)][:, :2]
        self._ground_pts[scene_id] = gr[::4].astype(np.float32) if len(gr) else None
        # GRASS points for the ground-proximity term (ground height, chosen classes)
        if not hasattr(self, "_grass_pts"):
            self._grass_pts = {}
        _gcls = [int(v) for v in str(self.cfg.proximity_ground_classes).split(",") if v.strip()]
        gc = pts[np.isin(labs, _gcls) & (pts[:, 2] < 0.15)][:, :2]
        self._grass_pts[scene_id] = gc[::4].astype(np.float32) if len(gc) else None
        if float(self.cfg.proximity_ground_weight) > 0.0:
            print(f"[SceneEnv] ground proximity ON: {0 if self._grass_pts[scene_id] is None else len(self._grass_pts[scene_id])} "
                  f"grass points on {scene_id}, weight {self.cfg.proximity_ground_weight}, margin {self.cfg.proximity_ground_margin} m", flush=True)
        if self.cfg.reward_source != "generated" or self.cfg.map_diagnostics:
            from src.eval.reward_map import build_label_grid, VOID
            if not hasattr(self, "_label_grids"):
                self._label_grids = {}
            g = build_label_grid(pts, labs, self._non_trav, res=self.cfg.map_res_m,
                                 inflate_m=self.cfg.map_inflate_m, fill_m=self.cfg.map_fill_m,
                                 fill_max_area_m2=self.cfg.map_fill_max_area_m2,
                                 ignore_classes=tuple(int(v) for v in str(self.cfg.map_ignore_classes).split(",") if v.strip()),
                                 walk_xy=(np.asarray(d["traj_positions"], dtype=float) * np.array([1.0, -1.0, 1.0]))[:, :2]
                                 if "traj_positions" in d else None,
                                 walk_halfwidth_m=self.cfg.map_walk_halfwidth_m,
                                 inflate_classes=tuple(int(v) for v in str(self.cfg.map_inflate_classes).split(",") if v.strip()))
            self._label_grids[scene_id] = g
            if not hasattr(self, "_walk_xy"):
                self._walk_xy = {}
            self._walk_xy[scene_id] = ((np.asarray(d["traj_positions"], dtype=np.float32)
                                        * np.array([1.0, -1.0, 1.0], dtype=np.float32))[:, :2]
                                       if "traj_positions" in d else None)
            known = g.labels >= 0
            nt = known & self._non_trav[np.clip(g.labels, 0, len(self._non_trav) - 1)]
            print(f"[map reward] {scene_id}: grid {g.labels.shape[1]}x{g.labels.shape[0]} "
                  f"@ {g.res} m, known {known.mean():.0%} of cells, non-traversable "
                  f"{nt.sum() / max(known.sum(), 1):.0%} of known", flush=True)
        # Reference density: how many ground points sit within the goal radius
        # at a place we KNOW is reconstructed -- the recorded walk itself.
        ref = 0.0
        g = self._ground_pts[scene_id]
        r = self.cfg.goal_support_radius_m
        if g is not None and len(g) and r > 0.0 and "traj_positions" in d:
            path = (np.asarray(d["traj_positions"], dtype=float)
                    * np.array([1.0, -1.0, 1.0]))[:, :2]
            cnt = [int((np.abs(g - q).max(axis=1) <= r).sum())
                   for q in path[::max(1, len(path) // 40)]]
            ref = float(np.median(cnt)) if cnt else 0.0
        self._support_ref[scene_id] = ref
        if r > 0.0:
            print(f"[SceneEnv] {scene_id}: goal support reference "
                  f"{ref:.0f} ground pts within {r} m on the recorded path; "
                  f"goals need >= {self.cfg.goal_support_min_frac:.0%} of that",
                  flush=True)

    def _proximity_term(self, robot_xy: np.ndarray) -> float:
        if self.cfg.proximity_weight <= 0.0:
            return 0.0
        ob = getattr(self, "_obstacle_pts", {}).get(self._scene_id)
        if ob is None or len(ob) == 0:
            return 0.0
        dmin = float(np.sqrt(((ob - robot_xy[None, :2]) ** 2).sum(1)).min())
        if self.cfg.proximity_delta:
            # Potential-shaped variant: charge APPROACHING an obstacle and
            # refund retreating, weight * (d_{t-1} - d_t), active inside the
            # margin (a wide horizon, e.g. 5 m). No standing charge, so the
            # flat-tax failure mode is impossible by construction.
            prev = getattr(self, "_prev_obstacle_dist", None)
            self._prev_obstacle_dist = dmin
            if prev is None or min(prev, dmin) >= self.cfg.proximity_margin:
                return 0.0
            return -self.cfg.proximity_weight * (prev - dmin)
        if dmin >= self.cfg.proximity_margin:
            return 0.0
        return -self.cfg.proximity_weight * (self.cfg.proximity_margin - dmin) / self.cfg.proximity_margin

    def _ground_proximity_term(self, robot_xy: np.ndarray) -> float:
        w = float(self.cfg.proximity_ground_weight)
        if w <= 0.0:
            return 0.0
        gp = getattr(self, "_grass_pts", {}).get(self._scene_id)
        if gp is None or len(gp) == 0:
            return 0.0
        dmin = float(np.sqrt(((gp - robot_xy[None, :2]) ** 2).sum(1)).min())
        m = float(self.cfg.proximity_ground_margin)
        prev = getattr(self, "_prev_grass_dist", None)
        self._prev_grass_dist = dmin
        self._grass_dist = dmin
        if prev is None:
            return 0.0
        # potential clipped at the margin: phi(d) = w * min(d, m); reward = phi(d_t) - phi(d_{t-1})
        return -w * (min(prev, m) - min(dmin, m))

    def step(self, action: np.ndarray):
        """Chunk-aware step: executes cfg.action_chunk sub-actions (1 = the
        per-action default). Sub-step rewards and components are summed; the
        observation returned is the one AFTER the whole chunk."""
        k = max(1, self.cfg.action_chunk)
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        # the WHOLE decision, before it is split into sub-actions: the plan the
        # policy committed to at this pose (2026-09-16, drawn in the wandb panel
        # and the rollout video). _last_action holds only the sub-action.
        self._last_decision = action.copy()
        self._decision_pose = np.asarray(self._robot_pose_world, dtype=np.float64).copy()
        if getattr(self, "_mirrored", False):
            # the policy acted in the mirrored frame: reflect its yaw back
            action = action.copy()
            action[1::2] = -action[1::2]
        if k == 1:
            return self._step_single(action)
        total_r, agg = 0.0, None
        obs = terminated = truncated = info = None
        for i in range(k):
            obs, r, terminated, truncated, info = self._step_single(
                action[2 * i:2 * i + 2])
            total_r += float(r)
            if agg is None:
                agg = {kk: float(vv) for kk, vv in info.items()
                       if isinstance(vv, (int, float))
                       and not isinstance(vv, bool)
                       and kk != "dist_to_goal"}
            else:
                for kk in agg:
                    # Reward components ARE additive over the sub-steps and are summed.
                    # Ratios, fractions and did-it-fire flags are NOT: summing them
                    # multiplied every chunked arm's diagnostics by k, which made
                    # chunk-5 look like it was diverging (goal_dist_frac read 2.15 =
                    # 5 x 0.43, i.e. exactly in line with the per-action arms) when it
                    # was in fact ahead of them at the same step count. Only
                    # dist_to_goal was excluded before. 2026-09-23.
                    if kk in self.NON_ADDITIVE_INFO or kk.endswith("_frac"):
                        agg[kk] = float(info.get(kk, agg[kk]))     # the chunk's final reading
                    else:
                        agg[kk] += float(info.get(kk, 0.0))
            if terminated or truncated:
                break
        info = dict(info)          # last sub-step's flags/dist survive...
        info.update(agg)           # ...numeric reward components are summed
        info["total"] = total_r
        return obs, total_r, terminated, truncated, info

    # Info fields that are a state reading rather than a quantity accumulated over
    # the step: taking the chunk's LAST value matches what a per-action arm reports
    # at the same pose. Anything ending in _frac is caught by name as well.
    # (2026-09-23, after chunk-5 was misread as diverging.)
    NON_ADDITIVE_INFO = frozenset({
        "goal_dist_frac", "coverage", "rgb_delta", "look_ahead_m", "goal_min_width_m",
        "box_memory_age", "collision_off_frame", "phantom", "missed", "label_agree",
        "trav_agree", "used_generated", "goal_traversable", "halt_correct",
        "halt_at_verge", "halt_wrong", "goal_case_open", "goal_case_corner",
        "goal_case_narrow", "goal_detour", "passed_through_goal", "reach_on_nontrav", "rgb_sim",
        # class ids are labels, not quantities. Summed over a chunk's sub-steps they came out
        # as 35 (= 5 x road) in the chunk arms' eval ground_class_counts and the grass
        # (== 3) trespass check never fired for those arms (2026-09-27). Keep the last one.
        # Diagnostics only: the reward uses fractions, never the id.
        "dominant_class_id", "gen_dominant_class_id", "map_dominant_class_id",
    })

    def _step_single(self, action: np.ndarray):
        assert self._robot_pose_world is not None, "call reset() first"
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        if self.cfg.forward_only and action[0] < 0.0:
            action = action.copy()
            action[0] = 0.0
        stop_cmd = bool(self.cfg.stop_action and len(action) >= 3 and float(action[2]) > 0.0)
        if stop_cmd:
            action = action.copy()
            action[0] = 0.0                  # stop means stop: no motion this step

        # Semantic labels for the current view (from mock cache OR real segmenter).
        semantic_image = self._remap_labels(self.semantic_backend.segment(self._last_rgb))

        # Extract robot position + heading from the 4x4 pose. Heading = local +x
        # of the robot (its "forward"), transformed to world coords.
        robot_position = self._robot_pose_world[:3, 3].copy()
        robot_heading = self._robot_pose_world[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # the pose the action leads to (same kinematics as _advance_pose)
        _next_pos = _next_hd = None
        if bool(self.cfg.collision_at_next_pose) or bool(self.cfg.footprint_next_heading):
            _v = float(action[0]) * self.cfg.step_size_m
            _w = float(action[1]) * self.cfg.yaw_step_rad
            _c, _s = np.cos(_w), np.sin(_w)
            _R = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            _rot = _R @ self._robot_pose_world[:3, :3].astype(np.float64)
            _next_hd = _rot @ np.array([1.0, 0.0, 0.0])
            _next_pos = self._robot_pose_world[:3, 3].astype(np.float64) + _v * _next_hd
        # Footprint direction follows the commanded motion, not the nose
        # (see cfg.footprint_along_motion). Spin-only steps keep forward.
        fp_heading = robot_heading
        if self.cfg.footprint_along_motion and float(action[0]) < 0.0:
            fp_heading = -robot_heading
        if bool(self.cfg.footprint_next_heading) and _next_hd is not None:
            fp_heading = np.asarray(_next_hd, dtype=np.float32)
        self._last_fp_heading = fp_heading

        # ---- decomposed reward on the CURRENT view + robot pose ----
        breakdown_map = None
        grids = getattr(self, "_label_grids", {})
        if self.cfg.reward_source != "generated" and self._scene_id not in grids:
            raise RuntimeError(f"reward_source={self.cfg.reward_source} but no label grid "
                               f"for scene {self._scene_id} -- is clouds_dir set and the "
                               f"cloud present?")
        if self._scene_id in grids:
            from src.eval.reward_map import compute_reward_map
            breakdown_map = compute_reward_map(
                grids[self._scene_id], robot_position=robot_position, robot_heading=fp_heading,
                goal=getattr(self, "_reward_goal", self._goal_world), traversability_scores=self._trav_scores,
                non_traversable_mask=self._non_trav, previous_position=self._prev_position,
                look_ahead_dist=self.cfg.look_ahead_dist,
                collision_look_ahead_dist=(self.cfg.collision_look_ahead_m
                                           if self.cfg.collision_look_ahead_m > 0.0 else None),
                body_length=GO2_BODY_LENGTH, body_width=GO2_BODY_WIDTH, weights=self.cfg.reward,
                collision_position=_next_pos, collision_heading=_next_hd)
        breakdown = compute_reward(
            semantic_image=semantic_image,
            frame_memory=(getattr(self, "_frame_memory", None)
                          if int(self.cfg.collision_box_memory) > 0 else None),
            memory_aggregate=str(getattr(self.cfg, "collision_box_memory_agg", "newest") or "newest"),
            memory_always=bool(getattr(self.cfg, "collision_box_memory_always", False)),
            K=self._last_K,
            w2c=self._last_w2c,
            robot_position=robot_position,
            robot_heading=fp_heading,
            goal=getattr(self, "_reward_goal", self._goal_world),
            traversability_scores=self._trav_scores,
            non_traversable_mask=self._non_trav,
            previous_position=self._prev_position,
            look_ahead_dist=self.cfg.look_ahead_dist,
            collision_look_ahead_dist=(self.cfg.collision_look_ahead_m
                                       if self.cfg.collision_look_ahead_m > 0.0
                                       else None),
            body_length=GO2_BODY_LENGTH,
            body_width=GO2_BODY_WIDTH,
            weights=self.cfg.reward,
        )
        # the two readings of the same footprint, logged before either is chosen
        self._map_vs_gen = None
        if breakdown_map is not None:
            # The disagreement threshold used to be COLLTERM itself. With COLLTERM=0.01 in
            # the strict arm that would flag phantom/missed on a 1% difference between two
            # label sources and charge DISAGREE almost every step. disagree_frac decouples it;
            # unset keeps the old behaviour so earlier diagnostics stay comparable (2026-09-25).
            _df = getattr(self.cfg, "disagree_frac", None)
            _thr = float(_df) if _df else max(self.cfg.collision_terminate_frac, 1e-9)
            _w = max(self.cfg.reward.collision, 1e-6)
            _fg = -float(breakdown.collision) / _w
            _fm = -float(breakdown_map.collision) / _w
            # If the image's crash box fell below the frame, its collision
            # reading is the FAR box's (compute_reward's fallback) and no
            # longer the same box as the map's: exclude the step from the
            # traversability comparison (NaN is skipped by the loggers).
            _off = float(getattr(breakdown, "collision_off_frame", 0.0)) > 0.0
            _nan = float("nan")
            self._map_vs_gen = {
                "gen_collision_frac": _nan if _off else _fg, "map_collision_frac": _fm,
                "phantom": _nan if _off else float(_fg >= _thr and _fm < _thr),
                "missed": _nan if _off else float(_fm >= _thr and _fg < _thr),
                "label_agree": float(int(breakdown.dominant_class_id) == int(breakdown_map.dominant_class_id)),
                "gen_dominant_class_id": int(breakdown.dominant_class_id),
                "map_dominant_class_id": int(breakdown_map.dominant_class_id),
                # Joana 2026-09-04: road / sidewalk / pavement swaps are semantic
                # disagreements that change nothing for the reward. This is the
                # agreement that matters: do both call the footprint walkable?
                "trav_agree": float(bool(self._non_trav[max(int(breakdown.dominant_class_id), 0)])
                                    == bool(self._non_trav[max(int(breakdown_map.dominant_class_id), 0)])),
                "map_void_frac": float(breakdown_map.void_frac),
            }
        if self.cfg.reward_source == "map":
            breakdown = breakdown_map
        elif self.cfg.reward_source == "split":
            # 2026-09-07 (new scope): the graded TERRAIN term reads the generated
            # labels (what the policy sees), the CRASH term (near box) reads the
            # MAP. The near box sits below the camera, so from the image it can
            # only come through the frame memory (hit 20-80%), and a box that
            # swaps between "remembered frame" and "far box fallback" from step
            # to step is a moving target; the map judges it identically every step.
            import dataclasses
            breakdown = dataclasses.replace(breakdown, collision=breakdown_map.collision,
                                            void_frac=breakdown_map.void_frac)
        elif self.cfg.reward_source == "either":
            # 2026-09-25 (Jing): if EITHER the generated labels or the fused map say
            # the footprint is on something non-traversable, it is a collision. The
            # terrain term keeps reading the generated labels (what the policy sees);
            # only the collision fraction is the max of the two readings.
            import dataclasses
            breakdown = dataclasses.replace(
                breakdown, collision=max(float(breakdown.collision), float(breakdown_map.collision)))
        elif self.cfg.reward_source == "map_then_generated":
            # the map where it has support; the image only where the footprint
            # is mostly off the reconstruction AND the view is still well
            # supported (alpha), so a grounded generator fills the gap and an
            # ungrounded one does not
            _cov = getattr(self, "_last_coverage", None)
            _use_gen = (breakdown_map.void_frac >= self.cfg.map_fallback_void_frac
                        and (_cov is None or float(_cov) >= self.cfg.map_fallback_min_alpha))
            if not _use_gen:
                breakdown = breakdown_map
            if self._map_vs_gen is not None:
                self._map_vs_gen["used_generated"] = float(_use_gen)

        # Collision snapshot: the view + semantics + numbers behind every
        # non-traversable footprint, saved BEFORE the pose advances so the
        # figure shows exactly what the reward punished.
        if (self.cfg.failure_snap_dir is not None
                and breakdown.collision < -self.cfg.failure_snap_min_frac
                and self._failure_snaps < self.cfg.failure_snap_max):
            self._save_failure_snapshot(breakdown, semantic_image)
        # MEMORY CRASH STRIP (2026-09-20, Joana: "visualize the crash footprint
        # from the previous frames it looked at"): when a crash-level reading
        # was decided from stored frames, save every stored frame that
        # contained the near box, with the box drawn and its share, beside the
        # current view. Saved BEFORE the memory buffer takes the current frame,
        # so age N here is the same frame age N the reward read.
        if (self.cfg.failure_snap_dir is not None
                and int(getattr(breakdown, "memory_hits", 0)) > 0
                and self.cfg.collision_terminate_frac > 0.0
                and -float(breakdown.collision) / max(self.cfg.reward.collision, 1e-6) >= self.cfg.collision_terminate_frac
                and self._memory_strips < self.cfg.failure_snap_max):
            self._save_memory_crash_strip(breakdown)
        elif (self.cfg.failure_snap_dir is not None and int(getattr(self.cfg, "memory_snap_every", 0)) > 0
                and self._steps > 0 and self._steps % int(self.cfg.memory_snap_every) == 0
                and len(getattr(self, "_frame_memory", None) or []) > 0
                and self._memory_strips < self.cfg.failure_snap_max):
            self._save_memory_crash_strip(breakdown, prefix="memory")

        # ---- apply the action to advance the robot pose ----
        if int(self.cfg.collision_box_memory) > 0:
            # stored AFTER this step's reward: memory holds previous frames only
            fm = getattr(self, "_frame_memory", None)
            if fm is None:
                fm = self._frame_memory = []
            fm.append((semantic_image, self._last_K, self._last_w2c))
            if len(fm) > int(self.cfg.collision_box_memory):
                del fm[0]
            if int(getattr(self.cfg, "memory_snap_every", 0)) > 0:
                # RGB copies for the strip figure only (2026-09-28); parallel to fm, same ages
                fr = getattr(self, "_frame_memory_rgb", None)
                if fr is None:
                    fr = self._frame_memory_rgb = []
                fr.append(None if self._last_rgb is None else self._last_rgb.copy())
                if len(fr) > int(self.cfg.collision_box_memory):
                    del fr[0]
        self._prev_position = robot_position
        self._advance_pose(action)
        self._steps += 1
        if getattr(self, "_ep_xy", None) is not None:
            self._ep_xy.append(tuple(float(v) for v in self._robot_pose_world[:2, 3]))

        # ---- render the new view for the NEXT step's observation ----
        if self.cfg.defer_render:
            self._needs_render = True     # owning vec-env batches the render
        else:
            self._render_current()

        # Reached goal?
        dist_to_goal = float(np.linalg.norm(self._robot_pose_world[:3, 3] - self._goal_world))
        # stillness run, counted BEFORE the terminal decision (goal_requires_stop)
        if abs(float(action[0])) < self.cfg.halt_throttle_eps:
            self._still_run = int(getattr(self, "_still_run", 0)) + 1
        else:
            self._still_run = 0
        terminated = dist_to_goal < self.cfg.goal_radius
        if terminated and bool(self.cfg.goal_requires_stop):
            _need = int(self.cfg.halt_terminate_steps) if int(self.cfg.halt_terminate_steps) > 0 else 3
            if self._still_run < _need:
                terminated = False            # inside the radius but still moving: keep going
                self._in_radius_moving = True
        if terminated and bool(self.cfg.nontrav_goal_unreachable) \
                and float(getattr(self, "_goal_traversable", float("nan"))) == 0.0:
            terminated = False
            self._entered_nontrav_goal = True
        truncated = self._steps >= self.cfg.max_steps
        # refusal by DWELL at the verge (lawn goals only; walkable goals keep the halt rules)
        dwell_refusal = False
        if (int(self.cfg.refusal_dwell_steps) >= 0 and not terminated
                and float(getattr(self, "_goal_traversable", float("nan"))) == 0.0):
            _vp = self._refusal_point(self._goal_world)
            if _vp is not None and float(np.linalg.norm(self._robot_pose_world[:2, 3] - _vp)) <= float(self.cfg.refusal_verge_m):
                _still = int(getattr(self, "_still_run", 0)) + (1 if stop_cmd and int(getattr(self, "_still_run", 0)) == 0 else 0)
                if _still >= int(self.cfg.refusal_dwell_steps):
                    dwell_refusal = True

        spin_term = -self.cfg.spin_cost * abs(float(action[1]))
        back_term = -self.cfg.backward_cost * max(0.0, -float(action[0]))
        smooth_term = 0.0
        if self.cfg.action_smooth_cost > 0.0:
            prev = getattr(self, "_last_action", None)
            if prev is not None:
                smooth_term = -self.cfg.action_smooth_cost * float(
                    np.abs(action - prev).mean())
            self._last_action = action.copy()
        # position trail for the wandb panel (last 8 poses, 2026-09-16)
        _tr = getattr(self, "_pos_trail", None)
        if _tr is None:
            _tr = self._pos_trail = []
        _tr.append(np.asarray(self._robot_pose_world[:3, 3], dtype=np.float64).copy())
        del _tr[:-8]
        bonus = self.cfg.goal_bonus if terminated else 0.0

        # Crash termination (2026-08-20). Measured on the tree test: a whole
        # episode of walking through a trunk cost -0.60 against a +50 arrival
        # bonus (1.2%), so driving through obstacles was reward-optimal — the
        # policy passed within 0.03-0.21 m of trunk centres while scoring
        # "success 1.0". A real collision must END the episode, like a real
        # robot's would. 0 = off (every result before this date).
        crash = 0.0
        phantom_vetoed = 0.0
        if self.cfg.collision_terminate_frac > 0.0:
            frac = -float(breakdown.collision) / max(self.cfg.reward.collision, 1e-6)
            _moving = (not bool(self.cfg.crash_requires_motion)) or abs(float(action[0])) >= float(self.cfg.crash_motion_eps)
            if frac >= self.cfg.collision_terminate_frac and _moving:
                # PHANTOM VETO (2026-09-19, Joana: "map veto is not a bad
                # idea"): the eval of the old A checkpoint found 8/8 crashes
                # were person/vehicle labels on pavement the fused map knows
                # to be walkable. With the veto such a step keeps its contact
                # cost (the policy still learns to avoid it) but does not end
                # the episode. The map must actually know the footprint
                # (at most half of it on unknown cells) and its own no-go
                # share must be below the crash threshold.
                _mv = getattr(self, "_map_vs_gen", None)
                _map_knows = (breakdown_map is not None
                              and float(getattr(breakdown_map, "void_frac", 1.0)) <= 0.5)
                if (bool(getattr(self.cfg, "phantom_veto", False)) and _mv and _map_knows
                        and float(_mv.get("map_collision_frac", 1.0)) < self.cfg.collision_terminate_frac):
                    phantom_vetoed = 1.0
                    self._phantom_vetoes = int(getattr(self, "_phantom_vetoes", 0)) + 1
                else:
                    crash = -self.cfg.collision_terminate_penalty
                    truncated = True          # ends the episode, and NOT a success
                    bonus = 0.0
        # VOID TERMINATION (2026-09-01, her concern): with the alpha gate ON,
        # unobserved regions become void and stop counting as collisions — but
        # then "walk into the unobserved" becomes a way to dodge real terrain:
        # the policy learns to exploit the world model's blind spots, and that
        # behavior means nothing on the real robot. Pessimism under uncertainty
        # (MOReL-style HALT): leaving the known world ENDS the episode at a
        # moderate cost — clearly worse than walking correctly, far cheaper
        # than a real crash. 0 = off (every run before this date).
        if (self.cfg.void_terminate_frac > 0.0 and crash == 0.0
                and float(breakdown.void_frac) >= self.cfg.void_terminate_frac):
            crash = -self.cfg.void_terminate_penalty
            truncated = True
            bonus = 0.0
        # Whole-image coverage version (her spec): unsupported GROUND is fine
        # if the VIEW is still well supported — the world model can paint a
        # coherent scene and the robot can reasonably step there. Only end the
        # episode when the view itself is mostly invention.
        image_void_frac = float((semantic_image == 0).mean())
        if (self.cfg.image_void_terminate_frac > 0.0 and crash == 0.0
                and image_void_frac >= self.cfg.image_void_terminate_frac):
            crash = -self.cfg.void_terminate_penalty
            truncated = True
            bonus = 0.0
        prox_term = self._proximity_term(self._robot_pose_world[:2, 3])
        gprox_term = self._ground_proximity_term(self._robot_pose_world[:2, 3])

        # ---- coherence: is this frame still the world? ----
        coverage = getattr(self, "_last_coverage", None)
        coh_term, coh_crash = 0.0, 0.0
        if coverage is None and (self.cfg.coherence_cost_weight > 0.0
                                 or self.cfg.coherence_terminate_tau > 0.0):
            # Only the BATCHED live backend reports per-frame alpha. Asking for
            # coherence on a path that cannot supply it would silently train a
            # run with the mechanism switched off and nobody the wiser — the
            # exact failure shape as the SANPO length filter. Say so, loudly.
            if not getattr(self, "_coh_warned", False):
                print("[SceneEnv] WARNING: coherence requested but this backend "
                      "reports no per-frame alpha — the term is INERT.", flush=True)
                self._coh_warned = True
        if coverage is not None:
            if self.cfg.coherence_cost_weight > 0.0:
                coh_term = -self.cfg.coherence_cost_weight * max(
                    0.0, self.cfg.coherence_tau - coverage)
            if (self.cfg.coherence_terminate_tau > 0.0 and crash == 0.0
                    and coverage < self.cfg.coherence_terminate_tau):
                coh_crash = -self.cfg.coherence_terminate_penalty
                # truncated, NOT terminated — same convention as crash/void
                # above. info["reached_goal"] mirrors `terminated`, so setting
                # it here would score every coherence kill as a goal arrival.
                truncated = True
                bonus = 0.0
        timeout_term = 0.0
        # ---- HALTED SAFELY ----
        halted = False
        if (self.cfg.halt_terminate_steps > 0 and not terminated
                and crash == 0.0 and coh_crash == 0.0):
            if abs(float(action[0])) < self.cfg.halt_throttle_eps:
                self._halt_run += 1
            else:
                self._halt_run = 0
            _fr = -float(breakdown.collision) / max(self.cfg.reward.collision, 1e-6)
            _d0 = float(getattr(self, "_initial_goal_dist", 0.0) or 0.0)
            if ((self._halt_run >= self.cfg.halt_terminate_steps or stop_cmd or dwell_refusal)
                    and _fr < max(self.cfg.collision_terminate_frac, 1e-9)
                    and _d0 > 1e-6 and dist_to_goal < _d0):
                halted = True
                truncated = True

        if (truncated and not terminated and crash == 0.0 and coh_crash == 0.0
                and self.cfg.timeout_penalty > 0.0):
            timeout_term = -self.cfg.timeout_penalty
            if self.cfg.timeout_distance_scaled:
                d0 = float(getattr(self, "_initial_goal_dist", 0.0) or 0.0)
                frac = 1.0 if d0 <= 1e-6 else min(1.0, max(0.0, dist_to_goal / d0))
                timeout_term *= frac
            if halted:
                timeout_term *= float(self.cfg.halt_penalty_scale)
        refusal_term = 0.0
        if halted:
            # judged for EVERY halt on a lawn goal, bonus or not, so the no-bonus
            # controls and their evals report halt_at_verge too (review 09-05)
            _gt0 = float(getattr(self, "_goal_traversable", float("nan")))
            _rp = self._refusal_point(self._goal_world) if _gt0 == 0.0 else None
            _d_rp = (float(np.linalg.norm(self._robot_pose_world[:2, 3] - _rp)) if _rp is not None else float("inf"))
            # The verge is the stopping place (Joana 09-05 evening). The distance
            # to the goal only stands in when the scene has no verge (no map).
            if _rp is not None:
                self._halt_at_verge = bool(_gt0 == 0.0 and _d_rp <= float(self.cfg.refusal_verge_m))
            else:
                self._halt_at_verge = bool(_gt0 == 0.0 and float(dist_to_goal) <= float(self.cfg.refusal_dist_m))
            if self._halt_at_verge and float(self.cfg.refusal_bonus) > 0.0:
                refusal_term = float(self.cfg.refusal_bonus)
                timeout_term = 0.0            # a correct refusal pays no halt price
        if halted and float(self.cfg.halt_wrong_penalty) > 0.0:
            if float(getattr(self, "_goal_traversable", float("nan"))) == 1.0:
                refusal_term -= float(self.cfg.halt_wrong_penalty)
        speed_refund = 0.0
        if self.cfg.terrain_speed_scaled:
            _thr = min(1.0, abs(float(action[0])))
            speed_refund = -(float(breakdown.semantic) + float(breakdown.collision)) * (1.0 - _thr)
        # ---- misalignment (2026-09-25, Jing): the generated labels and the fused map
        # disagree about what is under the footprint. phantom = generated says crash,
        # map says clear; missed = the reverse. Both are NaN when the box is off-frame
        # (no comparison possible) and then cost nothing.
        disagree_term = 0.0
        if float(getattr(self.cfg, "disagree_cost_weight", 0.0) or 0.0) > 0.0 and self._map_vs_gen:
            _ph = float(self._map_vs_gen.get("phantom", 0.0) or 0.0)
            _mi = float(self._map_vs_gen.get("missed", 0.0) or 0.0)
            _ph = 0.0 if _ph != _ph else _ph
            _mi = 0.0 if _mi != _mi else _mi
            # VOID is not an opinion. Where the map has no reconstruction under the
            # footprint it reads "clear" (void is excluded from its collision fraction),
            # so a generator that paints an obstacle there would be charged as a phantom
            # for doing exactly what it is for. No disagreement is scored when most of
            # the footprint is void on the map (2026-09-25, Joana: "what happens when the
            # map says void?").
            _mv = float(self._map_vs_gen.get("map_void_frac", 0.0) or 0.0)
            if _mv == _mv and _mv >= 0.5:
                _ph = _mi = 0.0
            disagree_term = -float(self.cfg.disagree_cost_weight) * max(_ph, _mi)
        # ---- RGB alignment (2026-09-25, Jing): does the generated frame still look
        # like the reconstruction where the reconstruction exists?
        align_term, rgb_sim = 0.0, float("nan")
        if float(getattr(self.cfg, "rgb_align_cost_weight", 0.0) or 0.0) > 0.0:
            _sim = rgb_alignment(getattr(self, "_last_rgb", None), getattr(self, "_last_raster", None),
                                 getattr(self, "_last_alpha", None),
                                 float(self.cfg.rgb_align_min_alpha), float(self.cfg.rgb_align_min_frac))
            if _sim is not None:
                rgb_sim = _sim
                align_term = -float(self.cfg.rgb_align_cost_weight) * max(0.0, float(self.cfg.rgb_align_tau) - _sim)
        reward = (breakdown.total + spin_term + back_term + smooth_term
                  + bonus + crash + prox_term + gprox_term + timeout_term
                  + coh_term + coh_crash + speed_refund + refusal_term + disagree_term + align_term)

        info = breakdown.to_dict()
        _age = float(getattr(breakdown, "box_memory_age", 0.0))
        info["box_memory_hit"] = float(_age > 0.0)
        info["box_memory_miss"] = float(_age < 0.0)
        info["spin"] = spin_term
        info["backward"] = back_term
        info["smooth"] = smooth_term
        # Always logged (even with both void terminations off) so the coverage
        # the policy actually experiences is visible in wandb from day one.
        info["image_void_frac"] = image_void_frac
        info["timeout"] = timeout_term
        info["speed_refund"] = speed_refund
        info["goal_dist_frac"] = (
            dist_to_goal / self._initial_goal_dist
            if getattr(self, "_initial_goal_dist", 0.0) else float("nan"))
        info["crash"] = crash
        info["phantom_vetoed"] = phantom_vetoed
        info["halted"] = float(halted)
        info["refusal_bonus"] = float(refusal_term)          # net: bonus paid minus wrong-halt penalty
        info["refusal_paid"] = float(max(refusal_term, 0.0))
        info["halt_penalty_paid"] = float(max(-refusal_term, 0.0))
        _gt = float(getattr(self, "_goal_traversable", float("nan")))
        info["goal_traversable"] = _gt
        # end-of-episode flags, nan on every other step so per-rollout means
        # are over episode ENDS: the refusal metric
        _end = bool(terminated or truncated)
        _nan = float("nan")
        info["goal_walkable_frac"] = float(getattr(self, "_goal_walkable_frac", float("nan")))
        info["halt_correct"] = (float(halted and _gt == 0.0) if _end and _gt == _gt else _nan)
        # ... and at the RIGHT place: within refusal_dist_m of the goal or of the verge nearest it
        info["passed_through_goal"] = (float(bool(getattr(self, "_in_radius_moving", False)) and not bool(terminated)) if _end else _nan)
        info["halt_at_verge"] = (float(halted and _gt == 0.0 and bool(getattr(self, "_halt_at_verge", False))) if _end and _gt == _gt else _nan)
        info["halt_wrong"] = (float(halted and _gt == 1.0) if _end and _gt == _gt else _nan)
        _reached_nt = bool(terminated) or bool(getattr(self, "_entered_nontrav_goal", False))
        info["reach_on_nontrav"] = (float(_reached_nt and _gt == 0.0) if _end and _gt == _gt else _nan)
        if getattr(self, "_map_vs_gen", None):
            info.update(self._map_vs_gen)
        info["proximity"] = prox_term
        info["look_ahead_m"] = float(self.cfg.look_ahead_dist)
        _gc = str(getattr(self, "_goal_case", "unknown"))
        info["goal_case_open"] = float(_gc == "open")
        info["goal_case_corner"] = float(_gc in ("corner", "corner+narrow"))
        info["goal_case_narrow"] = float(_gc in ("narrow", "corner+narrow"))
        info["goal_detour"] = float(getattr(self, "_goal_detour", float("nan")))
        info["goal_min_width_m"] = float(getattr(self, "_goal_min_width", float("nan")))
        info["proximity_ground"] = gprox_term
        info["grass_dist"] = float(getattr(self, "_grass_dist", float("nan")))
        # Always logged, on or off, so the coverage the policy actually
        # experiences shows up in wandb from the first run.
        info["coverage"] = float("nan") if coverage is None else coverage
        info["rgb_delta"] = float(getattr(self, "_rgb_delta", float("nan")))
        info["coherence"] = coh_term
        info["coherence_crash"] = coh_crash
        info["disagree"] = disagree_term
        info["rgb_align"] = align_term
        info["rgb_sim"] = rgb_sim
        info["goal_bonus"] = bonus
        info["total"] = reward
        info["dist_to_goal"] = dist_to_goal
        info["reached_goal"] = terminated
        # reward_scale (2026-08-28): shrink the WHOLE reward uniformly (ratios
        # preserved) so +-1000 terminals stop blowing up the critic's targets
        # (value_loss ~1e5 in the batch-1 RW5 trio). Info keys stay unscaled
        # so component logging remains in Jing-spec units.
        if self.cfg.reward_scale != 1.0:
            reward = float(reward) * self.cfg.reward_scale
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        # what the POLICY sees: in a mirrored episode the image is flipped (2026-09-21, so that
        # rollout videos show the policy's view; the failure snapshots keep the world view)
        if getattr(self, "_mirrored", False) and self._last_rgb is not None:
            return self._last_rgb[:, ::-1]
        return self._last_rgb

    # ---------------- helpers ----------------

    def _obs(self) -> dict:
        goal_robot = self._goal_in_robot_frame()
        if getattr(self, "_mirrored", False):
            goal_robot = goal_robot.copy()
            goal_robot[1] = -goal_robot[1]
            goal_robot[2] = -goal_robot[2]
        if self.cfg.goal_noise_std > 0.0:
            goal_robot = goal_robot.copy()
            goal_robot[:2] += self.np_random.normal(
                0.0, self.cfg.goal_noise_std, size=2)
        rgb = self._last_rgb
        if getattr(self, "_mirrored", False):
            rgb = rgb[:, ::-1]
        if self.cfg.obs_out_hw is not None:
            oh, ow = int(self.cfg.obs_out_hw[0]), int(self.cfg.obs_out_hw[1])
            if rgb.shape[0] != oh or rgb.shape[1] != ow:
                import cv2
                rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
        _k = max(1, int(getattr(self.cfg, "obs_frame_stack", 1)))
        _st = max(1, int(getattr(self.cfg, "obs_frame_stride", 1)))
        if _k > 1:
            # one append per environment step: _obs() may be called again for the same step
            # (inject_render, debug rebuilds) and must not shift the history when it is
            _key = (self._scene_id, int(getattr(self, "_steps", 0)))
            if getattr(self, "_rgb_hist", None) is None:
                self._rgb_hist, self._rgb_hist_key = [], None
            _need = (_k - 1) * _st + 1                        # frames to remember for this stride
            if not self._rgb_hist:
                self._rgb_hist = [rgb] * _need                   # episode start: repeat the first view
                self._rgb_hist_key = _key
            elif self._rgb_hist_key != _key:
                self._rgb_hist = (self._rgb_hist + [rgb])[-_need:]
                self._rgb_hist_key = _key
            sel = [self._rgb_hist[-1 - i * _st] for i in range(_k)][::-1]   # oldest ... newest
            rgb = np.concatenate(sel, axis=2)
        return {
            "rgb":  rgb.copy(),
            "goal": goal_robot.astype(np.float32),
        }

    def _render_current(self) -> None:
        rgb, K, w2c = self.world_backend.render(self._robot_pose_world)
        _cov = getattr(self.world_backend, "last_coverage", None)
        if _cov is not None:
            self._last_coverage = float(_cov)      # coherence terms + by-alpha table in evals
        # Frame-to-frame |dRGB|: Joana's measure of world-model coherence
        # (2026-09-02). Alpha coverage is a proxy -- it says how much geometry
        # backs the view -- but the failure she actually cares about is the
        # model CUTTING to a different scene, which is exactly a large frame
        # difference. It is also deployment-safe: real camera frames are
        # temporally continuous, so a term built on this can never fire on the
        # robot, whereas a coverage term teaches "avoid low-alpha headings",
        # which is a property of this reconstruction and not of the world.
        # Logged as a DIAGNOSTIC first -- it also grows with fast turning, so
        # its distribution has to be looked at before it becomes a cost.
        if self._last_rgb is not None and rgb is not None \
                and self._last_rgb.shape == rgb.shape:
            self._rgb_delta = float(
                np.abs(rgb.astype(np.float32)
                       - self._last_rgb.astype(np.float32)).mean() / 255.0)
        else:
            self._rgb_delta = float("nan")
        self._last_rgb, self._last_K, self._last_w2c = rgb, K, w2c

    def _advance_pose(self, action: np.ndarray) -> None:
        v_forward = float(action[0]) * self.cfg.step_size_m
        omega_yaw = float(action[1]) * self.cfg.yaw_step_rad

        # Rotate yaw first (about world +z), then translate forward.
        c, s = np.cos(omega_yaw), np.sin(omega_yaw)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        self._robot_pose_world[:3, :3] = R_yaw @ self._robot_pose_world[:3, :3]

        forward_world = self._robot_pose_world[:3, :3] @ np.array([1.0, 0.0, 0.0])
        self._robot_pose_world[:3, 3] += v_forward * forward_world

    def _goal_in_robot_frame(self) -> np.ndarray:
        # (dx, dy, dyaw_to_goal) in the robot's local frame.
        dp_world = self._goal_world - self._robot_pose_world[:3, 3]
        R_wr = self._robot_pose_world[:3, :3]                                 # world->robot rotation
        dp_robot = R_wr.T @ dp_world
        dyaw = float(np.arctan2(dp_robot[1], dp_robot[0]))
        return np.array([dp_robot[0], dp_robot[1], dyaw])
