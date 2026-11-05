"""PPO first version on a REAL scene (Milestone B/C glue — Thursday deliverable).

One RUGD scene, rasterizer-only observations (cheap, holey — fine for the loop
smoke), reward from Gaussian-rasterized labels (real observed geometry; never
generated RGB). All distances in REAL METERS via the extract_poses calibration.

Usage (Marlowe, GPU):
    python scripts/train_ppo_real.py \
        --scene rugd_trail_00 \
        --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
        --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
        --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
        --total_steps 10000 \
        --output_dir outputs/ppo_real_trail00

Outputs: ppo_final.zip, checkpoints/, tensorboard/, rollout.mp4 (eval episode).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


class HaltPriceCurriculum(BaseCallback):
    """The halt is earned (2026-09-04). halt_penalty_scale starts at `start`
    (1.0 = a halt pays the full distance-scaled timeout) and notches down by
    `notch` toward `end` each time the last `window` episodes win >= threshold,
    re-earning each notch -- the same rule as GoalRadiusCurriculum. Warm arms
    reach `end` within an hour; a cold arm must learn to reach goals before
    refusing becomes cheap, which is what stops the freeze."""

    def __init__(self, start: float, end: float, window: int = 100,
                 threshold: float = 0.5, notch: float = 0.1, gate: bool = False):
        super().__init__()
        self.start, self.end, self.window, self.threshold, self.notch = start, end, window, threshold, notch
        self.s = start
        self._wins: list = []
        # gate (2026-09-05): halting is switched OFF until the first threshold
        # pass, then on at `start` price, then notches as before.
        self.gate = bool(gate)
        self._enabled = not self.gate

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._wins.append(1.0 if info.get("goal_bonus", 0.0) > 0 else 0.0)
        return True

    def _on_rollout_start(self) -> None:
        recent = self._wins[-self.window:]
        passed = (len(recent) >= self.window // 2 and float(np.mean(recent)) >= self.threshold)
        if self.gate and not self._enabled:
            if passed:
                self._enabled = True
                self._wins.clear()
        elif passed and self.s > self.end:
            self.s = max(self.end, self.s - self.notch)
            self._wins.clear()
        if self.gate:
            self.training_env.env_method("set_halt_enabled", bool(self._enabled))
        self.training_env.env_method("set_halt_penalty_scale", self.s)
        if self.logger is not None:
            self.logger.record("curriculum/halt_scale", self.s)
            if self.gate:
                self.logger.record("curriculum/halt_enabled", float(self._enabled))


class HaltWrongRamp(BaseCallback):
    """Wrong-halt penalty ramp (2026-09-06): 0 at the start, linear to `end`
    over `ramp_steps` environment steps, then held. With a stop action or a
    one-step halt the policy halts constantly at first; a full penalty from
    step 0 would teach 'never halt' before a single verge bonus is seen."""

    def __init__(self, end: float, ramp_steps: int):
        super().__init__()
        self.end, self.ramp = float(end), max(1, int(ramp_steps))
        self._t0 = None

    def _on_training_start(self) -> None:
        # a warm continuation starts at the parent's step count: ramp from HERE
        self._t0 = int(self.num_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self._t0 is None:
            self._t0 = int(self.num_timesteps)
        p = self.end * min(1.0, float(self.num_timesteps - self._t0) / self.ramp)
        self.training_env.env_method("set_halt_wrong_penalty", p)
        if self.logger is not None:
            self.logger.record("curriculum/halt_wrong_penalty", p)


class RefusalDwellCurriculum(BaseCallback):
    """Joana's dwell curriculum (2026-09-06). The still steps required inside
    the verge zone start at `start` (0 = arrival alone refuses) and rise by
    one, up to `end` (3 = the real halt), each time at least half of the last
    `window` lawn-goal episodes ended with a refusal at the verge. Logs
    curriculum/refusal_dwell."""

    def __init__(self, start: int, end: int = 3, window: int = 100, threshold: float = 0.5):
        super().__init__()
        self.n, self.end, self.window, self.threshold = int(start), int(end), window, threshold
        self._hits: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and info.get("goal_traversable", float("nan")) == 0.0:
                v = info.get("halt_at_verge", float("nan"))
                if v == v:
                    self._hits.append(1.0 if v > 0 else 0.0)
        return True

    def _on_rollout_start(self) -> None:
        recent = self._hits[-self.window:]
        if (len(recent) >= self.window // 2 and float(np.mean(recent)) >= self.threshold
                and self.n < self.end):
            self.n += 1
            self._hits.clear()
        self.training_env.env_method("set_refusal_dwell", self.n)
        if self.logger is not None:
            self.logger.record("curriculum/refusal_dwell", float(self.n))


class VergeRadiusCurriculum(BaseCallback):
    """The verge radius is earned (2026-09-05, Joana). Starts at `start` m and
    notches by `notch` toward `end` each time at least half of the last
    `window` lawn-goal episodes ended with a halt at the verge; holds
    otherwise. Logs curriculum/verge_radius."""

    def __init__(self, start: float, end: float, window: int = 100,
                 threshold: float = 0.5, notch: float = 0.25):
        super().__init__()
        self.start, self.end, self.window, self.threshold, self.notch = start, end, window, threshold, notch
        self.r = start
        self._hits: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and info.get("goal_traversable", float("nan")) == 0.0:
                v = info.get("halt_at_verge", float("nan"))
                if v == v:
                    self._hits.append(1.0 if v > 0 else 0.0)
        return True

    def _on_rollout_start(self) -> None:
        recent = self._hits[-self.window:]
        if (len(recent) >= self.window // 2 and float(np.mean(recent)) >= self.threshold
                and self.r > self.end):
            self.r = max(self.end, self.r - self.notch)
            self._hits.clear()
        self.training_env.env_method("set_refusal_verge", self.r)
        if self.logger is not None:
            self.logger.record("curriculum/verge_radius", self.r)


class GoalRadiusCurriculum(BaseCallback):
    """Anneal the goal-capture radius start -> end (advisor spec 2026-08-27).
    Attacks terminal-capture: a 1.0 m disc is reachable by a fresh policy;
    0.5 m is the graduation exam.

    mode="success" (default, Joana 2026-08-28): the radius only SHRINKS once
    the policy wins >= threshold of its recent episodes at the current radius
    (5 cm notches), and holds while it struggles — a timer would starve a
    policy that hasn't learned the big disc yet. mode="time": linear anneal
    over `steps` timesteps (predictable fallback)."""

    def __init__(self, start: float, end: float, steps: int,
                 mode: str = "success", window: int = 100,
                 threshold: float = 0.5, notch: float = 0.05):
        super().__init__()
        self.start, self.end, self.steps = start, end, steps
        self.mode, self.window, self.threshold, self.notch = (
            mode, window, threshold, notch)
        self.r = start
        self._wins: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:            # episode ended at this step
                self._wins.append(1.0 if info.get("goal_bonus", 0.0) > 0
                                  else 0.0)
        return True

    def _on_rollout_start(self) -> None:
        if self.mode == "time":
            t = min(1.0, self.model.num_timesteps / max(1, self.steps))
            self.r = self.start + (self.end - self.start) * t
        else:
            recent = self._wins[-self.window:]
            if (len(recent) >= self.window // 2
                    and float(np.mean(recent)) >= self.threshold
                    and self.r > self.end):
                self.r = max(self.end, self.r - self.notch)
                self._wins.clear()           # re-earn the next notch
        self.training_env.env_method("set_goal_radius", self.r)
        if self.logger is not None:
            self.logger.record("curriculum/goal_radius", self.r)
            if self._wins:
                self.logger.record("curriculum/recent_success",
                                   float(np.mean(self._wins[-self.window:])))


class GoalDistCurriculum(BaseCallback):
    """Success-gated goal-DISTANCE growth (2026-08-29, Joana's catch: fixed
    6 m goals removed the easy-win bootstrap that made Arm B learn — B's
    frame-range goals included near-spawn gimmes that fed its curriculum).
    Goals start close and grow a notch each time the policy earns >= threshold
    recent wins, up to the target distance. Runs alongside the radius
    curriculum; both gates draw on the same win stream."""

    def __init__(self, start: float, end: float, window: int = 100,
                 threshold: float = 0.5, notch: float = 0.5,
                 state_path: "Path | None" = None, turn_from=None, turn_mix=0.0,
                 min_episodes: "int | None" = None, per_scene: bool = False):
        super().__init__()
        self.d, self.end = start, end
        self.window, self.threshold, self.notch = window, threshold, notch
        # PER-SCENE gate (2026-09-25, Joana: "can the policy only work on one easy
        # scene and still make the curriculum advance?"). With one shared win list,
        # yes: a plaza whose episodes are short and mostly won fills the window
        # faster than a corner whose episodes are long and mostly lost, so the
        # band climbs on the plaza's record. With per_scene the notch is earned
        # only when EVERY scene that has enough recent episodes clears the
        # threshold -- the worst scene decides, not the average.
        self.per_scene = bool(per_scene)
        self._wins_by_scene: dict = {}
        # How many recent episodes must be in the window before the gate may pass.
        # The old rule was window // 2 = 50, which one 2048-step rollout supplies
        # (~40 steps per episode), so the band could climb 0.5 m EVERY rollout on a
        # coin-flip bar: every arm reached 10 m in 40-60k steps while still at
        # ~50% success and stayed at 50% for the rest of training (2026-09-25,
        # Joana reading the curves: "the curriculum went up too fast").
        self.min_episodes = int(min_episodes) if min_episodes else window // 2
        self._wins: list = []
        # Where the curriculum ACTUALLY got to, on disk beside the checkpoints.
        # env_config.json records `goal_dist_range` from ARGS -- the range the
        # run STARTED at (2,8) -- so an eval adopting it would test a policy on
        # a distribution it stopped training on. And the end point is not
        # predictable from the config either: the far end only advances when
        # the policy earns notches, so the truth has to be written as it moves.
        self.state_path = state_path
        self.turn_from = turn_from      # band distance at which corner goals start appearing
        self.turn_mix = float(turn_mix)  # the mix reached at the ceiling

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                win = 1.0 if info.get("goal_bonus", 0.0) > 0 else 0.0
                self._wins.append(win)
                if self.per_scene:
                    sk = info.get("scene_idx", info.get("scene_id", "?"))
                    try:
                        sk = int(round(float(sk)))
                    except (TypeError, ValueError):
                        pass
                    self._wins_by_scene.setdefault(sk, []).append(win)
        return True

    def _gate_passes(self) -> bool:
        recent = self._wins[-self.window:]
        if len(recent) < self.min_episodes or self.d >= self.end:
            return False
        if not self.per_scene:
            return float(np.mean(recent)) >= self.threshold
        # EVERY scene seen since the last notch must (a) have enough recent episodes
        # and (b) clear the bar. A scene with too few episodes BLOCKS -- it does not
        # abstain. The first version let it abstain, and the test showed a plaza
        # earning a notch every two rollouts while the corner's count was wiped
        # before it could ever be judged (2026-09-25). The slowest scene sets the pace.
        seen = {k: v[-self.window:] for k, v in self._wins_by_scene.items() if v}
        if not seen:
            return False
        need = max(10, self.min_episodes // len(seen))
        if any(len(v) < need for v in seen.values()):
            return False
        judged = {k: float(np.mean(v)) for k, v in seen.items()}
        if self.logger is not None:
            self.logger.record("curriculum/worst_scene_success", min(judged.values()))
            self.logger.record("curriculum/scenes_judged", float(len(judged)))
        return min(judged.values()) >= self.threshold

    def _on_rollout_start(self) -> None:
        if self._gate_passes():
            self.d = min(self.end, self.d + self.notch)
            self._wins.clear()               # re-earn the next notch
            self._wins_by_scene.clear()
        rngs = self.training_env.env_method("set_goal_dist", self.d)
        # corner rule ramped on the SAME clock: no corners until the band reaches turn_from,
        # then linearly to turn_mix by the ceiling. turn_from=None leaves the mix alone.
        if self.turn_from is not None and self.turn_mix > 0.0:
            span = max(1e-6, float(self.end) - float(self.turn_from))
            frac = min(1.0, max(0.0, (float(self.d) - float(self.turn_from)) / span))
            mix = float(self.turn_mix) * frac
            self.training_env.env_method("set_goal_turn_mix", mix)
            if self.logger is not None:
                self.logger.record("curriculum/goal_turn_mix", mix)
        if self.logger is not None:
            self.logger.record("curriculum/goal_dist", self.d)
            # The far end alone is not the task: with a sliding window the near
            # end moves too, and "goals are 2-8 m" vs "goals are 5-10 m" are
            # different problems with the same `goal_dist`.
            rng0 = next((r for r in rngs if r), None)
            if rng0:
                self.logger.record("curriculum/goal_dist_lo", float(rng0[0]))
                self.logger.record("curriculum/goal_dist_hi", float(rng0[1]))
        rng0 = next((r for r in rngs if r), None)
        if self.state_path is not None and rng0:
            try:
                import json as _json
                self.state_path.write_text(_json.dumps({
                    "goal_dist": float(self.d),
                    "goal_dist_range": [float(rng0[0]), float(rng0[1])],
                    "num_timesteps": int(self.num_timesteps),
                }, indent=2))
            except Exception:
                pass


class RewardComponentsCallback(BaseCallback):
    """Log per-component reward means each rollout (meeting item 2026-08-13:
    'visualize different reward vectors on wandb / see the range the reward
    falls in'). Would have exposed Run A's zero-semantic-collision anomaly
    during training instead of at eval."""

    KEYS = ("semantic", "goal", "collision", "step", "void", "spin",
            "backward", "smooth", "timeout", "crash", "proximity", "proximity_ground",
            "goal_bonus", "speed_refund", "refusal_bonus", "refusal_paid", "halt_penalty_paid", "total",
            # 2026-09-01: the world model's own uncertainty, logged from day
            # one so any void THRESHOLD gets chosen from the measured
            # distribution instead of guessed. void_frac = footprint support;
            # image_void_frac = whole-view support (her measure). The spin
            # `cov` number is MEAN ALPHA, a different statistic — it cannot be
            # converted into these without the alpha histogram.
            "void_frac",
            # 2026-09-02: these were set on `info` by SceneEnv but MISSING from
            # this list, so 460539/460540 ran 17 h with the coherence term
            # inside `total` and invisible everywhere else. The only way to
            # recover its size was to subtract the other components from the
            # total by hand. `coverage` is the quantity the whole mechanism is
            # about -- log it whether or not the term is switched on, so the
            # thresholds get set from the distribution the policy ACTUALLY
            # visits rather than from a spin sweep.
            "coherence", "coherence_crash",
            # goal_dist_frac = final distance / starting distance. Without it a
            # drop in reward/timeout is ambiguous: FEWER timeouts and CHEAPER
            # timeouts look identical, and those are opposite conclusions about
            # `--timeout_distance_scaled`. Needed to read H (461357) at all.
            )

    # Per-step means blur the terminal quantities -- what matters for the
    # proportional timeout is how close the robot was WHEN THE EPISODE ENDED,
    # not averaged over every step it took getting there. Logged separately as
    # reward/end_*.
    TERMINAL_KEYS = ("goal_dist_frac", "coverage", "rgb_delta")

    # These are NOT reward terms and logging them under reward/ made the panel
    # unreadable -- `coverage` is mean alpha, `collision_off_frame` is a
    # did-the-mechanism-fire flag, `goal_dist_frac` is a distance ratio. They
    # go under diag/ so reward/ contains only things that are summed into the
    # return (her ask 2026-09-02).
    DIAG_KEYS = ("goal_case_open", "goal_case_corner", "goal_case_narrow", "goal_detour", "goal_min_width_m", "look_ahead_m",
                 "coverage", "collision_off_frame", "box_memory_age", "box_memory_hit", "box_memory_miss", "goal_dist_frac",
                 # 2026-09-04: generator vs map reading of the same footprint
                 "phantom", "missed", "label_agree", "trav_agree", "gen_collision_frac", "map_collision_frac",
                 "used_generated", "map_void_frac",
                 # refusal metric (per episode end): goal on traversable ground?
                 # halted on a non-traversable goal = correct; on a traversable one = the freeze
                 "goal_traversable", "goal_walkable_frac", "halt_correct", "halt_at_verge", "halt_wrong", "reach_on_nontrav", "passed_through_goal",
                 # 1.0 on the step a HALTED-SAFELY terminal fires, so
                 # diag/halted is the RATE of correct stops -- the first
                 # metric for the behaviour this project is about.
                 "halted", "phantom_vetoed",
                 "image_void_frac", "scene_idx", "rgb_delta",
                 # THE one that was missing: fraction of steps where the
                 # footprint did not project into the image at all. Two evals
                 # ran 20 episodes each with it at 1.0 -- the reward blind for
                 # every step -- and nothing said so (2026-09-02).
                 "off_frame_frac")

    def __init__(self, step_size_m: float = 0.3):
        super().__init__()
        self.step_size_m = float(step_size_m)
        self._sums = {k: 0.0 for k in self.KEYS + self.DIAG_KEYS}
        self._cnt = {k: 0 for k in self.KEYS + self.DIAG_KEYS}
        self._end_sums = {k: 0.0 for k in self.TERMINAL_KEYS}
        self._end_cnt = {k: 0 for k in self.TERMINAL_KEYS}
        self._thr_sum, self._thr_n = 0.0, 0

    def _on_step(self) -> bool:
        # THROTTLE. step_size_m is the MAXIMUM; the linear action in [0,1]
        # scales it. Measured 2026-09-02: ppo_240704 sighted runs at 28% of
        # maximum -- 0.083 m/step -- so in 60 steps it covers 5 m against goals
        # at 5-10 m, and most of its "timeouts" were the clock rather than a
        # decision to stop. Creeping minimises crash risk exactly the way
        # freezing does, and nothing logged it.
        acts = self.locals.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=float).reshape(-1, np.asarray(acts).shape[-1])
            self._thr_sum += float(np.clip(a[:, 0], 0.0, 1.0).mean())
            self._thr_n += 1
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            if "total" not in info:
                continue
            for k in self.KEYS + self.DIAG_KEYS:
                v = float(info.get(k, 0.0))
                # coverage and goal_dist_frac are nan when unavailable; one nan
                # would poison the running mean for the whole rollout.
                if v != v:
                    continue
                self._sums[k] += v
                self._cnt[k] += 1
            if i < len(dones) and dones[i]:
                for k in self.TERMINAL_KEYS:
                    v = float(info.get(k, float("nan")))
                    if v != v:
                        continue
                    self._end_sums[k] += v
                    self._end_cnt[k] += 1
        return True

    def _on_rollout_end(self) -> None:
        # The FIRST rollout goes to stdout as well as wandb: it is the earliest
        # moment the run can tell you whether moving pays, and it sits directly
        # under the frozen probe in the log for comparison. Waiting eight hours
        # to discover the budget is wrong is how today went.
        if not getattr(self, "_printed_first", False) and self._cnt.get("total"):
            self._printed_first = True
            line = "-" * 78
            print(f"\n{line}\nFIRST ROLLOUT -- reward per step for the POLICY "
                  f"({self._cnt['total']} steps)\n{line}")
            for k in self.KEYS:
                if self._cnt[k]:
                    print(f"  {k:<16} {self._sums[k] / self._cnt[k]:+9.4f}")
            for k in self.DIAG_KEYS:
                if self._cnt[k]:
                    print(f"  [diag] {k:<10} {self._sums[k] / self._cnt[k]:+9.4f}")
            if self._thr_n:
                _t = self._thr_sum / self._thr_n
                print(f"  [diag] throttle    {_t:+9.4f}   "
                      f"({_t * self.step_size_m:.3f} m/step, "
                      f"{100 * _t:.0f}% of maximum)")
            print(f"{line}\n  Compare `total` with the frozen probe above: if "
                  f"moving is not clearly better,\n  the policy's best strategy "
                  f"is to stand still and it will find that. Watch throttle "
                  f"too:\n  creeping at 28% is how ppo_240704 avoided crashing "
                  f"without stopping.\n{line}", flush=True)
        for k in self.KEYS:
            if self._cnt[k]:
                self.logger.record(f"reward/{k}", self._sums[k] / self._cnt[k])
        for k in self.DIAG_KEYS:
            if self._cnt[k]:
                self.logger.record(f"diag/{k}", self._sums[k] / self._cnt[k])
        if self._thr_n:
            thr = self._thr_sum / self._thr_n
            self.logger.record("diag/throttle", thr)
            # what that throttle actually buys, in metres
            self.logger.record("diag/step_m", thr * self.step_size_m)
        self._thr_sum, self._thr_n = 0.0, 0
        for k in self.TERMINAL_KEYS:
            if self._end_cnt[k]:
                self.logger.record(f"reward/end_{k}",
                                   self._end_sums[k] / self._end_cnt[k])
        self._sums = {k: 0.0 for k in self.KEYS + self.DIAG_KEYS}
        self._cnt = {k: 0 for k in self.KEYS + self.DIAG_KEYS}
        self._end_sums = {k: 0.0 for k in self.TERMINAL_KEYS}
        self._end_cnt = {k: 0 for k in self.TERMINAL_KEYS}

from src.env.scene_env import SceneEnv, SceneEnvConfig
from src.env.real_calibrated import (
    CalibratedRealWorldBackend, CalibratedBackendConfig, GaussianLabelBackend,
)
from src.eval.reward_2d import RewardWeights


class WandbImagePanel(BaseCallback):
    """Every N rollouts, log what the policy sees and what it is scored on:
    the first robot of every GPU worker: its current observation, its injected
    generated labels (colorized), and the coverage the gate read. Images to wandb so a run can
    be judged by eye without pulling rollout videos."""

    V14 = np.array([
        (0, 0, 0), (200, 225, 245), (150, 100, 55), (75, 190, 80), (95, 65, 35),
        (50, 120, 200), (210, 210, 210), (70, 70, 85), (235, 205, 150),
        (220, 140, 80), (185, 55, 50), (170, 200, 55), (205, 70, 145), (110, 130, 220),
    ], np.uint8)

    def __init__(self, every: int, sem_palette: int = 0):
        super().__init__()
        self.every = max(int(every), 1)
        self._n = 0
        # 2026-09-17 (Joana): colour the labels and the memory strip with the
        # palette the run decodes with (SEMPAL), the same table the surveys and
        # the rollout videos use, so a colour means the same class in every
        # picture of the pipeline. The class-level V14 table above is the
        # fallback when the taxonomy module is not importable.
        if int(sem_palette) > 0:
            try:
                from diffsynth.utils.class_taxonomy import v14_palette
                self.V14 = (v14_palette(int(sem_palette)).numpy() * 255).astype(np.uint8)
            except Exception as e:
                print(f"[WandbImagePanel] palette {sem_palette} unavailable ({e}); display table", flush=True)

    def _on_step(self) -> bool:
        return True

    @staticmethod
    def topdown(grid, walk_xy, paths, goal_xy, scene, size=360, non_trav=None):
        """One top-down picture: the scene's label grid (walkable light, wall dark,
        void grey), the recorded walk (thin grey), and the episode paths -- the
        last FINISHED episode in blue, the one in progress in orange -- with the
        goal circled. Same conventions as scene_corner_scan.py --plot, so a path
        here and a path in the offline overheads read the same way (2026-09-25)."""
        import cv2
        pts = [np.asarray(p, float) for p in paths if p is not None and len(p) > 1]
        allxy = np.concatenate(pts + ([np.asarray([goal_xy], float)] if goal_xy is not None else []), axis=0) \
            if pts or goal_xy is not None else None
        if grid is not None:
            L = grid.labels; res = float(grid.res); x0, y0 = float(grid.x0), float(grid.y0)
            img = np.full((*L.shape, 3), 70, np.uint8)
            known = L >= 0
            img[known] = (232, 232, 228)
            # non-traversable cells (walls, lawn, people) dark red, from the env's own mask;
            # the first version painted every known cell light and a wall placed in the test
            # was invisible (2026-09-25). Without the mask the panel is walk + paths only.
            if non_trav is not None:
                nt = np.asarray(non_trav, bool)
                bad = known & nt[np.clip(L, 0, len(nt) - 1)]
                img[bad] = (60, 60, 190)
            if allxy is not None:
                M = 5.0
                i0 = max(0, int((allxy[:, 1].min() - y0 - M) / res)); i1 = min(L.shape[0], int((allxy[:, 1].max() - y0 + M) / res))
                j0 = max(0, int((allxy[:, 0].min() - x0 - M) / res)); j1 = min(L.shape[1], int((allxy[:, 0].max() - x0 + M) / res))
            else:
                i0, i1, j0, j1 = 0, L.shape[0], 0, L.shape[1]
            img = img[i0:i1, j0:j1]
            if img.size == 0:
                return None
            sc = max(1, int(round(size / max(1, img.shape[0]))))
            img = cv2.resize(img, (img.shape[1] * sc, img.shape[0] * sc), interpolation=cv2.INTER_NEAREST)
            px = lambda xy: (int(((xy[0] - x0) / res - j0) * sc), int(((xy[1] - y0) / res - i0) * sc))
        else:
            if allxy is None:
                return None
            lo = allxy.min(0) - 3.0; hi = allxy.max(0) + 3.0
            res = max((hi - lo).max() / size, 1e-3)
            W2 = max(int((hi[0] - lo[0]) / res) + 1, 440)      # wide enough for the caption
            img = np.full((int((hi[1] - lo[1]) / res) + 1, W2, 3), 232, np.uint8)
            px = lambda xy: (int((xy[0] - lo[0]) / res), int((xy[1] - lo[1]) / res))
        if walk_xy is not None:
            w = np.asarray(walk_xy, float)
            for k in range(len(w) - 1):
                cv2.line(img, px(w[k]), px(w[k + 1]), (150, 150, 150), 1, cv2.LINE_AA)
        colours = [(190, 110, 40), (40, 140, 230)]        # BGR: blue = last finished, orange = in progress
        for path, col in zip(pts, colours):
            for k in range(len(path) - 1):
                cv2.line(img, px(path[k]), px(path[k + 1]), col, 2, cv2.LINE_AA)
            cv2.circle(img, px(path[0]), 4, col, -1)
            cv2.circle(img, px(path[-1]), 4, (30, 30, 30), -1)
        if goal_xy is not None:
            cv2.circle(img, px(goal_xy), max(4, int(0.75 / res) if grid is not None else 6), (40, 160, 40), 2)
        img = cv2.flip(img, 0)
        cv2.putText(img, f"{scene}  blue=last episode  orange=current  dot=end", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        return img[:, :, ::-1]                              # BGR -> RGB for wandb

    def _on_rollout_end(self) -> None:
        self._n += 1
        if self._n % self.every:
            return
        try:
            import wandb
            env = self.training_env
            # One row per GPU worker (2026-09-16): each worker holds its own
            # scene, so the first robot of every worker shows every scene in
            # play. Single-GPU envs have one worker -> one row, as before.
            n_workers = int(getattr(env, "n_workers", 1) or 1)
            stride = max(int(env.num_envs) // n_workers, 1)
            idx = [k * stride for k in range(n_workers)]
            rows, scenes, covs, tops = [], [], [], []
            for i in idx:
                rgb = env.get_attr("_last_rgb", indices=[i])[0]
                lab = env.get_attr("_injected_labels", indices=[i])[0]
                try:
                    cov = env.get_attr("_last_coverage", indices=[i])[0]
                except Exception:
                    cov = None
                _nm = "?"
                for _attr in ("_scene_id", "_current_scene_id"):
                    try:
                        _v = env.get_attr(_attr, indices=[i])[0]
                        if _v:
                            _nm = str(_v); break
                    except Exception:
                        pass
                scenes.append(_nm)
                if rgb is None:
                    continue
                base = np.ascontiguousarray(np.asarray(rgb)[..., :3]).copy()
                # PLAN on the observation, drawn by the SAME code as the offline
                # survey (src/eval/panels.py, 2026-09-16): orange path + the reward
                # footprint at each planned pose. One step for a per-step policy,
                # the whole chunk for a chunked one, from the pose it was decided at.
                try:
                    from src.eval.panels import integrate_plan, draw_plan
                    _K = env.get_attr("_last_K", indices=[i])[0]
                    _w2c = env.get_attr("_last_w2c", indices=[i])[0]
                    _cfg = env.get_attr("cfg", indices=[i])[0]
                    _act = None
                    for _attr in ("_last_decision", "_last_action"):
                        try:
                            _v = env.get_attr(_attr, indices=[i])[0]
                        except Exception:
                            _v = None
                        if _v is not None and np.asarray(_v).size >= 2:
                            _act = np.asarray(_v); break
                    try:
                        _pp = env.get_attr("_decision_pose", indices=[i])[0]
                    except Exception:
                        _pp = None
                    if _pp is None:
                        _pp = env.get_attr("_robot_pose_world", indices=[i])[0]
                    if _K is not None and _w2c is not None and _pp is not None and _act is not None:
                        _plan = integrate_plan(np.asarray(_pp), _act, float(_cfg.step_size_m), float(_cfg.yaw_step_rad))
                        draw_plan(base, np.asarray(_pp), _plan, _K, _w2c,
                                  look_ahead_dist=float(_cfg.look_ahead_dist))
                except Exception as _e:
                    print(f"[WandbImagePanel] plan skipped: {type(_e).__name__}: {_e}", flush=True)
                panel = [base]
                if lab is not None:
                    col = self.V14[np.clip(np.asarray(lab).astype(np.int64), 0, 13)]
                    if col.shape[:2] != panel[0].shape[:2]:
                        import cv2
                        col = cv2.resize(col, (panel[0].shape[1], panel[0].shape[0]), interpolation=cv2.INTER_NEAREST)
                    panel.append((0.55 * col + 0.45 * panel[0]).astype(np.uint8))
                # MEMORY STRIP, same code as the survey (src/eval/panels.py)
                try:
                    from src.eval.panels import memory_strip
                    from src.eval.reward_2d import _footprint_corners_world, GO2_BODY_LENGTH, GO2_BODY_WIDTH
                    _fm = env.get_attr("_frame_memory", indices=[i])[0]
                    if _fm:
                        _pose = env.get_attr("_robot_pose_world", indices=[i])[0]
                        _cfg = env.get_attr("cfg", indices=[i])[0]
                        _nt = env.get_attr("_non_trav", indices=[i])[0]
                        _hd = np.asarray(_pose[:3, :3]) @ np.array([1.0, 0.0, 0.0])
                        _la = float(getattr(_cfg, "collision_look_ahead_m", 0.0) or 0.0) or float(_cfg.look_ahead_dist)
                        _cw = _footprint_corners_world(np.asarray(_pose[:3, 3], float), _hd, look_ahead_dist=_la,
                                                       length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
                        _strip = memory_strip(_fm, _cw, self.V14, np.asarray(_nt),
                                              np.asarray(_pose[:3, 3], float),
                                              out_h=panel[0].shape[0], out_w=panel[0].shape[1],
                                              horizontal=True)
                        if _strip is not None:
                            panel.append(_strip)
                except Exception as _e:
                    print(f"[WandbImagePanel] memory strip skipped: {type(_e).__name__}: {_e}", flush=True)
                rows.append(np.concatenate(panel, axis=1))
                covs.append(cov)
                # TOP-DOWN: where this worker's robot actually went (2026-09-25)
                try:
                    _sid = env.get_attr("_scene_id", indices=[i])[0]
                    _g = None
                    try:
                        _g = (env.get_attr("_label_grids", indices=[i])[0] or {}).get(_sid)
                    except Exception:
                        _g = None
                    _walk = None
                    try:
                        _walk = (env.get_attr("_walk_xy", indices=[i])[0] or {}).get(_sid)
                    except Exception:
                        _walk = None
                    def _ga(name):                       # attribute may not exist before the first reset
                        try:
                            return env.get_attr(name, indices=[i])[0]
                        except Exception:
                            return None
                    _last, _cur, _goal = _ga("_last_ep_xy"), _ga("_ep_xy"), _ga("_ep_goal")
                    _td = self.topdown(_g, _walk, [_last, _cur], _goal, _nm, non_trav=_ga("_non_trav"))
                    if _td is not None:
                        tops.append(_td)
                except Exception as _e:
                    print(f"[WandbImagePanel] topdown skipped: {type(_e).__name__}: {_e}", flush=True)
            if not rows:
                return
            w = max(r.shape[1] for r in rows)
            rows = [r if r.shape[1] == w else np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
            img = np.concatenate(rows, axis=0)
            cap = f"rollout {self._n} | rows = GPU workers: " + ", ".join(
                f"{s}" + (f" cov {float(c):.2f}" if c is not None else "") for s, c in zip(scenes, covs))
            _log = {"panels/obs_labels": wandb.Image(img, caption=cap)}
            if tops:
                h = max(t.shape[0] for t in tops)
                tops = [t if t.shape[0] == h else np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tops]
                _log["panels/topdown"] = wandb.Image(np.concatenate(tops, axis=1),
                                                     caption=f"rollout {self._n} | one map per GPU worker: " + ", ".join(scenes))
            wandb.log(_log, step=self.num_timesteps)
        except Exception as e:  # never let logging kill a run
            print(f"[WandbImagePanel] skipped: {type(e).__name__}: {e}", flush=True)


def make_env(args):
    # rung 7: --scenes trains one policy over several scenes (round-robin per
    # episode in SceneEnv). Single --scene remains the default path.
    scenes = args.scenes if getattr(args, "scenes", None) else [args.scene]
    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in scenes},
        goal_frame=args.goal_frame,
        goal_frame_range=tuple(args.goal_frame_range) if args.goal_frame_range else None,
        goal_min_sep_m=args.goal_min_sep,
        spawn_max_frame=args.spawn_max_frame,
        goal_xy_override=(tuple(float(v) for v in args.goal_xy.split(","))
                          if getattr(args, "goal_xy", None) else None),
        goal_dir_360=getattr(args, "goal_dir_360", False),
        goal_cone_deg=getattr(args, "goal_cone_deg", 360.0),
        goal_dist_range=(tuple(float(v) for v in args.goal_dist_range.split(","))
                         if getattr(args, "goal_dist_range", None) else None),
        goal_dist_window_m=getattr(args, "goal_dist_window", None),
        goal_turn_deg=float(getattr(args, "goal_turn_deg", 0.0) or 0.0),
        goal_turn_mix=float(getattr(args, "goal_turn_mix", 0.0) or 0.0),
        goal_turn_beyond_m=float(getattr(args, "goal_turn_beyond_m", 2.0)),
        spawn_label_classes=(tuple(int(v) for v in args.spawn_classes.split(","))
                             if getattr(args, "spawn_classes", None) else None),
        spawn_yaw_jitter_deg=getattr(args, "spawn_yaw_jitter", 0.0),
        spawn_lat_jitter_m=getattr(args, "spawn_lat_jitter", 0.0),
        spawn_heading_from_walk=bool(getattr(args, "spawn_heading_from_walk", False)),
        spawn_frames_by_scene=str(getattr(args, "spawn_frames", "") or ""),
        sem_palette_version=getattr(args, "sem_palette", 1), static_scene=bool(getattr(args, "static_scene", False)),
        static_movers=str(getattr(args, "static_movers", "") or ""),
        render_window=int(getattr(args, "render_window", 0) or 0),
        coverage_window=int(getattr(args, "coverage_window", 0) or 0),
        render_mode="rasterizer_only",       # cheap per-step; diffusion later
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=getattr(args, "render_height", None) or getattr(args, "obs_height", 336),
        W=getattr(args, "render_width", None) or getattr(args, "obs_width", 560),
        goal_dist_m=getattr(args, "goal_dist", None),
        spawn_min_frame=getattr(args, "spawn_min", 0),
    )
    # Dump BEFORE the backend is built. Building it loads the diffusion
    # pipeline, which takes the better part of an hour, so dumping afterwards
    # meant the file describing a run did not exist while the run was starting
    # -- on 2026-09-03 seven arms had been up for 55 minutes with no
    # env_config.json anywhere on disk, and nothing could be audited against
    # what they were actually configured with.
    env_cfg = _scene_env_cfg(args)
    _dump_env_config(args, env_cfg)

    if getattr(args, "live", False):
        # Live per-action diffusion: the policy queries the generative model at
        # its own CONTINUOUS pose every step — no cache, no grid snapping.
        # Observations and reward labels come from the same diffusion call.
        from src.env.live_backend import LiveDiffusedBackend
        world = LiveDiffusedBackend(cfg, checkpoint=args.live_ckpt,
                                    live_frames=args.live_frames,
                                    alpha_gate=not getattr(args, "no_alpha_gate", False),
        alpha_gate_tau=getattr(args, "alpha_gate_tau", 0.5),
        raster_obs=bool(getattr(args, "raster_obs", False)))
    elif getattr(args, "obs_cache", None):
        # Ribbon-cache mode: observations = precomputed diffused views (v10 +
        # reader), reward labels = the cache's alpha-masked diffused labels.
        # No GPU rendering during training at all.
        from src.env.cached_backend import CachedDiffusedBackend
        world = CachedDiffusedBackend(cfg, cache_root=args.obs_cache,
                                      alpha_gate=not getattr(args, "no_alpha_gate", False),
        alpha_gate_tau=getattr(args, "alpha_gate_tau", 0.5))
    else:
        world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    env = SceneEnv(world_backend=world, semantic_backend=sem,
                   scene_ids=scenes, cfg=env_cfg)
    return Monitor(env)


def _dump_env_config(args, cfg):
    """Record the env the policy is ACTUALLY learning in, beside its
    checkpoints. Evaluating with different kinematics runs the policy outside
    its own action model — on 2026-09-01 the eval used step 0.25 / yaw 0.3
    against training's 0.30 / 0.50 with reverse re-enabled, and it only
    surfaced when a rollout video showed the robot backing up. A knob needs a
    human to remember; a file does not."""
    try:
        import json as _json
        out = Path(args.output_dir) / "env_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps({
            "step_size_m": cfg.step_size_m,
            "yaw_step_rad": cfg.yaw_step_rad,
            "forward_only": bool(getattr(cfg, "forward_only", False)),
            "look_ahead_dist": cfg.look_ahead_dist,
            "collision_look_ahead_m": cfg.collision_look_ahead_m,
            "collision_box_memory": getattr(cfg, "collision_box_memory", 0),
            "collision_box_memory_agg": getattr(cfg, "collision_box_memory_agg", "newest"),
            "collision_box_memory_always": bool(getattr(cfg, "collision_box_memory_always", False)),
            "goal_turn_deg": float(getattr(cfg, "goal_turn_deg", 0.0) or 0.0),
            "goal_turn_mix": float(getattr(cfg, "goal_turn_mix", 0.0) or 0.0),
            "goal_turn_beyond_m": float(getattr(cfg, "goal_turn_beyond_m", 2.0)),
            "goal_turn_from": getattr(args, "goal_turn_from", None),
            "max_steps_base": int(getattr(args, "max_steps_base", 40) or 40),
            "max_steps_per_m": float(getattr(args, "max_steps_per_m", 0.0) or 0.0),
            "obs_frame_stack": int(getattr(args, "obs_frame_stack", 1) or 1),
            "obs_frame_stride": int(getattr(args, "obs_frame_stride", 1) or 1),
            # the encoder was only ever in the directory name, and only for dinov2, so an eval
            # of an older run cannot tell which backbone it was (2026-09-22)
            "encoder": str(getattr(args, "encoder", "nature")),
            # the parent checkpoint, so a continued run's full curve can be chained (2026-09-22)
            "warmstart": (str(getattr(args, "warmstart", "")) if getattr(args, "warmstart", None) else ""),
            "image_norm_fix": bool(getattr(args, "image_norm_fix", False)),
            "reset_image_head": bool(getattr(args, "reset_image_head", False)),
            "collision_at_next_pose": bool(getattr(cfg, "collision_at_next_pose", False)),
            "look_ahead_auto": bool(getattr(cfg, "look_ahead_auto", False)),
            "footprint_next_heading": bool(getattr(cfg, "footprint_next_heading", False)),
            "crash_requires_motion": bool(getattr(cfg, "crash_requires_motion", False)),
            "goal_support_radius_m": cfg.goal_support_radius_m,
            "goal_support_min_frac": getattr(cfg, "goal_support_min_frac", 0.25),
            "goal_radius": cfg.goal_radius,
            "collision_threshold": cfg.collision_threshold,
            "collision_terminate_frac": cfg.collision_terminate_frac,
            "phantom_veto": bool(getattr(cfg, "phantom_veto", False)),
            "collision_terminate_penalty": cfg.collision_terminate_penalty,
            "trav_path": cfg.trav_path,
            # 2026-09-02, her question: "for eval, why not do the same spawns
            # and goals method as in training?" -- because these live on the
            # BACKEND config, not SceneEnvConfig, so they were never dumped and
            # eval could not reproduce them at all. It could only pin one
            # goal_xy or use the default goal frame, which is how an eval ended
            # up with d_start varying uncontrolled and some episodes
            # unreachable before the policy acted. Dumped here so eval rebuilds
            # the training distribution from the file instead of from knobs
            # somebody has to remember.
            "goal_dir_360": bool(getattr(args, "goal_dir_360", False)),
            "goal_dist_range": getattr(args, "goal_dist_range", None),
            # 2026-09-22: goal_dist was never recorded, so eval could not adopt the target distance
            # and fell into the random-frame branch -- 13.8 m goals for a policy trained at 2-4 m.
            "goal_dist": getattr(args, "goal_dist", None),
            "cur_threshold": getattr(args, "cur_threshold", None),
            "cur_window": getattr(args, "cur_window", None),
            "cur_min_episodes": getattr(args, "cur_min_episodes", None),
            "cur_notch": getattr(args, "cur_notch", None),
            "cur_per_scene": bool(getattr(args, "cur_per_scene", False)),
            "disagree_cost_weight": float(getattr(args, "disagree_cost", 0.0) or 0.0),
            "rgb_align_cost_weight": float(getattr(args, "rgb_align_cost", 0.0) or 0.0),
            "disagree_frac": getattr(args, "disagree_frac", None),
            "rgb_align_tau": float(getattr(args, "rgb_align_tau", 0.8)),
            # and the fusion window, which decides what world the policy is even shown
            "render_window": int(getattr(args, "render_window", 0) or 0),
            "coverage_window": int(getattr(args, "coverage_window", 0) or 0),
            "goal_dist_window": getattr(args, "goal_dist_window", None),
            "goal_cone_deg": getattr(args, "goal_cone_deg", None),
            "goal_frame_range": getattr(args, "goal_frame_range", None),
            "spawn_classes": getattr(args, "spawn_classes", None),
            "spawn_yaw_jitter": getattr(args, "spawn_yaw_jitter", None),
            "spawn_lat_jitter": getattr(args, "spawn_lat_jitter", None),
            "spawn_min": getattr(args, "spawn_min", None),
            "action_chunk": getattr(cfg, "action_chunk", 1),
            "max_steps": cfg.max_steps,
            # Reward too, not just kinematics. Eval built its own weights
            # (semantic 1.0 / goal 1.5 / bonus 50, no reward_scale, no
            # proximity) against training's 5 / 10 / 1000 / 0.01 — so an eval
            # "return" was never on the same scale as anything in training and
            # could not be compared to it. Behaviour is unaffected (the policy
            # does not see the eval reward) but every reported number was.
            "semantic_weight": cfg.reward.semantic,
            "goal_weight": cfg.reward.goal,
            "collision_weight": cfg.reward.collision,
            "step_cost": cfg.reward.step_cost,
            "void_cost": cfg.reward.void_cost,
            "terrain_as_cost": bool(cfg.reward.terrain_as_cost),
            "spin_cost": cfg.spin_cost,
            "backward_cost": cfg.backward_cost,
            "action_smooth_cost": getattr(cfg, "action_smooth_cost", 0.0),
            "goal_bonus": cfg.goal_bonus,
            "timeout_penalty": getattr(cfg, "timeout_penalty", 0.0),
            "timeout_distance_scaled": bool(getattr(cfg, "timeout_distance_scaled", False)),
            "proximity_weight": getattr(cfg, "proximity_weight", 0.0),
            "proximity_margin": getattr(cfg, "proximity_margin", 1.0),
            "proximity_delta": bool(getattr(cfg, "proximity_delta", False)),
            "proximity_ground_weight": getattr(cfg, "proximity_ground_weight", 0.0),
            "proximity_ground_margin": getattr(cfg, "proximity_ground_margin", 1.2),
            "proximity_ground_classes": getattr(cfg, "proximity_ground_classes", "3,4,5"),
            "goal_min_score": getattr(cfg, "goal_min_score", 0.0),
            "goal_lat_jitter_m": getattr(cfg, "goal_lat_jitter_m", 0.0),
            "reward_scale": getattr(cfg, "reward_scale", 1.0),
            "coherence_cost_weight": getattr(cfg, "coherence_cost_weight", 0.0),
            "coherence_tau": getattr(cfg, "coherence_tau", 0.4),
            "coherence_terminate_tau": getattr(cfg, "coherence_terminate_tau", 0.0),
            # THE ALPHA GATE. Training runs NOGATE=1 (raw labels); eval
            # defaulted to GATED, which turns low-coverage pixels into void --
            # and void is excluded from the collision fraction when
            # void_cost > 0, so gated evals simply do not crash where training
            # would. Joana caught it in a video panel: "sem RAW" solid
            # non-traversable, "sem REWARD (gated)" black, episode running on
            # (2026-09-03).
            "no_alpha_gate": bool(getattr(args, "no_alpha_gate", False)),
            "alpha_gate_tau": float(getattr(args, "alpha_gate_tau", 0.5)),
            # 2026-09-06: the policy saw the splat raster, not a diffused
            # frame. Eval must adopt it or it scores the policy on images it
            # never trained on.
            "raster_obs": bool(getattr(args, "raster_obs", False)),
            "static_scene": bool(getattr(args, "static_scene", False)),
            "static_movers": str(getattr(args, "static_movers", "") or ""),
            "spawn_heading_from_walk": bool(getattr(args, "spawn_heading_from_walk", False)),
            "spawn_frames": str(getattr(args, "spawn_frames", "") or ""),
            "goal_case_mix": str(getattr(args, "goal_case_mix", "") or ""),
            "label_remap": str(getattr(args, "label_remap", "") or ""),
            "goal_case_tries": int(getattr(args, "goal_case_tries", 24)),
            "goal_center_clear_m": float(getattr(args, "goal_center_clear_m", 0.0) or 0.0),
            "mirror_prob": float(getattr(args, "mirror_prob", 0.0) or 0.0),
            # 2026-09-07: the palette the generator's conditioning is colorized
            # with. Evals adopt it; they ran v1 against training's v4 until today.
            "sem_palette": int(getattr(args, "sem_palette", 1)),
            # 2026-09-22 (Joana: "what semantics version did you use for this?"):
            # the palette was recorded but the world model that generated every
            # observation was not, so no eval folder could say which semantic
            # checkpoint produced its frames. Recorded for the ledger only; eval
            # does not adopt it (it is passed on the eval command line).
            "live_ckpt": str(getattr(args, "live_ckpt", "") or ""),
            "halt_terminate_steps": getattr(cfg, "halt_terminate_steps", 0),
            "halt_throttle_eps": getattr(cfg, "halt_throttle_eps", 0.05),
            "halt_penalty_scale": getattr(cfg, "halt_penalty_scale", 1.0),
            "refusal_bonus": getattr(cfg, "refusal_bonus", 0.0),
            "refusal_dist_m": getattr(cfg, "refusal_dist_m", 2.0),
            "refusal_verge_m": getattr(cfg, "refusal_verge_m", 1.5),
            "halt_wrong_penalty": getattr(cfg, "halt_wrong_penalty", 0.0),
            "nontrav_goal_unreachable": getattr(cfg, "nontrav_goal_unreachable", False),
            "goal_requires_stop": getattr(cfg, "goal_requires_stop", False),
            "stop_action": getattr(cfg, "stop_action", False),
            "refusal_dwell_steps": getattr(cfg, "refusal_dwell_steps", -1),
            "lawn_progress_to_verge": getattr(cfg, "lawn_progress_to_verge", False),
            "terrain_speed_scaled": bool(getattr(cfg, "terrain_speed_scaled", False)),
            "reward_source": getattr(cfg, "reward_source", "generated"),
            "map_diagnostics": bool(getattr(cfg, "map_diagnostics", True)),
            "map_fallback_void_frac": getattr(cfg, "map_fallback_void_frac", 0.5),
            "map_fallback_min_alpha": getattr(cfg, "map_fallback_min_alpha", 0.4),
            "goal_traversable_mix": getattr(cfg, "goal_traversable_mix", 0.0),
            "spawn_support_tries": getattr(cfg, "spawn_support_tries", 0),
            "goal_mix_map_draw": getattr(cfg, "goal_mix_map_draw", False),
            "goal_nontrav_classes": getattr(cfg, "goal_nontrav_classes", "3,4,5"),
            "goal_nontrav_edge_m": getattr(cfg, "goal_nontrav_edge_m", 0.0),
            "goal_nontrav_tries": getattr(cfg, "goal_nontrav_tries", 0),
            "goal_nontrav_cone_deg": getattr(cfg, "goal_nontrav_cone_deg", 0.0),
            "map_res_m": getattr(cfg, "map_res_m", 0.1),
        }, indent=2))
        print(f"[train] env recorded for eval: {out}", flush=True)
    except Exception as e:
        print(f"[train] could not write env_config.json: {e}", flush=True)



def _reward_banner(args, cfg, env) -> None:
    """Print what the run is ACTUALLY configured with, and what freezing costs.

    Six of the eleven things this run depends on had no log at all on
    2026-09-02 -- the loaded traversability values, whether the curricula move,
    whether goal-support rejects anything, whether the near collision box is in
    frame. Two mechanisms (the coherence term and the distance curriculum) had
    already been silently inert for weeks before anyone noticed. A banner is
    cheap; discovering after eight hours that road was still 0.5 is not.
    """
    try:
        _reward_banner_body(args, cfg, env)
    except Exception as e:
        print(f"[banner] could not print reward configuration: {e}", flush=True)


def _reward_banner_body(args, cfg, env) -> None:
    from src.eval.traversability import load_traversability
    V14 = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk",
           "road", "pavement", "stairs", "obstacle", "vegetation", "person",
           "vehicle"]
    line = "=" * 78
    print(f"\n{line}\nREWARD CONFIGURATION -- what this run is actually "
          f"running\n{line}", flush=True)
    try:
        trav = load_traversability(Path(cfg.trav_path))
        pairs = "  ".join(f"{V14[i]}={trav[i]:.2f}" for i in range(min(14, len(trav))))
        print(f"traversability [{cfg.trav_path}]\n  {pairs}", flush=True)
    except Exception as e:
        print(f"traversability: COULD NOT LOAD {cfg.trav_path} ({e})", flush=True)
    r = cfg.reward
    print(f"reward weights   semantic={r.semantic} goal={r.goal} "
          f"collision={r.collision} step={r.step_cost} void={r.void_cost} "
          f"terrain_as_cost={r.terrain_as_cost} scale={cfg.reward_scale}")
    print(f"terminals        goal_bonus={cfg.goal_bonus} @ r={cfg.goal_radius}m | "
          f"crash={cfg.collision_terminate_penalty} @ "
          f"{cfg.collision_terminate_frac} | timeout={cfg.timeout_penalty} "
          f"scaled={cfg.timeout_distance_scaled}")
    print(f"per-step costs   smooth={cfg.action_smooth_cost} "
          f"spin={cfg.spin_cost} proximity={cfg.proximity_weight} "
          f"margin={cfg.proximity_margin}")
    ca = cfg.collision_look_ahead_m
    print(f"footprints       shaping={cfg.look_ahead_dist}m  collision="
          f"{ca if ca > 0 else cfg.look_ahead_dist}m"
          f"{'  (SPLIT)' if ca > 0 else '  (shared -- stop-at-edge is NOT learnable)'}")
    # Built OUTSIDE the f-string. Nesting quotes inside f-string braces is
    # legal on 3.12+ (PEP 701) and a SyntaxError on older Pythons -- jobs
    # 464400/464402 died in 4 seconds because a local parse check ran on 3.13
    # and the cluster does not (2026-09-03).
    _h = int(getattr(cfg, "halt_terminate_steps", 0) or 0)
    if _h:
        _halt_msg = ("ON after %d stopped steps "
                     "(pays the distance-scaled timeout)" % _h)
    else:
        _halt_msg = ("OFF -- a correctly-refused goal is indistinguishable "
                     "from a timeout")
    print("halted-safely    " + _halt_msg)
    print(f"coherence        weight={cfg.coherence_cost_weight} "
          f"tau={cfg.coherence_tau} terminate_tau={cfg.coherence_terminate_tau}"
          f"{'' if cfg.coherence_cost_weight > 0 else '   (OFF)'}")
    print(f"episode          max_steps={cfg.max_steps} step={cfg.step_size_m}m "
          f"yaw={cfg.yaw_step_rad}rad forward_only={cfg.forward_only}  "
          f"-> max reach {cfg.max_steps * cfg.step_size_m:.1f} m")
    gsr = cfg.goal_support_radius_m
    print(f"goal support     {gsr}m" + ("" if gsr > 0 else
          "   (OFF -- ~14.5% of goals have no reconstruction under them)"))
    rc = args.goal_radius_start
    dc = args.goal_dist_start
    print(f"curricula        radius={rc if rc else 'OFF'} -> {cfg.goal_radius} | "
          f"distance={dc if dc else 'OFF'} -> {args.goal_dist}")
    # Print the scenes the ENV actually has, not the ones that were asked for.
    # make_live_vec_env PRUNES scenes that do not fit in GPU memory -- on
    # 2026-09-03 every arm silently trained on 5 of the 6 requested (gnd_AUw60
    # was dropped for OOM), while this line kept claiming 6. A banner that
    # reports the request instead of the reality is worse than no banner.
    _live = None
    try:
        _live = env.get_attr("scene_ids")[0]
    except Exception:
        pass
    _dropped = ([s for s in args.scenes if s not in _live] if _live else [])
    print(f"entropy          ent_coef={args.ent_coef}   "
          f"scenes={_live if _live else args.scenes} "
          f"rotate_every={args.scene_rotate}")
    if _dropped:
        print(f"                 !!! REQUESTED BUT NOT LOADED (pruned for "
              f"GPU memory): {_dropped} -- this run trains on "
              f"{len(_live)} of {len(args.scenes)} scenes")
    print(line, flush=True)


def _goal_cone_banner(env, n: int = 500) -> None:
    """Prove, in THIS run, that no goal starts behind the robot.

    Costs milliseconds (no rendering) and prints before the first rollout, so a
    48-hour arm cannot spend its whole life sampling unwinnable episodes
    without saying so. A forward-only robot given a goal behind it can only
    time out, and that penalty is independent of anything the policy does --
    pure noise in the gradient.
    """
    try:
        res = env.env_method("goal_cone_probe", n)[0]
    except Exception as e:
        print(f"[goal-cone] probe unavailable: {e}", flush=True)
        return
    print("\n=== GOAL CONE CHECK (no rendering; "
          f"{n} spawn/goal draws per scene) ===")
    print(f"{'scene':<16}{'mean':>8}{'max':>8}{'BEHIND >90deg':>16}")
    worst = 0.0
    for sid, (mean, mx, behind) in sorted(res.items()):
        print(f"{sid:<16}{mean:8.1f}{mx:8.1f}{behind:15.2f}%")
        worst = max(worst, behind)
    if worst > 0.0:
        print(f"!!! {worst:.2f}% of episodes start with the goal BEHIND a "
              f"forward-only robot -- those can only time out. The cone is "
              f"NOT centred on the recorded spawn heading in this run.")
    else:
        print("OK: no goal starts behind the robot on any scene.")
    # corner episodes (2026-09-21): share of draws that spawned before a bend and aimed beyond it,
    # the mean bend between spawn and goal, and three examples (spawn frame, goal frame, bend)
    try:
        cr = env.env_method("goal_corner_probe_result")[0]
        rows = [(sid, v) for sid, v in cr.items() if not sid.endswith("_examples")]
        if rows and any(v[0] > 0 for _, v in rows):
            print("=== CORNER EPISODES (same draws) ===")
            print(f"{'scene':<16}{'corner %':>10}{'mean bend':>11}   examples (spawn->goal frame, bend deg)")
            for sid, (pct, mb) in sorted(rows):
                ex = cr.get(sid + "_examples", [])
                print(f"{sid:<16}{pct:9.0f}%{mb:10.0f}   {ex}")
            tot = float(np.mean([v[0] for _, v in rows]))
            print(f"corner episodes over all scenes: {tot:.0f}% of draws (bend scenes fire, straight walks fall back)")
    except Exception as _e:
        print(f"[corner probe] unavailable: {_e}")
    print(flush=True)


def _frozen_probe(env, steps: int = 120) -> None:
    """What does the reward pay a robot that does NOTHING?

    Every claim about the policy freezing rested on my arithmetic for a
    "frozen on clean sidewalk" baseline that no policy had been observed to
    follow (Joana caught this, 2026-09-02). So measure it: pin the action to
    zero, run, and print the per-component budget. If standing still is cheap,
    the freezing trap is real and the goal term has to outweigh it. If it is
    expensive, my analysis was wrong and the weights need a different fix.
    """
    import numpy as _np
    keys = ("semantic", "goal", "collision", "step", "spin", "smooth",
            "proximity", "proximity_ground", "coherence", "crash", "timeout", "total")
    sums = {k: 0.0 for k in keys}
    n = 0
    try:
        env.reset()
        act = _np.zeros((env.num_envs,) + env.action_space.shape,
                        dtype=_np.float32)
        for _ in range(steps // max(1, env.num_envs)):
            _, _, _, infos = env.step(act)
            for info in infos:
                if "total" not in info:
                    continue
                for k in keys:
                    v = float(info.get(k, 0.0))
                    if v == v:
                        sums[k] += v
                n += 1
    except Exception as e:
        print(f"[frozen probe] skipped ({e})", flush=True)
        return
    if not n:
        print("[frozen probe] no steps recorded", flush=True)
        return
    line = "-" * 78
    print(f"\n{line}\nFROZEN PROBE -- reward per step for action=0, "
          f"{n} steps (MEASURED, not assumed)\n{line}")
    for k in keys:
        print(f"  {k:<12} {sums[k] / n:+9.4f}")
    print(f"\n  A policy that never moves earns {sums['total'] / n:+.4f} per "
          f"step, so {sums['total'] / n * 120:+.1f} over a\n  120-step episode "
          f"before the timeout penalty. Moving has to beat that.\n{line}",
          flush=True)


def _scene_env_cfg(args):
    # Shaping v2 (approved 2026-07-23): void split from obstacles (mild penalty),
    # collision 5->1, goal pull 0.5->1.5, anti-spin tax, spawn curriculum,
    # goals ~3 real meters. Corrected metric scale (camera 0.25 m): blind zone
    # starts ~0.6 m, so look_ahead 1.5 m is safely visible.
    return SceneEnvConfig(
        max_steps=args.max_steps,
        step_size_m=0.25,                    # matches demo action scale; 0.15 clipped 43% of demos
        yaw_step_rad=0.3,
        reward=RewardWeights(semantic=getattr(args, "semantic_weight", 1.0),
                             goal=getattr(args, "goal_weight", 1.5),
                             collision=1.0,
                             step_cost=0.05, void_cost=0.3,
                             terrain_as_cost=True),          # v4
        look_ahead_dist=1.5,
        # 0.0 = old single-box behaviour. The camera sits at 0.25 m and the
        # blind zone starts ~0.6 m (see above), so with a 0.7 m body the
        # closest FULLY VISIBLE collision box is centred at 0.95 m -- 1.0 is
        # the practical floor, not the 0.4 m the geometry alone would suggest.
        collision_look_ahead_m=getattr(args, "collision_look_ahead", 0.0),
        collision_box_memory=int(getattr(args, "collision_box_memory", 0)),
        collision_box_memory_agg=str(getattr(args, "collision_box_memory_agg", "newest") or "newest"),
        collision_box_memory_always=bool(getattr(args, "collision_box_memory_always", False)),
        collision_at_next_pose=bool(getattr(args, "collision_at_next_pose", False)),
        look_ahead_auto=bool(getattr(args, "look_ahead_auto", False)),
        footprint_next_heading=bool(getattr(args, "footprint_next_heading", False)),
        crash_requires_motion=bool(getattr(args, "crash_requires_motion", False)),
        goal_support_radius_m=getattr(args, "goal_support_radius", 0.0),
        goal_support_min_frac=getattr(args, "goal_support_min_frac", 0.25),
        goal_radius=getattr(args, "goal_radius", 0.75),
        collision_threshold=0.1,
        spin_cost=getattr(args, "spin_cost", 0.05),
        backward_cost=getattr(args, "backward_cost", 0.0),
        collision_terminate_frac=getattr(args, "collision_terminate_frac", 0.0),
        phantom_veto=bool(getattr(args, "phantom_veto", False)),
        collision_terminate_penalty=getattr(args, "collision_terminate_penalty", 20.0),
        void_terminate_frac=getattr(args, "void_terminate_frac", 0.0),
        void_terminate_penalty=getattr(args, "void_terminate_penalty", 100.0),
        image_void_terminate_frac=getattr(args, "image_void_terminate_frac", 0.0),
        coherence_cost_weight=getattr(args, "coherence_cost_weight", 0.0),
        coherence_tau=getattr(args, "coherence_tau", 0.4),
        coherence_terminate_tau=getattr(args, "coherence_terminate_tau", 0.0),
        coherence_terminate_penalty=getattr(args, "coherence_terminate_penalty", 100.0),
        disagree_cost_weight=float(getattr(args, "disagree_cost", 0.0) or 0.0),
        rgb_align_cost_weight=float(getattr(args, "rgb_align_cost", 0.0) or 0.0),
        disagree_frac=getattr(args, "disagree_frac", None),
        rgb_align_tau=float(getattr(args, "rgb_align_tau", 0.8)),
        proximity_weight=getattr(args, "proximity_weight", 0.0),
        proximity_margin=getattr(args, "proximity_margin", 1.0),
        clouds_dir=getattr(args, "clouds_dir", None),
        action_chunk=getattr(args, "action_chunk", 1),
        footprint_along_motion=getattr(args, "footprint_along_motion", False),
        forward_only=getattr(args, "forward_only", False),
        action_smooth_cost=getattr(args, "action_smooth_cost", 0.0),
        goal_bonus=getattr(args, "goal_bonus", 50.0),        # v4 default
        obs_out_hw=((getattr(args, "obs_height", 336), getattr(args, "obs_width", 560))
                    if (getattr(args, "render_height", None)
                        or getattr(args, "render_width", None)) else None),
        goal_noise_std=getattr(args, "goal_noise_std", 0.0),
        proximity_delta=getattr(args, "proximity_delta", False),
        proximity_ground_weight=float(getattr(args, "proximity_ground_weight", 0.0)),
        proximity_ground_margin=float(getattr(args, "proximity_ground_margin", 1.2)),
        proximity_ground_classes=str(getattr(args, "proximity_ground_classes", "3,4,5") or "3,4,5"),
        goal_min_score=float(getattr(args, "goal_min_score", 0.0) or 0.0),
        goal_lat_jitter_m=float(getattr(args, "goal_lat_jitter_m", 0.0) or 0.0),
        goal_lat_jitter_min_share=float(getattr(args, "goal_lat_jitter_min_share", 0.6) or 0.6),
        timeout_penalty=getattr(args, "timeout_penalty", 0.0),
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
        # 2026-09-22: these are SceneEnvConfig fields (the stacking happens in
        # SceneEnv._obs). They were passed to CalibratedBackendConfig instead,
        # which has no such field, so every arm carrying FRAMESTACK died in 8 s
        # with a TypeError. Eval had them on the right config all along.
        max_steps_base=int(getattr(args, "max_steps_base", 40) or 40),
        max_steps_per_m=float(getattr(args, "max_steps_per_m", 0.0) or 0.0),
        obs_frame_stack=int(getattr(args, "obs_frame_stack", 1) or 1),
        obs_frame_stride=int(getattr(args, "obs_frame_stride", 1) or 1),
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
        goal_case_tries=int(getattr(args, "goal_case_tries", 24)),
        goal_center_clear_m=float(getattr(args, "goal_center_clear_m", 0.0) or 0.0),
        mirror_prob=float(getattr(args, "mirror_prob", 0.0) or 0.0),
        map_walk_halfwidth_m=float(getattr(args, "map_walk_halfwidth_m", 0.4)),
        map_ignore_classes=str(getattr(args, "map_ignore_classes", "")),
        label_remap=str(getattr(args, "label_remap", "") or ""),
        timeout_distance_scaled=getattr(args, "timeout_distance_scaled", False),
        reward_scale=getattr(args, "reward_scale", 1.0),
        random_spawn=True,
        trav_path=getattr(args, "trav_path", None),
        failure_snap_dir=str(args.output_dir / "failures"),
    )


def make_live_vec_env(args):
    """Batched live training: N robots in ONE scene sharing one pipe. Builds
    the same configs as make_env, then N defer-render SceneEnvs around a
    single BatchedLiveDiffusedBackend and the LiveVecEnv that batches every
    render into one generation call."""
    from src.env.vec_live_env import (
        BatchedLiveDiffusedBackend, InjectedLabelBackend, LiveVecEnv,
    )
    from stable_baselines3.common.vec_env import VecMonitor

    # Multi-scene rotation (2026-08-29): all robots still share ONE resident
    # scene at a time; LiveVecEnv rotates it every --scene_rotate steps.
    scenes = args.scenes if getattr(args, "scenes", None) else [args.scene]
    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in scenes},
        goal_frame=args.goal_frame,
        goal_frame_range=tuple(args.goal_frame_range) if args.goal_frame_range else None,
        goal_min_sep_m=args.goal_min_sep,
        spawn_max_frame=args.spawn_max_frame,
        goal_xy_override=(tuple(float(v) for v in args.goal_xy.split(","))
                          if getattr(args, "goal_xy", None) else None),
        goal_dir_360=getattr(args, "goal_dir_360", False),
        goal_cone_deg=getattr(args, "goal_cone_deg", 360.0),
        goal_dist_range=(tuple(float(v) for v in args.goal_dist_range.split(","))
                         if getattr(args, "goal_dist_range", None) else None),
        goal_dist_window_m=getattr(args, "goal_dist_window", None),
        goal_turn_deg=float(getattr(args, "goal_turn_deg", 0.0) or 0.0),
        goal_turn_mix=float(getattr(args, "goal_turn_mix", 0.0) or 0.0),
        goal_turn_beyond_m=float(getattr(args, "goal_turn_beyond_m", 2.0)),
        spawn_label_classes=(tuple(int(v) for v in args.spawn_classes.split(","))
                             if getattr(args, "spawn_classes", None) else None),
        spawn_yaw_jitter_deg=getattr(args, "spawn_yaw_jitter", 0.0),
        spawn_lat_jitter_m=getattr(args, "spawn_lat_jitter", 0.0),
        spawn_heading_from_walk=bool(getattr(args, "spawn_heading_from_walk", False)),
        spawn_frames_by_scene=str(getattr(args, "spawn_frames", "") or ""),
        sem_palette_version=getattr(args, "sem_palette", 1), static_scene=bool(getattr(args, "static_scene", False)),
        static_movers=str(getattr(args, "static_movers", "") or ""),
        render_window=int(getattr(args, "render_window", 0) or 0),
        coverage_window=int(getattr(args, "coverage_window", 0) or 0),
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=getattr(args, "render_height", None) or getattr(args, "obs_height", 336),
        W=getattr(args, "render_width", None) or getattr(args, "obs_width", 560),
        goal_dist_m=getattr(args, "goal_dist", None),
        spawn_min_frame=getattr(args, "spawn_min", 0),
    )
    world = BatchedLiveDiffusedBackend(
        cfg, checkpoint=args.live_ckpt, live_frames=args.live_frames,
        alpha_gate=not getattr(args, "no_alpha_gate", False),
        alpha_gate_tau=getattr(args, "alpha_gate_tau", 0.5),
        raster_obs=bool(getattr(args, "raster_obs", False)))
    world.num_inference_steps = args.live_steps
    # Pre-reconstruct EVERY rotation scene now, while GPU headroom is maximal
    # (2026-08-30, G12 saga: first-visit reconstruction mid-training OOMs at
    # 560-render — training state + the resident scene's splats eat the ~7.3GB
    # the reconstructor needs). Each fresh cache is evicted to CPU immediately
    # so the next reconstruction also sees a clean GPU; rotation later never
    # reconstructs, it only uploads from the cache.
    if len(scenes) > 1:
        import gc as _gc
        import torch as _torch
        from src.env.real_backend import _move_tree_to
        done = []
        for s in scenes:
            free0 = _torch.cuda.mem_get_info()[0] / 1e9
            try:
                world.load_scene(s)
            except _torch.cuda.OutOfMemoryError:
                # 2026-08-30: ~2 GB/scene leaks through eviction at 560-render
                # (measured ladder 32.6->18.7 over 8 scenes). Until the leak is
                # found, survive: train on the scenes that fit, say so loudly.
                print(f"[make_live_vec_env] OOM pre-reconstructing {s} — "
                      f"PRUNING scene list to the {len(done)} that fit: "
                      f"{done}", flush=True)
                break
            world._cache[s] = _move_tree_to(world._cache[s], "cpu")
            # Clear the rasterizer's own splat reference (leak suspect #1) and
            # collect ref cycles before measuring.
            runner = getattr(getattr(getattr(world._reconstructor, "gs_renderer", None),
                                     "rasterizer", None), "runner", None)
            if runner is not None and hasattr(runner, "splats"):
                runner.splats = None
            _gc.collect()
            _torch.cuda.empty_cache()
            free1 = _torch.cuda.mem_get_info()[0] / 1e9
            done.append(s)
            print(f"[make_live_vec_env] pre-reconstructed {s}: free "
                  f"{free0:.1f} -> {free1:.1f} GB (leak {free0 - free1:.2f})",
                  flush=True)
        if done and len(done) < len(scenes):
            scenes = done
    world.load_scene(scenes[0])

    envs = []
    for _ in range(args.live_batch):
        env_cfg = _scene_env_cfg(args)
        env_cfg.defer_render = True
        e = SceneEnv(world_backend=world, semantic_backend=None,
                     scene_ids=scenes, cfg=env_cfg)
        e.semantic_backend = InjectedLabelBackend(e)
        envs.append(e)
    _start = int(getattr(args, "scene_start_idx", 0) or 0) % max(1, len(scenes))
    if _start:
        world.load_scene(scenes[_start])
        for e in envs:
            e.scene_ids = [scenes[_start]]
        print(f"[make_live_vec_env] worker starts on {scenes[_start]} (index {_start})", flush=True)
    return VecMonitor(LiveVecEnv(envs, world, scenes=scenes,
                                 rotate_every=getattr(args, "scene_rotate", 0),
                                 start_idx=_start))


def _live_worker_args(args, worker_idx: int, n_workers: int):
    """The args one GPU worker trains with. With exactly n_workers scenes,
    worker i gets scene i and no rotation; otherwise every worker gets the
    full list and rotates on its own clock."""
    import copy
    a = copy.copy(args)
    scenes = list(args.scenes) if getattr(args, "scenes", None) else None
    if scenes and len(scenes) == n_workers:
        a.scenes = [scenes[worker_idx]]
        a.scene_rotate = 0
    elif scenes and n_workers > 1:
        # stagger: worker i starts a proportional way into the list so the K
        # workers hold K DIFFERENT scenes at once (2026-09-23).
        a.scene_start_idx = (worker_idx * len(scenes)) // n_workers
    return a


def make_multi_gpu_live_env(args):
    """K live vec-envs, one per GPU, behind one VecEnv (--live_gpus K).

    Each worker process pins itself to one GPU of the job, then builds the
    same VecMonitor(LiveVecEnv) as make_live_vec_env, with --live_batch
    robots. Throughput scales ~K x because the diffusion calls run on K
    cards at once; PPO's rollout is n_steps x (K x live_batch) env steps.
    """
    from src.env.multi_gpu_vec_env import MultiGPUVecEnv
    K = int(args.live_gpus)
    B = int(getattr(args, "live_batch", 1))
    scenes = list(args.scenes) if getattr(args, "scenes", None) else [args.scene]
    if len(scenes) not in (1, K) and getattr(args, "scene_rotate", 0) <= 0:
        raise SystemExit(f"--live_gpus {K} with {len(scenes)} scenes needs either "
                         f"exactly {K} scenes (one per GPU) or --scene_rotate > 0")

    def _factory(i: int):
        return make_live_vec_env(_live_worker_args(args, i, K))

    env = MultiGPUVecEnv(_factory, n_workers=K)
    per_worker = ([scenes[i] for i in range(K)] if len(scenes) == K
                  else [scenes] * K)
    print(f"[live_gpus] {K} GPUs x {B} robots = {env.num_envs} envs; "
          f"scene per worker: {per_worker}; rollout = 128 x {env.num_envs} = "
          f"{128 * env.num_envs} env steps; checkpoint every "
          f"{int(args.ckpt_every_calls) * env.num_envs} env steps", flush=True)
    return env


def save_rollout_video(model, env, out_path: Path, max_frames=120,
                       seed: "int | None" = None,
                       sem_palette: int = 4):
    """Eval rollout with a HUD (step/action/reward/dist) + top-down map inset:
    agent path (white), recorded real trajectory (gray), goal (green). Makes
    off-manifold wandering legible — black frames = outside the reconstructed
    volume, and the map shows exactly where the agent is relative to the trail."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw
    from src.eval.palette import CLASS_COLORS_V14_255
    from src.eval.reward_2d import (
        _footprint_corners_world, _project_points, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
    )

    def _colorize(lab, H, W, tag, footprint_uv=None):
        import cv2
        # v14 runs carry ids 0-13; legacy raster runs carry the 30-class ids —
        # pick the palette that actually matches the ids or colors lie.
        # Use the SAME palette the surveys and the reward decode use --
        # v14_palette(sem_palette). This drew from src/eval/palette.py instead,
        # a different table, so eval videos were coloured differently from
        # every other visual in the pipeline (2026-09-02). Falls back if the
        # taxonomy import is unavailable.
        try:
            from diffsynth.utils.class_taxonomy import v14_palette
            pal = (v14_palette(sem_palette).numpy() * 255).astype(np.uint8)
        except Exception:
            from src.eval.palette import CLASS_COLORS_255
            pal = (CLASS_COLORS_V14_255 if int(np.max(lab)) < 14
                   else CLASS_COLORS_255)
        col = pal[np.clip(lab, 0, len(pal) - 1)]
        if col.shape[:2] != (H, W):
            col = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
        col = col.copy()
        if footprint_uv is not None:
            # the exact quad the reward scores this step (look-ahead footprint)
            cv2.polylines(col, [footprint_uv.astype(np.int32)], True,
                          (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(col, tag, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return col

    def _footprint_uv(base_env, world, action=None):
        """Project the reward's footprint quad into the CURRENT view. Same math
        as compute_reward: corners from pose+heading, camera from the cached
        view actually served (so the box lands where the reward truly read).

        2026-09-08: the box is drawn BEFORE env.step(), so `_last_fp_heading`
        is the PREVIOUS step's post-turn heading. With footprint_next_heading
        the reward scores the box at THIS step's post-turn heading, and the box
        is only 0.3 m wide 1.5 m ahead: a yaw command of 0.7 (0.2 rad) shifts
        it 0.31 m = one full box width, so the drawn box and the scored box did
        not overlap whenever the policy turned (Joana: "how is that not a
        collision" on a frame whose drawn box sat on grass while the reward
        read 0.06). Rotate by the action about to be taken, exactly as
        SceneEnv.step does."""
        try:
            pose = base_env._robot_pose_world
            pos = pose[:3, 3]
            hd = None
            if action is not None and bool(getattr(base_env.cfg, "footprint_next_heading", False)):
                _w = float(np.asarray(action).reshape(-1)[1]) * float(base_env.cfg.yaw_step_rad)
                _c, _s = np.cos(_w), np.sin(_w)
                _R = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                hd = (_R @ pose[:3, :3].astype(np.float64)) @ np.array([1.0, 0.0, 0.0])
            if hd is None:
                hd = getattr(base_env, "_last_fp_heading", None)
            if hd is None:
                hd = pose[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            corners = _footprint_corners_world(
                pos, hd, look_ahead_dist=base_env.cfg.look_ahead_dist,
                length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
            uv, in_front = _project_points(corners, base_env._last_K, base_env._last_w2c)
            if not in_front.all():
                return None
            return uv
        except Exception:
            return None

    def semantic_panel(world, H, W, footprint_uv=None):
        """Colorized semantics: what the reward reads, and (gated runs) also
        the model's RAW belief before voiding — the semantics-model test view."""
        used = getattr(world, "_last_semantic_image", None)
        if used is None:
            return None
        raw = getattr(world, "_last_semantic_raw", None)
        panels = []
        if raw is not None and raw is not used:
            panels.append(_colorize(raw, H, W, "sem RAW (model belief)", footprint_uv))
        # Say what this panel IS: under a map reward the generated semantics
        # are an observation, not the reward, and the panel used to say REWARD.
        _src = str(getattr(getattr(base_env, "cfg", None), "reward_source", "generated"))
        _lab = ("sem REWARD" if _src == "generated" else
                "sem GENERATED (reward = MAP)" if _src == "map" else
                "sem GENERATED (reward = map, gen fallback)")
        panels.append(_colorize(used, H, W, (_lab + " (gated)") if len(panels) else _lab, footprint_uv))
        return np.concatenate(panels, axis=1)

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    frames, path_xy, rows = [], [], []
    _cams = []
    # Seeded so the video is THE SAME EPISODE as the scored one, and so two
    # policies compared on the same seed get identical spawns and goals.
    # Unseeded, eval drew fresh spawns every run and no two evaluations
    # were comparable episode-by-episode (2026-09-02).
    obs, _ = (env.reset(seed=seed) if seed is not None else env.reset())
    # MUST precede calib access: reset loads the scene

    def _xyyaw():
        P = base_env._robot_pose_world
        return [round(float(P[0, 3]), 3), round(float(P[1, 3]), 3),
                round(float(np.arctan2(P[1, 0], P[0, 0])), 3)]
    traj = [_xyyaw() + [0]]       # [x, y, yaw, collision] — same format as eval
    world = base_env.world_backend
    cal = world._calib[base_env.scene_ids[0]] if hasattr(world, "_calib") else None
    ref_traj = cal.positions[:, :2] if cal is not None else None
    goal = base_env._goal_world[:2].copy()
    done, r, info = False, 0.0, {}
    while not done and len(frames) < max_frames:
        pose = base_env._robot_pose_world
        path_xy.append(pose[:2, 3].copy())
        action, _ = model.predict(obs, deterministic=True)

        img = Image.fromarray(env.render().copy())
        draw = ImageDraw.Draw(img, "RGBA")
        H, W = img.height, img.width

        # --- top-down map inset (bottom-right), fixed world window ---
        m = 110
        all_pts = np.array(path_xy + ([goal] if goal is not None else []))
        if ref_traj is not None:
            all_pts = np.vstack([all_pts, ref_traj])
        lo, hi = all_pts.min(0) - 1.0, all_pts.max(0) + 1.0
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-3))
        def to_px(p):
            return (W - m - 6 + (p[0] - lo[0]) / span * m,
                    H - 6 - (p[1] - lo[1]) / span * m)
        draw.rectangle([W - m - 10, H - m - 10, W - 2, H - 2], fill=(0, 0, 0, 170))
        if ref_traj is not None:
            draw.line([to_px(p) for p in ref_traj[::4]], fill=(160, 160, 160, 255), width=1)
        if len(path_xy) > 1:
            draw.line([to_px(p) for p in path_xy], fill=(255, 255, 255, 255), width=2)
        gx, gy = to_px(goal)
        # success disc (goal_radius) so terminations look right, then the dot
        rr = base_env.cfg.goal_radius / span * m
        draw.ellipse([gx - rr, gy - rr, gx + rr, gy + rr],
                     outline=(0, 255, 0, 200), width=1)
        draw.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=(0, 255, 0, 255))
        ax, ay = to_px(path_xy[-1])
        draw.ellipse([ax - 2, ay - 2, ax + 2, ay + 2], fill=(255, 80, 80, 255))
        # heading arrow: where the robot LOOKS (local +x in world), 0.8 m long.
        # If this arrow ever disagrees with where the white path advances next
        # frame, the pose/obs frames are inconsistent — the suspected frame bug.
        hd = pose[:2, 0] / (np.linalg.norm(pose[:2, 0]) + 1e-9)
        # 2.5 m and thicker: at 0.8 m the arrow was a few pixels on the
        # mini-map and unreadable in the episode videos (Joana, 2026-09-07).
        hx, hy = to_px(path_xy[-1] + hd * 2.5)
        draw.line([(ax, ay), (hx, hy)], fill=(255, 160, 0, 255), width=3)
        _ph = np.array([-hd[1], hd[0]]); _tip = path_xy[-1] + hd * 2.5
        for _sgn in (1.0, -1.0):
            _bx, _by = to_px(_tip - hd * 0.6 + _sgn * _ph * 0.4)
            draw.line([(hx, hy), (_bx, _by)], fill=(255, 160, 0, 255), width=3)

        # goal-bearing compass on the FPV: obs dyaw, 0 = dead ahead (up),
        # positive = goal to the LEFT. The needle should swing to center as
        # the policy turns toward the goal.
        gr = base_env._goal_in_robot_frame()
        cx, cy = W // 2, H - 22
        tipx = cx - 16 * np.sin(gr[2])
        tipy = cy - 16 * np.cos(gr[2])
        draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(0, 0, 0, 130))
        draw.line([(cx, cy), (tipx, tipy)], fill=(0, 255, 0, 255), width=2)
        draw.ellipse([tipx - 2, tipy - 2, tipx + 2, tipy + 2], fill=(0, 255, 0, 255))

        # --- HUD banner ---
        draw.rectangle([0, 0, W, 14], fill=(0, 0, 0, 180))
        if bool(getattr(base_env, "_mirrored", False)):
            draw.text((W - 190, 2), "MIRRORED EPISODE (policy view)", fill=(255, 80, 80, 255))
        draw.text((4, 2), f"t={len(frames):3d} v={float(action[0]):+.2f} "
                          f"w={float(action[1]):+.2f} r={float(r):+.2f} "
                          f"dist={info.get('dist_to_goal', float('nan')):.1f}m "
                          f"goal={np.degrees(gr[2]):+.0f}deg"
                          + (f" cache={world._last_lookup[0]*100:.0f}cm/"
                             f"{world._last_lookup[1]:.0f}deg"
                             if getattr(world, "_last_lookup", None) else ""),
                  fill=(255, 255, 255, 255))
        frame = np.array(img.convert("RGB"))
        fp_uv = _footprint_uv(base_env, world, action)
        if fp_uv is not None and bool(getattr(base_env, "_mirrored", False)):
            fp_uv = fp_uv.copy(); fp_uv[:, 0] = (frame.shape[1] - 1) - fp_uv[:, 0]   # mirrored view
        if fp_uv is not None:
            import cv2 as _cv
            _cv.polylines(frame, [fp_uv.astype(np.int32)], True, (255, 255, 0), 2, _cv.LINE_AA)
        sem = semantic_panel(world, H, W, fp_uv)
        if sem is not None:
            frame = np.concatenate([frame, sem], axis=1)
        frames.append(frame)
        # PATH OVERLAY bookkeeping (2026-09-16, Joana): the camera of this frame
        # and the action taken from it, so the EXECUTED next poses (drawn after
        # the episode) and, for chunked policies, the PLANNED chunk can be
        # projected onto this very frame.
        try:
            _cams.append((np.asarray(base_env._last_K, dtype=np.float64).copy(),
                          np.asarray(base_env._last_w2c, dtype=np.float64).copy(),
                          pose.astype(np.float64).copy(), np.asarray(action, dtype=np.float64).ravel().copy()))
            # (action here IS the whole decision: model.predict returns the chunk)
        except Exception:
            _cams.append(None)

        obs, r, terminated, truncated, info = env.step(action)
        traj.append(_xyyaw() +
                    [round(float(max(0.0, -info.get("collision", 0.0))), 3)])
        # Stamp the frame we just saved with what the crash box read AT THAT
        # POSE (the reward of this step is charged before the move). Without
        # it a map-reward crash is invisible: the robot stands on the walk and
        # the box 1.5 m ahead is on the lawn (2026-09-04, eval 466977).
        try:
            import cv2 as _cv
            _cf = float(max(0.0, -info.get("collision", 0.0)))
            _mf = float(info.get("map_collision_frac", float("nan")))
            _gf = float(info.get("gen_collision_frac", float("nan")))
            _thr = float(getattr(base_env.cfg, "collision_terminate_frac", 0.35))
            _ahead = float(getattr(base_env.cfg, "collision_look_ahead_m", 0.0) or 0.0) or float(base_env.cfg.look_ahead_dist)
            _age = float(info.get("box_memory_age", 0.0) or 0.0)
            _mem = ("  mem miss" if _age < 0 else (f"  mem {int(_age)}" if _age > 0 else ""))
            _txt = (f"crash box {_ahead:.1f}m ahead: reward {_cf:.2f}" + _mem
                    + (f"  map {_mf:.2f}" if _mf == _mf else "")
                    + (f"  gen {_gf:.2f}" if _gf == _gf else "")
                    + ("  CRASH" if (_thr > 0 and _cf >= _thr) else ""))
            _cv.putText(frames[-1], _txt, (4, 30), _cv.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 60, 60) if (_thr > 0 and _cf >= _thr) else (255, 255, 255), 1, _cv.LINE_AA)
        except Exception:
            pass
        done = terminated or truncated

    # PATH OVERLAY (2026-09-16, Joana's ask): on frame t draw where the robot
    # actually went next (green: poses t+1 .. t+8 projected into frame t's
    # camera, dots = steps) and, for a chunked policy, where it PLANNED to go
    # (orange: the action chunk integrated from pose t with the env's step and
    # yaw constants). The two agree by construction for a chunked policy that
    # is executed open-loop; for a per-step policy only the green path exists.
    try:
        import cv2 as _cv
        _all_pos = [np.array([p[0], p[1], 0.0]) for p in path_xy]
        _sz = float(base_env.cfg.step_size_m); _yz = float(base_env.cfg.yaw_step_rad)
        for _t, _cam in enumerate(_cams):
            if _cam is None or _t >= len(frames):
                continue
            _K, _w2c, _pose, _act = _cam
            _fr = frames[_t]
            # executed path: the next 8 recorded positions
            _fut = np.array(_all_pos[_t + 1:_t + 9]) if _t + 1 < len(_all_pos) else np.zeros((0, 3))
            if len(_fut):
                _uv, _front = _project_points(_fut, _K, _w2c)
                _pts = [tuple(np.round(_uv[i]).astype(int)) for i in range(len(_fut)) if _front[i]]
                for _a, _b in zip(_pts[:-1], _pts[1:]):
                    _cv.line(_fr, _a, _b, (0, 230, 0), 2, _cv.LINE_AA)
                for _q in _pts:
                    _cv.circle(_fr, _q, 3, (0, 230, 0), -1, _cv.LINE_AA)
            # planned chunk: integrate the actions from pose t. NAVRL_FAKE_CHUNK=N
            # (2026-09-16) draws the current per-step action repeated N times as
            # a fake plan, to verify the projection before any chunked policy exists.
            _fake = int(os.environ.get("NAVRL_FAKE_CHUNK", "0") or 0)
            if _fake > 0 and _act.size == 2:
                _act = np.tile(_act, _fake)
            if _act.size >= 4 and _act.size % 2 == 0:
                _x, _y = float(_pose[0, 3]), float(_pose[1, 3])
                _yaw = float(np.arctan2(_pose[1, 0], _pose[0, 0]))
                _plan = []
                for _k in range(_act.size // 2):
                    _v, _w = float(_act[2 * _k]) * _sz, float(_act[2 * _k + 1]) * _yz
                    _yaw += _w
                    _x += _v * np.cos(_yaw); _y += _v * np.sin(_yaw)
                    _plan.append([_x, _y, 0.0])
                _uv, _front = _project_points(np.array(_plan), _K, _w2c)
                _pts = [tuple(np.round(_uv[i]).astype(int)) for i in range(len(_plan)) if _front[i]]
                for _a, _b in zip(_pts[:-1], _pts[1:]):
                    _cv.line(_fr, _a, _b, (255, 160, 0), 2, _cv.LINE_AA)
                for _q in _pts:
                    _cv.circle(_fr, _q, 3, (255, 160, 0), -1, _cv.LINE_AA)
                _cv.putText(_fr, "orange = planned chunk   green = executed path", (4, 46),
                            _cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, _cv.LINE_AA)
            elif len(_fut):
                _cv.putText(_fr, "green = executed path (next 8 steps)", (4, 46),
                            _cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, _cv.LINE_AA)
    except Exception as _e:
        print(f"[rollout video] path overlay skipped: {type(_e).__name__}: {_e}", flush=True)

    # Arrival frame: the loop above draws BEFORE stepping, so the terminal pose
    # (entering the goal disc) was never rendered — append it.
    # WHY the episode ended, burned in. "reached_goal=False" conflated
    # crashed-at-step-8, walked-to-the-boundary-and-held, and never-moved --
    # three different policies with the same label (her ask, 2026-09-02).
    if info.get("reached_goal"):
        outcome, ocol = "GOAL", (0, 255, 0, 255)
    elif float(info.get("crash", 0.0)) != 0.0:
        outcome, ocol = "CRASH", (255, 60, 60, 255)
    elif float(info.get("coherence_crash", 0.0)) != 0.0:
        outcome, ocol = "INCOHERENT (left the world model)", (255, 160, 0, 255)
    elif float(info.get("halted", 0.0)) != 0.0:
        outcome, ocol = "HALTED (stopped safely, short of the goal)", (80, 200, 255, 255)
    else:
        outcome, ocol = "TIMEOUT (ran out of steps)", (255, 255, 0, 255)
    final = Image.fromarray(env.render().copy())
    fd = ImageDraw.Draw(final, "RGBA")
    fd.rectangle([0, 0, final.width, 26], fill=(0, 0, 0, 200))
    fd.text((4, 2), f"t={len(frames):3d}  ENDED: {outcome}", fill=ocol)
    fd.text((4, 14), f"dist_to_goal={info.get('dist_to_goal', float('nan')):.2f}m "
                     f"r={float(r):+.2f}", fill=(220, 220, 220, 255))
    last = np.array(final.convert("RGB"))
    sem = semantic_panel(world, last.shape[0], last.shape[1])
    if sem is not None:
        last = np.concatenate([last, sem], axis=1)
    # hold the final frame ~1 s so the outcome is readable at 8 fps
    frames.extend([last] * 8)

    iio.imwrite(str(out_path), np.stack(frames), fps=8,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[train_ppo_real] rollout video: {out_path} "
          f"({len(frames)} frames, reached_goal={info.get('reached_goal')})")
    return {"traj": traj,
            "goal_xy": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
            "outcome": outcome.split(" ")[0],
            "success": bool(info.get("reached_goal", False))}


def expand_action_head(model_new, ckpt_path: Path, stop_bias: float, stop_log_std: float = -0.7):
    """Warm-start a policy that gained a STOP output (2026-09-06). Every
    parameter with a matching shape is copied from the checkpoint; the three
    that changed shape -- action_net.weight [3,H] vs [2,H], action_net.bias
    [3] vs [2], log_std [3] vs [2] -- keep the two driving rows and get a new
    stop row with zero weights and bias `stop_bias` (-2 => P(stop) ~2% per
    step under unit noise), so the reacher keeps reaching and stops rarely
    until the bonus teaches it where. Fails loudly on any other mismatch."""
    import torch as th
    old = PPO.load(str(ckpt_path), device=model_new.device)
    src = old.policy.state_dict()
    dst = model_new.policy.state_dict()
    copied, expanded, missing = [], [], []
    with th.no_grad():
        for k, v in dst.items():
            if k not in src:
                missing.append(k); continue
            s = src[k]
            if tuple(s.shape) == tuple(v.shape):
                v.copy_(s); copied.append(k)
            elif k == "action_net.weight" and s.shape[0] + 1 == v.shape[0] and s.shape[1:] == v.shape[1:]:
                v.zero_(); v[:s.shape[0]].copy_(s); expanded.append(k)
            elif k == "action_net.bias" and s.shape[0] + 1 == v.shape[0]:
                v[:s.shape[0]].copy_(s); v[s.shape[0]] = float(stop_bias); expanded.append(k)
            elif k == "log_std" and s.shape[0] + 1 == v.shape[0]:
                v[:s.shape[0]].copy_(s); v[s.shape[0]] = float(stop_log_std); expanded.append(k)
            else:
                raise RuntimeError(f"[expand_action_head] cannot map {k}: checkpoint {tuple(s.shape)} vs new {tuple(v.shape)}")
    if missing:
        raise RuntimeError(f"[expand_action_head] keys absent from the checkpoint: {missing[:8]}")
    model_new.policy.load_state_dict(dst)
    print(f"[expand_action_head] copied {len(copied)} tensors, expanded {expanded} from {ckpt_path.name}; "
          f"stop bias {stop_bias:+.1f} (P(stop) ~{100 * 0.5 * (1 + __import__('math').erf(stop_bias / 2 ** 0.5)):.0f}% per step at unit noise)", flush=True)


def _bc_stop_metrics(act_true: np.ndarray, act_pred: np.ndarray, stop_action: bool,
                     eps: float = 0.15) -> dict:
    """Stop recall / false-stop rate / accuracy of predicted actions against
    demo actions. A demo stop is act[2] > 0 with the stop action, else a zero
    throttle; a predicted stop uses the same rule (throttle < eps)."""
    if stop_action and act_true.shape[1] >= 3:
        t_stop = act_true[:, 2] > 0.0
        p_stop = act_pred[:, 2] > 0.0
    else:
        t_stop = np.abs(act_true[:, 0]) < eps
        p_stop = np.abs(act_pred[:, 0]) < eps
    n_stop, n_drive = int(t_stop.sum()), int((~t_stop).sum())
    drive = ~t_stop
    return {
        "n": int(len(act_true)), "n_stop": n_stop, "n_drive": n_drive,
        "stop_recall": (float((p_stop & t_stop).sum()) / n_stop) if n_stop else float("nan"),
        "false_stop": (float((p_stop & drive).sum()) / n_drive) if n_drive else float("nan"),
        "accuracy": float((p_stop == t_stop).mean()) if len(t_stop) else float("nan"),
        "throttle_mae": float(np.abs(act_pred[drive, 0] - act_true[drive, 0]).mean()) if n_drive else float("nan"),
        "yaw_mae": float(np.abs(act_pred[drive, 1] - act_true[drive, 1]).mean()) if n_drive else float("nan"),
    }


def bc_pretrain(model, demos_path: Path, epochs: int = 25, batch_size: int = 64,
                holdout: float = 0.2):
    """Behavior-clone the PPO policy on demonstrations.

    Maximizes log-prob of demo actions under the policy (SB3 evaluate_actions
    handles preprocessing internally, incl. image normalization). The policy
    then enters PPO already knowing roughly how to drive.

    2026-09-06: a HELD-OUT split (whole episodes when the npz carries an
    `episode` key, else the last `holdout` share of samples) reports, every
    epoch, the stop recall and false-stop rate of the DETERMINISTIC policy
    action. This is the learnability number Joana asked for: can the frame +
    goal vector tell where the map's verge is, with RL out of the loop.
    """
    import torch as th
    d = np.load(demos_path, allow_pickle=True)
    obs_rgb = d["obs"]                      # [N,H,W,3] uint8
    goal = d["goal"].astype(np.float32)
    act = d["act"].astype(np.float32)
    n = len(act)
    stop_action = bool(int(model.action_space.shape[0]) == 3)
    if int(model.action_space.shape[0]) != int(act.shape[1]):
        raise SystemExit(f"[bc] REFUSED: demos carry {act.shape[1]}-dim actions but the policy has "
                         f"{int(model.action_space.shape[0])} (record the demos under the same STOPACT setting)")
    # held-out split
    if "episode" in d and holdout > 0.0:
        eps_ids = np.asarray(d["episode"]); uniq = np.unique(eps_ids)
        n_hold = max(1, int(round(len(uniq) * holdout))) if len(uniq) > 1 else 0
        hold_eps = set(uniq[-n_hold:].tolist()) if n_hold else set()
        hold = np.array([e in hold_eps for e in eps_ids], dtype=bool)
    elif holdout > 0.0 and n > 10:
        hold = np.zeros(n, dtype=bool); hold[int(n * (1.0 - holdout)):] = True
    else:
        hold = np.zeros(n, dtype=bool)
    tr_idx, ho_idx = np.nonzero(~hold)[0], np.nonzero(hold)[0]
    device = model.policy.device
    # Model observation space is channel-first (SB3 VecTransposeImage).
    rgb_chw = np.transpose(obs_rgb, (0, 3, 1, 2))
    _stops = int(((act[:, 2] > 0) if stop_action else (np.abs(act[:, 0]) < 0.15)).sum())
    print(f"[bc] {n} demos ({_stops} stop steps), train {len(tr_idx)} / held-out {len(ho_idx)}, "
          f"{epochs} epochs, action dim {act.shape[1]}", flush=True)

    def _predict(idx: np.ndarray) -> np.ndarray:
        outs = []
        with th.no_grad():
            for s0 in range(0, len(idx), 256):
                b = idx[s0:s0 + 256]
                obs_t = {"rgb": th.as_tensor(rgb_chw[b]).to(device),
                         "goal": th.as_tensor(goal[b]).to(device)}
                a = model.policy._predict(obs_t, deterministic=True)
                outs.append(a.detach().cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, act.shape[1]), np.float32)

    def _report(tag: str):
        if not len(ho_idx):
            return
        m = _bc_stop_metrics(act[ho_idx], _predict(ho_idx), stop_action)
        print(f"[bc] HELD-OUT {tag}: stop recall {m['stop_recall']:.2f} ({m['n_stop']} stops)  "
              f"false-stop {m['false_stop']:.3f} ({m['n_drive']} drives)  accuracy {m['accuracy']:.2f}  "
              f"throttle MAE {m['throttle_mae']:.2f}  yaw MAE {m['yaw_mae']:.2f}", flush=True)

    _report("before")
    opt = th.optim.Adam(model.policy.parameters(), lr=3e-4)
    idx = tr_idx.copy()
    for ep in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for s0 in range(0, len(idx), batch_size):
            b = idx[s0:s0 + batch_size]
            obs_t = {
                "rgb": th.as_tensor(rgb_chw[b]).to(device),
                "goal": th.as_tensor(goal[b]).to(device),
            }
            act_t = th.as_tensor(act[b]).to(device)
            _, log_prob, entropy = model.policy.evaluate_actions(obs_t, act_t)
            loss = -log_prob.mean() - 1e-3 * entropy.mean()
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        print(f"[bc] epoch {ep+1}/{epochs} loss {np.mean(losses):.3f}", flush=True)
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            _report(f"epoch {ep+1}")


def resolve_warmstart(warmstart, goal_dist_start=None, log=print):
    """(checkpoint_path, goal_dist_start) for a --warmstart that may be a DIRECTORY.

    A directory is resolved to its newest ppo_<N>_steps.zip HERE, when the job starts,
    not when it was submitted -- that is what lets a continuation be queued with
    --dependency before its parent has written the checkpoint it continues from.
    A continuation also gets a fresh output_dir, so the curriculum would restart at
    goal_dist_start (2 m) and re-climb ground the parent already earned; the parent
    wrote where it actually got to, so resume from there. 2026-09-23.
    """
    from pathlib import Path as _P
    if warmstart is None:
        return warmstart, goal_dist_start
    _d = _P(str(warmstart))
    if not _d.is_dir():
        return str(warmstart), goal_dist_start
    _cands = sorted(_d.glob("ppo_*_steps.zip"), key=lambda q: int(q.stem.split("_")[1]))
    if not _cands:
        raise SystemExit(f"REFUSED: --warmstart {_d} is a directory with no ppo_*_steps.zip in it")
    ckpt = str(_cands[-1])
    log(f"[warmstart] {_d} -> newest checkpoint {_P(ckpt).name} "
        f"(of {len(_cands)}: {', '.join(q.stem.split('_')[1] for q in _cands[-4:])})")
    _cs = _d.parent / "curriculum_state.json"
    if _cs.is_file() and goal_dist_start is not None:
        try:
            import json as _j
            _prev = float(_j.loads(_cs.read_text())["goal_dist"])
            if _prev > float(goal_dist_start):
                log(f"[warmstart] curriculum resumes at {_prev:.2f} m (parent's last value) "
                    f"instead of {float(goal_dist_start):.2f}")
                goal_dist_start = _prev
        except Exception as _e:
            log(f"[warmstart] could not read {_cs} ({_e}); curriculum starts at {goal_dist_start}")
    return ckpt, goal_dist_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--total_steps", type=int, default=10_000)
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--spawn_max_frame", type=int, default=None,
                    help="cap spawn range; 3 = full traverses from the trail start (rung 5)")
    ap.add_argument("--goal_frame", type=int, default=30,
                    help="goal = real-trajectory position at this frame (~30 => ~3-4 real m)")
    ap.add_argument("--goal_frame_range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="rung 6: sample the goal frame per episode from [LO, HI]")
    ap.add_argument("--obs_cache", default=None,
                    help="ribbon-cache root (outputs/ribbon_cache); enables cached diffused observations")
    ap.add_argument("--live", action="store_true",
                    help="live per-action diffusion observations (no cache): one "
                         "5-frame generation per env step at the exact pose")
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm07b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors",
                    help="semantic diffusion checkpoint the live backend serves")
    ap.add_argument("--live_frames", type=int, default=5,
                    help="frames per live call (4 history + current; 4k+1)")
    ap.add_argument("--warmstart", type=Path, default=None,
                    help="PPO checkpoint .zip to continue training from "
                         "(num_timesteps preserved; total_steps counts the NEW steps)")
    ap.add_argument("--void_terminate_frac", type=float, default=0.0,
                    help="end the episode when this fraction of the footprint "
                         "has no gaussian support (alpha-gated void) — stops "
                         "the policy exploiting the world model's blind spots")
    ap.add_argument("--image_void_terminate_frac", type=float, default=0.0,
                    help="end the episode when this fraction of the WHOLE "
                         "IMAGE has no gaussian support (= 1 - alpha coverage "
                         "with the gate on). Her measure: unsupported ground "
                         "is fine while the view stays paintable")
    ap.add_argument("--void_terminate_penalty", type=float, default=100.0,
                    help="cost applied on void termination (RW5 scale: a real "
                         "crash is 1000)")
    ap.add_argument("--timeout_distance_scaled", action="store_true",
                    help="scale the timeout penalty by remaining/initial "
                         "distance, so stopping close is cheap and stopping "
                         "far is not — makes 'approach as close as legally "
                         "possible and hold' the optimum without defining a "
                         "boundary")
    ap.add_argument("--coherence_cost_weight", type=float, default=0.0,
                    help="graded cost w*max(0, tau - coverage). Coverage is "
                         "mean alpha: how much of the render has real geometry "
                         "behind it. 0 = off")
    ap.add_argument("--coherence_tau", type=float, default=0.4,
                    help="read off the coverage ladder by eye, 2026-09-01")
    ap.add_argument("--coherence_terminate_tau", type=float, default=0.0,
                    help="end the episode below this coverage — frames with no "
                         "geometry at all. 0 = off")
    ap.add_argument("--coherence_terminate_penalty", type=float, default=100.0)
    ap.add_argument("--alpha_gate_tau", type=float, default=0.5,
                    help="alpha below this is relabelled void in the REWARD's "
                         "label map (never in the observation). Measured "
                         "2026-09-01: all phantom crashes gone by 0.1")
    ap.add_argument("--raster_obs", action="store_true",
                    help="ABLATION: the policy sees the reconstruction's raster and the "
                         "reward reads the splat labels (no diffusion call). Observation "
                         "and map then agree on where the edge is. Not deployable.")
    ap.add_argument("--no_alpha_gate", action="store_true",
                    help="UNGATED reward: trust diffused labels in invented regions too "
                         "(coherence-justified; the gated run is the safety-anchored twin)")
    ap.add_argument("--phantom_veto", action="store_true",
                    help="2026-09-19: a crash-level footprint reading on ground the fused map knows to be "
                         "walkable does not end the episode (the generated no-go label is a phantom); it "
                         "still pays the per-step contact cost. Needs the map (clouds_dir).")
    ap.add_argument("--collision_terminate_frac", type=float, default=0.0,
                    help="footprint overlap that ENDS the episode (0 = off). "
                         "Without it a whole episode of walking through a tree "
                         "costs ~1%% of the goal bonus, so crashing is optimal.")
    ap.add_argument("--collision_terminate_penalty", type=float, default=20.0)
    ap.add_argument("--backward_cost", type=float, default=0.0,
                    help="penalty * max(0,-v); tax the camera-blind backing gait")
    ap.add_argument("--footprint_along_motion", action="store_true",
                    help="score the footprint along the commanded motion "
                         "direction; motion onto unseen (rear) ground prices "
                         "as worst-case terrain instead of scoring free")
    ap.add_argument("--forward_only", action="store_true",
                    help="clamp negative velocity to 0 (stand-and-turn): kills "
                         "the backward-exploit class by construction")
    ap.add_argument("--spin_cost", type=float, default=0.05,
                    help="penalty * |yaw action| per step; raise to break the "
                         "bang-bang full-lock turning habit")
    ap.add_argument("--action_smooth_cost", type=float, default=0.0,
                    help="penalty * mean|a_t - a_{t-1}|; charges action "
                         "CHANGES (flip-flops), not turning itself")
    ap.add_argument("--frozen_probe", action="store_true",
                    help="before training, run zero-action steps and print the "
                         "per-component reward budget — the MEASURED cost of "
                         "standing still, which every freezing argument needs")
    ap.add_argument("--ent_coef", type=float, default=0.01,
                    help="PPO entropy bonus. 0.01 is what every run to date "
                         "used (it was hardcoded, not the SB3 default of 0). "
                         "Raise for a from-scratch arm that stops exploring "
                         "before it ever reaches a goal.")
    ap.add_argument("--goal_support_min_frac", type=float, default=0.25,
                    help="a goal needs this fraction of the ground-point "
                         "density found on the recorded path. 0 = accept any "
                         "single point (too weak).")
    ap.add_argument("--goal_support_radius", type=float, default=0.0,
                    help="metres; >0 rejects sampled goals with no ground "
                         "points within this radius and draws again. 14.5%% of "
                         "goals were measured off-cloud (goal_audit.py).")
    ap.add_argument("--collision_at_next_pose", action="store_true",
                    help="crash judged on the BODY footprint at the pose the action leads to (map)")
    ap.add_argument("--footprint_next_heading", action="store_true",
                    help="orient the visible box by the heading AFTER the commanded turn")
    ap.add_argument("--crash_requires_motion", action="store_true",
                    help="the box can only crash while the robot moves (throttle >= 0.15)")
    ap.add_argument("--look_ahead_auto", action="store_true",
                    help="graded box at each scene's nearest visible ground (camera height and focal length)")
    ap.add_argument("--collision_box_memory_agg", type=str, default="newest", choices=("newest", "mean"),
                    help="how the near box is read from the frame memory: the newest frame containing it, "
                         "or the MEAN over every stored frame containing it (2026-09-16, Joana)")
    ap.add_argument("--collision_box_memory_always", action="store_true",
                    help="average the current frame's box reading with the stored frames even when the box is visible now (2026-09-20)")
    ap.add_argument("--goal_turn_deg", type=float, default=0.0, help="corner goals: minimum bend of the walk between spawn and goal frame (0 = off)")
    ap.add_argument("--goal_turn_mix", type=float, default=0.0, help="share of episodes that draw a corner goal when one exists in the band")
    ap.add_argument("--goal_turn_from", type=float, default=None,
                    help="ramp the corner mix with the distance curriculum: 0 below this band "
                         "distance, rising to --goal_turn_mix at the ceiling (needs --goal_dist_start)")
    ap.add_argument("--goal_turn_beyond_m", type=float, default=2.0, help="corner goals: metres of walk past the first bend the goal must lie (0 = off)")
    ap.add_argument("--collision_box_memory", type=int, default=0,
                    help="read the near box from the newest of the last N generated frames that contains it (0 = off)")
    ap.add_argument("--collision_look_ahead", type=float, default=0.0,
                    help="metres; 0 = judge collision on the same 1.5 m box as "
                         "the graded semantic score (behaviour before "
                         "2026-09-02). ~1.0 moves only the lethal test in to "
                         "the body while shaping keeps its warning distance. "
                         "Below ~0.95 the box enters the camera blind zone.")
    ap.add_argument("--goal_bonus", type=float, default=50.0)
    ap.add_argument("--goal_radius", type=float, default=0.75)
    ap.add_argument("--goal_radius_start", type=float, default=None,
                    help="curriculum: capture radius anneals from this down "
                         "to --goal_radius over --curriculum_steps")
    ap.add_argument("--curriculum_steps", type=int, default=100_000)
    ap.add_argument("--curriculum_mode", default="success",
                    choices=["success", "time"],
                    help="success: radius shrinks only when the policy earns "
                         "it (>=50%% recent wins); time: linear anneal")
    ap.add_argument("--goal_dist", type=float, default=None,
                    help="fixed spawn->goal distance in meters (advisor spec)")
    ap.add_argument("--encoder", default="nature",
                    choices=["nature", "dinov2", "dinov2b", "resnet18", "both"],
                    help="policy visual encoder: SB3 NatureCNN (scratch) or "
                         "a frozen pretrained backbone (advisor ablation)")
    ap.add_argument("--max_steps_base", type=int, default=40,
                    help="episode budget = base + per_m * (top of the goal band)")
    ap.add_argument("--max_steps_per_m", type=float, default=0.0,
                    help="0 pins max_steps; 14 keeps a 1.15x margin over the measured cruise (2026-09-23)")
    ap.add_argument("--obs_frame_stack", type=int, default=1,
                    help="MULTIPLE IMAGES: the observation is the last K rendered views on the channel axis (1 = current frame only)")
    ap.add_argument("--obs_frame_stride", type=int, default=1,
                    help="steps between stacked views: stack 3 stride 3 spans 1.5 m instead of 0.5 m")
    ap.add_argument("--image_norm_fix", action="store_true",
                    help="build/patch the policy with normalize_images=False: SB3 divided the image by 255 "
                         "before the extractor divided it again (all DINO arms before 2026-09-22 were blind)")
    ap.add_argument("--reset_image_head", action="store_true",
                    help="with --warmstart: zero the image head (Linear+ReLU) so the policy starts exactly as "
                         "the warm-start behaves and grows the image pathway from zero")
    ap.add_argument("--encoder_grid", type=str, default="",
                    help='region grid for the frozen ViT patch tokens, e.g. "6x8" (default 3x4)')
    ap.add_argument("--rgb_align_cost", type=float, default=0.0,
                    help="per-step cost when the generated RGB drifts from the Gaussian raster where alpha is high")
    ap.add_argument("--rgb_align_tau", type=float, default=0.8,
                    help="similarity (1 - mean|rgb-raster|/255 over covered pixels) below which the cost applies")
    ap.add_argument("--disagree_frac", type=float, default=None,
                    help="footprint fraction that counts as a phantom/missed disagreement (default: COLLTERM)")
    ap.add_argument("--disagree_cost", type=float, default=0.0,
                    help="per-step cost when the generated labels and the fused map disagree under the footprint")
    ap.add_argument("--cur_threshold", type=float, default=0.5,
                    help="distance curriculum: recent success needed to earn a notch (default 0.5)")
    ap.add_argument("--cur_window", type=int, default=100,
                    help="distance curriculum: episodes the success is averaged over")
    ap.add_argument("--cur_min_episodes", type=int, default=None,
                    help="distance curriculum: episodes that must be in the window before the gate "
                         "may pass; default window//2 (the old, fast behaviour). Set = window to "
                         "require a full window.")
    ap.add_argument("--cur_per_scene", action="store_true",
                    help="distance curriculum: earn a notch only when the WORST scene with enough "
                         "recent episodes clears the threshold, not the average over scenes")
    ap.add_argument("--cur_notch", type=float, default=0.5,
                    help="distance curriculum: metres the band grows per earned notch")
    ap.add_argument("--goal_dist_start", type=float, default=None,
                    help="distance curriculum: goals start here and grow to "
                         "--goal_dist as the policy earns wins")
    ap.add_argument("--obs_height", type=int, default=336,
                    help="render/observation height (multiple of 112)")
    ap.add_argument("--obs_width", type=int, default=560,
                    help="render/observation width (multiple of 112)")
    ap.add_argument("--render_height", type=int, default=None,
                    help="render-high/observe-small: diffusion renders at this "
                         "height (multiple of 112) and obs['rgb'] is downsized "
                         "to --obs_height. Reward/labels stay at render size.")
    ap.add_argument("--render_width", type=int, default=None,
                    help="see --render_height")
    ap.add_argument("--live_steps", type=int, default=4,
                    help="diffusion sampler steps for live generation")
    ap.add_argument("--reward_scale", type=float, default=1.0,
                    help="uniform reward multiplier (e.g. 0.01 tames the "
                         "critic under +-1000 terminals; ratios preserved)")
    ap.add_argument("--spawn_min", type=int, default=0,
                    help="min spawn frame: keep out of the weak-recon clip "
                         "edges where live generation confabulates")
    ap.add_argument("--scene_rotate", type=int, default=0,
                    help="live multi-scene: rotate the resident world every "
                         "N robot-steps (0 = never; needs --scenes)")
    ap.add_argument("--goal_weight", type=float, default=1.5,
                    help="progress-to-goal shaping weight")
    ap.add_argument("--goal_dir_360", action="store_true",
                    help="J-spec: goals at random 360-degree bearing and "
                         "random distance (--goal_dist_range) from spawn; may "
                         "land on non-traversable ground BY DESIGN")
    ap.add_argument("--halt_terminate_steps", type=int, default=0,
                    help="END the episode when the policy has stopped (throttle "
                         "below --halt_throttle_eps) for this many consecutive "
                         "steps, is somewhere safe, and has closed some "
                         "distance. Pays the same distance-scaled timeout it "
                         "would have received anyway. 0 = off.")
    ap.add_argument("--halt_throttle_eps", type=float, default=0.05)
    ap.add_argument("--n_steps", type=int, default=128,
                    help="PPO rollout length per robot (update batch = n_steps x robots)")
    ap.add_argument("--ckpt_every_calls", type=int, default=2000,
                    help="checkpoint every N env calls (x n_envs = steps); 2000 x 4 = 8000 steps")
    ap.add_argument("--reward_source", default="generated",
                    choices=("generated", "map", "map_then_generated", "split", "either"),
                    help="labels the reward reads: the generated image, or the scene cloud map")
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
                    help="cone of the map-direct lawn draw (0 = the goal cone). Lawn lies BESIDE "
                         "the walk: the 50-degree cone held no lawn cell on most frames (2026-09-06)")
    ap.add_argument("--goal_nontrav_tries", type=int, default=0,
                    help="tries of the map-direct lawn draw (0 = goal_mix_tries, 12). "
                         "12 delivered 15%% lawn goals instead of 25%% (2026-09-06)")
    ap.add_argument("--goal_nontrav_edge_m", type=float, default=0.0,
                    help="map-direct lawn goals must have walkable ground within this distance (0 = anywhere on the lawn)")
    ap.add_argument("--spawn_support_tries", type=int, default=0,
                    help="redraw a spawn whose crash box is already at the crash threshold on the map, up to N times (0 = off)")
    ap.add_argument("--goal_traversable_mix", type=float, default=0.0,
                    help="P(goal on traversable ground by the map); 0 = the sampler's natural mix")
    ap.add_argument("--map_walk_halfwidth_m", type=float, default=0.4)
    ap.add_argument("--map_ignore_classes", default="")
    ap.add_argument("--mirror_prob", type=float, default=0.0,
                    help="probability an episode is presented MIRRORED (image flipped, goal lateral+bearing negated, yaw command negated): balances the 3:1 right-turn bias of the training scenes")
    ap.add_argument("--goal_center_clear_m", type=float, default=0.0,
                    help="reject goals with any non-walkable map cell within this distance of the centre (0 = off)")
    ap.add_argument("--label_remap", type=str, default="",
                    help='remap GENERATED label ids before the reward reads them, e.g. "12:0" (phantom person -> void; 2026-09-07)')
    ap.add_argument("--terrain_speed_scaled", action="store_true",
                    help="terrain cost x |throttle|: driving onto bad ground costs, facing it does not")
    ap.add_argument("--halt_gate", action="store_true",
                    help="with --halt_scale_start: halting is unavailable until recent wins pass the threshold, then priced from start")
    ap.add_argument("--halt_scale_start", type=float, default=None,
                    help="halt-price curriculum: start scale (1.0 = full timeout), notches to --halt_penalty_scale on wins")
    ap.add_argument("--refusal_bonus", type=float, default=0.0,
                    help="reward for HALTING on a non-traversable goal within --refusal_dist_m of it (0 = off)")
    ap.add_argument("--refusal_dist_m", type=float, default=2.0)
    ap.add_argument("--refusal_verge_m", type=float, default=1.5)
    ap.add_argument("--verge_start", type=float, default=None,
                    help="verge-radius curriculum: start radius (m), notches by 0.25 to --refusal_verge_m as refusals at the verge pass 50%")
    ap.add_argument("--refusal_dwell_start", type=int, default=None,
                    help="dwell curriculum: still steps required inside the verge zone at the start (0 = arrival refuses); rises to --refusal_dwell_end")
    ap.add_argument("--refusal_dwell_end", type=int, default=3)
    ap.add_argument("--stop_bias", type=float, default=-2.0,
                    help="initial bias of the stop output (mean of its Gaussian); -2 => ~2%% stops per step at unit noise")
    ap.add_argument("--stop_action", action="store_true",
                    help="third action output: > 0 halts now (cold start only, changes the action space)")
    ap.add_argument("--lawn_progress_to_verge", action="store_true",
                    help="on lawn-goal episodes the progress term measures distance to the verge, not the goal")
    ap.add_argument("--goal_requires_stop", action="store_true",
                    help="a goal counts as reached only after the robot is still for the halt steps inside its radius")
    ap.add_argument("--nontrav_goal_unreachable", action="store_true",
                    help="entering the radius of a non-traversable goal ends nothing and pays nothing; only a halt can earn there")
    ap.add_argument("--halt_wrong_ramp", type=int, default=0,
                    help="ramp the wrong-halt penalty from 0 to --halt_wrong_penalty over this many env steps (0 = full from the start)")
    ap.add_argument("--halt_wrong_penalty", type=float, default=0.0,
                    help="flat penalty for HALTING on a traversable goal (mirror of --refusal_bonus)")
    ap.add_argument("--halt_penalty_scale", type=float, default=1.0,
                    help="HALTED pays timeout x this (1.0 = same as a timeout)")
    ap.add_argument("--goal_dist_window", type=float, default=None,
                    help="sliding-window distance curriculum: keep the goal "
                         "range this wide instead of pinning its near end. "
                         "Unset (default) = near end stays where it started, "
                         "so 2 m goals persist to the end of training.")
    ap.add_argument("--goal_dist_range", default=None,
                    help="'lo,hi' meters for --goal_dir_360 (e.g. 5,10)")
    ap.add_argument("--goal_cone_deg", type=float, default=360.0,
                    help="constrain --goal_dir_360 bearings to +-this/2 of "
                         "the path tangent (single-pass capture only renders "
                         "a forward cone; 360 once pano scenes exist)")
    ap.add_argument("--spawn_classes", default=None,
                    help="comma class ids; only spawn on frames whose ground "
                         "patch is one of these (e.g. 6,8 = sidewalk/pavement)")
    ap.add_argument("--spawn_yaw_jitter", type=float, default=0.0,
                    help="rotate each spawn heading by U(-x,+x) degrees — "
                         "anti-memorization: stop-at-grass, not stop-at-"
                         "deviation (her J-v2 spec)")
    ap.add_argument("--spawn_lat_jitter", type=float, default=0.0,
                    help="slide each spawn laterally by U(-x,+x) meters")
    ap.add_argument("--spawn_frames", type=str, default="",
                    help='spawn only at these frames per scene: "scene:10,20,30;scene2:20,40"')
    ap.add_argument("--goal_case_mix", type=str, default="",
                    help='target goal-case mix, e.g. "corner:0.5,narrow:0.25,open:0.25" (empty = label only)')
    ap.add_argument("--goal_case_tries", type=int, default=24)
    ap.add_argument("--wandb_project", type=str, default="nav-rl",
                    help="wandb project; new-scope arms go to their own project so the graphs stay readable")
    ap.add_argument("--spawn_heading_from_walk", action="store_true",
                    help="spawn facing the walk's direction of travel instead of the recorded camera yaw")
    ap.add_argument("--static_scene", action="store_true",
                    help="reconstruct as a STATIC scene: every source frame's Gaussians render from any pose")
    ap.add_argument("--render_window", type=int, default=0,
                    help="fusion window: rasterize from the w recorded frames nearest the "
                         "query (0 = whole scene; 21 = the static/dynamic middle ground)")
    ap.add_argument("--coverage_window", type=int, default=0,
                    help="coverage statistic for the coherence gate from the w nearest "
                         "frames' Gaussians (0 = full-scene alpha, which saturates in static mode)")
    ap.add_argument("--static_movers", type=str, default="",
                    help='with --static_scene: classes kept per-frame, e.g. "12,13" (person, vehicle); must match the label head (v31+)')
    ap.add_argument("--sem_palette", type=int, default=1,
                    help="v14 palette version for the live semantic decode — "
                         "must match --live_ckpt's training palette "
                         "(v21=1, v24/v25 line=4)")
    ap.add_argument("--goal_xy", default=None,
                    help="'x,y': FIX the goal at this nav-frame point every "
                         "episode — obstacle-encounter training (2026-08-31, "
                         "Jing's non-traversable-areas directive): goals "
                         "placed behind an obstacle force avoidance learning")
    ap.add_argument("--goal_noise_std", type=float, default=0.0,
                    help="std (m) of Gaussian noise on the goal-vector obs xy "
                         "each step — the anti-odometry lever; reward and "
                         "termination still use the true goal")
    ap.add_argument("--semantic_weight", type=float, default=1.0,
                    help="terrain/semantic footprint weight (0 disables)")
    ap.add_argument("--timeout_penalty", type=float, default=0.0,
                    help="subtracted at max_steps without goal — makes "
                         "freezing strictly worse than trying")
    ap.add_argument("--proximity_delta", action="store_true",
                    help="potential-shaped proximity: weight*(d_prev - d_now) "
                         "inside the margin; approaching costs, retreating "
                         "refunds, standing is free")
    ap.add_argument("--proximity_weight", type=float, default=0.0,
                    help="cost/step for being within proximity_margin of a "
                         "GEOMETRIC obstacle (scene cloud) — undreamable, "
                         "unlike the semantic footprint. 0 = off")
    ap.add_argument("--proximity_margin", type=float, default=1.0)
    ap.add_argument("--proximity_ground_weight", type=float, default=0.0,
                    help="potential-shaped cost per metre of approach toward the map's GRASS points "
                         "inside proximity_ground_margin (0 = off). Needs --clouds_dir.")
    ap.add_argument("--proximity_ground_margin", type=float, default=1.2)
    ap.add_argument("--proximity_ground_classes", type=str, default="3,4,5")
    ap.add_argument("--goal_lat_jitter_m", type=float, default=0.0,
                    help="2026-09-17: move path goals sideways by up to this (m), keeping only draws whose "
                         "goal disc is walkable on the map (>= --goal_lat_jitter_min_share)")
    ap.add_argument("--goal_lat_jitter_min_share", type=float, default=0.6)
    ap.add_argument("--goal_min_score", type=float, default=0.0,
                    help="goals only where the map's traversability score is >= this (0 = crash threshold); "
                         "0.5 keeps goals off mulch beds (trail 0.35) and on pavement classes")
    ap.add_argument("--clouds_dir", default=None,
                    help="dir with <scene>_cloud.npz from dump_scene_cloud.py "
                         "(required for --proximity_weight > 0)")
    ap.add_argument("--action_chunk", type=int, default=1,
                    help="policy outputs k action pairs per decision and only "
                         "re-observes after all k execute (trajectory arm; "
                         "1 = per-action)")
    ap.add_argument("--trav_path", default=None,
                    help="traversability yaml override (config/traversability_v14.yaml for cached runs)")
    ap.add_argument("--goal_min_sep", type=float, default=1.0,
                    help="spawn-goal separation guard in meters (v6 used 1.5)")
    ap.add_argument("--target_kl", type=float, default=0.02,
                    help="PPO KL trust region; 0 disables (rung 7b)")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="rung 7: train across these scenes (overrides --scene)")
    ap.add_argument("--live_batch", type=int, default=1,
                    help="N robots sharing one pipe via batched generation "
                         "(live mode only; 1 = the classic single-robot path)")
    ap.add_argument("--live_gpus", type=int, default=1,
                    help="live mode: K worker processes, one GPU + one pipe each, "
                         "each running --live_batch robots; PPO sees K x live_batch "
                         "envs. With exactly K --scenes, worker i trains on scene i. "
                         "Submit with --gres=gpu:K. 1 = today's single-GPU path")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/ppo_real"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_label", default="",
                    help="short human-readable name for wandb. The output DIR "
                         "name is a 20-token path that still says trail00, "
                         "UNGATED and v21obs on runs that are none of those "
                         "things -- unreadable in a run list and sometimes "
                         "wrong. Paths keep the long name so existing globs "
                         "still work; only the display name changes.")
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_images_every", type=int, default=10,
                    help="log an obs | generated-labels panel for the first env every N "
                         "rollouts (0 = off). Images to wandb, Joana 2026-09-13")
    ap.add_argument("--bc_demos", type=Path, default=None,
                    help="npz from make_demo_dataset.py; if set, behavior-clone the "
                         "policy on real-trajectory demonstrations before PPO")
    ap.add_argument("--bc_epochs", type=int, default=25)
    ap.add_argument("--bc_holdout", type=float, default=0.2,
                    help="share of demo EPISODES held out of BC to report stop recall / false-stop")
    ap.add_argument("--parse_only", action="store_true",
                    help="exit 0 right after argument parsing (preflight on the login node)")
    args = ap.parse_args()
    if args.parse_only:
        print("PARSE_OK"); return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dump the env config HERE, not inside a builder. It lived in make_env(),
    # which the LIVE batched path never calls -- so every live arm ever run has
    # been missing env_config.json, and every eval of one silently fell back to
    # CLI defaults: crash termination OFF (frac 0), coherence termination OFF,
    # goal support OFF. That is why five evals on 2026-09-03 reported ZERO
    # crashes against a 72% crash rate in training. One call, on the one path
    # every run takes.
    _dump_env_config(args, _scene_env_cfg(args))

    # ONE env: each env holds a reconstructed scene on the GPU. Parallel envs
    # would multiply VRAM; not worth it for the smoke.
    if getattr(args, "live", False) and getattr(args, "live_gpus", 1) > 1:
        env = make_multi_gpu_live_env(args)
    elif getattr(args, "live", False) and getattr(args, "live_batch", 1) > 1:
        env = make_live_vec_env(args)
        if args.bc_demos:
            print("[live_batch] BC pretrain on the policy before batched training (2026-09-06)", flush=True)
    else:
        env = make_env(args)

    _expand = False
    args.warmstart, args.goal_dist_start = resolve_warmstart(
        args.warmstart, args.goal_dist_start, log=lambda m: print(m, flush=True))
    if args.warmstart is not None and getattr(args, "stop_action", False):
        try:
            _probe = PPO.load(str(args.warmstart), device="cpu")
            _expand = int(_probe.action_space.shape[0]) == int(env.action_space.shape[0]) - 1
            del _probe
        except Exception as _e:
            print(f"[train_ppo_real] could not probe {args.warmstart}: {_e}", flush=True)
    if args.warmstart is not None and not _expand:
        # Continue training an existing policy (e.g. the cache-trained champion
        # fine-tuning on live observations). SB3 keeps num_timesteps; learn()
        # below adds total_steps on top (reset_num_timesteps=False).
        model = PPO.load(str(args.warmstart), env=env,
                         tensorboard_log=str(args.output_dir / "tensorboard"))
        # The checkpoint carries its own target_kl — but a warm-start into a NEW
        # setting (new scenes / new obs source) shifts the policy hard on the
        # first updates, and the old leash aborts every round at step 0 (Run C
        # 2026-08-23: 300k steps, zero learning). Honor the CLI value instead.
        model.target_kl = (args.target_kl if args.target_kl > 0 else None)
        _parent_normalize = bool(getattr(model.policy, "normalize_images", True))
        if getattr(args, "image_norm_fix", False):
            # the checkpoint's pickled policy_kwargs carry normalize_images=True: patch the live
            # policy AND the kwargs that PPO.save writes, so evals of the new checkpoints match
            model.policy.normalize_images = False
            model.policy_kwargs = dict(model.policy_kwargs or {}); model.policy_kwargs["normalize_images"] = False
            print("[image check] warm-start patched: normalize_images=False (single /255 in the extractor)", flush=True)
        # Reset the image head only when the PARENT never used its camera. A pre-fix checkpoint was
        # trained with the double /255, so its head must start from zero; a post-fix checkpoint has
        # already learned to see and zeroing it would throw that away (2026-09-22, continuations).
        _parent_blind = bool(_parent_normalize)
        _do_reset = bool(getattr(args, "reset_image_head", False)) or (
            bool(getattr(args, "image_norm_fix", False)) and _parent_blind)
        print(f"[image check] parent normalize_images={_parent_normalize} -> image head reset: {_do_reset}"
              + ("" if _do_reset else "  (parent already sees; keeping its image pathway)"), flush=True)
        if _do_reset:
            # 2026-09-23: the guard below and env_config both read args.reset_image_head,
            # so an AUTOMATIC reset has to write itself back. Without this, two warm arms
            # (500181, 500182) had their head correctly zeroed and were then refused by
            # their own guard for not reacting to the image -- which a zeroed head cannot
            # do by construction -- and env_config recorded reset_image_head=false.
            args.reset_image_head = True
            import torch                                   # only imported lazily elsewhere in this script
            _n = 0
            for _ext in {id(e): e for e in (getattr(model.policy, "pi_features_extractor", None),
                                             getattr(model.policy, "vf_features_extractor", None),
                                             getattr(model.policy, "features_extractor", None)) if e is not None}.values():
                _head = getattr(_ext, "head", None)
                if _head is not None:
                    with torch.no_grad():
                        _head[0].weight.zero_(); _head[0].bias.zero_()
                    _n += 1
            print(f"[image check] image head zeroed on {_n} extractor(s): step 0 behaves like the warm-start, "
                  f"the image pathway grows from zero", flush=True)
        print(f"[train_ppo_real] warm-start from {args.warmstart} "
              f"(num_timesteps={model.num_timesteps}, target_kl={model.target_kl})")
    else:
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
        if getattr(args, "encoder", "nature") != "nature":
            from src.policy.encoders import FrozenBackboneExtractor
            _fk = dict(backbone=args.encoder)
            if getattr(args, "encoder_grid", ""):
                _fk["grid"] = tuple(int(v) for v in str(args.encoder_grid).lower().split("x"))
            policy_kwargs.update(
                features_extractor_class=FrozenBackboneExtractor,
                features_extractor_kwargs=_fk)
        if getattr(args, "image_norm_fix", False):
            policy_kwargs["normalize_images"] = False
        model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs=policy_kwargs,
        verbose=1, seed=args.seed,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        # NSTEPS (2026-09-17): rollout length per robot. PPO updates on
        # n_steps x robots env steps, so a 2-GPU arm (8 robots) with 128 updates
        # on 1024 steps while the 4-GPU run (16 robots) updates on 2048. Set
        # n_steps=256 on 2-GPU arms so every run updates on the same 2048.
        n_steps=int(getattr(args, "n_steps", 128)),
        # 2026-09-23: a K-frame stack sends K images per sample through the backbone, so a
        # fixed batch of 64 OOMed both stack-3 arms (500172, 500173) in the policy update.
        # Scale it down by K and keep it a multiple of 8. K=1 is unchanged at 64.
        batch_size=max(8, (64 // max(1, int(getattr(args, "obs_frame_stack", 1)))) // 8 * 8),
        # v6c: constant lr again. v6b's linear decay produced a NEVER-learned
        # policy (0% at 200k/306k/400k, goal-blind) — the random-goal task needs
        # full-rate learning throughout (v6 only found its peak at 306k). The KL
        # trust region stays as the sole collapse protection: it triggered just
        # 4x in v6b (harmless) yet caps the exact mechanism behind 3/3 observed
        # peak-then-collapse runs.
        learning_rate=1e-4,
        # target_kl: 0 disables. Single-scene: 0.02 harmless (4-110 triggers).
        # Multi-scene v7: 0.02 fired on 77% of update rounds (4837x) and
        # strangled learning -> rung 7b runs unleashed.
        target_kl=(args.target_kl if args.target_kl > 0 else None),
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2,
        # ent_coef has been 0.01 since this file was written -- NOT the SB3
        # default of 0.0, as I claimed on 2026-09-02 before reading this far
        # down the constructor. It is now a knob with the same default, so
        # every existing run is unchanged and a from-scratch arm can raise it.
        ent_coef=args.ent_coef,
    )
        if getattr(args, "stop_action", False):
            import torch as th
            if _expand:
                expand_action_head(model, args.warmstart, float(args.stop_bias))
            else:
                # cold start: a fresh head fires the stop ~50% of steps and a
                # random driver then learns 'stop at once' (cheaper than its
                # crashes); start the stop output rare instead
                with th.no_grad():
                    model.policy.action_net.bias[-1] = float(args.stop_bias)
                    model.policy.log_std[-1] = -0.7
                print(f"[train_ppo_real] stop action: cold start, stop bias {float(args.stop_bias):+.1f}", flush=True)

    if args.bc_demos is not None:
        bc_pretrain(model, args.bc_demos, epochs=args.bc_epochs,
                    holdout=float(getattr(args, "bc_holdout", 0.2)))
        model.save(str(args.output_dir / "policy_after_bc.zip"))

    callbacks = [CheckpointCallback(save_freq=int(args.ckpt_every_calls),
                                    save_path=str(args.output_dir / "checkpoints"),
                                    name_prefix="ppo"),
                 RewardComponentsCallback(
                     step_size_m=_scene_env_cfg(args).step_size_m)]
    if args.goal_radius_start is not None:
        callbacks.append(GoalRadiusCurriculum(
            args.goal_radius_start, args.goal_radius, args.curriculum_steps,
            mode=args.curriculum_mode))
    if int(getattr(args, "halt_wrong_ramp", 0) or 0) > 0 and float(getattr(args, "halt_wrong_penalty", 0.0)) > 0.0:
        callbacks.append(HaltWrongRamp(float(args.halt_wrong_penalty), int(args.halt_wrong_ramp)))
    if getattr(args, "refusal_dwell_start", None) is not None:
        callbacks.append(RefusalDwellCurriculum(int(args.refusal_dwell_start), int(args.refusal_dwell_end)))
    if getattr(args, "verge_start", None) is not None:
        callbacks.append(VergeRadiusCurriculum(float(args.verge_start), float(args.refusal_verge_m)))
    if getattr(args, "halt_scale_start", None) is not None:
        callbacks.append(HaltPriceCurriculum(float(args.halt_scale_start), float(args.halt_penalty_scale),
                                             gate=bool(getattr(args, "halt_gate", False))))
    if args.goal_dist_start is not None and args.goal_dist is not None:
        callbacks.append(GoalDistCurriculum(
            args.goal_dist_start, args.goal_dist,
            window=int(getattr(args, "cur_window", 100) or 100),
            threshold=float(getattr(args, "cur_threshold", 0.5) or 0.5),
            notch=float(getattr(args, "cur_notch", 0.5) or 0.5),
            min_episodes=getattr(args, "cur_min_episodes", None),
            per_scene=bool(getattr(args, "cur_per_scene", False)),
            state_path=args.output_dir / "curriculum_state.json",
            turn_from=getattr(args, "goal_turn_from", None),
            turn_mix=float(getattr(args, "goal_turn_mix", 0.0) or 0.0)))
    if args.use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(project=str(getattr(args, "wandb_project", "nav-rl") or "nav-rl"),
                       name=(args.run_label or args.output_dir.name),
                       config=vars(args) | {"total_steps": args.total_steps},
                       sync_tensorboard=True, dir=str(args.output_dir))
            callbacks.append(WandbCallback())
            if int(getattr(args, "wandb_images_every", 0)) > 0:
                callbacks.append(WandbImagePanel(int(args.wandb_images_every),
                                                 sem_palette=int(getattr(args, "sem_palette", 0) or 0)))
        except Exception as e:
            print(f"[train_ppo_real] wandb unavailable ({e}); continuing without")

    print(f"[train_ppo_real] training {args.total_steps} steps on {args.scene} ...")
    _reward_banner(args, _scene_env_cfg(args), env)
    _goal_cone_banner(env)
    if args.frozen_probe:
        _frozen_probe(env)
    # ---- FRAME STACK CHECK (2026-09-22): the env must actually hand the policy K views ----
    _k = int(getattr(args, "obs_frame_stack", 1))
    _ch = int(env.observation_space["rgb"].shape[0] if env.observation_space["rgb"].shape[0] % 3 == 0
              and env.observation_space["rgb"].shape[0] <= 30 else env.observation_space["rgb"].shape[-1])
    if _ch != 3 * _k:
        raise SystemExit(f"REFUSED: --obs_frame_stack {_k} but obs['rgb'] carries {_ch} channels")
    if _k > 1:
        print(f"[frame stack] {_k} views per observation, {_ch} channels", flush=True)
    # ---- IMAGE CHECK (2026-09-22): does the policy react to its image at all? ----
    try:
        from src.policy.encoders import image_sensitivity
        _is = image_sensitivity(model)
        print(f"[image check] normalize_images={_is['normalize_images']}  |action(random image) - action(black)| = {_is['delta']:.4f}"
              f"  ({'uses the image' if _is['delta'] > 1e-3 else 'BLIND: image has no effect'})", flush=True)
        if getattr(args, "image_norm_fix", False) and _is["normalize_images"]:
            raise SystemExit("REFUSED: --image_norm_fix given but the policy still has normalize_images=True")
        if getattr(args, "image_norm_fix", False) and not getattr(args, "reset_image_head", False) and _is["delta"] <= 1e-3:
            raise SystemExit("REFUSED: --image_norm_fix given but the action does not react to the image")
    except SystemExit:
        raise
    except Exception as _e:
        print(f"[image check] skipped: {_e}", flush=True)
    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True,
                reset_num_timesteps=(args.warmstart is None))
    model.save(str(args.output_dir / "ppo_final.zip"))

    print("[train_ppo_real] eval rollout ...")
    if getattr(args, "live_batch", 1) > 1 or getattr(args, "live_gpus", 1) > 1:
        print("[live_batch] end-of-training rollout video skipped (vec env)")
    else:
        save_rollout_video(model, env, args.output_dir / "rollout.mp4")


if __name__ == "__main__":
    main()
