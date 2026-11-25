"""Startup gate for the 2026-11-24 runs: compare a run's env_config.json with the paper policy's
(configs/reference/env_config_paper_j503294.json) and say whether the run differs from the paper in
exactly the settings its arm is meant to change -- no more, no less.

    python scripts/check_env_config.py <run_dir_or_env_config.json> <arm>

A difference in a key the arm does not expect is printed as UNEXPECTED (the classic silent-knob bug,
e.g. PROX=0 overridden by RW5). An expected key that did NOT change is printed as MISSING."""
import json, sys
from pathlib import Path

REF = Path(__file__).resolve().parents[1] / "configs/reference/env_config_paper_j503294.json"
IGNORE = {"warmstart", "spawn_frames", "live_ckpt"}     # paths and derived lists, not settings
# keys the paper run's config predates (added 2026-09-25, commit 1251b3b): the paper run used the code's
# hard-wired values, which these defaults reproduce (GoalDistCurriculum(window=100, threshold=0.5, notch=0.5),
# min episodes window//2, mean over scenes)
PAPER_DEFAULTS = {"cur_threshold": 0.5, "cur_window": 100, "cur_notch": 0.5, "cur_min_episodes": None,
                  "cur_per_scene": False}
# settings that do nothing while another one is off: a difference there is reported but not a failure
INACTIVE = {
    "goal_turn_from": lambda r: not r.get("goal_turn_deg") and not r.get("goal_turn_mix"),   # corner rule off
    "goal_turn_beyond_m": lambda r: not r.get("goal_turn_deg") and not r.get("goal_turn_mix"),
    "rgb_align_tau": lambda r: not r.get("rgb_align_cost_weight"),                            # RGB-drift cost off
    "disagree_frac": lambda r: not r.get("disagree_cost_weight"),                             # disagreement cost off
}
EXPECT = {                                                # key -> value the arm must produce
    "b1":           {"throttle01": True},
    "b1_s1":        {"throttle01": True},
    "b1_s2":        {"throttle01": True},
    "b2_std05":     {"throttle01": True},                 # LOGSTD / ENT are PPO args: checked in the launch log
    "b1_stack3":    {"throttle01": True, "obs_frame_stack": 3, "obs_frame_stride": 3},
    "b1_chunk5":    {"throttle01": True, "action_chunk": 5},
    "b1_nocov":     {"throttle01": True, "coherence_cost_weight": 0.0, "coherence_terminate_tau": 0.0},
    "b1_onethresh": {"throttle01": True, "coherence_cost_weight": 0.0, "coherence_terminate_tau": 0.4},
    "b1_maplabels": {"throttle01": True, "reward_source": "map"},
    "b1_nomem":     {"throttle01": True, "collision_box_memory": 0},
    "b1_noprox":    {"throttle01": True, "proximity_weight": 0.0, "proximity_ground_weight": 0.0},
    "b1_interp":    {"throttle01": True, "interp_history": True},
    "b1_interp_nocov": {"throttle01": True, "interp_history": True, "coherence_cost_weight": 0.0, "coherence_terminate_tau": 0.0},
}

def main():
    p, arm = Path(sys.argv[1]), sys.argv[2]
    if p.is_dir():
        p = p / "env_config.json"
    run, ref, exp = json.load(open(p)), json.load(open(REF)), EXPECT[arm]
    ok = True
    for k in sorted(set(run) | set(ref) | set(exp)):
        if k in IGNORE:
            continue
        a, b = ref.get(k, "<absent>"), run.get(k, "<absent>")
        if k in exp:
            if b != exp[k]:
                print(f"MISSING     {k}: expected {exp[k]!r}, run has {b!r}"); ok = False
            else:
                print(f"expected    {k}: {a!r} -> {b!r}")
        elif a != b:
            if a == "<absent>" and b in (False, 0, 0.0, None, ""):
                continue                                  # a key added after the paper run, at its off value
            if a == "<absent>" and k in PAPER_DEFAULTS and b == PAPER_DEFAULTS[k]:
                print(f"paper-dflt  {k}: {b!r} (not saved in the paper run; same as the code then)")
                continue
            if k in INACTIVE and INACTIVE[k](run) and INACTIVE[k](ref):
                print(f"inactive    {k}: paper {a!r}, run {b!r} (has no effect: its switch is off in both)")
                continue
            print(f"UNEXPECTED  {k}: paper {a!r}, run {b!r}"); ok = False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
