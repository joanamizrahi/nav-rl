"""Reward audit: what does each term ACTUALLY contribute, step by step?

The reward has been balanced by argument so far (weights chosen by reasoning,
verified only through end behavior). This makes it empirical: run a policy
checkpoint for N episodes, collect every term the env already reports per step
(semantic/terrain, goal progress, collision, void, step cost, spin cost, goal
bonus), and produce:

  reward_audit.png   top: per-step term traces for the first episodes
                     bottom: mean |contribution| share per term (the "who is
                     actually steering this policy" bar chart)
  reward_audit.json  aggregate numbers per term (mean, share, per-episode)

Read it with two questions:
  1. Ranking: is the incentive ordering what we intended (goal pull dominant,
     terrain meaningful, housekeeping small)?
  2. Dead terms: a term that never moves is either satisfied (good: collision
     ~0 on a competent policy) or toothless (bad: weight too small to matter).

This is also the baseline for the conservative/aggressive knob: the knob will
scale the risk group (semantic deficit + collision + void) vs the goal group,
and this plot is how we see what a given knob value actually does.

Usage (Marlowe): sbatch scripts/slurm/audit_reward.sh  (CKPT=... required)
Expert-trajectory counterpart: validate_reward.py (reward along the real path).
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

TERMS = ["semantic", "goal", "collision", "void", "step", "spin", "goal_bonus"]


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from eval_policy import build_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--goal_frame", type=int, default=30)
    ap.add_argument("--spawn_max_frame", type=int, default=None)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm07b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(build_env(args))
    model = PPO.load(args.checkpoint, env=env, device="cuda")

    episodes = []          # per episode: {term: [per-step values]}
    for ep in range(args.episodes):
        obs, _ = env.reset()
        traces = {t: [] for t in TERMS}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            for t in TERMS:
                traces[t].append(float(info.get(t, 0.0)))
            done = term or trunc
        episodes.append(traces)
        print(f"ep {ep}: {len(traces['goal'])} steps, success={term}", flush=True)

    # Aggregates: mean per-step value and mean |contribution| share per term.
    agg = {}
    all_abs = {t: [] for t in TERMS}
    for traces in episodes:
        for t in TERMS:
            all_abs[t].extend(abs(v) for v in traces[t])
    total_abs = sum(np.sum(v) for v in all_abs.values()) or 1.0
    for t in TERMS:
        vals = [v for traces in episodes for v in traces[t]]
        agg[t] = {
            "mean_per_step": round(float(np.mean(vals)), 4),
            "abs_share": round(float(np.sum(all_abs[t]) / total_abs), 4),
        }
    with open(args.out_dir / "reward_audit.json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "episodes": args.episodes,
                   "terms": agg}, f, indent=2)

    print("\n=== TERM AGGREGATES (share = fraction of total |reward| moved) ===")
    for t in sorted(TERMS, key=lambda t: -agg[t]["abs_share"]):
        print(f"  {t:10s}  mean/step {agg[t]['mean_per_step']:+8.3f}   share {agg[t]['abs_share']:.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # goal_bonus (+50, one step) flattens every other trace to invisibility if
    # plotted together — it gets a title mention, not a line, and the share
    # chart shows the PER-STEP economy (the thing the weights actually tune).
    step_terms = [t for t in TERMS if t != "goal_bonus"]
    n_show = min(3, len(episodes))
    fig, axes = plt.subplots(n_show + 1, 1, figsize=(10, 3 * (n_show + 1)))
    for i in range(n_show):
        ax = axes[i]
        for t in step_terms:
            if any(abs(v) > 1e-6 for v in episodes[i][t]):
                ax.plot(episodes[i][t], label=t, alpha=0.8)
        got_bonus = any(v > 0 for v in episodes[i]["goal_bonus"])
        ax.set_title(f"episode {i}: per-step reward terms"
                     f" (goal_bonus +50 at end: {'yes' if got_bonus else 'NO'})")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=4)
    ax = axes[-1]
    step_total = sum(np.sum(all_abs[t]) for t in step_terms) or 1.0
    shares = [float(np.sum(all_abs[t]) / step_total) for t in step_terms]
    ax.bar(step_terms, shares)
    bonus_share = agg["goal_bonus"]["abs_share"]
    ax.set_title(f"per-step |contribution| share (terminal goal_bonus excluded; "
                 f"it is {bonus_share:.0%} of ALL reward moved)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.out_dir / "reward_audit.png", dpi=120)
    print(f"\nwrote {args.out_dir}/reward_audit.png + reward_audit.json")


if __name__ == "__main__":
    main()
