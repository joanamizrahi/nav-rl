"""Trend of one or more arms over their whole run, from the SB3 tables in the
.out log: one row per rollout dump (or every Nth), the keys you name. Use it
BEFORE deciding anything from the last dump alone (Joana, 2026-09-05).

    python scripts/arm_trend.py 467990 467993 --every 4 \
        --keys recent_success goal_dist_hi halt_enabled halt_scale \
               goal_traversable halt_wrong halt_correct reach_on_nontrav

Keys are matched by the row name inside any table section (rollout/, diag/,
curriculum/, reward/ ...). Missing keys print as '-'.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

LOG_DIR = Path("/scratch/m000204-pm07b/joana")
ROW = re.compile(r"^\|\s+([A-Za-z0-9_/]+)\s+\|\s+([-+0-9.e]+|nan|inf)\s+\|")


def parse(path: Path) -> list[dict]:
    dumps, cur = [], {}
    for line in path.open(errors="replace"):
        if line.startswith("----"):
            if cur:
                dumps.append(cur)
                cur = {}
            continue
        m = ROW.match(line)
        if m:
            try:
                cur[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    if cur:
        dumps.append(cur)
    return dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", type=int)
    ap.add_argument("--keys", nargs="+", default=["recent_success", "goal_dist_hi", "halt_enabled", "halt_scale",
                                                  "goal_traversable", "halt_wrong", "halt_correct", "reach_on_nontrav",
                                                  "ep_len_mean"])
    ap.add_argument("--every", type=int, default=4, help="print every Nth dump (the last one always)")
    ap.add_argument("--log_dir", default=str(LOG_DIR))
    args = ap.parse_args()
    for j in args.jobs:
        p = Path(args.log_dir) / f"slurm-ppo-real-{j}.out"
        if not p.exists():
            print(f"=== {j}: no log at {p}")
            continue
        dumps = parse(p)
        print(f"=== {j}: {len(dumps)} dumps  ({p.name})")
        head = f"{'dump':>4} {'steps':>7} " + " ".join(f"{k[:14]:>14}" for k in args.keys)
        print(head)
        idx = list(range(0, len(dumps), max(1, args.every)))
        if dumps and (len(dumps) - 1) not in idx:
            idx.append(len(dumps) - 1)
        for i in idx:
            d = dumps[i]
            steps = d.get("total_timesteps", float("nan"))
            cells = []
            for k in args.keys:
                v = d.get(k)
                cells.append(f"{'-':>14}" if v is None else f"{v:>14.3g}")
            print(f"{i:>4} {steps:>7.0f} " + " ".join(cells))
        print()


if __name__ == "__main__":
    main()
