"""What did the running arms ACTUALLY configure? Read every env_config.json.

The launch spec says what was intended; `env_config.json` is what the env was
built with. Those have disagreed before -- `RW=5` silently applied default
weights to six 48-hour arms. Anything audited against the spec instead of
against this file is auditing a wish.

Prints the goal/spawn/scene block for each recent run, side by side, and flags
any key on which the runs disagree (an arm meant to differ in ONE variable that
differs in two is not an ablation).

    python scripts/show_run_configs.py
    python scripts/show_run_configs.py --since "2026-09-02 21:00" --root /scratch/...
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

KEYS = ("scenes", "goal", "spawn", "cone", "max_steps", "step_size",
        "collision", "look_ahead")


def interesting(k: str) -> bool:
    return any(t in k for t in KEYS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/m000204-pm07b/joana")
    ap.add_argument("--since", default="2026-09-02 21:00",
                    help="only configs written after this (find -newermt)")
    args = ap.parse_args()

    out = subprocess.run(
        ["find", args.root, "-name", "env_config.json", "-newermt", args.since],
        capture_output=True, text=True).stdout.split()
    if not out:
        raise SystemExit(f"no env_config.json under {args.root} newer than "
                         f"{args.since} -- widen --since")

    runs = {}
    for f in sorted(out):
        try:
            runs[Path(f).parent.name] = json.load(open(f))
        except Exception as e:
            print(f"[skip] {f}: {e}")

    names = list(runs)
    keys = sorted({k for d in runs.values() for k in d if interesting(k)})
    w = max(len(k) for k in keys) + 2
    print(f"{len(names)} runs under {args.root} since {args.since}\n")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    print()
    # Plain .ljust/.rjust rather than nested f-string format specs: the cluster
    # runs an older Python than the machine these are written on, and nested
    # f-strings are a 3.12 feature (PEP 701).
    print("key".ljust(w) + "".join(("[%d]" % i).rjust(26)
                                   for i in range(len(names))))
    print("-" * (w + 26 * len(names)))
    for k in keys:
        vals = [str(runs[n].get(k, "--")) for n in names]
        flag = "  <-- DIFFERS" if len(set(vals)) > 1 else ""
        print(k.ljust(w) + "".join(v[:24].rjust(26) for v in vals) + flag)


if __name__ == "__main__":
    main()
