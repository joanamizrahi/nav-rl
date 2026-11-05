#!/usr/bin/env python3
"""Reward curves over an arm's WHOLE life, across every warm-started continuation.

Training writes no CSV -- only SB3's stdout tables and wandb -- and a warm start
continues the timestep counter, so one arm's curve is spread over several
slurm-ppo-real-<id>.out files linked by `--warmstart <parent dir>`. This walks each
lineage backwards from its newest job, parses every rollout table, and writes one
CSV with (arm, job, timesteps, ep_rew_mean, ep_len_mean, ...) so the curve can be
plotted on the laptop (2026-09-25, Joana: "training curves of reward mean for each
arm over their whole timesteps").

    python scripts/arm_curves.py 503293 503294 503295 ... --out arm_curves.csv
    python scripts/arm_curves.py --all --out arm_curves.csv      # every lineage tip
"""
import argparse, csv, glob, os, re, sys

LOGS = "/scratch/m000204-pm07b/joana"
OUT = "/scratch/m000204-pm07b/joana/outputs"
# std / entropy_loss / approx_kl / clip_fraction: whether the policy is still
# EXPLORING. A reward flat for 100k steps at a fixed band is either a local optimum
# or a collapsed action std -- and only the second shows up here (2026-09-25,
# Joana: "it was at 10 m for such a long time and never adapted again").
KEYS = ("ep_rew_mean", "ep_len_mean", "goal_dist", "std", "entropy_loss", "approx_kl", "clip_fraction",
        "explained_variance", "value_loss")


def out_file(job):
    f = os.path.join(LOGS, f"slurm-ppo-real-{job}.out")
    return f if os.path.exists(f) else None


def parent_of(job):
    """The job this one warm-started from, read off its own --warmstart argument."""
    f = out_file(job)
    if not f:
        return None
    with open(f, errors="replace") as fh:
        for line in fh:
            m = re.search(r"--warmstart\s+(\S+)", line)
            if m:
                j = re.search(r"_j(\d+)", m.group(1))
                return j.group(1) if j else None
            if "rollout/" in line:          # past the argument echo; no warm start
                break
    return None


def lineage(job):
    """Oldest ... newest."""
    chain, seen = [], set()
    while job and job not in seen:
        chain.append(job); seen.add(job)
        job = parent_of(job)
    return chain[::-1]


def parse_tables(job):
    """Every SB3 logger table in the .out -> list of dicts. A table is a block of
    '|    key    | value |' lines; total_timesteps closes it."""
    f = out_file(job)
    rows, cur = [], {}
    if not f:
        return rows
    with open(f, errors="replace") as fh:
        for line in fh:
            m = re.match(r"\|\s+([A-Za-z_/]+)\s+\|\s+([-+0-9.eE]+|nan)\s+\|", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            try:
                cur[k] = float(v)
            except ValueError:
                continue
            if k == "total_timesteps":
                if "ep_rew_mean" in cur:
                    rows.append(cur)
                cur = {}
    return rows


def arm_name(job):
    d = glob.glob(os.path.join(OUT, f"*_j{job}"))
    if not d:
        return f"j{job}"
    n = os.path.basename(d[0])
    tags = []
    if "chunk5" in n: tags.append("chunk5")
    if "chunk2" in n: tags.append("chunk2")
    if "stack3" in n or "fs3" in n: tags.append("stack3")
    if "bmamean_bm5" in n: tags.append("mem")
    if "nature" in n or "cnn" in n: tags.append("cnn")
    if "-imgfix" in n or "imgfix" in n: tags.append("dino")
    if "coh10" in n: tags.append("coh10")
    return "_".join(tags) or n[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", help="newest job of each lineage")
    ap.add_argument("--all", action="store_true", help="every job that is not somebody's parent")
    ap.add_argument("--out", default="arm_curves.csv")
    a = ap.parse_args()
    jobs = list(a.jobs)
    if a.all:
        every = sorted({re.search(r"-(\d+)\.out$", f).group(1)
                        for f in glob.glob(os.path.join(LOGS, "slurm-ppo-real-*.out"))})
        parents = {parent_of(j) for j in every}
        jobs = [j for j in every if j not in parents]
    if not jobs:
        sys.exit("no jobs given")
    n = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage_tip", "arm", "job", "timesteps"] + list(KEYS))
        for tip in jobs:
            chain = lineage(tip)
            name = arm_name(tip)
            print(f"{tip}  {name:<18} lineage: {' -> '.join(chain)}")
            for job in chain:
                rows = parse_tables(job)
                for r in rows:
                    w.writerow([tip, name, job, int(r["total_timesteps"])] + [r.get(k, "") for k in KEYS])
                n += len(rows)
                print(f"      {job}: {len(rows)} rollout rows"
                      + (f", {int(rows[0]['total_timesteps'])} -> {int(rows[-1]['total_timesteps'])} steps" if rows else ""))
    print(f"\n{n} rows -> {a.out}")


if __name__ == "__main__":
    main()
