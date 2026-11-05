"""Every running arm on one page: which one is learning the behaviour we want?

Reads the SB3 tables straight out of each arm's slurm .out -- no wandb API, no
GPU, seconds on the login node -- and overlays all arms on a small grid of the
signals that actually answer the question:

  goal_dist          the competence meter. It only advances when the policy
                     wins >= 50% of the last 100 episodes, so it is immune to
                     the reward's shifting scale AND to scene rotation.
  how episodes end   derived from the per-step reward means: goal_bonus,
                     crash and coherence_crash are one-shot terminals, so
                     (mean per step) * ep_len / penalty = fraction of episodes
                     ending that way. The remainder is timeout (or halted).
  ep_len_mean        surviving longer is the first thing that has to happen.
  throttle           creeping is freezing's cousin: ppo_240704 ran at 28%.
  halted             the new terminal -- the first direct metric for
                     "walked to the boundary and stopped".

Arms are named by what DISTINGUISHES them: the ledger label with every token
shared by all arms stripped, so "base" comes out as "s1" and the halt arm as
"halt5-s1" instead of eleven identical 80-character strings.

    python scripts/fleet_dashboard.py 463164 463165 ... --out fleet.png
    python scripts/fleet_dashboard.py --since "2026-09-03 00:00" --out fleet.png
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

LOGDIR = Path("/scratch/m000204-pm07b/joana")
LEDGER = LOGDIR / "launch_ledger.log"

SECTION = re.compile(r"^\|\s+(\w+)/\s+\|\s+\|\s*$")
KV = re.compile(r"^\|\s+(\S+)\s+\|\s+([-+]?[\d.]+(?:e[-+]?\d+)?|nan|inf)\s+\|\s*$")
TABLE_END = re.compile(r"^-{20,}\s*$")


def parse_tables(path: Path) -> dict:
    """{key: [values...]} with one entry per SB3 dump, keys like 'rollout/ep_len_mean'."""
    series: dict[str, list] = {}
    cur: dict[str, float] = {}
    section = ""
    for ln in path.read_text(errors="ignore").splitlines():
        m = SECTION.match(ln)
        if m:
            section = m.group(1) + "/"
            continue
        m = KV.match(ln)
        if m:
            try:
                cur[section + m.group(1)] = float(m.group(2))
            except ValueError:
                pass
            continue
        if TABLE_END.match(ln) and cur:
            # a dump is complete when we hit the closing rule with content
            if "time/total_timesteps" in cur:
                for k, v in cur.items():
                    series.setdefault(k, []).append(v)
                # keys absent from this dump get NaN so lengths stay aligned
                n = len(series["time/total_timesteps"])
                for k in series:
                    if len(series[k]) < n:
                        series[k].append(float("nan"))
            cur = {}
    return series


def ledger_labels() -> dict[int, str]:
    out: dict[int, str] = {}
    if not LEDGER.exists():
        return out
    job = None
    for ln in LEDGER.read_text(errors="ignore").splitlines():
        m = re.match(r"^=== job (\d+)", ln)
        if m:
            job = int(m.group(1))
        m = re.match(r"^\s+label\s+(\S+)", ln)
        if m and job is not None:
            out[job] = m.group(1)
    return out


def short_names(labels: dict[int, str]) -> dict[int, str]:
    """Strip every token shared by ALL arms; what is left is what differs."""
    toks = {j: l.split("-") for j, l in labels.items()}
    if not toks:
        return {}
    common = set.intersection(*(set(t) for t in toks.values()))
    out = {}
    for j, t in toks.items():
        kept = [x for x in t if x not in common]
        out[j] = "-".join(kept) if kept else "base"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", type=int)
    ap.add_argument("--since", default="",
                    help="if no jobs given: every ppo-real log modified after this")
    ap.add_argument("--out", default="fleet.png")
    ap.add_argument("--scenes", default="gnd_AU_180,gnd_AUd210,gnd_AUw210,gnd_AUw360,gnd_AUw330",
                    help="scene names in diag/scene_idx order -- the PRUNED list the "
                         "env actually loaded (the banner prints it), not the one "
                         "requested")
    ap.add_argument("--ceilings", default="36.8,55.8,68.1,86.1,89.8",
                    help="REACHABLE %% per scene from goal_audit (2-8 m, new cone, "
                         "2026-09-03). A goal rate is only readable against this: "
                         "on gnd_AU_180 nearly half the goals sit on grass.")
    ap.add_argument("--at_steps", type=int, default=0,
                    help="also print a table with every arm read at the dump "
                         "nearest this many steps THIS RUN, so arms launched "
                         "hours apart are compared at the same training "
                         "budget rather than at whatever step each is on now")
    ap.add_argument("--recent", type=int, default=6,
                    help="per-scene panels use the last N dumps on each scene")
    args = ap.parse_args()
    scene_names = args.scenes.split(",")
    ceilings = [float(v) for v in args.ceilings.split(",")]

    jobs = list(args.jobs)
    if not jobs:
        cmd = ["find", str(LOGDIR), "-maxdepth", "1", "-name", "slurm-ppo-real-*.out"]
        if args.since:
            cmd += ["-newermt", args.since]
        for f in subprocess.run(cmd, capture_output=True, text=True).stdout.split():
            m = re.search(r"slurm-ppo-real-(\d+)\.out$", f)
            if m:
                jobs.append(int(m.group(1)))
        jobs.sort()
    if not jobs:
        raise SystemExit("no jobs given and none found")

    labels = ledger_labels()
    # Job id FIRST. Two arms launched before the label carried COHTERM had
    # byte-identical labels (463164 and 463170), so their legend entries
    # merged and a colour got misread as the wrong arm (2026-09-03).
    # 2026-09-03: 463224 trains with CRASHPEN=200 and every fraction below
    # divided by 1000, so its crash rate read 12% when it was ~61%. The
    # penalty is per arm; read it off the label token -cp<N>.
    def crash_pen(j) -> float:
        m = re.search(r"-cp(\d+(?:\.\d+)?)", labels.get(j, ""))
        return float(m.group(1)) if m else 1000.0

    names = {j: "%d %s" % (j, n)
             for j, n in short_names({j: labels.get(j, str(j)) for j in jobs}).items()}

    data = {}
    for j in jobs:
        p = LOGDIR / f"slurm-ppo-real-{j}.out"
        if not p.exists():
            print(f"[skip] {j}: no log"); continue
        s = parse_tables(p)
        if "time/total_timesteps" not in s:
            print(f"[skip] {j}: no SB3 tables yet (still loading?)"); continue
        data[j] = s

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    colors = {j: cmap(i % 20) for i, j in enumerate(data)}

    def get(s, k):
        return np.array(s.get(k, [float("nan")] * len(s["time/total_timesteps"])), dtype=float)

    fig, axes = plt.subplots(4, 4, figsize=(22, 17))
    axes = axes.ravel()
    panels = [
        ("curriculum/goal_dist", "goal distance curriculum (m)\nadvances only on >=50% wins", None),
        # how episodes end, ONE outcome per panel (48 lines in one box was
        # unreadable). Derived from per-step reward means x episode length /
        # penalty; the two windows differ, so single dumps can overshoot 1.
        ("END_goal", "episodes ending at the GOAL (fraction)", (0, 1.05)),
        ("END_crash", "episodes ending in a CRASH (fraction)\nper-arm penalty (1000, or 200 on cp200)", (0, 1.05)),
        ("END_stopped", "episodes that STOPPED (fraction)\nneither goal nor crash nor incoherent: timeout or halted", (0, 1.05)),
        ("rollout/ep_len_mean", "episode length (steps)", None),
        ("curriculum/recent_success", "recent success (100-ep window)", (0, 1)),
        ("diag/throttle", "mean commanded throttle\n(ppo_240704 crept at 0.28)", (0, 1)),
        ("HALTED_EP", "episodes ending HALTED (fraction)\nstopped deliberately, short of the goal, on safe ground", None),
        # the GRADED coherence cost, per step -- not the terminal. Nonzero
        # means coverage is below tau_coh (0.4) and the policy is being
        # charged for looking where the reconstruction is thin.
        ("reward/coherence", "incoherence COST per step (unscaled)\n0 = coverage above tau_coh everywhere", None),
        ("diag/coverage", "reconstruction coverage seen by the policy\n(mean alpha; tau_coh 0.4, kill 0.1)", (0, 1)),
        ("reward/semantic", "terrain cost per step (unscaled)\n0 = perfect ground, -5 = all grass", None),
    ]
    for ax, (key, title, ylim) in zip(axes, panels):
        for j, s in data.items():
            x = get(s, "time/total_timesteps")
            x = x - x[0]                       # steps THIS run, warm arms included
            c, nm = colors[j], names.get(j, str(j))
            if key.startswith("END_"):
                L = get(s, "rollout/ep_len_mean")
                goal = get(s, "reward/goal_bonus") * L / 1000.0
                crash = -get(s, "reward/crash") * L / crash_pen(j)
                coh = -get(s, "reward/coherence_crash") * L / 100.0
                y = {"END_goal": goal, "END_crash": crash,
                     "END_stopped": np.clip(1.0 - goal - crash - coh, 0.0, 1.0)}[key]
                ax.plot(x, y, "-o", c=c, lw=1.6, ms=2.5, label=nm)
            elif key == "HALTED_EP":
                # diag/halted is a PER-STEP rate; times episode length it is
                # the fraction of episodes that ended halted. Arms without the
                # terminal log nothing here and are skipped.
                h = get(s, "diag/halted")
                if np.all(np.isnan(h)):
                    continue
                ax.plot(x, h * get(s, "rollout/ep_len_mean"), "-o", c=c, lw=1.6, ms=2.5, label=nm)
            else:
                y = get(s, key)
                if np.all(np.isnan(y)):
                    continue
                # markers, so an arm with a single dump still shows up: the
                # halt arm's first point was invisible as a one-vertex line
                ax.plot(x, y, "-o", c=c, lw=1.6, ms=2.5, label=nm)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env steps this run")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[j], lw=2, label=names.get(j, str(j)))
               for j in data]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    # ---- BY SCENE. Behaviour depends heavily on which scene is resident:
    # gnd_AU_180 has 47.7% of goals on grass and a 36.8% reachability ceiling,
    # gnd_AUw330 has 1% and 89.8%. A fleet-wide goal rate blends them and
    # arms that started an hour apart are on different scenes at the same
    # step. All four envs share ONE scene per rollout (vec_live_env rotates
    # it), so diag/scene_idx is a clean integer per dump and this is valid.
    def by_scene(s, key_fn, j):
        sc = np.rint(get(s, "diag/scene_idx")).astype(int)
        vals = key_fn(s, j)
        out = []
        for k in range(len(scene_names)):
            m = np.where(sc == k)[0][-args.recent:]
            out.append(float(np.nanmean(vals[m])) if len(m) else float("nan"))
        return np.array(out)

    goal_fn = lambda s, j: get(s, "reward/goal_bonus") * get(s, "rollout/ep_len_mean") / 10.0
    crash_fn = lambda s, j: -get(s, "reward/crash") * get(s, "rollout/ep_len_mean") / (crash_pen(j) / 100.0)
    xs = np.arange(len(scene_names))
    narm = max(len(data), 1)
    for ax, fn, title in ((axes[11], goal_fn, "goal rate BY SCENE (%%), last %d dumps\nblack tick = reachability ceiling" % args.recent),
                          (axes[12], crash_fn, "crash rate BY SCENE (%%), last %d dumps" % args.recent)):
        for i, (j, s) in enumerate(data.items()):
            y = by_scene(s, fn, j)
            off = (i - (narm - 1) / 2) * (0.7 / narm)
            ax.plot(xs + off, y, "o", ms=5, c=colors[j], mec="k", mew=0.3,
                    label=names.get(j, str(j)))
        if fn is goal_fn:
            for k, c in enumerate(ceilings[:len(scene_names)]):
                ax.plot([k - 0.4, k + 0.4], [c, c], "-", c="k", lw=1.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace("gnd_", "") for n in scene_names], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    # 12th panel: fraction of episodes ended by the COHERENCE TERMINAL, alone.
    # It is the dotted line in "how episodes end", but Joana asked to see it
    # by itself: on the arms that actually move (463170, 463879) it ends
    # 15-25% of episodes even at tau_kill 0.05.
    ax = axes[13]
    for j, s_ in data.items():
        x = get(s_, "time/total_timesteps"); x = x - x[0]
        y = -get(s_, "reward/coherence_crash") * get(s_, "rollout/ep_len_mean") / 100.0
        ax.plot(x, y, "-o", c=colors[j], lw=1.4, ms=2.5)
    ax.set_title("episodes ended by the COHERENCE TERMINAL (fraction)\n"
                 "coverage < tau_kill; 0.1 on most arms, 0.05 on 463170", fontsize=10)
    ax.set_xlabel("env steps this run"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    # 2026-09-04: the generator's reading of the footprint against the map's,
    # logged live on every arm that has a cloud. phantom = generator says
    # crash (>= collision_terminate_frac) while the map says clear.
    for k, key, title in ((14, "diag/phantom", "PHANTOM rate per step\n(generator says crash, map says clear)"),
                          (15, "diag/trav_agree", "footprint TRAVERSABILITY agreement\n(generator and map both walkable, or both not)")):
        ax = axes[k]
        drawn = False
        for j, s_ in data.items():
            y = get(s_, key)
            if np.all(np.isnan(y)):
                continue
            x = get(s_, "time/total_timesteps"); x = x - x[0]
            ax.plot(x, y, "-o", c=colors[j], lw=1.4, ms=2.5); drawn = True
        ax.set_title(title, fontsize=10); ax.set_xlabel("env steps this run"); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        if not drawn:
            ax.text(0.5, 0.5, "no arm logs this yet\n(arms launched after 2026-09-04 01:30)", ha="center", va="center", fontsize=9, transform=ax.transAxes)
    fig.suptitle(f"fleet of {len(data)} arms — {Path(args.out).stem}", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"==> {args.out}")

    # ---- the latest row per arm, so the picture does not have to be squinted at
    def row(j, s, i):
        """One table row for arm j at dump index i. halt% = halted rate per
        step x ep_len: the fraction of episodes that ENDED by the halt
        terminal, on the same footing as goal% and crash%."""
        L = get(s, "rollout/ep_len_mean")[i]
        gb, cr, ch = (get(s, "reward/goal_bonus")[i], get(s, "reward/crash")[i],
                      get(s, "reward/coherence_crash")[i])
        x = get(s, "time/total_timesteps"); x = x - x[0]
        return (f"{names.get(j, str(j))[:34]:<34} {int(x[i]):>7} "
                f"{get(s, 'curriculum/goal_dist')[i]:>6.1f} "
                f"{get(s, 'curriculum/recent_success')[i]:>5.2f} {L:>6.1f} "
                f"{100 * gb * L / 1000:>6.1f} {100 * -cr * L / crash_pen(j):>7.1f} "
                f"{100 * -ch * L / 100:>7.1f} {100 * get(s, 'diag/halted')[i] * L:>6.1f} "
                f"{get(s, 'diag/throttle')[i]:>5.2f}")

    hdr = (f"{'arm':<34} {'steps':>7} {'gdist':>6} {'succ':>5} {'eplen':>6} "
           f"{'goal%':>6} {'crash%':>7} {'incoh%':>7} {'halt%':>6} {'thr':>5}")
    print("\nLATEST dump per arm (arms are at DIFFERENT steps -- see --at_steps):")
    print(hdr)
    for j, s in data.items():
        print(row(j, s, -1))
    if args.at_steps:
        print(f"\nEVERY arm at the dump nearest {args.at_steps} steps this run:")
        print(hdr)
        for j, s in data.items():
            x = get(s, "time/total_timesteps"); x = x - x[0]
            if x[-1] < args.at_steps * 0.8:
                print(f"{names.get(j, str(j))[:34]:<34}   (only {int(x[-1])} steps so far)")
                continue
            print(row(j, s, int(np.argmin(np.abs(x - args.at_steps)))))


if __name__ == "__main__":
    main()
