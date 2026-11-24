"""Pre-launch gate (2026-11-24): run `submit_arm.sh <arm> --dry` for b1 and for each arm, take the
python arguments each would really pass (PRINTARGS), and check that every arm differs from b1 in
exactly the flags it is meant to change -- and that b1 itself carries --throttle01. Catches a knob
that never reaches python (THROTTLE01 sat in the CACHE branch; PROX=0 is overridden by RW5)
BEFORE anything is submitted. Run on Marlowe from the repo root:

    python scripts/preflight_arms.py b1_s1 b1_nocov ...      (no names = every known arm)
"""
import shlex, subprocess, sys

EXPECT = {   # flag -> value the arm must end up with (None = flag must be ABSENT)
    "b1_s1": {"--seed": "1"},
    "b1_s2": {"--seed": "2"},
    "b2_std05": {"--log_std_init": "-0.69", "--ent_coef": "0"},
    "b1_stack3": {"--obs_frame_stack": "3", "--obs_frame_stride": "3"},
    "b1_chunk5": {"--action_chunk": "5"},
    "b1_nocov": {"--coherence_cost_weight": "0", "--coherence_terminate_tau": "0"},
    "b1_onethresh": {"--coherence_cost_weight": "0", "--coherence_terminate_tau": "0.4"},
    "b1_maplabels": {"--reward_source": "map"},
    "b1_nomem": {"--collision_box_memory": None, "--collision_box_memory_agg": None},
    "b1_noprox": {"--proximity_weight": "0", "--proximity_ground_weight": "0"},
    "b1_interp": {"--interp_history": ""},
    "b1_interp_nocov": {"--interp_history": "", "--coherence_cost_weight": "0", "--coherence_terminate_tau": "0"},
}
IGNORE = {"--seed"} - set()           # compared only where expected


def args_of(arm):
    r = subprocess.run(["bash", "scripts/submit_arm.sh", arm, "--dry"], capture_output=True, text=True,
                       env={**__import__("os").environ, "PREFLIGHT_RAW": "1"})
    out = r.stdout + r.stderr
    lines = [l.strip() for l in out.splitlines()]
    try:
        i = next(n for n, l in enumerate(lines) if l.startswith("ARGS:"))
        j = next(n for n, l in enumerate(lines) if l.startswith("OUT:"))
    except StopIteration:
        sys.exit(f"{arm}: no ARGS line in the dry run -- output was:\n{out}")
    toks = shlex.split(" ".join(lines[i:j])[len("ARGS:"):])
    last = {}
    for n, t in enumerate(toks):
        if t.startswith("--"):
            v = toks[n + 1] if n + 1 < len(toks) and not toks[n + 1].startswith("--") else ""
            last[t] = v                                   # argparse keeps the LAST value
    return last


def main():
    arms = sys.argv[1:] or list(EXPECT)
    base = args_of("b1")
    ok = True
    print(f"b1: --throttle01 {'present' if '--throttle01' in base else 'MISSING'}")
    ok &= "--throttle01" in base
    for arm in arms:
        a, exp = args_of(arm), EXPECT[arm]
        bad = []
        for k, v in exp.items():
            if v is None and k in a:
                bad.append(f"{k} should be absent, is {a[k]!r}")
            elif v is not None and a.get(k, "<absent>") != v:
                bad.append(f"{k} should be {v!r}, is {a.get(k, '<absent>')!r}")
        for k in sorted(set(a) | set(base)):
            if k in exp:
                continue
            if a.get(k, "<absent>") != base.get(k, "<absent>"):
                bad.append(f"UNEXPECTED {k}: b1 {base.get(k, '<absent>')!r}, {arm} {a.get(k, '<absent>')!r}")
        print(f"{arm:16s} {'PASS' if not bad else 'FAIL'}")
        for b in bad:
            print(f"    {b}")
        ok &= not bad
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
