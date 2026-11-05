"""Every PPO run we have ever launched, decoded — what it trained on, what
reward it used, how far it got, and whether anybody ever looked at it.

The launcher encodes a run's entire configuration in its output directory name
(scripts/slurm/train_ppo_real.sh appends a token per knob). That makes the
names authoritative but unreadable — `ppo_live_gnd_AUw360_UNGATED_x4_ms6_
v21obs_warm_rw5_semw5_trstrict_g360_gr5-10_gc50_spcls_sjy20_sjl0.4_r336x224_
rr560x336_rs0.01_smin10` is a full experiment spec nobody can read at a glance.
This decodes them back into columns, adds progress and whether an eval exists,
and sorts so the never-evaluated runs are easy to find.

Usage (login node, no GPU):
    python scripts/fleet_inventory.py
    python scripts/fleet_inventory.py --running        # only live jobs
    python scripts/fleet_inventory.py --csv /tmp/fleet.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import re
import subprocess
import time
from pathlib import Path

OUTPUTS = "/scratch/m000204-pm07b/joana/outputs"

# token -> (column, how to read it). Order matters: longest prefix first so
# `_smax` is not eaten by `_sm`, `_sjy`/`_sjl` not eaten by `_s`, etc.
TOKENS = [
    ("UNGATED",  lambda v: ("gate", "OFF")),
    ("SMOKE",    lambda v: ("smoke", "yes")),
    ("noBC",     lambda v: ("bc", "none")),
    ("bc2",      lambda v: ("bc", "demos2")),
    ("hybrid",   lambda v: ("cache", "hybrid")),
    ("trstrict", lambda v: ("trav", "override")),
    ("g360",     lambda v: ("goal_dir", "360")),
    ("spcls",    lambda v: ("spawn_filter", "on")),
    ("v21obs",   lambda v: ("semantics", "v21")),
    ("warm",     lambda v: ("warmstart", "yes")),
    ("rw5",      lambda v: ("reward", "rw5")),
    ("cur",      lambda v: ("curriculum", "on")),
    ("mf",       lambda v: ("footprint", "along-motion")),
    ("fwd",      lambda v: ("forward_only", "yes")),
    ("both",     lambda v: ("encoder", "both")),
    ("dinov2",   lambda v: ("encoder", "dinov2")),
    ("resnet18", lambda v: ("encoder", "resnet18")),
]
# prefix -> column, for tokens that carry a value
VALUED = [
    ("tau", "gate_tau"), ("smax", "spawn_max"), ("smin", "spawn_min"), ("sjy", "spawn_jit_yaw"),
    ("sjl", "spawn_jit_lat"), ("gxy", "goal_xy"), ("gds", "goal_dist_start"),
    ("gd", "goal_dist"), ("gr", "goal_range"), ("gc", "goal_cone"),
    ("gn", "goal_noise"), ("ivt", "imgvoidterm"), ("vt", "voidterm"),
    ("semw", "semantic_w"), ("rs", "reward_scale"), ("pal", "palette"),
    ("prox", "proximity"), ("ct", "coll_term"), ("bk", "backward_cost"),
    ("sp", "spin_cost"), ("ls", "live_steps"), ("chunk", "action_chunk"),
    ("seed", "seed"), ("ms", "multi_scene"), ("multi", "multi_scene"),
    ("x", "n_robots"), ("rr", "render_res"), ("r", "obs_res"),
    ("sm", "smooth_cost"),
]


def decode(name: str) -> dict:
    """Directory name -> readable configuration."""
    cfg = {"gate": "on", "reward": "default", "bc": "demos1",
           "semantics": "v10", "encoder": "nature"}
    body = name
    for pre in ("ppo_live_", "ppo_v14diff_", "ppo_"):
        if body.startswith(pre):
            cfg["mode"] = {"ppo_live_": "live", "ppo_v14diff_": "cache"}.get(pre, "raster")
            body = body[len(pre):]
            break
    parts = body.split("_")
    scene, i = [], 0
    while i < len(parts) and not _is_token(parts[i]):
        scene.append(parts[i])
        i += 1
    cfg["scene"] = "_".join(scene) or "?"
    for p in parts[i:]:
        hit = False
        for tok, fn in TOKENS:
            if p == tok:
                k, v = fn(p)
                cfg[k] = v
                hit = True
                break
        if hit:
            continue
        for pre, col in VALUED:
            if p.startswith(pre) and len(p) > len(pre):
                cfg[col] = p[len(pre):]
                hit = True
                break
        if not hit and p.isdigit():
            cfg["steps_override"] = p
    return cfg


def _is_token(p: str) -> bool:
    if any(p == t for t, _ in TOKENS):
        return True
    return any(p.startswith(pre) and len(p) > len(pre) and
               any(c.isdigit() for c in p[len(pre):]) for pre, _ in VALUED)


def eval_index(root: Path) -> dict:
    """run dir -> list of (eval dir, success_rate).

    eval_policy.py writes to its OWN --out_dir (outputs/eval_*), not into the
    run directory, and records the checkpoint path inside metrics.json. So the
    only way to know whether a run was ever evaluated is to read every
    metrics.json and map its checkpoint back to the run it came from.
    """
    import json
    idx = {}
    for mp in root.glob("*/metrics.json"):
        try:
            s = json.loads(mp.read_text()).get("summary", {})
        except Exception:
            continue
        ck = str(s.get("checkpoint", ""))
        for d in root.glob("ppo_*"):
            if d.is_dir() and (str(d) + "/") in ck:
                idx.setdefault(d.name, []).append(
                    (mp.parent.name, s.get("success_rate")))
                break
    return idx


def running_dirs() -> dict:
    """job id -> output dir, read from each job's launcher echo."""
    out = {}
    try:
        q = subprocess.run(["squeue", "-u", "jmizrahi", "-h", "-o", "%i %j"],
                           capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return out
    for line in q.strip().splitlines():
        bits = line.split()
        if len(bits) < 2 or "ppo" not in bits[1]:
            continue
        jid = bits[0]
        log = Path(f"/scratch/m000204-pm07b/joana/slurm-ppo-real-{jid}.out")
        if not log.exists():
            continue
        for ln in log.read_text(errors="ignore").splitlines()[:400]:
            if ln.startswith("==> rung:") and " out: " in ln:
                out[jid] = ln.split(" out: ")[-1].strip()
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=OUTPUTS)
    ap.add_argument("--running", action="store_true",
                    help="only the runs that are on a node right now")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    live = running_dirs()
    live_by_dir = {v: k for k, v in live.items()}
    evals = eval_index(Path(args.root))

    rows = []
    for d in sorted(Path(args.root).glob("ppo_*")):
        if not d.is_dir():
            continue
        if args.running and str(d) not in live_by_dir:
            continue
        cfg = decode(d.name)
        ck = sorted(d.glob("checkpoints/ppo_*_steps.zip"))
        last = 0
        for c in ck:
            m = re.search(r"_(\d+)_steps", c.name)
            if m:
                last = max(last, int(m.group(1)))
        fails = len(list((d / "failures").glob("*.png"))) if (d / "failures").is_dir() else 0
        has_crash_csv = (d / "failures" / "crash_poses.csv").exists()
        # failure_snap_max is 200 PER ENV (scene_env.py), so an x8 run caps at
        # ~1600 and the column saturates. Flag it — it is not a crash count.
        nrob = int(cfg.get("n_robots", "1") or 1)
        cap = 200 * max(nrob, 1)
        ev = evals.get(d.name, [])
        best = max((e[1] for e in ev if e[1] is not None), default=None)
        final = (d / "ppo_final.zip").exists()
        age_h = (time.time() - d.stat().st_mtime) / 3600.0
        rows.append(dict(
            job=live_by_dir.get(str(d), ""), name=d.name,
            mode=cfg.get("mode", "?"),
            # ms6/ms12 runs rotate over N scenes; the member list lives only in
            # the launcher's "==> rung:" line, never in the directory name — so
            # printing the base scene here would be a lie.
            scene=(f"multi-{cfg['multi_scene']}" if cfg.get("multi_scene")
                   else cfg.get("scene", "?")),
            gate=cfg.get("gate"), reward=cfg.get("reward"),
            sem=cfg.get("semantics"), semw=cfg.get("semantic_w", ""),
            trav=cfg.get("trav", "v14"), cone=cfg.get("goal_cone", ""),
            voidterm=cfg.get("voidterm", ""), ivt=cfg.get("imgvoidterm", ""),
            enc=cfg.get("encoder"), warm=cfg.get("warmstart", ""),
            n_robots=cfg.get("n_robots", "1"),
            ckpt_steps=last, n_ckpt=len(ck), crashes=fails,
            crash_csv="yes" if has_crash_csv else "",
            done="yes" if final else "",
            evals=len(ev), best_sr="" if best is None else best,
            capped="CAP" if fails >= 0.95 * cap else "",
            idle_h=round(age_h, 1)))

    rows.sort(key=lambda r: (r["evals"] > 0, -r["ckpt_steps"]))

    hdr = ["job", "scene", "mode", "gate", "reward", "sem", "semw", "trav",
           "cone", "gate_tau", "voidterm", "ivt", "enc", "warm", "ckpt_steps",
           "crashes",
           "capped", "done", "evals", "best_sr", "idle_h"]
    w = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) if rows else len(h)
         for h in hdr}
    print("  ".join(h.ljust(w[h]) for h in hdr))
    print("  ".join("-" * w[h] for h in hdr))
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(w[h]) for h in hdr))
    print(f"\n{len(rows)} runs   "
          f"{sum(1 for r in rows if r['evals'])} evaluated   "
          f"{sum(1 for r in rows if not r['evals'] and r['ckpt_steps'] > 0)} "
          f"trained but NEVER evaluated")
    print("NOTE  'crashes' counts snapshot PNGs and saturates at 200 per robot "
          "(scene_env.failure_snap_max);\n      'CAP' means it hit the ceiling "
          "— it is a sample of crashes, not a crash rate.")
    print("\nfull names:")
    for r in rows:
        print(f"  {r['job'] or '   -  '}  {r['name']}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = _csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else hdr)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n==> {args.csv}")


if __name__ == "__main__":
    main()
