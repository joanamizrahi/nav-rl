"""One line per scene from check_rewards_many logs: focal, blind zone, footprint
void / obstacle fraction and crash-level steps on the recorded walk, survey
video present. Replaces the paste/sed one-liner that misaligned (2026-09-16).

    python scripts/check_rewards_table.py /scratch/.../slurm-check-rew-many-4886*.out \
        [--survey_root /scratch/.../outputs/check_rew_campus]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SECTION = re.compile(r"^=== SCENE (\S+) ===\s*$", re.M)
FY = re.compile(r"fy\s+([\d.]+)\s+cy\s+([\d.]+)")
BLIND = re.compile(r"ground closer than\s+([\d.]+)\s*m")
HEIGHT = re.compile(r"camera height for the geometry block:\s+([\d.]+) m")
VOID = re.compile(r"footprint void\s+([\d.]+)")
COLL = re.compile(r"footprint coll\s+([\d.]+)\s+steps[^:]*:\s*(\d+)/(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--survey_root", default="/scratch/m000204-pm07b/joana/outputs/check_rew_campus")
    ap.add_argument("--look_ahead", type=float, default=1.5)
    args = ap.parse_args()
    rows = {}
    for p in args.logs:
        txt = Path(p).read_text(errors="ignore")
        marks = list(SECTION.finditer(txt))
        for i, m in enumerate(marks):
            name = m.group(1)
            sec = txt[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(txt))]
            fy = FY.search(sec); bl = BLIND.search(sec); h = HEIGHT.search(sec)
            voids = [float(v) for v in VOID.findall(sec)]
            colls = COLL.findall(sec)
            failed = "!!! FAILED" in sec
            survey = (Path(args.survey_root) / name / f"SURVEY_{name}_path.mp4").exists()
            rows[name] = dict(
                fy=float(fy.group(1)) if fy else float("nan"),
                blind=float(bl.group(1)) if bl else float("nan"),
                h=float(h.group(1)) if h else float("nan"),
                void=max(voids) if voids else float("nan"),
                coll=max(float(c[0]) for c in colls) if colls else float("nan"),
                crash=f"{max(int(c[1]) for c in colls)}/{colls[0][2]}" if colls else "-",
                failed=failed, survey=survey, log=Path(p).name)
    print(f"{'scene':<14}{'fy':>6}{'h':>6}{'blind':>7}{'void':>7}{'coll':>7}{'crash':>8}  survey  status")
    for name, r in sorted(rows.items(), key=lambda kv: -(kv[1]['coll'] if kv[1]['coll'] == kv[1]['coll'] else -1)):
        status = "FAILED" if r["failed"] else ("no geometry" if r["fy"] != r["fy"] else
                 ("blind>look-ahead" if r["blind"] + 0.15 > args.look_ahead else "ok"))
        print(f"{name:<14}{r['fy']:6.0f}{r['h']:6.2f}{r['blind']:7.2f}{r['void']:7.3f}{r['coll']:7.3f}{r['crash']:>8}  {'yes' if r['survey'] else 'NO':<6}  {status}")
    n = len(rows); done = sum(1 for r in rows.values() if r["fy"] == r["fy"] and not r["failed"])
    print(f"\n{done}/{n} scenes with a complete geometry block; incomplete/failed: "
          + ", ".join(k for k, r in rows.items() if r["failed"] or r["fy"] != r["fy"]))


if __name__ == "__main__":
    main()
