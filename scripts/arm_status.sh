#!/usr/bin/env bash
# Every arm, its state, and what is MISSING for it. Written 2026-09-23 after three
# finished arms (chunk5, memory_cnn, memDINO) sat with no continuation queued and
# nobody noticed, because the inventory lived in my head instead of in a command.
#
#   bash scripts/arm_status.sh
set -uo pipefail
OUT=/scratch/m000204-pm07b/joana/outputs
cd "$(dirname "$0")/.."

name_of () {   # training job id -> arm name
  case "$1" in
    497929) echo A_dino_old;;        497930) echo memDINO_old;;
    497931) echo chunk5_old;;        497932) echo memCNN_baseline;;
    499454) echo A_dino;;            499455) echo corner_ramped;;
    499460) echo corner_ramped_cnn;; 500171) echo chunk5;;
    500179) echo memory_cnn;;        500180) echo memory_dino;;
    500776) echo A_dino_warmold;;    500777) echo memDINO_warmold;;
    500778) echo stack3;;            500779) echo stack3_stride3;;
    500937) echo A_dino_cont;;       500938) echo corner_ramped_cont;;
    500939) echo corner_ramped_cnn_cont;;
    *) echo "j$1";;
  esac
}
# Which parents already HAVE a continuation. Read from the launch lines of every
# job, not from a hardcoded list -- the list went stale within a day and reported
# MISSING for four arms that were already continued (2026-09-24).
PARENTS=$(grep -ho -- "--warmstart [^ ]*" /scratch/m000204-pm07b/joana/slurm-ppo-real-*.out 2>/dev/null \
          | grep -o "_j[0-9]\+" | tr -d '_j' | sort -u)
covered () {
  echo "$PARENTS" | grep -qx "$1" && echo yes || echo no
}

printf "%-22s %-8s %-11s %8s %6s  %-12s %s\n" ARM JOB STATE STEPS CURR CONTINUATION EVAL-CELLS
TODO=()
for d in $(ls -d $OUT/*_j[0-9]* 2>/dev/null); do
  J=${d##*_j}
  [ "$J" -ge 499400 ] 2>/dev/null || continue          # today's arms only
  # a job that died at startup wrote no checkpoint and is not an arm; the eight
  # from this morning were drowning the real list (2026-09-23)
  [ -d "$d/checkpoints" ] || continue
  ls "$d/checkpoints"/ppo_*_steps.zip >/dev/null 2>&1 || continue
  NAME=$(name_of "$J")
  ST=$(sacct -j "$J" -X -n -o State%12 2>/dev/null | head -1 | tr -d ' '); ST=${ST:-?}
  CK=$(cd "$d/checkpoints" 2>/dev/null && ls ppo_*_steps.zip 2>/dev/null | sort -t_ -k2 -n | tail -1)
  STEPS=$(sed 's/[^0-9]//g' <<< "${CK:-}"); STEPS=${STEPS:-0}
  CUR=$(python3 -c "import json;print(json.load(open('$d/curriculum_state.json'))['goal_dist'])" 2>/dev/null || echo "-")
  N_OLD=$(ls -d $OUT/eval_*_j$J* 2>/dev/null | wc -l | tr -d ' ')
  N_ANY=$(grep -rl "_j$J/" $OUT/eval_*/metrics.json 2>/dev/null | wc -l | tr -d ' ')
  CONT=$(covered "$J")
  case "$ST" in
    RUNNING|PENDING) MARK="(training)";;
    *) if [ "$CONT" = "no" ]; then MARK="MISSING"; TODO+=("$NAME ($J): no continuation queued");
       else MARK="continued"; fi;;
  esac
  [ "$ST" != "RUNNING" ] && [ "$ST" != "PENDING" ] && [ "${N_ANY:-0}" -lt 3 ] \
      && TODO+=("$NAME ($J): only ${N_ANY:-0} eval cell(s)")
  printf "%-22s %-8s %-11s %8s %6s  %-12s %s\n" "$NAME" "$J" "$ST" "$STEPS" "$CUR" "$MARK" "${N_ANY:-0} cells"
done

echo ""
if [ ${#TODO[@]} -eq 0 ]; then
  echo "nothing outstanding"
else
  echo "OUTSTANDING:"
  printf '  - %s\n' "${TODO[@]}"
  echo ""
  echo "  continuation:  GPUS=4 bash scripts/submit_arm.sh <armconfig> --warm \$OUT/<rundir>/checkpoints"
  echo "  evals:         bash scripts/submit_eval.sh <jobid>"
fi
echo ""
echo "queue: $(squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' ') job(s); the account limit is about 32"
