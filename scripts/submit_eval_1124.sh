#!/usr/bin/env bash
# The 2026-11-24 evaluation protocol -- IDENTICAL for every arm (and for the paper policy), so no run is
# graded by its own training rules:
#   scenarios   EVAL_1124: lawn corner (quad2_00), dirt corner (sequoia1_21), pedestrian (quad2_04_ped),
#               second pedestrian scene (quad1_09_ped); goals, spawns and step budgets from configs/eval_scenes.env
#   episodes    16 per cell, deterministic (mean action), same seeds for every arm, all 16 recorded as videos
#   contact     judged on the RECORDING's fused label map (REWSRC=map) for every arm, at any throttle (CRASHANY)
#   coverage    no coverage cost or termination in eval (COH=0, COHTERM=0) for every arm; coverage is
#               logged per step instead, so low-coverage driving shows up as a number, not as an early end
# Everything else (throttle map, step and turn size, memory, stack, chunk) is adopted from the run's own
# env_config.json, because it is the policy's interface.
#
#   bash scripts/submit_eval_1124.sh <arm> <trainjobid> [--ckpts 60000,120000,last] [--scenes "quad2_00 ..."] [--dry]
# Output: outputs/evals/wave1124/<arm>/<scene>_<ckpt>_goal<NN>/ (metrics.json + episode_<k>.mp4)
set -euo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?usage: submit_eval_1124.sh <arm> <trainjobid> [--ckpts a,b,last] [--scenes "..."] [--dry]}
JOB=${2:?usage: submit_eval_1124.sh <arm> <trainjobid> ...}; shift 2
. configs/eval_scenes.env
CKPTS="60000 120000 last"; SC="$EVAL_1124"; DRY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --ckpts)  CKPTS="${2//,/ }"; shift 2;;
        --scenes) SC="$2"; shift 2;;
        --dry)    DRY="--dry"; shift;;
        *) echo "unknown option $1"; exit 1;;
    esac
done
export EVALTAG="wave1124/$ARM" VIDEOS=0 RECORD=16 EPISODES=16 CRASHANY=1 \
       REWSRC=map COH=0 COHTAU=0.4 COHTERM=0 \
       FORCEKEYS="reward_source,coherence_cost_weight,coherence_terminate_tau"
echo "=== protocol 1124: $ARM (job $JOB)   checkpoints: $CKPTS   scenarios: $SC"
for c in $CKPTS; do
    if [ "$c" = "last" ]; then
        bash scripts/submit_eval.sh "$JOB" $SC $DRY
    else
        bash scripts/submit_eval.sh "$JOB" $SC --ckpt "$c" $DRY
    fi
done
