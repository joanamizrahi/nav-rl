#!/usr/bin/env bash
# One place that knows how to launch an eval, so nothing gets dropped by hand.
#
#   bash scripts/submit_eval.sh <trainjobid> [scene ...] [--dry]
#   bash scripts/submit_eval.sh 499460 --ckpt 40000        # a specific checkpoint
#   bash scripts/submit_eval.sh 499460 --all               # all 5 scenes, not just the core 3
#
# Defaults to the 3-scene core set. Reads goal + spawn bound from configs/eval_scenes.env.
set -euo pipefail
cd "$(dirname "$0")/.."
. configs/eval_scenes.env

JOB=${1:?usage: submit_eval.sh <trainjobid> [scenes...] [--ckpt N] [--dry] [--all]}; shift
DRY=0; CKPT_WANT=""; SCENES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry)  DRY=1; shift;;
        --all)  SCENES="$(awk 'NF && $1 !~ /^#/ {print $1}' <<< "$EVAL_SCENES")"; shift;;
        --ckpt) CKPT_WANT=$2; shift 2;;
        *)      SCENES="$SCENES $1"; shift;;
    esac
done
[ -z "${SCENES// }" ] && SCENES="$EVAL_CORE"

OUT=/scratch/m000204-pm07b/joana/outputs
SEM=${LIVECKPT:-/scratch/m000204-pm07b/joana/runs/train_semantic_v35b_p6_ppl5/checkpoint-epoch-3.safetensors}
D=$(ls -d $OUT/*_j${JOB} 2>/dev/null | head -1) || true
[ -n "${D:-}" ] || { echo "REFUSED: no run directory for job $JOB"; exit 1; }
if [ -n "$CKPT_WANT" ]; then
    C=$D/checkpoints/ppo_${CKPT_WANT}_steps.zip
else
    LATEST=$(cd "$D/checkpoints" 2>/dev/null && ls ppo_*_steps.zip 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -z "$LATEST" ]; then
        # a continuation writes nothing until it passes its first checkpoint boundary
        NOW=$(python3 -c "import json;print(int(json.load(open('$D/curriculum_state.json'))['num_timesteps']))" 2>/dev/null || echo "")
        echo "REFUSED: job $JOB has written no checkpoint yet."
        [ -n "$NOW" ] && echo "         it is at ${NOW} steps; checkpoints are every 10000, so the first lands at $(( (NOW/10000 + 1) * 10000 ))."
        exit 1
    fi
    C=$D/checkpoints/$LATEST
fi
[ -f "$C" ] || { echo "REFUSED: no checkpoint $C"; exit 1; }
CUR=$(python3 -c "import json;print(json.load(open('$D/curriculum_state.json'))['goal_dist'])" 2>/dev/null || echo "none")

echo "=== eval job $JOB   $(basename "$C")   curriculum reached ${CUR} m"
for S in $SCENES; do
    LINE=$(awk -v s="$S" '$1==s {print; exit}' <<< "$EVAL_SCENES")
    [ -n "$LINE" ] || { echo "REFUSED: $S is not in configs/eval_scenes.env"; exit 1; }
    GOAL=$(awk '{print $2}' <<< "$LINE"); SPAWN=$(awk '{print $3}' <<< "$LINE")
    # a row may name a different REAL scene, so "quad2_00_far" is a variant of quad2_00
    REAL=$(awk '{for(i=4;i<=NF;i++) if ($i ~ /^SCENE=/) {sub(/^SCENE=/,"",$i); print $i; exit}}' <<< "$LINE")
    [ -n "$REAL" ] && S_REAL="$REAL" || S_REAL="$S"
    # gnd clips do not live in campus_clips; a row may name its own directory
    CLIPSUB=$(awk '{for(i=4;i<=NF;i++) if ($i ~ /^CLIPS=/) {sub(/^CLIPS=/,"",$i); print $i; exit}}' <<< "$LINE")
    CLIPDIR=/scratch/m000204-pm07b/joana/data/${CLIPSUB:-campus_clips}
    # the distance THIS test spans, measured; sizes the budget so every arm gets the
    # same one on a fixed-goal scene (2026-09-24, chunk5 timed out on its budget)
    TDIST=$(awk '{for(i=4;i<=NF;i++) if ($i ~ /^TESTDIST=/) {sub(/^TESTDIST=/,"",$i); print $i; exit}}' <<< "$LINE")
    unset GOAL_FRAME GOALFRAMERANGE SPAWN_MAX SPAWNMAX TESTDIST   # never leak between scenes
    [ -n "$TDIST" ] && export TESTDIST="$TDIST"
    # 16 episodes, not 8: a single 8-episode cell is a thin population, and we found
    # that the RECORDED rollout is not the SCORED one -- same seed, deterministic
    # actions, and episode 0 came back GOAL when scored and TIMEOUT when recorded.
    # So the scored episodes are the numbers and the recorded ones are the pictures.
    # VIDEOS is capped below EPISODES because each recording is a SECOND rollout and
    # doubles the cost of the cell. 2026-09-24.
    export LIVE=1 LIVECKPT="$SEM" CKPT="$C" SCENE="$S_REAL" EPISODES="${EPISODES:-16}" \
           VIDEOS="${VIDEOS:-8}" \
           CLIPS_DIR="$CLIPDIR" \
           MAXSTEPSPERM="${MAXSTEPSPERM:-16}" MAXSTEPSBASE="${MAXSTEPSBASE:-40}"
    eval "export $GOAL"
    if [ "$SPAWN" != "-" ]; then
        for kv in ${SPAWN//,/ }; do eval "export $kv"; done
    fi
    printf "    %-13s %-22s %s%s\n" "$S" "$GOAL" "${SPAWN/-/spawn unbounded (goal range, safe)}" \
        "$([ -n "$REAL" ] && echo "   [scene $REAL]")$([ -n "$CLIPSUB" ] && echo "   [clips $CLIPSUB]")$([ -n "$TDIST" ] && echo "   [${TDIST} m budget]")"
    [ "$DRY" = "1" ] && continue
    sbatch --export=ALL scripts/slurm/eval_policy.sh >/dev/null 2>&1 \
        && echo "        submitted" || echo "        LIMIT REACHED -- retry later"
done
[ "$DRY" = "1" ] && echo "    (dry run: nothing submitted)"
