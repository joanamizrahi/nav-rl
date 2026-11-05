#!/usr/bin/env bash
# Submit one policy arm from config FILES instead of a forty-line terminal block
# (2026-09-22, Joana: "can we keep the configs in some files instead of relying on the terminal
# commands each time?"). Several bugs this month entered through a retyped block: a knob dropped,
# an encoder pasted onto the wrong arm, a scene list that no longer matched the one being claimed.
# A file is reviewable, diffable and identical between two launches.
#
#   scripts/submit_arm.sh <arm>                      # from scratch
#   scripts/submit_arm.sh <arm> --warm <checkpoint>  # continue a policy
#   scripts/submit_arm.sh <arm> --dry                # print everything, submit nothing
#
# <arm> names a file in configs/arms/. It is layered on configs/arms/_base_campus.env, and any
# knob it sets that the base does not define is REFUSED: a typo cannot silently do nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?usage: submit_arm.sh <arm> [--warm <ckpt>] [--dry]}; shift
WARM=""; DRY=0; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --warm) WARM=$2; shift 2;;
        --dry)  DRY=1; shift;;
        *)      EXTRA+=("$1"); shift;;
    esac
done
BASE=configs/arms/_base_campus.env
ARMF=configs/arms/${ARM}.env
SCENESF=configs/arms/scenes_campus13.env
for f in "$BASE" "$ARMF" "$SCENESF"; do [ -f "$f" ] || { echo "no such config: $f"; exit 1; }; done

keys() { grep -oE '^[A-Z][A-Z0-9_]*=' "$1" | tr -d '=' ; }
# every knob the arm sets must exist in the base, except the few the submitter owns
OWNED="GPUS TIME MEM CPUS NSTEPS"
for k in $(keys "$ARMF"); do
    if ! keys "$BASE" | grep -qx "$k" && ! echo "$OWNED" | tr ' ' '\n' | grep -qx "$k"; then
        echo "REFUSED: $ARMF sets $k, which the base does not define. Typo, or add it to $BASE."; exit 1
    fi
done
# 2026-09-22: sourcing the arm file OVERWRITES the caller's environment, so
# `GPUS=2 submit_arm.sh A_dino` silently got the arm file's GPUS=4. Resource
# knobs typed on the command line must win; everything else still comes from
# the files. Snapshot before sourcing, restore after.
for k in $(cat <(keys "$BASE") <(keys "$ARMF") <(keys "$SCENESF") | sort -u); do
    if [ -n "${!k+x}" ]; then eval "_CALLER_$k=\$$k"; fi
done
_CALL_GPUS=${GPUS:-}; _CALL_TIME=${TIME:-}; _CALL_MEM=${MEM:-}; _CALL_CPUS=${CPUS:-}; _CALL_NSTEPS=${NSTEPS:-}
set -a; . "$BASE"; . "$SCENESF"; . "$ARMF"; set +a
[ -n "$_CALL_GPUS" ]   && GPUS=$_CALL_GPUS
[ -n "$_CALL_TIME" ]   && TIME=$_CALL_TIME
[ -n "$_CALL_MEM" ]    && MEM=$_CALL_MEM
[ -n "$_CALL_CPUS" ]   && CPUS=$_CALL_CPUS
[ -n "$_CALL_NSTEPS" ] && NSTEPS=$_CALL_NSTEPS
# 2026-09-23: a knob typed on the command line that the arm file also sets was
# SILENTLY discarded -- `CHUNK=2 submit_arm.sh trajectory_chunk5` submitted a
# chunk-5 arm. Resource knobs are the caller's to override (restored above);
# everything else belongs to the arm file, so a disagreement is refused rather
# than ignored. The fix for "I want a different knob" is a new arm file.
for k in $(cat <(keys "$BASE") <(keys "$ARMF") <(keys "$SCENESF") | sort -u); do
    case " $OWNED " in *" $k "*) continue;; esac
    eval "_pre=\${_CALLER_$k-__unset__}"
    [ "$_pre" = "__unset__" ] && continue
    eval "_post=\$$k"
    if [ "$_pre" != "$_post" ]; then
        echo "REFUSED: you set $k=$_pre on the command line but $ARMF sets $k=$_post."
        echo "         Only $OWNED may be overridden that way. To vary $k, make an arm file:"
        echo "           cp $ARMF configs/arms/<new-name>.env   # then edit $k there"
        exit 1
    fi
done
GPUS=${GPUS:-4}; TIME=${TIME:-12:00:00}
MEM=${MEM:-$([ "$GPUS" -ge 4 ] && echo 192G || echo 96G)}
CPUS=${CPUS:-$([ "$GPUS" -ge 4 ] && echo 16 || echo 8)}
[ "$GPUS" -lt 4 ] && NSTEPS=${NSTEPS:-256}       # keep 2048 env steps per PPO update on 8 robots
export LIVEGPUS=$GPUS SCENES="$(echo $SCENES_LIST | tr ' ' ',')" TAG=${TAG:-$ARM}
if [ -n "${NSTEPS:-}" ]; then export NSTEPS; fi
if [ -n "$WARM" ]; then export WARMSTART="$WARM"; fi

# spawn frames are DERIVED from the pose screen, never typed
export SPAWNFRAMES=$(TRAIN="$SCENES_LIST" python - <<'PY'
import json, os
out = []
for s in os.environ["TRAIN"].split():
    fl = set(json.load(open(f"/scratch/m000204-pm07b/joana/outputs/pose_vs_odom_kprior/{s}_pose_vs_odom.json"))["flagged_frames"])
    out.append(f"{s}:{','.join(str(f) for f in range(5, 61) if f not in fl and (f-1) not in fl and (f+1) not in fl)}")
print(";".join(out))
PY
)
n_scenes=$(echo $SCENES_LIST | wc -w); n_spawn=$(echo "$SPAWNFRAMES" | tr ';' '\n' | wc -l)
echo "=== arm: $ARM"
echo "    encoder      ${ENCODER}   image fix ${IMGFIX:-0}   memory ${BOXMEM:-off}   chunk ${CHUNK:-1}   stack ${FRAMESTACK:-1}"
echo "    goals        band ${GOALRANGE} -> ${GOALDIST} m, curriculum from ${GOALDIST_START:-OFF}"
if [ -n "${GOALTURN:-}" ]; then
    echo "    corners      ${GOALTURN} deg, mix ${GOALTURNMIX}${GOALTURNFROM:+, ramped 0 at ${GOALTURNFROM} m -> ${GOALTURNMIX} at ${GOALDIST} m}"
else
    echo "    corners      none"
fi
echo "    spawns       jitter ${SPAWNJYAW} deg / ${SPAWNJLAT} m, mirror ${MIRROR}"
echo "    scenes       ${n_scenes} (${n_spawn} spawn lists)"
echo "    start        ${WARM:-from scratch}"
echo "    resources    ${GPUS} GPU, ${MEM}, ${CPUS} cpu, ${TIME}"
[ "$n_scenes" = "$n_spawn" ] || { echo "REFUSED: $n_scenes scenes but $n_spawn spawn lists"; exit 1; }
[ "${ENCODER}" = "nature" ] && [ "${IMGFIX:-0}" = "1" ] && { echo "REFUSED: IMGFIX on a nature CNN starves it"; exit 1; }
[ -n "${GOALTURNFROM:-}" ] && [ -z "${GOALDIST_START:-}" ] && { echo "REFUSED: GOALTURNFROM needs the distance curriculum"; exit 1; }
# a DIRECTORY is allowed: the job picks the newest ppo_*_steps.zip inside it at
# START time, so a continuation can be queued with --dependency before its parent
# has finished writing (2026-09-23).
# A continuation chained with --dependency may be submitted before its parent has
# written checkpoints/ (created at the first checkpoint, ~20 min in). Accept a
# missing checkpoints/ when the parent RUN directory exists and a --dependency is
# given: the job resolves the newest checkpoint at START, by which time it is there.
# Without a --dependency the old refusal stands (2026-09-25).
_dep=0; for _x in "${EXTRA[@]}"; do case "$_x" in --dependency=*) _dep=1;; esac; done
if [ -n "$WARM" ] && [ ! -e "$WARM" ] && [ "$_dep" = 1 ] && [ -d "$(dirname "$WARM")" ] && [ "$(basename "$WARM")" = "checkpoints" ]; then
    echo "    (checkpoints/ not written yet; accepted because the run exists and the job is chained)"
elif [ -n "$WARM" ] && [ ! -f "$WARM" ] && [ ! -L "$WARM" ] && [ ! -d "$WARM" ]; then
    echo "REFUSED: no such checkpoint or directory $WARM"; exit 1
fi
[ -n "$WARM" ] && [ -d "$WARM" ] && echo "    (directory: the newest ppo_*_steps.zip in it is chosen when the job starts)"
if [ "$DRY" = 1 ]; then echo "    (dry run, nothing submitted)"; exit 0; fi
sbatch --gres=gpu:$GPUS --mem=$MEM --cpus-per-task=$CPUS --time=$TIME "${EXTRA[@]}" scripts/slurm/train_ppo_real.sh
