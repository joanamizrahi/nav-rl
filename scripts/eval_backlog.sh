#!/usr/bin/env bash
# Submit a list of eval cells as the account's job limit allows, instead of
# watching squeue and retrying by hand (2026-09-23: 8 cells, 1 got in).
#
#   bash scripts/eval_backlog.sh cells.txt            # drains it, then exits
#   nohup bash scripts/eval_backlog.sh cells.txt &    # in the background
#
# cells.txt: one per line, "<trainjobid> <scene> <GOALKEY=value>", # comments ok.
# A submitted line is removed from the file, so the file IS the remaining work
# and the script is safe to interrupt and restart.
set -uo pipefail
BACKLOG=${1:?usage: eval_backlog.sh <cells file>}
EVERY=${EVERY:-240}                 # seconds between passes
MAXPASS=${MAXPASS:-180}             # give up after this many passes (12 h at 240 s)
SEM=${SEM:-/scratch/m000204-pm07b/joana/runs/train_semantic_v35b_p6_ppl5/checkpoint-epoch-3.safetensors}
OUTROOT=/scratch/m000204-pm07b/joana/outputs
cd "$(dirname "$0")/.."

for ((pass=1; pass<=MAXPASS; pass++)); do
    left=$(grep -cvE '^\s*(#|$)' "$BACKLOG" 2>/dev/null || echo 0)
    [ "$left" -eq 0 ] && { echo "$(date +%H:%M) backlog empty"; exit 0; }
    echo "$(date +%H:%M) pass $pass, $left cell(s) left"
    while read -r J S G; do
        case "$J" in ''|\#*) continue;; esac
        D=$(ls -d $OUTROOT/*_j$J 2>/dev/null | head -1)
        [ -z "$D" ] && { echo "  skip $J (no run dir)"; continue; }
        C=$D/checkpoints/$(cd "$D/checkpoints" && ls ppo_*_steps.zip 2>/dev/null | sort -t_ -k2 -n | tail -1)
        [ -f "$C" ] || { echo "  skip $J (no checkpoint)"; continue; }
        unset GOAL_FRAME GOALFRAMERANGE          # one scene's goal must not leak into the next
        export LIVE=1 LIVECKPT="$SEM" CKPT="$C" SCENE="$S" EPISODES="${EPISODES:-8}" \
               CLIPS_DIR=/scratch/m000204-pm07b/joana/data/campus_clips \
               MAXSTEPSPERM="${MAXSTEPSPERM:-16}" MAXSTEPSBASE="${MAXSTEPSBASE:-40}"
        eval "export $G"
        if id=$(sbatch --parsable --export=ALL scripts/slurm/eval_policy.sh 2>/dev/null); then
            echo "  submitted $id   $J $S $G"
            # drop this exact line from the backlog so a restart never duplicates it
            grep -vxF "$J $S $G" "$BACKLOG" > "$BACKLOG.tmp" && mv "$BACKLOG.tmp" "$BACKLOG"
        fi
    done < <(grep -vE '^\s*(#|$)' "$BACKLOG")
    left=$(grep -cvE '^\s*(#|$)' "$BACKLOG" 2>/dev/null || echo 0)
    [ "$left" -eq 0 ] && { echo "$(date +%H:%M) backlog empty"; exit 0; }
    sleep "$EVERY"
done
echo "gave up after $MAXPASS passes; $(grep -cvE '^\s*(#|$)' "$BACKLOG") cell(s) still queued in $BACKLOG"
