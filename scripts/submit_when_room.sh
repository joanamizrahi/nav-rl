#!/usr/bin/env bash
# Submit evals one scene at a time, retrying while the account is at its job-submit
# limit, so an overnight batch drains itself (2026-09-27, last night of cluster access).
#   nohup bash scripts/submit_when_room.sh <log> "<env knobs>" <trainjob> <ckpt> <scene> [scene ...] &
# Every line of <log> says what was submitted; "LIMIT" lines are retried every 5 min.
LOG=$1; KNOBS=$2; JOB=$3; CKPT=$4; shift 4
cd /scratch/m000204-pm07b/joana/nav-rl
CK=(); [ -n "$CKPT" ] && [ "$CKPT" != "latest" ] && CK=(--ckpt "$CKPT")   # "latest" = newest checkpoint at submit time
for S in "$@"; do
    while true; do
        out=$(env $KNOBS bash scripts/submit_eval.sh "$JOB" "$S" "${CK[@]}" 2>&1)
        if grep -q "LIMIT REACHED" <<< "$out"; then echo "$(date +%H:%M) LIMIT $JOB $S, retry in 5 min" >> "$LOG"; sleep 300; continue; fi
        echo "$(date +%H:%M) $JOB $CKPT $S: $(grep -o 'submitted\|REFUSED.*' <<< "$out" | head -1)" >> "$LOG"; break
    done
done
echo "$(date +%H:%M) DONE $JOB $CKPT" >> "$LOG"
