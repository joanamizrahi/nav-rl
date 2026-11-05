#!/usr/bin/env bash
# Every gate a freshly started training job must clear, in one place.
#
# Written 2026-09-23 after a day in which four separate things went wrong at
# startup and each was found by hand, hours late: obs_frame_stack on the wrong
# config (10 arms dead in 8 s), the image-head guard firing on a correctly reset
# head, a CUDA OOM from the frame stack, and four GPU workers all sitting on the
# same scene so a run only ever saw 6 of 13 scenes.
#
#   bash scripts/check_run.sh 500937 500938 ...    # or with no args, every job
#
# WAIT means the job has not got that far yet, not that it failed.
LOGDIR=${LOGDIR:-/scratch/m000204-pm07b/joana}
JOBS=${@:-$(squeue -u "$USER" -h -o "%i" 2>/dev/null)}
[ -z "$JOBS" ] && { echo "no jobs"; exit 0; }

p(){ printf "  [\033[32mPASS\033[0m] %s\n" "$1"; }
f(){ printf "  [\033[31mFAIL\033[0m] %s\n" "$1"; BAD=$((BAD+1)); }
w(){ printf "  [wait] %s\n" "$1"; }
i(){ printf "         %s\n" "$1"; }
BAD=0

for J in $JOBS; do
  OUT=$LOGDIR/slurm-ppo-real-$J.out; ERR=$LOGDIR/slurm-ppo-real-$J.err
  ST=$(sacct -j "$J" -X -n -o State%20 2>/dev/null | head -1 | tr -d ' ')
  EL=$(sacct -j "$J" -X -n -o Elapsed 2>/dev/null | head -1 | tr -d ' ')
  NAME=$(grep -m1 -o "wandb label: [^ ]*" "$OUT" 2>/dev/null | cut -c14-70)
  echo ""
  echo "== $J   ${ST:-?}  ${EL:-}   ${NAME:-}"
  [ -f "$OUT" ] || { w "no .out yet (not started)"; continue; }

  # 1. did it die
  if grep -qE "Traceback|REFUSED|OutOfMemoryError|CUDA out of memory" "$ERR" 2>/dev/null; then
    f "error in .err:"; i "$(grep -hE 'REFUSED|Error:|OutOfMemoryError|TypeError|ValueError' "$ERR" | tail -2)"
  else p "no traceback, REFUSED or OOM in .err"; fi

  # 2. GPUs and robots
  G=$(grep -m1 -o "\[live_gpus\] [0-9]* GPUs x [0-9]* robots = [0-9]* envs" "$OUT")
  [ -n "$G" ] && p "${G#\[live_gpus\] }" || w "live_gpus line not printed yet"

  # 3. THE SCENE STAGGER (the 2026-09-23 fix)
  NW=$(grep -c "worker starts on" "$OUT" 2>/dev/null)
  NGPU=$(grep -m1 -o "\[live_gpus\] [0-9]*" "$OUT" | awk '{print $2}')
  if [ "${NW:-0}" -ge 1 ]; then
    p "scene stagger active: $NW workers start off index 0 (expect $((${NGPU:-4}-1)))"
    i "$(grep -m3 "worker starts on" "$OUT" | sed 's/.*worker starts on //' | tr '\n' ' ')"
  elif grep -q "scene rotate" "$OUT" 2>/dev/null; then
    f "NO stagger: all workers started at scene 0 -- this job has the old code"
  else w "stagger line not printed yet"; fi

  # 4. rotations must name DIFFERENT scenes (old code repeated one K times)
  R=$(grep -o "scene rotate -> [a-z0-9_]*" "$OUT" 2>/dev/null | awk '{print $NF}' | head -4)
  RN=$(echo "$R" | grep -c .); RU=$(echo "$R" | sort -u | grep -c .)
  if [ "${RN:-0}" -lt 4 ]; then w "fewer than 4 rotations so far (${RN:-0})"
  elif [ "$RU" -ge 3 ]; then p "first 4 rotations name $RU distinct scenes: $(echo $R | tr '\n' ' ')"
  else f "first 4 rotations are the same scene ($RU distinct) -- workers in lockstep"; fi

  # 5. scene list size
  NS=$(grep -m1 -o "\-\-scenes [a-z0-9_ ]*" "$OUT" | sed 's/--scenes //' | wc -w)
  [ "${NS:-0}" -gt 0 ] && { [ "$NS" -eq 13 ] && p "13 training scenes" || f "$NS scenes, expected 13"; } \
                       || w "scene list not printed yet"

  # 6. can the policy see (the double-/255 bug)
  IC=$(grep -m1 "\[image check\] normalize_images" "$OUT")
  if [ -n "$IC" ]; then
    D=$(echo "$IC" | grep -o "= [0-9.]*" | head -1 | tr -d '= ')
    if echo "$IC" | grep -q "uses the image"; then p "sees the image: delta $D"
    else f "BLIND: delta $D -- kill it, do not spend 12 h on this"; fi
  else w "image check not reached yet"; fi

  # 7. warm start resolution and curriculum resume
  #    WARM is read from the launch line, not from the [warmstart] log line, so a
  #    job running older code is still reported as warm rather than as cold.
  WARMP=$(grep -m1 -o -- "--warmstart [^ ]*" "$OUT" | awk '{print $2}')
  if [ -n "$WARMP" ]; then
    p "warm start from $(basename "$WARMP")"
    if grep -q "newest checkpoint" "$OUT" 2>/dev/null; then
      i "$(grep -m1 "newest checkpoint" "$OUT" | sed 's/.*\[warmstart\] //' | cut -c1-76)"
    fi
    CR=$(grep -m1 "curriculum resumes at" "$OUT")
    if [ -n "$CR" ]; then p "${CR#*\[warmstart\] }"
    else i "curriculum NOT resumed from the parent (older code, or parent wrote no state)"; fi
    HR=$(grep -m1 "image head reset" "$OUT")
    [ -n "$HR" ] && i "${HR#*\[image check\] }"
  else i "cold arm (no --warmstart on the launch line)"; fi

  # 8. the curriculum must actually be switched on
  if grep -qm1 -- "--goal_dist_start" "$OUT"; then p "distance curriculum active (--goal_dist_start present)"
  elif grep -qm1 "rung:" "$OUT"; then f "NO --goal_dist_start: the curriculum will never move off its floor"
  else w "launch line not printed yet"; fi

  # 9. is it actually training, and fast enough
  TS=$(grep "total_timesteps" "$OUT" | tail -1 | grep -o "[0-9]\+" | tail -1)
  TE=$(grep "time_elapsed" "$OUT" | tail -1 | grep -o "[0-9]\+" | tail -1)
  # a continuation keeps the parent's step counter, so only the steps THIS job
  # has run may be divided by the time THIS job has been up
  W0=$(grep -m1 -o "warm[0-9]\+" "$OUT" | head -1 | tr -dc '0-9')
  NEW=$TS; [ -n "$W0" ] && [ -n "$TS" ] && [ "$TS" -gt "$W0" ] && NEW=$((TS - W0))
  if [ -n "$NEW" ] && [ -n "$TE" ] && [ "$TE" -gt 0 ]; then
    RATE=$((NEW * 3600 / TE)); P12=$((RATE * 12))
    [ -n "$W0" ] && P12=$((P12 + W0))
    if [ "$P12" -ge 50000 ]; then p "throughput $RATE steps/h -> ~${P12} in 12 h (past the 50k mark)"
    else f "throughput $RATE steps/h -> only ~${P12} in 12 h, short of 50k"; fi
  else w "no rollout logged yet"; fi

  # 10. throttle must not be decaying toward a motionless robot
  TH=$(grep "diag/throttle" "$OUT" | tail -1 | tr -d ' ' | awk -F'|' '{print $3}')
  [ -n "$TH" ] && i "latest mean throttle $TH (must stay positive and not decay)"
done

echo ""
[ "$BAD" -eq 0 ] && echo "ALL CHECKS PASSED (or still waiting)" || echo "$BAD CHECK(S) FAILED -- see above"
