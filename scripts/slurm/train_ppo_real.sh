#!/usr/bin/env bash
#SBATCH --job-name=ppo-real-smoke
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# 8h was a leftover from the smoke-test era. At the measured 2.6k env steps/h a
# live run needs ~77 h for 200k steps, so an 8 h wall killed every arm before it
# had run overnight (caught 2026-09-02 when four freshly launched arms all came
# back with TIME_LIMIT 8:00:00 while the older ones had 1-2 days).
# Override per submission with `sbatch --time=HH:MM:SS`.
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-ppo-real-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-ppo-real-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# First-version RL loop (Thursday deliverable): PPO on rugd_trail_00 with
# rasterizer-only observations + Gaussian-label reward, all in real meters.
# Success = loop runs, reward logs, return trends up on the trivial task.

set -euo pipefail

# Snapshot the caller's knobs BEFORE any default is applied, so the ledger
# records what was actually typed rather than what the script filled in.
# 2026-09-03: seven 48-hour arms were running and NOTHING on disk said what any
# of them was -- the .out files do not echo the command, env_config.json is
# only written after the hour-long pipeline load, and shell history had scrolled
# away. An unrecorded experiment is not an experiment.
KNOB_NAMES='DISAGREEFRAC|RGBALIGN|RGBALIGNTAU|DISAGREE|CURPERSCENE|CURTHRESH|CURWIN|CURMINEP|CURNOTCH|LIVE|MAXSTEPSPERM|MAXSTEPSBASE|IMGFIX|RESETHEAD|FRAMESTACK|FRAMESTRIDE|PHANTOMVETO|BOXMEMALL|GOALTURN|GOALTURNMIX|GOALTURNBEYOND|GOALTURNFROM|NSTEPS|RW5|RW|RW5SMOOTH|PROXM|CKPTSTEPS|GOALJLAT|GOALMINSCORE|PROXGNDCLS|BOXMEMAGG|SMOOTHCOST|SEED|TARGETKL|STEPS|STEPS_OVERRIDE|PROBE|MAXSTEPS|GOALSUPPORT|GOAL360|NOGATE|TIMEOUTDIST|SEMW|SEMPAL|LIVEBATCH|LIVEFRAMES|ROTATE|GOALRANGE|GOALCONE|GOALNOISE|GOALXY|GOALDIST|GOALDIST_START|GOALDISTWIN|GOALFRAMERANGE|SPAWNCLS|SPAWNJYAW|SPAWNJLAT|SPAWNMIN|SPAWNMAX|HEIGHT|WIDTH|RENDERH|RENDERW|REWSCALE|TRAV|SCENES|SCENE|NSC|LIVECKPT|COH|COHTAU|COHTERM|COLLAHEAD|BOXMEM|NEXTPOSE|LOOKAUTO|NEXTHEAD|CRASHMOVE|RASTEROBS|STATICSCENE|SPAWNHEADWALK|SPAWNFRAMES|GOALCASE|LABELREMAP|ENCGRID|STATICMOVERS|GOALBONUS|GOALWEIGHT|GOALCLEAR|MIRROR|WANDB_PROJECT|GOALRADIUS|TIMEOUTPEN|COLLTERM|CURRICULUM|WARMSTART|CRASHPEN|HALT|HALTEPS|HALTSCALE|REFUSAL|REFUSALDIST|REFUSALVERGE|VERGECUR|HALTWRONG|HALTWRONGRAMP|NTGOAL|GOALSTOP|STOPACT|VERGEPROG|DWELLCUR|CKPTFREQ|SPEEDCOST|REWSRC|MAPINFL|MAPINFLCLS|HALTCUR|HALTGATE|GOALMIX|GOALMIXMAP|GOALNTTRIES|GOALNTCONE|SPAWNSUPPORT|ENT|CHUNK|PROX|PROXGND|PROXGNDM|GATETAU|TAG|LABEL|CLIPS_DIR|OBSCACHE'
# Whitelist, not a blacklist. The first version excluded known-noisy prefixes
# and still emitted lmod shell functions, base64 module tables and every SLURM
# variable -- thousands of unreadable characters per entry (2026-09-03, first
# real ledger dump). Only the launcher's OWN knobs are worth recording; the
# resolved `rung` line below already carries everything the run actually got.
LAUNCH_ENV="$(env | grep -E "^(${KNOB_NAMES})=" | sort | tr '\n' ' ')"
# A LABEL typed at launch time is a prefix on the wandb name (the launcher
# rebuilds LABEL from knobs below and used to drop it silently: the six
# 2026-09-07 corner arms were all typed with LABEL=cornerX-... and none of
# those names reached wandb).
USER_LABEL="${LABEL:-}"

# PORTABLE (2026-09-08): the same launcher must run on Marlowe (module
# system + the shared conda env) and on a rented cloud GPU (neither).
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
[ -d /users/jmizrahi/.conda/envs/neoverse/bin ] && export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH

export PYTHONNOUSERSITE=1
hash -r

# RL deps (one-time; no-ops afterwards)
python - <<'PY' || python -m pip install --quiet "stable-baselines3[extra]" gymnasium
import stable_baselines3, gymnasium
PY

NAVRL_ROOT=/scratch/m000204-pm07b/joana/nav-rl
cd "$NAVRL_ROOT"

# Ladder-aware: if the demos file exists we run the BC rung (v3); otherwise the
# pure shaped rung (v2). Output dirs are versioned accordingly — no overwrites.
# DEMOS_FILE overrides which demo set BC uses (e.g. demos_v2.npz = post-fix
# convention + cached diffused obs). Default demos_v1.npz is from 2026-07-27 —
# PRE-yaw-fix (mirrored turns, raster obs): poisons cached runs. NOBC=1 skips BC.
DEMOS=${DEMOS_FILE:-/scratch/m000204-pm07b/joana/outputs/demos_v1.npz}
if [ "${NOBC:-0}" = "1" ]; then
    DEMOS=/nonexistent
fi
if [ -f "$DEMOS" ]; then
    BC_ARGS="--bc_demos $DEMOS"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v4_cost_trail00
else
    BC_ARGS=""
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v2_shaped_trail00
fi
STEPS=200000
SEED=${SEED:-0}
if [ "${LIVE:-0}" = "1" ]; then
    # LIVE per-action diffusion (no cache, 2026-08-23): one 5-frame semantic
    # generation per env step at the exact continuous pose. Measured 1.35 s/step
    # at 560x336 -> 50k steps ~19 h, 200k ~75 h. Submit with:
    #   sbatch --mem=96G --time=36:00:00 ... (fine-tune)  /  --time=96:00:00 (200k)
    # SMOKE=1 -> 500-step gate run (obs sanity + timing + VRAM before real runs).
    # IMGEVERY=N (2026-09-16): obs / generated-labels panel to wandb every N rollouts (default 10; use 1 on a smoke).
    BC_ARGS="--live --goal_frame_range 15 70 --goal_min_sep 1.5 --trav_path config/traversability_v14.yaml --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_live_trail00
    # SCENE: live-train on another world (gnd_*, sitex_*); pair with CLIPS_DIR.
    if [ -n "${SCENE:-}" ] && [ "$SCENE" != "rugd_trail_00" ]; then
        OUT=/scratch/m000204-pm07b/joana/outputs/ppo_live_${SCENE}
    fi
    STEPS=200000
    if [ "${NOGATE:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --no_alpha_gate"
        OUT=${OUT}_UNGATED
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        OUT=${OUT}_SMOKE
        STEPS=500
    fi
    # LIVEBATCH: N robots sharing the pipe via batched generation (the
    # parallel-training plan). 1/unset = classic single-robot live.
    if [ -n "${LIVEBATCH:-}" ] && [ "${LIVEBATCH}" != "1" ]; then
        BC_ARGS="$BC_ARGS --live_batch $LIVEBATCH"
        OUT=${OUT}_x${LIVEBATCH}
    fi
    # RENDWIN / COVWIN (2026-09-13): fusion window for the rasters, and the
    # window the coherence gate's coverage is computed from (see
    # src/env/window.py). Unset = today's full-scene behaviour.
    if [ -n "${RENDWIN:-}" ] && [ "${RENDWIN}" != "0" ]; then
        BC_ARGS="$BC_ARGS --render_window $RENDWIN"
        OUT=${OUT}_rw${RENDWIN}
    fi
    if [ -n "${COVWIN:-}" ] && [ "${COVWIN}" != "0" ]; then
        BC_ARGS="$BC_ARGS --coverage_window $COVWIN"
        OUT=${OUT}_cw${COVWIN}
    fi
    # LIVEGPUS: K worker processes, one GPU + one pipe each, LIVEBATCH robots
    # per worker, one PPO over all of them (2026-09-12). Throughput ~K x.
    # The sbatch header asks for ONE gpu, so submit with
    #   sbatch --gres=gpu:$K --mem=$((48*K))G --cpus-per-task=$((4*K)) ...
    # The check below refuses to start if the job did not actually get K.
    if [ -n "${LIVEGPUS:-}" ] && [ "${LIVEGPUS}" != "1" ]; then
        NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        if [ "${NGPU:-0}" -lt "${LIVEGPUS}" ]; then
            echo "REFUSED: LIVEGPUS=${LIVEGPUS} but this job has ${NGPU} GPU(s)." \
                 "Submit with: sbatch --gres=gpu:${LIVEGPUS} --mem=$((48*LIVEGPUS))G" \
                 "--cpus-per-task=$((4*LIVEGPUS)) ..."
            exit 3
        fi
        echo "==> LIVEGPUS=${LIVEGPUS}: $(nvidia-smi -L | tr '\n' ';')"
        BC_ARGS="$BC_ARGS --live_gpus $LIVEGPUS"
        OUT=${OUT}_g${LIVEGPUS}
    fi
    # LIVE_DEMOS: BC-prime the live run on LIVE-rendered demos (B-prime rescue
    # for the cold-start dream-marination). NEVER pass raster/cached demo files
    # here — observation source must match training.
    if [ -n "${LIVE_DEMOS:-}" ]; then
        # LIVE_DEMOS=auto: the newest expert-recorded demos.npz at JOB START,
        # so the BC arm can be queued with --dependency=afterok:<recorder>
        # before the recorder has run (2026-09-06). Refuses if none exists.
        if [ "$LIVE_DEMOS" = "auto" ]; then
            # eval dir names over 200 chars are hashed and lose the _expertmap
            # suffix (469970 did), so match on the demos file itself
            LIVE_DEMOS=$(ls -t /scratch/m000204-pm07b/joana/outputs/eval_*/demos.npz 2>/dev/null | head -1)
            if [ -z "$LIVE_DEMOS" ]; then
                echo "REFUSED: LIVE_DEMOS=auto but no outputs/eval_*/demos.npz exists"
                exit 3
            fi
            echo "==> LIVE_DEMOS resolved to $LIVE_DEMOS"
        fi
        BC_ARGS="$BC_ARGS --bc_demos $LIVE_DEMOS"
        OUT=${OUT}_bc
    fi
    # SCENES (comma list) + ROTATE: live multi-scene rotation (2026-08-29).
    # All robots share one resident world; it swaps every ROTATE robot-steps.
    if [ -n "${SCENES:-}" ]; then
        BC_ARGS="$BC_ARGS --scenes ${SCENES//,/ } --scene_rotate ${ROTATE:-4000}"
        NSC=$(echo "$SCENES" | awk -F, '{print NF}')
        OUT=${OUT}_ms${NSC}
    fi
    # LIVECKPT: semantics checkpoint for live generation (default v10; set to
    # the v21 all-GT checkpoint for urban-scene training).
    if [ -n "${LIVECKPT:-}" ]; then
        BC_ARGS="$BC_ARGS --live_ckpt $LIVECKPT"
        OUT=${OUT}_v21obs
    fi
elif [ "${CACHE:-0}" = "1" ]; then
    # v14-DIFFUSED (2026-08-15, headline ask): the 6d recipe on the
    # corrected right-handed frame, observations from the ribbon cache (v10 +
    # reader diffused views), reward from the cache's alpha-masked diffused
    # labels scored by the v14 table. Requires cache_gen.sh to have populated
    # outputs/ribbon_cache/<scene>. SMOKE=1 -> 10k-step gate run.
    # OBS_CACHE selects which cache to train on: default = v1 path-threaded
    # cache (ribbon_cache); OBS_CACHE=ribbon_cache_spin -> spin cache v2.
    # Non-default caches get their tag appended to the output dir.
    OBS_CACHE=${OBS_CACHE:-ribbon_cache}
    # comma list = HYBRID (multiple caches loaded, sticky-family lookup)
    CACHE_PATHS=$(echo "$OBS_CACHE" | sed 's#[^,]*#/scratch/m000204-pm07b/joana/outputs/&#g')
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5 --obs_cache ${CACHE_PATHS} --trav_path config/traversability_v14.yaml --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v14diff_trail00
    if [[ "$OBS_CACHE" == *,* ]]; then
        OUT=${OUT}_hybrid
    elif [ "$OBS_CACHE" != "ribbon_cache" ]; then
        OUT=${OUT}_${OBS_CACHE#ribbon_cache_}
    fi
    STEPS=800000
    if [ "${NOGATE:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --no_alpha_gate"
        OUT=${OUT}_UNGATED
    fi
    if [ -n "${COLLTERM:-}" ]; then
        # crash termination: footprint overlap >= COLLTERM ends the episode
        # (no goal bonus, -COLLPEN). Without it, walking through a tree costs
        # ~1% of the arrival bonus and the policy correctly ignores obstacles.
        BC_ARGS="$BC_ARGS --collision_terminate_frac $COLLTERM --collision_terminate_penalty ${COLLPEN:-20}"
        OUT="${OUT}_ct${COLLTERM}"
    fi
    if [ -n "${BACKCOST:-}" ]; then
        BC_ARGS="$BC_ARGS --backward_cost $BACKCOST"
        OUT=${OUT}_bk${BACKCOST}
    fi
    # MOTIONFOOT=1: footprint scores along the commanded motion direction;
    # reversing prices as worst-case terrain (unseen ground) instead of free.
    if [ "${MOTIONFOOT:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --footprint_along_motion"
        OUT=${OUT}_mf
    fi
    # FWDONLY=1: negative velocity clamps to 0 — no reverse, by construction.
    if [ "${FWDONLY:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --forward_only"
        OUT=${OUT}_fwd
    fi
    # SPINCOST: |yaw action| tax override (default 0.05; 0.15 breaks bang-bang)
    if [ -n "${SPINCOST:-}" ]; then
        BC_ARGS="$BC_ARGS --spin_cost $SPINCOST"
        OUT=${OUT}_sp${SPINCOST}
    fi
    # SMOOTHCOST: action-change tax (targets flip-flops, not turning)
    if [ -n "${SMOOTHCOST:-}" ]; then
        BC_ARGS="$BC_ARGS --action_smooth_cost $SMOOTHCOST"
        OUT=${OUT}_sm${SMOOTHCOST}
    fi
    if [ "${NOBC:-0}" = "1" ]; then
        OUT=${OUT}_noBC
    elif [ -n "${DEMOS_FILE:-}" ]; then
        OUT=${OUT}_bc2
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        OUT=${OUT}_SMOKE
        STEPS=10000
    fi
    # Run C (2026-08-23): SCENES="s1 s2 ..." trains ONE policy over several
    # cached scenes (round-robin per episode; cache loaded per scene from the
    # same root). RAM scales with scene count -> submit with --mem=96G.
    if [ -n "${SCENES:-}" ]; then
        BC_ARGS="$BC_ARGS --scenes $SCENES"
        OUT=${OUT}_multi$(echo "$SCENES" | wc -w | tr -d ' ')
    fi
elif [ "${RUNG7B:-0}" = "1" ]; then
    # 7b = 7 with the KL leash OFF (it truncated 77% of update rounds).
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5 --target_kl 0 --scenes rugd_trail_00 rugd_park-1_00 rugd_park-2_00 rugd_trail-4_00"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v7b_multiscene4_unleashed
    STEPS=800000
elif [ "${RUNG7:-0}" = "1" ]; then
    # v7 = multi-scene: the proven 6d recipe (random goals, min_sep 1.5) over
    # four scenes at once. LRU cache eviction keeps GPU residency bounded.
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5 --scenes rugd_trail_00 rugd_park-1_00 rugd_park-2_00 rugd_trail-4_00"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v7_multiscene4
    STEPS=800000
elif [ "${RUNG6D:-0}" = "1" ]; then
    # v6d = v6c with min_sep back at v6's 1.5 (the one env difference from the
    # only config that ever learned). Paired with a 6c seed-rerun, this splits
    # "min_sep poisoned it" from "PPO seed variance".
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v6d_minsep15_trail00
    STEPS=400000
elif [ "${RUNG6C:-0}" = "1" ]; then
    # v6c = v6b curriculum (random goals 15-70, min_sep 1.0) at CONSTANT lr,
    # target_kl as the only collapse protection (v6b post-mortem: decay -> 0%).
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v6c_randomgoal_trail00
    STEPS=400000
elif [ "${RUNG6B:-0}" = "1" ]; then
    # v6b = v6 recipe + close-range exposure (min_sep 1.0) + collapse protection
    # (lr decay + target_kl). Code-side changes; only the out dir differs here.
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v6b_randomgoal_trail00
    STEPS=400000
elif [ "${RUNG6:-0}" = "1" ]; then
    # random goals (15-70) + spawns over the whole trail; harder task -> 2x steps
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v6_randomgoal_trail00
    STEPS=400000
elif [ "${RUNG5:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --spawn_max_frame 3"
    OUT=/scratch/m000204-pm07b/joana/outputs/ppo_v5_traverse_trail00
fi
if [ "$SEED" != "0" ]; then
    BC_ARGS="$BC_ARGS --seed $SEED"
    OUT="${OUT}_seed${SEED}"
fi
# CHUNK: trajectory arm (plan B) — policy outputs CHUNK action pairs
# per decision, re-observes only after the chunk executes. 1/unset = per-action.
if [ -n "${CHUNK:-}" ] && [ "${CHUNK}" != "1" ]; then
    BC_ARGS="$BC_ARGS --action_chunk $CHUNK"
    OUT="${OUT}_chunk${CHUNK}"
fi
# PROX: proximity-cost weight (geometric stay-away term; needs scene clouds
# from dump_scene_cloud.py under outputs/scene_clouds/clouds). The clearance
# fix motivated by Run A's grazing exploit.
if [ -n "${PROX:-}" ]; then
    BC_ARGS="$BC_ARGS --proximity_weight $PROX --proximity_margin ${PROXMARGIN:-0.6} --clouds_dir /scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds"
    OUT="${OUT}_prox${PROX}"
fi
# TARGETKL: override the PPO KL leash (0 = off). Multi-scene and warm-start
# runs need 0 — the leash aborts every update when the policy must shift.
if [ -n "${TARGETKL:-}" ]; then
    BC_ARGS="$BC_ARGS --target_kl $TARGETKL"
fi
# WARMSTART: continue training an existing policy (.zip). Works with any rung
# (live fine-tune of the cached champion = LIVE=1 WARMSTART=<champion ckpt>).
# STEPS then counts the ADDITIONAL steps on top of the checkpoint's counter.
# WARMSTART=<run dir>/checkpoints/LATEST resolves to that run's newest
# checkpoint WHEN THIS JOB STARTS, so a continuation can be submitted with
# --dependency=afterany:<job> and pick up the checkpoint the parent ends on.
if [[ "${WARMSTART:-}" == */LATEST ]]; then
    _wdir="${WARMSTART%/LATEST}"
    WARMSTART=$(ls -t "$_wdir"/ppo_*_steps.zip 2>/dev/null | head -1 || true)
    [[ -f "$WARMSTART" ]] || { echo "REFUSED: no checkpoint under $_wdir to continue from"; exit 2; }
    echo "==> continuing from $WARMSTART"
    # 2026-09-19: the goal-distance curriculum is written to the run's
    # curriculum_state.json but never read back, so a continuation restarted
    # at GOALDIST_START. Resume it from the run being continued unless the
    # caller set GOALDIST_START explicitly.
    _cst="$(dirname "$_wdir")/curriculum_state.json"
    if [ -z "${GOALDIST_START:-}" ] && [ -f "$_cst" ]; then
        GOALDIST_START=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['goal_dist'])" "$_cst" 2>/dev/null || true)
        [ -n "$GOALDIST_START" ] && echo "==> curriculum resumed at goal_dist=$GOALDIST_START (from $_cst)"
    fi
fi
if [ -n "${WARMSTART:-}" ]; then
    BC_ARGS="$BC_ARGS --warmstart $WARMSTART"
    # The SOURCE goes in the directory name, not just "_warm". Two arms warm
    # -started from DIFFERENT checkpoints (463168 from ppo_240704, 463169 from
    # ppo_256704, 2026-09-03) resolved to the same OUT and spent nine hours
    # writing checkpoints into one directory. They only stayed distinguishable
    # by accident -- warm runs continue their step counter, so the two were
    # offset by 16k steps -- and 463168 was hours away from overwriting
    # 463169's history file by file. LABEL already carried the source; OUT did
    # not, and OUT is what actually collides on disk.
    OUT="${OUT}_warm$(basename "$WARMSTART" .zip | sed 's/ppo_//; s/_steps//')"
fi
# RW5=1: reward design v5 (2026-08-27) — big terminal rewards (+-1000),
# timeout penalty, potential-shaped obstacle cost (5 m horizon), strong
# smoothness, forward-only, crash forfeits the bonus AND charges. Semantic
# terrain term stays at default weight (sidewalk-over-driveway requirement).
# Works in any branch incl. LIVE. Do not combine with the PROX knob.
# RW=5 is the spelling everyone reaches for and it silently did NOTHING --
# 2026-09-02 launched six 48-hour arms that way and every one got DEFAULT
# weights: goal_bonus 50 instead of 1000, collision_terminate_frac 0.0 (so no
# crash termination AT ALL), timeout 0, forward_only False, step 0.25. The
# banner is what caught it. Accept both spellings, and shout when neither is
# given.
if [ "${RW:-}" = "5" ]; then RW5=1; fi
if [ "${RW5:-0}" != "1" ] && [ "${LIVE:-0}" = "1" ]; then
    echo "==> WARNING: neither RW5=1 nor RW=5 given -- this run uses DEFAULT" \
         "reward weights (goal_bonus 50, NO crash termination, no timeout," \
         "reverse enabled). Almost certainly not what you want."
fi
if [ "${RW5:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --goal_bonus 1000 --goal_radius 0.5 --goal_weight 10 --timeout_penalty 100 --proximity_delta --proximity_weight 10 --proximity_margin 5 --clouds_dir /scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds --action_smooth_cost 5 --forward_only --collision_terminate_frac 0.35 --collision_terminate_penalty 1000"
    OUT="${OUT}_rw5"
fi
# SEMW: semantic terrain-reward weight override (2026-08-31 preference pair:
# SEMW=5 vs SEMW=0 warm-started twins answer "does the semantic term teach
# terrain values"). Default (unset) keeps 1.0.
if [ -n "${SEMW:-}" ]; then
    BC_ARGS="$BC_ARGS --semantic_weight $SEMW"
    OUT="${OUT}_semw${SEMW}"
fi
# GOALNOISE: std (meters) of Gaussian noise added to the goal-vector obs each
# step (2026-08-31 anti-odometry lever: a noisy compass forces the policy to
# use vision for reliable navigation).
if [ -n "${GOALNOISE:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_noise_std $GOALNOISE"
    OUT="${OUT}_gn${GOALNOISE}"
fi
# GOALXY: fix the goal at 'x,y' every episode — obstacle-encounter training
# (goals behind an obstacle). SPAWNMAX caps spawn frames (pair with SPAWNMIN
# for a tight before-the-obstacle spawn zone).
if [ -n "${GOALXY:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_xy $GOALXY"
    OUT="${OUT}_gxy${GOALXY/,/_}"
fi
if [ -n "${SPAWNMAX:-}" ]; then
    BC_ARGS="$BC_ARGS --spawn_max_frame $SPAWNMAX"
    OUT="${OUT}_smax${SPAWNMAX}"
fi
# J-spec knobs (2026-08-31 Jing meeting): strict traversability table, 360
# random goals at 5-10m, spawn-validity filter. TRAV overrides RW5's table
# (argparse last-wins).
if [ -n "${TRAV:-}" ]; then
    BC_ARGS="$BC_ARGS --trav_path $TRAV"
    OUT="${OUT}_trstrict"
fi
if [ "${GOAL360:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --goal_dir_360"
    OUT="${OUT}_g360"
fi
# ---- LABEL: the short name wandb shows -----------------------------------
# The output directory name is 20 tokens long, leads with `trail00` and
# `UNGATED` (true of every run since July, so they distinguish nothing), and
# carries `v21obs` even on runs using v26 -- so a wandb run list is a wall of
# identical prefixes with the differences truncated off the right-hand end.
# This builds a name out of only what actually varies between arms.
LABEL_SEM=$(basename "${LIVECKPT:-none}" .safetensors | sed 's/checkpoint-epoch-/e/')
LABEL_RUN=$(basename "$(dirname "${LIVECKPT:-/none/none}")" | sed 's/train_semantic_//; s/_campus//; s/_dino/D/')
LABEL="${LABEL_RUN}${LABEL_SEM}"
# Encode WHICH checkpoint a warm arm inherits from -- 2026-09-02 launched two
# warm arms from different priors and they got identical labels, so wandb could
# not tell them apart at all.
if [ -n "${WARMSTART:-}" ]; then
    LABEL="${LABEL}-warm$(basename "$WARMSTART" .zip | sed 's/ppo_//; s/_steps//')"
else
    LABEL="${LABEL}-cold"
fi
[ -n "${NSC:-}" ] && LABEL="${LABEL}-${NSC}sc"
[ -n "${GOALRANGE:-}" ] && LABEL="${LABEL}-g${GOALRANGE/,/to}"
[ -n "${MAXSTEPS:-}" ] && LABEL="${LABEL}-ms${MAXSTEPS}"
[ -n "${RW5SMOOTH:-}" ] && LABEL="${LABEL}-sm${RW5SMOOTH}"
[ -n "${COLLAHEAD:-}" ] && LABEL="${LABEL}-ca${COLLAHEAD}"
[ -n "${BOXMEM:-}" ] && LABEL="${LABEL}-bm${BOXMEM}"
[ "${BOXMEMALL:-0}" = "1" ] && LABEL="${LABEL}-bmall"
[ "${RASTEROBS:-0}" = "1" ] && LABEL="${LABEL}-rast"
[ "${STATICSCENE:-0}" = "1" ] && LABEL="${LABEL}-static"
[ -n "${STATICMOVERS:-}" ] && LABEL="${LABEL}-mv${STATICMOVERS//,/-}"
[ "${SPAWNHEADWALK:-0}" = "1" ] && LABEL="${LABEL}-shw"
[ -n "${GOALCASE:-}" ] && LABEL="${LABEL}-case"
[ -n "${GOALCLEAR:-}" ] && LABEL="${LABEL}-gclr${GOALCLEAR}"
[ -n "${MIRROR:-}" ] && LABEL="${LABEL}-mir${MIRROR}"
[ "${NEXTPOSE:-0}" = "1" ] && LABEL="${LABEL}-np"
[ "${LOOKAUTO:-0}" = "1" ] && LABEL="${LABEL}-la"
[ "${NEXTHEAD:-0}" = "1" ] && LABEL="${LABEL}-nh"
[ "${CRASHMOVE:-0}" = "1" ] && LABEL="${LABEL}-cm"
[ -n "${GOALNTTRIES:-}" ] && LABEL="${LABEL}-nt${GOALNTTRIES}"
[ -n "${GOALNTCONE:-}" ] && LABEL="${LABEL}-nc${GOALNTCONE}"
[ -n "${PROXGND:-}" ] && LABEL="${LABEL}-pg${PROXGND}m${PROXGNDM:-1.2}"
[ -n "${GOALSUPPORT:-}" ] && LABEL="${LABEL}-gs${GOALSUPPORT}"
# COHTERM and COLLTERM were MISSING from the label, so on 2026-09-03 three
# arms that differed only in those (base / cohterm 0.05 / collterm 0.5) all
# published to wandb under one identical name and could not be told apart in
# the run list. The label is the run's identity -- every variable an arm
# isolates has to appear in it.
[ -n "${COH:-}" ] && LABEL="${LABEL}-coh${COH}t${COHTAU:-0.4}"
[ -n "${COHTERM:-}" ] && LABEL="${LABEL}ct${COHTERM}"
[ -n "${COLLTERM:-}" ] && LABEL="${LABEL}-collt${COLLTERM}"
[ "${TIMEOUTDIST:-0}" = "1" ] && LABEL="${LABEL}-todist"
[ -n "${CRASHPEN:-}" ] && LABEL="${LABEL}-cp${CRASHPEN}"
[ -n "${GOALBONUS:-}" ] && LABEL="${LABEL}-gb${GOALBONUS}"
[ -n "${GOALWEIGHT:-}" ] && LABEL="${LABEL}-gw${GOALWEIGHT}"
[ "${CURRICULUM:-0}" = "1" ] && LABEL="${LABEL}-curR"
[ -n "${GOALDIST_START:-}" ] && LABEL="${LABEL}-curD${GOALDIST_START}"
[ -n "${GOALDISTWIN:-}" ] && LABEL="${LABEL}-win${GOALDISTWIN}"
[ -n "${GOALTURN:-}" ] && LABEL="${LABEL}-turn${GOALTURN}m${GOALTURNMIX:-0.5}b${GOALTURNBEYOND:-2.0}"
[ "${IMGFIX:-0}" = "1" ] && LABEL="${LABEL}-imgfix"
[ -n "${FRAMESTACK:-}" ] && [ "${FRAMESTACK}" != "1" ] && LABEL="${LABEL}-stack${FRAMESTACK}"
[ -n "${NSTEPS:-}" ] && LABEL="${LABEL}-ns${NSTEPS}"
[ "${PHANTOMVETO:-0}" = "1" ] && LABEL="${LABEL}-pveto"
[ -n "${HALT:-}" ] && LABEL="${LABEL}-halt${HALT}"
[ -n "${HALTSCALE:-}" ] && LABEL="${LABEL}-hs${HALTSCALE}"
[ -n "${REFUSAL:-}" ] && LABEL="${LABEL}-rb${REFUSAL}"
[ -n "${VERGECUR:-}" ] && LABEL="${LABEL}-vc${VERGECUR}"
[ -n "${HALTWRONG:-}" ] && LABEL="${LABEL}-hw${HALTWRONG}"
[ -n "${HALTWRONGRAMP:-}" ] && LABEL="${LABEL}-hwr${HALTWRONGRAMP}"
[ "${NTGOAL:-0}" = "1" ] && LABEL="${LABEL}-ntu"
[ "${GOALSTOP:-0}" = "1" ] && LABEL="${LABEL}-gstop"
[ "${STOPACT:-0}" = "1" ] && LABEL="${LABEL}-stopact"
[ "${VERGEPROG:-0}" = "1" ] && LABEL="${LABEL}-vprog"
[ -n "${DWELLCUR:-}" ] && LABEL="${LABEL}-dw${DWELLCUR}"
[ -n "${HALTEPS:-}" ] && LABEL="${LABEL}-he${HALTEPS}"
[ -n "${HALTCUR:-}" ] && LABEL="${LABEL}-hc${HALTCUR}"
[ "${HALTGATE:-0}" = "1" ] && LABEL="${LABEL}-hg"
[ "${SPEEDCOST:-0}" = "1" ] && LABEL="${LABEL}-spd"
[ -n "${REWSRC:-}" ] && LABEL="${LABEL}-r${REWSRC/map_then_generated/hyb}"
[ -n "${MAPINFL:-}" ] && LABEL="${LABEL}-mi${MAPINFL}"
[ -n "${MAPINFLCLS:-}" ] && LABEL="${LABEL}-mic${MAPINFLCLS//,/-}"
[ -n "${LABELREMAP:-}" ] && LABEL="${LABEL}-lr${LABELREMAP//[:,]/-}"
[ -n "${GOALMIX:-}" ] && LABEL="${LABEL}-gm${GOALMIX}"
[ -n "${SPAWNSUPPORT:-}" ] && LABEL="${LABEL}-ss${SPAWNSUPPORT}"
[ "${GOALMIXMAP:-0}" = "1" ] && LABEL="${LABEL}-gmm"
[ -n "${ENT:-}" ] && LABEL="${LABEL}-ent${ENT}"
# The encoder, the action chunk and the goal cone were MISSING from the label:
# on 2026-09-07 the dinov2 / resnet18 / nature / chunk10 / cone90 arms all
# published under ONE wandb name. Every variable an arm isolates goes here.
[ -n "${ENCODER:-}" ] && [ "${ENCODER}" != "nature" ] && LABEL="${LABEL}-${ENCODER}"
[ -n "${ENCGRID:-}" ] && LABEL="${LABEL}-g${ENCGRID}"
[ -n "${CHUNK:-}" ] && LABEL="${LABEL}-chunk${CHUNK}"
[ -n "${GOALCONE:-}" ] && LABEL="${LABEL}-gc${GOALCONE}"
LABEL="${LABEL}-s${SEED:-0}"
[ -n "${USER_LABEL:-}" ] && LABEL="${USER_LABEL}_${LABEL}"
echo "==> wandb label: $LABEL"

# CRASHPEN: override RW5's baked-in crash penalty of 1000. This block sits
# AFTER the RW5 block so the last --collision_terminate_penalty wins (COLLPEN
# at the top does NOT work with RW5 -- it is overridden silently).
#
# Why it matters: crash -1000 and goal bonus +1000 are symmetric, but only
# ~51% of sampled goals are reachable (goal_audit.py) while crash fires on
# 19-25% of steps. The expected value of driving at a goal is therefore
# NEGATIVE, and the highest-return behaviour is to survive without committing
# -- exactly what arm B learned (0% goal arrivals, longest episodes, lowest
# crash). Lowering the crash side tilts that back.
if [ -n "${CRASHPEN:-}" ]; then
    BC_ARGS="$BC_ARGS --collision_terminate_penalty $CRASHPEN"
    OUT="${OUT}_cp${CRASHPEN}"
fi
# GOALBONUS / GOALWEIGHT (2026-09-07): the goal bonus and the progress weight
# were only ever set inside the RW5 block. Every corner arm ran without RW5
# and therefore on the script defaults: goal_bonus 50, goal_weight 1.5,
# crash 20 -- a timeout (100) cost five crashes and two goals. Explicit knobs
# so the intended 1000 / 10 / 1000 can be set without RW5's other baggage.
if [ -n "${GOALBONUS:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_bonus $GOALBONUS"
    OUT="${OUT}_gb${GOALBONUS}"
fi
if [ -n "${GOALWEIGHT:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_weight $GOALWEIGHT"
    OUT="${OUT}_gw${GOALWEIGHT}"
fi

# PROBE=1: before training, run zero-action steps and print the MEASURED
# per-step reward for standing still, then print the policy's first rollout
# underneath it. Every argument about the policy freezing rests on whether
# moving beats doing nothing, and until 2026-09-02 that comparison was my
# arithmetic rather than a measurement.
if [ "${PROBE:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --frozen_probe"
fi

# ENT: PPO entropy bonus (default 0 = SB3 default, no bonus). ~0.01 for cold.
if [ -n "${ENT:-}" ]; then
    BC_ARGS="$BC_ARGS --ent_coef $ENT"
    OUT="${OUT}_ent${ENT}"
fi

# MAXSTEPS: episode step budget (default 60). At 0.3 m/step a 10 m goal needs
# 34 steps at FULL linear speed, so below ~57% mean speed the far half of
# GOALRANGE=5,10 is unreachable on time no matter how good the policy is.
if [ -n "${MAXSTEPS:-}" ]; then
    BC_ARGS="$BC_ARGS --max_steps $MAXSTEPS"
    OUT="${OUT}_ms${MAXSTEPS}"
fi

# GOALSUPPORT: resample goals that have no reconstruction under them.
if [ -n "${GOALSUPPORT:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_support_radius $GOALSUPPORT"
    OUT="${OUT}_gsup${GOALSUPPORT}"
fi

# BOXMEM=<n>: image path only. The near box (COLLAHEAD) is below the camera in
# the current view; read it from the newest of the last n generated frames that
# contains it whole (Joana's t-2 idea, 2026-09-04). diag/box_memory_age says
# how far back the reading came from; -1 = fell back to the far box.
if [ -n "${BOXMEM:-}" ]; then
    BC_ARGS="$BC_ARGS --collision_box_memory ${BOXMEM}"
    # BOXMEMAGG=mean (2026-09-16): average the near box over all stored frames that contain it
    [ -n "${BOXMEMAGG:-}" ] && BC_ARGS="$BC_ARGS --collision_box_memory_agg ${BOXMEMAGG}" && OUT="${OUT}_bma${BOXMEMAGG}"
    # BOXMEMALL=1 (2026-09-20, Joana: "use memory all the time"): the current frame is one vote among the stored frames
    [ "${BOXMEMALL:-0}" = "1" ] && BC_ARGS="$BC_ARGS --collision_box_memory_always" && OUT="${OUT}_bmall"
    OUT="${OUT}_bm${BOXMEM}"
fi
# RASTEROBS=1: the policy SEES the reconstruction's raster (no diffusion call)
# and the reward reads the splat labels, so the observation and the map put
# the sidewalk edge in the same place. Ablation for the refusal question
# (2026-09-06); roughly ten times cheaper per step. Pair with REWSRC=map.
if [ "${RASTEROBS:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --raster_obs"
    OUT="${OUT}_rast"
fi
# STATICSCENE=1: reconstruct every scene as STATIC -- all source frames'
# Gaussians render from any pose (dense off-walk coverage, no per-frame
# switching); moving people become trails. 2026-09-07.
if [ "${STATICSCENE:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --static_scene"
    OUT="${OUT}_static"
fi
# STATICMOVERS=12,13 (with STATICSCENE=1): person/vehicle Gaussians stay
# per-frame (no trails). Must match the label head: v31/v32 yes, v26/v30 no.
if [ -n "${STATICMOVERS:-}" ]; then
    BC_ARGS="$BC_ARGS --static_movers ${STATICMOVERS}"
    OUT="${OUT}_mv${STATICMOVERS//,/-}"
fi
# SPAWNHEADWALK=1: spawn facing the walk's direction of travel (recorded
# camera yaws are 30-150 deg off on turns and on some scenes throughout).
if [ "${SPAWNHEADWALK:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --spawn_heading_from_walk"
    OUT="${OUT}_shw"
fi
# SPAWNFRAMES="scene:10,20,30;scene2:20,40": spawn only at those frames.
# GOALCASE="corner:0.5,narrow:0.25,open:0.25": draw goals to that case mix.
# WANDB_PROJECT: separate wandb project for the new-scope arms.
if [ -n "${SPAWNFRAMES:-}" ]; then
    BC_ARGS="$BC_ARGS --spawn_frames $SPAWNFRAMES"
    OUT="${OUT}_sf"
fi
# GOALCLEAR=0.5: no non-walkable map cell within 0.5 m of the goal centre
# (set it to the arrival radius before any radius reduction). 0/unset = off.
# MIRROR=0.5: half the episodes are presented mirrored (image flipped, goal
# lateral+bearing negated, yaw command negated). The training scenes have
# 158 right-turn corner goals against 52 left-turn ones.
if [ -n "${MIRROR:-}" ]; then
    BC_ARGS="$BC_ARGS --mirror_prob ${MIRROR}"
    OUT="${OUT}_mir${MIRROR}"
fi
if [ -n "${GOALCLEAR:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_center_clear_m ${GOALCLEAR}"
    OUT="${OUT}_gclr${GOALCLEAR}"
fi
if [ -n "${GOALCASE:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_case_mix $GOALCASE"
    OUT="${OUT}_case"
fi
if [ -n "${WANDB_PROJECT:-}" ]; then
    BC_ARGS="$BC_ARGS --wandb_project $WANDB_PROJECT"
fi
# GOALRADIUS / TIMEOUTPEN: fixed values appended AFTER the base
# args (argparse keeps the last occurrence). New-scope arms: radius 1.0.
[ -n "${GOALRADIUS:-}" ] && { BC_ARGS="$BC_ARGS --goal_radius $GOALRADIUS"; OUT="${OUT}_r${GOALRADIUS}"; }
[ -n "${TIMEOUTPEN:-}" ] && BC_ARGS="$BC_ARGS --timeout_penalty $TIMEOUTPEN"
# NEXTPOSE=1: crash on the body footprint at the pose the action leads to (map).
# LOOKAUTO=1: graded box at each scene's nearest visible ground. (2026-09-07)
if [ "${NEXTPOSE:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --collision_at_next_pose"
    OUT="${OUT}_np"
fi
if [ "${LOOKAUTO:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --look_ahead_auto"
    OUT="${OUT}_la"
fi
# NEXTHEAD=1: the visible box points where the commanded turn leads.
# CRASHMOVE=1: that box crashes only while moving (throttle >= 0.15).
if [ "${NEXTHEAD:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --footprint_next_heading"
    OUT="${OUT}_nh"
fi
if [ "${CRASHMOVE:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --crash_requires_motion"
    OUT="${OUT}_cm"
fi
# GOALNTTRIES=<n>: tries of the map-direct lawn draw (default 12 delivered
# 15% lawn goals instead of 25%; 5% on AUw360 alone, 2026-09-06).
if [ -n "${GOALNTTRIES:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_nontrav_tries ${GOALNTTRIES}"
    OUT="${OUT}_nt${GOALNTTRIES}"
fi
# GOALNTCONE=<deg>: cone of the map-direct lawn draw. Lawn lies beside the
# walk; the 50-degree goal cone held no lawn cell on most frames (funnel
# 2026-09-06), so lawn goals silently became walkable ones.
if [ -n "${GOALNTCONE:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_nontrav_cone_deg ${GOALNTCONE}"
    OUT="${OUT}_nc${GOALNTCONE}"
fi
# PROXGND=<weight> [PROXGNDM=<margin m>]: potential-shaped proximity to the
# map's GRASS points (2026-09-06, Joana). With VERGEPROG it makes a standoff
# about one margin short of the lawn = a natural halt inside the verge radius.
if [ -n "${PROXGND:-}" ]; then
    BC_ARGS="$BC_ARGS --proximity_ground_weight ${PROXGND} --proximity_ground_margin ${PROXGNDM:-1.2}"
    OUT="${OUT}_pg${PROXGND}m${PROXGNDM:-1.2}"
fi
# COLLAHEAD: judge COLLISION (and crash termination) on its own footprint this
# many metres ahead, while the graded semantic score keeps looking 1.5 m out.
# Unset = both share the 1.5 m box, the behaviour of every run before
# 2026-09-02, so the control arm stays bit-identical. Campus cameras sit at
# 0.60 m, so the blind zone is FAR deeper than the 0.25 m RUGD rigs the old
# comment assumed -- read reward/collision_off_frame on wandb, it must be ~0.
if [ -n "${COLLAHEAD:-}" ]; then
    BC_ARGS="$BC_ARGS --collision_look_ahead $COLLAHEAD"
    OUT="${OUT}_ca${COLLAHEAD}"
fi

if [ -n "${GOALRANGE:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_dist_range $GOALRANGE"
    OUT="${OUT}_gr${GOALRANGE/,/-}"
fi
if [ -n "${GOALCONE:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_cone_deg $GOALCONE"
    OUT="${OUT}_gc${GOALCONE}"
fi
if [ -n "${SPAWNCLS:-}" ]; then
    BC_ARGS="$BC_ARGS --spawn_classes $SPAWNCLS"
    OUT="${OUT}_spcls"
fi
if [ -n "${IMGVOIDTERM:-}" ]; then
    BC_ARGS="$BC_ARGS --image_void_terminate_frac $IMGVOIDTERM"
    OUT="${OUT}_ivt${IMGVOIDTERM}"
fi
if [ -n "${VOIDTERM:-}" ]; then
    BC_ARGS="$BC_ARGS --void_terminate_frac $VOIDTERM"
    OUT="${OUT}_vt${VOIDTERM}"
fi
# TIMEOUTDIST=1: timeout penalty scales with remaining/initial distance.
if [ "${TIMEOUTDIST:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --timeout_distance_scaled"
    OUT="${OUT}_todist"
fi
# COH: coherence cost weight — w*max(0, COHTAU - coverage), coverage = mean
# alpha. Her design 2026-09-01: coherence is a WHOLE-FRAME property, so this
# replaces the per-pixel void terms as the main uncertainty signal.
# COHTERM: terminate below this coverage (frames with no geometry at all).
if [ -n "${COH:-}" ]; then
    BC_ARGS="$BC_ARGS --coherence_cost_weight $COH --coherence_tau ${COHTAU:-0.4}"
    OUT="${OUT}_coh${COH}t${COHTAU:-0.4}"
fi
if [ -n "${COHTERM:-}" ]; then
    BC_ARGS="$BC_ARGS --coherence_terminate_tau $COHTERM"
    OUT="${OUT}_cohterm${COHTERM}"
fi
# GATETAU: alpha-gate threshold. Measured 2026-09-01 on gnd_AUw360 — alpha is
# bimodal (24% <0.1, 48% >0.9) and every phantom crash is already gone at 0.1,
# so 0.5 was paying 37% of total reward for zero extra protection.
if [ -n "${GATETAU:-}" ]; then
    BC_ARGS="$BC_ARGS --alpha_gate_tau $GATETAU"
    OUT="${OUT}_tau${GATETAU}"
fi
if [ -n "${SEMPAL:-}" ]; then
    BC_ARGS="$BC_ARGS --sem_palette $SEMPAL"
    OUT="${OUT}_pal${SEMPAL}"
fi
if [ -n "${SPAWNJYAW:-}" ]; then
    BC_ARGS="$BC_ARGS --spawn_yaw_jitter $SPAWNJYAW"
    OUT="${OUT}_sjy${SPAWNJYAW}"
fi
if [ -n "${SPAWNJLAT:-}" ]; then
    BC_ARGS="$BC_ARGS --spawn_lat_jitter $SPAWNJLAT"
    OUT="${OUT}_sjl${SPAWNJLAT}"
fi
# HEIGHT/WIDTH: native render+observation resolution (multiples of 112).
# The policy CNN sizes itself to this — checkpoints do NOT warm-start across
# resolutions. Speed rung measured 2026-08-28: 336x224 ~2.2x faster.
if [ -n "${HEIGHT:-}" ] || [ -n "${WIDTH:-}" ]; then
    BC_ARGS="$BC_ARGS --obs_height ${HEIGHT:-336} --obs_width ${WIDTH:-560}"
    OUT="${OUT}_r${WIDTH:-560}x${HEIGHT:-336}"
fi
# RENDERH/RENDERW: render-high/observe-small (2026-08-30, her design) — the
# diffusion renders at this res (~2x slower at 560), obs["rgb"] is downsized
# to HEIGHT/WIDTH for the policy; reward/labels stay at render res.
if [ -n "${RENDERH:-}" ] || [ -n "${RENDERW:-}" ]; then
    BC_ARGS="$BC_ARGS --render_height ${RENDERH:-336} --render_width ${RENDERW:-560}"
    OUT="${OUT}_rr${RENDERW:-560}x${RENDERH:-336}"
fi
# LIVESTEPS: diffusion sampler steps for live generation (default 4; 2 = the
# measured ~2x rung, quality-gated by the drive previews).
if [ -n "${LIVESTEPS:-}" ] && [ "${LIVESTEPS}" != "4" ]; then
    BC_ARGS="$BC_ARGS --live_steps $LIVESTEPS"
    OUT="${OUT}_ls${LIVESTEPS}"
fi
# CURRICULUM=1: goal-capture radius anneals 1.0 -> --goal_radius over the
# first 100k steps (terminal-capture fix, advisor spec 2026-08-27).
# HALT: end the episode when the policy has deliberately stopped somewhere
# safe, having made progress -- and charge the same distance-scaled timeout it
# would have paid anyway. Without it a correctly-refused goal is scored exactly
# like freezing, and the robot pays the terrain cost for every step it waits.
if [ -n "${HALT:-}" ]; then
    BC_ARGS="$BC_ARGS --halt_terminate_steps ${HALT}"
    OUT="${OUT}_halt${HALT}"
fi
# HALTEPS: |throttle| below this counts as "still" for the halt terminal
# (default 0.05). A policy driving at 0.97 with unit exploration noise almost
# never samples five consecutive throttles under 0.05, so at the default the
# terminal is off the exploration path and its value is never learned.
if [ -n "${HALTEPS:-}" ]; then
    BC_ARGS="$BC_ARGS --halt_throttle_eps ${HALTEPS}"
    OUT="${OUT}_he${HALTEPS}"
fi
# HALTGATE=1 (with HALTCUR): halting is UNAVAILABLE until recent wins pass 50%,
# then enabled at the HALTCUR price and notched as usual. W cold'' froze at
# halt price 1.0 (halt_wrong 0.56, 2026-09-04): a policy that cannot reach
# goals halts to dodge the crash, whatever the halt costs.
if [ "${HALTGATE:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --halt_gate"
    OUT="${OUT}_hg"
fi
# HALTCUR=<start>: the halt is earned -- pays <start> x timeout (1.0 = full)
# and notches down to HALTSCALE as recent wins pass 50%. Evals use HALTSCALE
# (the final value), like the radius.
if [ -n "${HALTCUR:-}" ]; then
    BC_ARGS="$BC_ARGS --halt_scale_start ${HALTCUR}"
    OUT="${OUT}_hc${HALTCUR}"
fi
# SPEEDCOST=1: terrain cost x |throttle|. Driving onto bad ground costs,
# standing still facing it does not -- a dense signal for slowing down
# before any terminal is involved. Logged as reward/speed_refund.
if [ "${SPEEDCOST:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --terrain_speed_scaled"
    OUT="${OUT}_spd"
fi
# REWSRC=map: the reward reads the scene cloud's top-down label grid instead
# of the generated semantic image (map_then_generated = map with image
# fallback where the footprint is mostly off the reconstruction). Needs the
# clouds_dir RW5 already passes.
if [ -n "${REWSRC:-}" ]; then
    BC_ARGS="$BC_ARGS --reward_source ${REWSRC}"
    OUT="${OUT}_r${REWSRC/map_then_generated/hyb}"
fi
# MAPINFL: how far non-traversable map cells grow (m). Default 0.2 = the robot
# half-width; 0.1 still trips the 0.35 crash test on a one-cell wall (3 of 6
# footprint rows) while narrowing walkways by only 10 cm a side.
if [ -n "${MAPINFL:-}" ]; then
    BC_ARGS="$BC_ARGS --map_inflate_m ${MAPINFL}"
    OUT="${OUT}_mi${MAPINFL}"
fi
# MAPINFLCLS: which classes inflate, e.g. 10,11,13 = obstacle/vegetation/vehicle
# only; grass and the other ground classes keep the edge SAM3 saw. Empty =
# every non-traversable class (the 2026-09-03/04 arms).
if [ -n "${MAPINFLCLS:-}" ]; then
    BC_ARGS="$BC_ARGS --map_inflate_classes ${MAPINFLCLS}"
    OUT="${OUT}_mic${MAPINFLCLS//,/-}"
fi
# LABELREMAP: remap GENERATED label ids before the reward reads them, e.g.
# 12:0 = phantom "person" -> void (2026-09-07, v26 e10 on static renders).
if [ -n "${LABELREMAP:-}" ]; then
    BC_ARGS="$BC_ARGS --label_remap ${LABELREMAP}"
    OUT="${OUT}_lr${LABELREMAP//[:,]/-}"
fi
# GOALMIX=<p>: P(goal on traversable ground by the map). Needs a map (clouds
# dir), which RW5 passes. Evals of such arms must carry the same GOALMIX.
if [ -n "${GOALMIX:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_traversable_mix ${GOALMIX}"
    OUT="${OUT}_gm${GOALMIX}"
fi
# GOALMIXMAP=1: the non-traversable share of GOALMIX is drawn straight from
# map cells (grass/rough/water) in the window and cone, instead of hoping the
# walk sampler lands on lawn (it found none in 12 draws on short windows and
# on two scenes -> gen 3 trained at 87-96% traversable, 2026-09-04).
if [ "${GOALMIXMAP:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --goal_mix_map_draw"
    OUT="${OUT}_gmm"
fi
# SPAWNSUPPORT=<n>: redraw a spawn whose crash box (this arm's: near or far)
# is already >= the crash threshold on the map, up to n times. Needs a map.
# Evals of such arms must carry the same SPAWNSUPPORT.
if [ -n "${SPAWNSUPPORT:-}" ]; then
    BC_ARGS="$BC_ARGS --spawn_support_tries ${SPAWNSUPPORT}"
    OUT="${OUT}_ss${SPAWNSUPPORT}"
fi
# REFUSAL=<bonus>: a halt on a NON-traversable goal within REFUSALDIST m (default
# 2.0) of it earns <bonus> instead of the halt price. Gen 3b (2026-09-05, 10 h):
# no arm distinguished lawn goals from hard goals because a correct refusal
# earned nothing. Evals of such arms must carry the same REFUSAL/REFUSALDIST.
if [ -n "${REFUSAL:-}" ]; then
    BC_ARGS="$BC_ARGS --refusal_bonus ${REFUSAL}"
    OUT="${OUT}_rb${REFUSAL}"
    if [ -n "${REFUSALDIST:-}" ]; then
        BC_ARGS="$BC_ARGS --refusal_dist_m ${REFUSALDIST}"
        OUT="${OUT}_rd${REFUSALDIST}"
    fi
    if [ -n "${REFUSALVERGE:-}" ]; then
        BC_ARGS="$BC_ARGS --refusal_verge_m ${REFUSALVERGE}"
        OUT="${OUT}_rv${REFUSALVERGE}"
    fi
    # VERGECUR=<start>: verge radius earned down from <start> m to REFUSALVERGE
    # (default 1.5) as halts at the verge pass 50% of lawn-goal episodes.
    if [ -n "${VERGECUR:-}" ]; then
        BC_ARGS="$BC_ARGS --verge_start ${VERGECUR}"
        OUT="${OUT}_vc${VERGECUR}"
    fi
fi
# STOPACT=1: third action output, > 0 = halt now (cold start only). VERGEPROG=1:
# on lawn-goal episodes the progress reward pulls toward the verge nearest the
# goal instead of onto the grass (2026-09-06).
if [ "${STOPACT:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --stop_action"
    OUT="${OUT}_stopact"
fi
if [ "${VERGEPROG:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --lawn_progress_to_verge"
    OUT="${OUT}_vprog"
fi
# DWELLCUR=<start>: Joana's dwell curriculum -- still steps required inside the
# verge zone to refuse, from <start> (0 = arrival) up to 3 (the halt), each
# stage earned at 50% of lawn-goal episodes (2026-09-06).
if [ -n "${DWELLCUR:-}" ]; then
    BC_ARGS="$BC_ARGS --refusal_dwell_start ${DWELLCUR}"
    OUT="${OUT}_dw${DWELLCUR}"
fi
# GOALSTOP=1: a goal is reached only after the robot is STILL for the halt steps
# inside its radius; passing through pays nothing. Teaches the stop that refusal
# needs on every successful episode (Joana, 2026-09-05).
if [ "${GOALSTOP:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --goal_requires_stop"
    OUT="${OUT}_gstop"
fi
# NTGOAL=1: a non-traversable goal cannot be reached -- entering its radius ends
# nothing and pays nothing (radius 1.0 at the start of the curriculum let the
# robot collect lawn goals from the pavement, 2026-09-05). Only a halt earns there.
if [ "${NTGOAL:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --nontrav_goal_unreachable"
    OUT="${OUT}_ntu"
fi
# HALTWRONG=<pen>: flat penalty for a halt on a TRAVERSABLE goal, the mirror of
# REFUSAL. Makes the halt decision about the ground at the goal, not about
# dodging the crash price (2026-09-05).
if [ -n "${HALTWRONG:-}" ]; then
    BC_ARGS="$BC_ARGS --halt_wrong_penalty ${HALTWRONG}"
    OUT="${OUT}_hw${HALTWRONG}"
    # HALTWRONGRAMP=<steps>: penalty ramps 0 -> HALTWRONG over these env steps
    # (2026-09-06: with a stop action the full penalty kills halting before
    # the verge bonus is ever seen). Evals use the full HALTWRONG.
    if [ -n "${HALTWRONGRAMP:-}" ]; then
        BC_ARGS="$BC_ARGS --halt_wrong_ramp ${HALTWRONGRAMP}"
        OUT="${OUT}_hwr${HALTWRONGRAMP}"
    fi
fi
# HALTSCALE: what a HALTED episode pays, as a multiple of the distance-scaled
# timeout. 1 (default) = the tie; 0.3 = refusing an unreachable goal is the
# cheapest terminal after reaching it.
if [ -n "${HALTSCALE:-}" ]; then
    BC_ARGS="$BC_ARGS --halt_penalty_scale ${HALTSCALE}"
    OUT="${OUT}_hs${HALTSCALE}"
fi
# CKPTFREQ: checkpoint every N env calls (x4 envs = steps). Default 2000
# (= 8000 steps, ~4.5 h). 500 = every 2000 steps, for arms that must be
# evaluated the same night they were launched.
# CKPTSTEPS (2026-09-17): checkpoint every N ENV STEPS, converted to calls here
# from the number of robots (LIVEGPUS x LIVEBATCH). Run 1 (489497) was launched
# with CKPTFREQ=10000 read as steps; that is 10000 x 16 robots = 160k steps, so
# 14 hours and 86k steps ended with no checkpoint at all. Use this knob.
if [ -n "${CKPTSTEPS:-}" ]; then
    _nrob=$(( ${LIVEGPUS:-1} * ${LIVEBATCH:-4} ))
    CKPTFREQ=$(( CKPTSTEPS / _nrob )); [ "$CKPTFREQ" -lt 1 ] && CKPTFREQ=1
    echo "==> CKPTSTEPS=${CKPTSTEPS} with ${_nrob} robots -> checkpoint every ${CKPTFREQ} calls"
fi
if [ -n "${CKPTFREQ:-}" ]; then
    BC_ARGS="$BC_ARGS --ckpt_every_calls ${CKPTFREQ}"
fi
# NSTEPS (2026-09-17): PPO rollout length per robot; the update batch is
# NSTEPS x LIVEGPUS x LIVEBATCH env steps. 4 GPUs x 4 robots x 128 = 2048;
# a 2-GPU arm needs NSTEPS=256 to update on the same 2048.
if [ -n "${NSTEPS:-}" ]; then
    BC_ARGS="$BC_ARGS --n_steps ${NSTEPS}"
    OUT="${OUT}_ns${NSTEPS}"
fi
# PHANTOMVETO=1 (2026-09-19): a crash-level reading on ground the fused map
# knows to be walkable does not end the episode (the generated no-go label is a
# phantom); the step still pays the contact cost. Eval 09-19: 8/8 crashes of
# the old A checkpoint were phantoms (person on pavement).
if [ "${PHANTOMVETO:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --phantom_veto"
    OUT="${OUT}_pveto"
fi
if [ -n "${GOALDISTWIN:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_dist_window ${GOALDISTWIN}"
    OUT="${OUT}_win${GOALDISTWIN}"
fi
# FRAMESTACK=K (2026-09-22, Jing's "multiple images" arm): the observation is the last K rendered
# views stacked on the channel axis; every frame goes through the frozen backbone and their features
# concatenate. K=1 (or unset) is the single-frame policy.
# MAXSTEPSPERM (2026-09-23): the episode budget grows with the goal band instead of
# sitting at 90 while the curriculum climbs to 10 m. 12 steps/m is the measured
# cruise of the one policy that reaches goals (14 keeps a 1.15x margin at 10 m).
if [ -n "${MAXSTEPSPERM:-}" ] && [ "${MAXSTEPSPERM}" != "0" ]; then
    BC_ARGS="$BC_ARGS --max_steps_per_m ${MAXSTEPSPERM} --max_steps_base ${MAXSTEPSBASE:-40}"
    OUT="${OUT}_ms${MAXSTEPSBASE:-40}p${MAXSTEPSPERM}"
fi
if [ -n "${FRAMESTACK:-}" ] && [ "${FRAMESTACK}" != "1" ]; then
    BC_ARGS="$BC_ARGS --obs_frame_stack ${FRAMESTACK}"
    OUT="${OUT}_stack${FRAMESTACK}"
    # FRAMESTRIDE=N: steps between the stacked views (1 = consecutive, ~0.25 m apart)
    if [ -n "${FRAMESTRIDE:-}" ] && [ "${FRAMESTRIDE}" != "1" ]; then
        BC_ARGS="$BC_ARGS --obs_frame_stride ${FRAMESTRIDE}"
        OUT="${OUT}s${FRAMESTRIDE}"
    fi
fi
# IMGFIX=1 (2026-09-22): policy built/patched with normalize_images=False (SB3 divided the image
# by 255 before the extractor divided it again: every DINO arm before this date was BLIND, actions
# changed by <0.003 between a real frame and a black one). With WARMSTART the image head is zeroed
# so step 0 behaves like the warm-start and the image pathway grows from zero.
if [ "${IMGFIX:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --image_norm_fix"
    # the head reset is decided from the parent checkpoint inside train_ppo_real: a pre-fix parent
    # gets a zeroed image head, a parent that already sees keeps it. RESETHEAD=1 forces the reset.
    [ "${RESETHEAD:-0}" = "1" ] && BC_ARGS="$BC_ARGS --reset_image_head"
    OUT="${OUT}_imgfix"
fi
# GOALTURN=<deg> [GOALTURNMIX=<0-1>] (2026-09-21): corner goals -- a share of episodes must draw a
# path goal beyond a bend of the walk of at least GOALTURN degrees (falls back on straight walks)
if [ -n "${GOALTURN:-}" ]; then
    # GOALTURNBEYOND=<m> (default 2.0): the goal lies at least this much walk past the bend
    BC_ARGS="$BC_ARGS --goal_turn_deg ${GOALTURN} --goal_turn_mix ${GOALTURNMIX:-0.5} --goal_turn_beyond_m ${GOALTURNBEYOND:-2.0}"
    # GOALTURNFROM=<m>: ramp the mix from 0 at this band distance to GOALTURNMIX at GOALDIST
    [ -n "${GOALTURNFROM:-}" ] && BC_ARGS="$BC_ARGS --goal_turn_from ${GOALTURNFROM}" && OUT="${OUT}f${GOALTURNFROM}"
    OUT="${OUT}_turn${GOALTURN}m${GOALTURNMIX:-0.5}b${GOALTURNBEYOND:-2.0}"
fi
if [ "${CURRICULUM:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --goal_radius_start 1.0"
    OUT="${OUT}_cur"
fi
# GOALDIST: fixed spawn->goal distance in meters (advisor spec).
if [ -n "${GOALDIST:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_dist $GOALDIST"
    OUT="${OUT}_gd${GOALDIST}"
fi
# REWSCALE: uniform reward multiplier (0.01 recommended if value_loss stays
# ~1e5 — ratios preserved, critic targets tamed).
if [ -n "${REWSCALE:-}" ]; then
    BC_ARGS="$BC_ARGS --reward_scale $REWSCALE"
    OUT="${OUT}_rs${REWSCALE}"
fi
# RW5SMOOTH: override RW5's baked-in smoothness 5 (this block sits AFTER the
# RW5 block so the last --action_smooth_cost wins) — failure-mode-1 arm.
# COLLTERM on the LIVE path. The original COLLTERM block lives inside the
# CACHED (obs_cache) branch, so with LIVE=1 it was silently ignored: job 463226
# was launched as the collision-threshold arm on 2026-09-03, applied nothing,
# and wrote to the SAME output directory as the baseline 463164 -- two jobs
# clobbering one another's checkpoints. Third instance of the RW=5 shape.
# Sits AFTER the RW5 block so it overrides RW5's 0.35, and sets ONLY the
# fraction -- the penalty stays with RW5/CRASHPEN.
if [ -n "${COLLTERM:-}" ]; then
    BC_ARGS="$BC_ARGS --collision_terminate_frac $COLLTERM"
    case "$OUT" in
        *_ct${COLLTERM}) ;;
        *) OUT="${OUT}_ct${COLLTERM}" ;;
    esac
fi

if [ -n "${RW5SMOOTH:-}" ]; then
    BC_ARGS="$BC_ARGS --action_smooth_cost $RW5SMOOTH"
    OUT="${OUT}_sm${RW5SMOOTH}"
fi
# GOALJLAT (2026-09-17): lateral jitter of path goals in metres, map-checked.
if [ -n "${GOALJLAT:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_lat_jitter_m $GOALJLAT"
    OUT="${OUT}_gjl${GOALJLAT}"
fi
# GOALMINSCORE (2026-09-16): goals only on map cells with traversability >= this
# (0.5 = pavement classes only; mulch beds at 0.35 are excluded).
if [ -n "${GOALMINSCORE:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_min_score $GOALMINSCORE"
    OUT="${OUT}_gms${GOALMINSCORE}"
fi
# PROXGNDCLS (2026-09-16): classes the ground-proximity cost keeps away from
# (default 3,4,5 = grass, rough, water; campus: 2,3,4,5 adds trail = mulch beds).
if [ -n "${PROXGNDCLS:-}" ]; then
    BC_ARGS="$BC_ARGS --proximity_ground_classes $PROXGNDCLS"
    OUT="${OUT}_pgc${PROXGNDCLS//,/-}"
fi
# PROXM (2026-09-16): override RW5's baked-in obstacle proximity margin of 5 m
# (half the campus is within 5 m of a wall). Sits after the RW5 block so it wins.
if [ -n "${PROXM:-}" ]; then
    BC_ARGS="$BC_ARGS --proximity_margin $PROXM"
    OUT="${OUT}_pm${PROXM}"
fi
# ENCODER: policy visual encoder (nature | dinov2 | resnet18) — frozen
# pretrained backbones for the advisor's encoder ablation (2026-08-30).
if [ -n "${ENCODER:-}" ] && [ "${ENCODER}" != "nature" ]; then
    BC_ARGS="$BC_ARGS --encoder $ENCODER"
    OUT="${OUT}_${ENCODER}"
fi
# ENCGRID: region grid of the frozen ViT tokens, e.g. 6x8 (default 3x4).
if [ -n "${ENCGRID:-}" ]; then
    BC_ARGS="$BC_ARGS --encoder_grid $ENCGRID"
    OUT="${OUT}_g${ENCGRID}"
fi
# CURTHRESH / CURWIN / CURMINEP / CURNOTCH (2026-09-25): the distance-curriculum
# gate. Defaults reproduce the old behaviour (0.5 / 100 / 50 / 0.5), under which
# every arm climbed to 10 m in 40-60k steps at coin-flip success and plateaued.
# A slow gate is CURTHRESH=0.75 CURMINEP=100 CURNOTCH=0.25.
[ -n "${CURTHRESH:-}" ] && { BC_ARGS="$BC_ARGS --cur_threshold $CURTHRESH"; LABEL="${LABEL}-ct${CURTHRESH}"; }
[ -n "${CURWIN:-}" ]    && BC_ARGS="$BC_ARGS --cur_window $CURWIN"
[ -n "${CURMINEP:-}" ]  && { BC_ARGS="$BC_ARGS --cur_min_episodes $CURMINEP"; LABEL="${LABEL}-cm${CURMINEP}"; }
[ -n "${CURNOTCH:-}" ]  && { BC_ARGS="$BC_ARGS --cur_notch $CURNOTCH"; LABEL="${LABEL}-cn${CURNOTCH}"; }
[ "${CURPERSCENE:-0}" = "1" ] && { BC_ARGS="$BC_ARGS --cur_per_scene"; LABEL="${LABEL}-cps"; }
# DISAGREE=<w> (2026-09-25, Jing): per-step cost when generated labels and the map disagree
[ -n "${DISAGREE:-}" ] && { BC_ARGS="$BC_ARGS --disagree_cost $DISAGREE"; LABEL="${LABEL}-dis${DISAGREE}"; }
[ -n "${DISAGREEFRAC:-}" ] && BC_ARGS="$BC_ARGS --disagree_frac $DISAGREEFRAC"
# RGBALIGN=<w> [RGBALIGNTAU=<tau>] (2026-09-25, Jing): cost when the generated frame drifts from the Gaussian raster
[ -n "${RGBALIGN:-}" ] && { BC_ARGS="$BC_ARGS --rgb_align_cost $RGBALIGN --rgb_align_tau ${RGBALIGNTAU:-0.8}"; LABEL="${LABEL}-ra${RGBALIGN}t${RGBALIGNTAU:-0.8}"; }
# GOALDIST_START: distance curriculum — goals start here, grow to GOALDIST
# as the policy earns wins (the bootstrap E was missing).
if [ -n "${GOALDIST_START:-}" ]; then
    BC_ARGS="$BC_ARGS --goal_dist_start $GOALDIST_START"
    OUT="${OUT}_gds${GOALDIST_START}"
fi
# SPAWNMIN: keep spawns out of the weak-recon clip edges (confabulation
# zone, measured 2026-08-29).
if [ -n "${SPAWNMIN:-}" ]; then
    BC_ARGS="$BC_ARGS --spawn_min $SPAWNMIN"
    OUT="${OUT}_smin${SPAWNMIN}"
fi
# STEPS_OVERRIDE: extend any rung without a new branch (v6d was still climbing
# at its 400k cap -> 800k continuation). Output dir gets the step count.
if [ -n "${STEPS_OVERRIDE:-}" ]; then
    STEPS=$STEPS_OVERRIDE
    OUT="${OUT}_${STEPS}"
fi
echo "==> rung: ${BC_ARGS:-pure-shaped}  steps: $STEPS  out: $OUT"

# --- launch ledger. One append-only file for the whole project, plus a copy
# beside the checkpoints. Written NOW, before the pipeline load, so a job that
# dies during startup still leaves a record of what it was.
LEDGER="/scratch/m000204-pm07b/joana/launch_ledger.log"
# Run-dir names grew past the filesystem's 255-char limit on 2026-09-04
# (three arms died at mkdir). Keep the head of the name readable and hash
# the rest; the full knob list is in launch.txt and the ledger anyway.
_base=$(basename "$OUT")
if [ ${#_base} -gt 200 ]; then
    _h=$(printf '%s' "$_base" | md5sum | cut -c1-8)
    OUT="$(dirname "$OUT")/${_base:0:170}_h${_h}"
    echo "==> run dir name shortened from ${#_base} chars: $(basename "$OUT")"
fi
# 2026-09-19: the run dir is a function of the knobs, so a RELAUNCH with the
# same knobs (491641 after 491267) landed in the same folder and overwrote the
# first run's checkpoints and tensorboard events -- "old A" evals were in fact
# the new run at 50k. Every job now gets its own folder and wandb name.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    OUT="${OUT}_j${SLURM_JOB_ID}"
    LABEL="${LABEL:+${LABEL}-j${SLURM_JOB_ID}}"
fi
mkdir -p "$OUT" 2>/dev/null || true
LEDGER_ENTRY="$(cat <<EOF
=== job ${SLURM_JOB_ID:-nojob}   $(date -Is)   node ${SLURMD_NODENAME:-?}
    git    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    out    $OUT
    label  ${LABEL:-}
    steps  $STEPS
    rung   ${BC_ARGS:-pure-shaped}
    knobs  $LAUNCH_ENV
EOF
)"
printf '%s\n\n' "$LEDGER_ENTRY" >> "$LEDGER" 2>/dev/null || true
printf '%s\n'   "$LEDGER_ENTRY" >  "$OUT/launch.txt" 2>/dev/null || true
echo "==> ledger: $LEDGER"

# PREFLIGHT=1 (2026-09-26): print the exact argument line this job would run and stop.
# tests/preflight_arms.py feeds it to the real argparse, so an "invalid choice" or
# "unrecognized argument" is caught on the login node instead of costing a node and
# a queue slot (all ten fixed arms died in 6 s on 26 Sept: reward_source=either was
# implemented in the env but missing from the parser's choices).
if [ "${PREFLIGHT:-0}" = "1" ]; then
    printf 'PREFLIGHT_ARGS %s\n' "--scene ${SCENE:-rugd_trail_00} --clips_dir ${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips} --poses_dir /scratch/m000204-pm07b/joana/outputs/poses --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels --total_steps $STEPS --output_dir $OUT $BC_ARGS"
    exit 0
fi
python scripts/train_ppo_real.py \
    --scene "${SCENE:-rugd_trail_00}" \
    --clips_dir "${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}" \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
    --total_steps $STEPS \
    --output_dir "$OUT" \
    $BC_ARGS \
    --run_label "$LABEL" \
    ${IMGEVERY:+--wandb_images_every "$IMGEVERY"} \
    --use_wandb

echo "==> done: $OUT (rollout.mp4 + curves)"
