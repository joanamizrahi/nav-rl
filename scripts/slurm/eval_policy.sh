#!/usr/bin/env bash
#SBATCH --job-name=eval-policy
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# 01:00:00 loses the run to the PIPELINE LOAD. A live eval spends 25-40 min
# loading the diffusion pipeline before the first episode, so a 1 h wall leaves
# ~20 min for 20 episodes and job 463931 (2026-09-03) timed out having written
# videos but no metrics -- while its BLIND partner, which happened to load
# faster, completed. A pair where only one half survives is worse than no pair.
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-eval-policy-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-eval-policy-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
set -euo pipefail
# PORTABLE (2026-09-08): the same launcher must run on Marlowe (module
# system + the shared conda env) and on a rented cloud GPU (neither).
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
[ -d /users/jmizrahi/.conda/envs/neoverse/bin ] && export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH

export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.zip via --export=CKPT=...}
# 2026-09-03: jobs 464831/464832 ran the cloud RASTERIZER instead of the world
# model because the launch was `env $EV ...` from a shell where EV was empty
# (an export does not survive a new login). They finished in one minute,
# reported mean_coverage None and raw label ids -- and read like a result.
# A policy trained on live observations is evaluated on live observations,
# and a live eval always names the semantics checkpoint the policy trained on.
if [[ "$CKPT" == *ppo_live_* && "${LIVE:-0}" != "1" ]]; then
    echo "REFUSED: $(basename "$(dirname "$(dirname "$CKPT")")") trained on live observations but LIVE is not 1."
    echo "         Did the EV export survive this shell? Check: echo \"\$EV\" | wc -w   (expect about 30)"
    exit 2
fi
if [[ "${LIVE:-0}" == "1" && -z "${LIVECKPT:-}" ]]; then
    echo "REFUSED: LIVE=1 without LIVECKPT. Name the semantics checkpoint the policy trained on."
    exit 2
fi
# eval dir carries the run name (ppo_v4_... vs ppo_v5_...) so evals never collide
RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
# 2026-09-02: eval_<run>_<ckpt><flags> blew past the 255-byte filename limit
# (OSError 36) and every eval of the new arms would have failed the same way --
# their names are longer still. Drop the tokens that are identical on EVERY run
# and therefore distinguish nothing, then hard-cap as a backstop. The full name
# stays in the log line above, so nothing is lost.
RUN_SHORT=$(echo "$RUN_NAME" \
    | sed -e 's/ppo_live_trail00_UNGATED_//' -e 's/ppo_live_trail00_//' \
          -e 's/_r336x224//' -e 's/_rr560x336//' -e 's/_smin10//' \
          -e 's/_spcls//' -e 's/_gc50//' -e 's/_sjy20//' -e 's/_sjl0\.4//' \
          -e 's/_pal4//' -e 's/v21obs_//' -e 's/_trstrict//' -e 's/_semw5//' \
          -e 's/_rs0\.01//' -e 's/_g360//')
RUN_SHORT=${RUN_SHORT:0:110}
# an expert recording keeps its tag at the FRONT so the hashed name still says so
[ -n "${EXPERT:-}" ] && RUN_SHORT="expert${EXPERT}_${RUN_SHORT}"
EXTRA_ARGS=()
# Accept the training spellings too. train_ppo_real.sh uses SPAWNMIN/SPAWNMAX,
# this script used SPAWN_MIN/SPAWN_MAX, and a mismatched name is silently
# ignored -- the same trap as MAXSTEPS vs MAX_STEPS (2026-09-03).
SPAWN_MAX="${SPAWN_MAX:-${SPAWNMAX:-}}"
SPAWN_MIN="${SPAWN_MIN:-${SPAWNMIN:-}}"
if [[ "${SPAWN_MAX:-}" != "" ]]; then
    EXTRA_ARGS+=(--spawn_max_frame "$SPAWN_MAX")   # match the training rung

fi
# SPAWN_MIN pins the near end of the spawn range. Essential with GOAL_XY: the
# goal is a fixed world point, so without a floor the spawns spread back to the
# start of the path and most episodes run out of steps before reaching it.
# Kinematics + reverse clamp must mirror the training rung or the policy is
# evaluated outside the action model it learned (2026-09-01).
if [[ "${STEP:-}" != "" ]]; then
    EXTRA_ARGS+=(--step_size_m "$STEP")
fi
if [[ "${YAW:-}" != "" ]]; then
    EXTRA_ARGS+=(--yaw_step_rad "$YAW")
fi
if [[ "${FWDONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--forward_only)
fi
if [[ "${SPAWN_MIN:-}" != "" ]]; then
    EXTRA_ARGS+=(--spawn_min_frame "$SPAWN_MIN")
fi
OUT_SUFFIX=""
# 2026-09-28: the spawn window was NOT in the output name, so two tests on one scene with
# the same goal (quad2_04_ped 29-36 and quad2_04_pednear 39-43) overwrote each other
# overnight. Name the window. (Must come AFTER OUT_SUFFIX is initialised: set -u.)
if [[ "${GOAL_FRAME:-}" != "" ]]; then
    EXTRA_ARGS+=(--goal_frame "$GOAL_FRAME")       # generalization: goal the policy never trained on
    OUT_SUFFIX="_goal${GOAL_FRAME}"
fi
# (the goal line ASSIGNS the suffix, so the window must be appended after it -- the 11:00 version
# appended before and was silently overwritten; today's ped/pednear cells collided again)
[[ "${SPAWN_MAX:-}" != "" ]] && OUT_SUFFIX="${OUT_SUFFIX}_sp${SPAWN_MIN:-0}-${SPAWN_MAX}"
# GOAL360 / GOALRANGE / GOALCONE / GOALFRAMERANGE: sample goals the way TRAINING
# does instead of pinning one goal_xy. Eval could previously only use a fixed
# goal or the default goal frame, so it never reproduced the distribution the
# policy learned, and with a wide spawn range d_start varied uncontrolled.
EXTRA_ARGS+=(--sem_palette "${SEMPAL:-4}")
if [ "${GOAL360:-0}" = "1" ]; then
    EXTRA_ARGS+=(--goal_dir_360)
fi
if [ -n "${GOALRANGE:-}" ]; then
    EXTRA_ARGS+=(--goal_dist_range "${GOALRANGE}")
fi
if [ -n "${GOALCONE:-}" ]; then
    EXTRA_ARGS+=(--goal_cone_deg "${GOALCONE}")
fi
if [ -n "${GOALFRAMERANGE:-}" ]; then
    EXTRA_ARGS+=(--goal_frame_range "${GOALFRAMERANGE}")
    OUT_SUFFIX="${OUT_SUFFIX}_gfr${GOALFRAMERANGE/,/-}"
fi
# BLIND=1: zero the rgb observation (goal vector intact) — the does-the-policy-
# actually-look ablation. Videos still record the real frames.
if [[ "${BLIND:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--blind)
    OUT_SUFFIX="${OUT_SUFFIX}_blind"
fi
# HEIGHT/WIDTH: obs resolution — MUST match the checkpoint's training res
# (the CNN is size-locked; RW5-v2 arms train at 336x224).
if [[ -n "${HEIGHT:-}" || -n "${WIDTH:-}" ]]; then
    EXTRA_ARGS+=(--obs_height "${HEIGHT:-336}" --obs_width "${WIDTH:-560}")
    OUT_SUFFIX="${OUT_SUFFIX}_r${WIDTH:-560}x${HEIGHT:-336}"
fi
if [[ -n "${RENDERH:-}" || -n "${RENDERW:-}" ]]; then
    EXTRA_ARGS+=(--render_height "${RENDERH:-336}" --render_width "${RENDERW:-560}")
    OUT_SUFFIX="${OUT_SUFFIX}_rr${RENDERW:-560}x${RENDERH:-336}"
fi
[ -n "${SPAWNJYAW:-}" ] && EXTRA_ARGS+=(--spawn_yaw_jitter "$SPAWNJYAW")
[ -n "${SPAWNJLAT:-}" ] && EXTRA_ARGS+=(--spawn_lat_jitter "$SPAWNJLAT")
[ -n "${VOIDTERM:-}" ]  && EXTRA_ARGS+=(--void_terminate_frac "$VOIDTERM")
[ -n "${HALT:-}" ] && EXTRA_ARGS+=(--halt_terminate_steps "$HALT")
[ -n "${HALTSCALE:-}" ] && EXTRA_ARGS+=(--halt_penalty_scale "$HALTSCALE")
[ -n "${REFUSAL:-}" ] && { EXTRA_ARGS+=(--refusal_bonus "$REFUSAL"); OUT_SUFFIX="${OUT_SUFFIX}_rb${REFUSAL}"; }
[ -n "${REFUSALDIST:-}" ] && { EXTRA_ARGS+=(--refusal_dist_m "$REFUSALDIST"); OUT_SUFFIX="${OUT_SUFFIX}_rd${REFUSALDIST}"; }
[ -n "${REFUSALVERGE:-}" ] && { EXTRA_ARGS+=(--refusal_verge_m "$REFUSALVERGE"); OUT_SUFFIX="${OUT_SUFFIX}_rv${REFUSALVERGE}"; }
[ -n "${HALTWRONG:-}" ] && { EXTRA_ARGS+=(--halt_wrong_penalty "$HALTWRONG"); OUT_SUFFIX="${OUT_SUFFIX}_hw${HALTWRONG}"; }
[ "${NTGOAL:-0}" = "1" ] && { EXTRA_ARGS+=(--nontrav_goal_unreachable); OUT_SUFFIX="${OUT_SUFFIX}_ntu"; }
[ "${GOALSTOP:-0}" = "1" ] && { EXTRA_ARGS+=(--goal_requires_stop); OUT_SUFFIX="${OUT_SUFFIX}_gstop"; }
[ "${STOPACT:-0}" = "1" ] && { EXTRA_ARGS+=(--stop_action); OUT_SUFFIX="${OUT_SUFFIX}_stopact"; }
[ "${VERGEPROG:-0}" = "1" ] && { EXTRA_ARGS+=(--lawn_progress_to_verge); OUT_SUFFIX="${OUT_SUFFIX}_vprog"; }
[ -n "${HALTEPS:-}" ] && EXTRA_ARGS+=(--halt_throttle_eps "$HALTEPS")
[ "${SPEEDCOST:-0}" = "1" ] && EXTRA_ARGS+=(--terrain_speed_scaled)
[ -n "${REWSRC:-}" ] && { EXTRA_ARGS+=(--reward_source "$REWSRC"); OUT_SUFFIX="${OUT_SUFFIX}_r${REWSRC/map_then_generated/hyb}"; }
[ -n "${MAPINFL:-}" ] && { EXTRA_ARGS+=(--map_inflate_m "$MAPINFL"); OUT_SUFFIX="${OUT_SUFFIX}_mi${MAPINFL}"; }
[ -n "${MAPINFLCLS:-}" ] && { EXTRA_ARGS+=(--map_inflate_classes "$MAPINFLCLS"); OUT_SUFFIX="${OUT_SUFFIX}_mic${MAPINFLCLS//,/-}"; }
[ -n "${GOALMIX:-}" ] && { EXTRA_ARGS+=(--goal_traversable_mix "$GOALMIX"); OUT_SUFFIX="${OUT_SUFFIX}_gm${GOALMIX}"; }
[ -n "${SPAWNSUPPORT:-}" ] && { EXTRA_ARGS+=(--spawn_support_tries "$SPAWNSUPPORT"); OUT_SUFFIX="${OUT_SUFFIX}_ss${SPAWNSUPPORT}"; }
[ "${GOALMIXMAP:-0}" = "1" ] && { EXTRA_ARGS+=(--goal_mix_map_draw); OUT_SUFFIX="${OUT_SUFFIX}_gmm"; }
[ -n "${GOALNTTRIES:-}" ] && { EXTRA_ARGS+=(--goal_nontrav_tries "$GOALNTTRIES"); OUT_SUFFIX="${OUT_SUFFIX}_nt${GOALNTTRIES}"; }
[ -n "${GOALNTCONE:-}" ] && { EXTRA_ARGS+=(--goal_nontrav_cone_deg "$GOALNTCONE"); OUT_SUFFIX="${OUT_SUFFIX}_nc${GOALNTCONE}"; }
[ -n "${COLLAHEAD:-}" ] && { EXTRA_ARGS+=(--collision_look_ahead "$COLLAHEAD"); OUT_SUFFIX="${OUT_SUFFIX}_ca${COLLAHEAD}"; }
[ -n "${BOXMEM:-}" ] && { EXTRA_ARGS+=(--collision_box_memory "$BOXMEM"); OUT_SUFFIX="${OUT_SUFFIX}_bm${BOXMEM}"; }
[ "${RASTEROBS:-0}" = "1" ] && { EXTRA_ARGS+=(--raster_obs); OUT_SUFFIX="${OUT_SUFFIX}_rast"; }
[ "${STATICSCENE:-0}" = "1" ] && { EXTRA_ARGS+=(--static_scene); OUT_SUFFIX="${OUT_SUFFIX}_static"; }
[ -n "${STATICMOVERS:-}" ] && { EXTRA_ARGS+=(--static_movers "$STATICMOVERS"); OUT_SUFFIX="${OUT_SUFFIX}_mv${STATICMOVERS//,/-}"; }
[ "${SPAWNHEADWALK:-0}" = "1" ] && { EXTRA_ARGS+=(--spawn_heading_from_walk); OUT_SUFFIX="${OUT_SUFFIX}_shw"; }
[ -n "${SPAWNFRAMES:-}" ] && { EXTRA_ARGS+=(--spawn_frames "$SPAWNFRAMES"); OUT_SUFFIX="${OUT_SUFFIX}_sf"; }
[ -n "${GOALCASE:-}" ] && { EXTRA_ARGS+=(--goal_case_mix "$GOALCASE"); OUT_SUFFIX="${OUT_SUFFIX}_case"; }
[ -n "${LABELREMAP:-}" ] && { EXTRA_ARGS+=(--label_remap "$LABELREMAP"); OUT_SUFFIX="${OUT_SUFFIX}_lr${LABELREMAP//[:,]/-}"; }
[ -n "${GOALCLEAR:-}" ] && { EXTRA_ARGS+=(--goal_center_clear_m "$GOALCLEAR"); OUT_SUFFIX="${OUT_SUFFIX}_gclr${GOALCLEAR}"; }
[ -n "${MIRROR:-}" ] && { EXTRA_ARGS+=(--mirror_prob "$MIRROR"); OUT_SUFFIX="${OUT_SUFFIX}_mir${MIRROR}"; }
[ "${NEXTPOSE:-0}" = "1" ] && { EXTRA_ARGS+=(--collision_at_next_pose); OUT_SUFFIX="${OUT_SUFFIX}_np"; }
[ "${LOOKAUTO:-0}" = "1" ] && { EXTRA_ARGS+=(--look_ahead_auto); OUT_SUFFIX="${OUT_SUFFIX}_la"; }
[ "${NEXTHEAD:-0}" = "1" ] && { EXTRA_ARGS+=(--footprint_next_heading); OUT_SUFFIX="${OUT_SUFFIX}_nh"; }
[ "${CRASHMOVE:-0}" = "1" ] && { EXTRA_ARGS+=(--crash_requires_motion); OUT_SUFFIX="${OUT_SUFFIX}_cm"; }
# EXPERT=map: the scripted map-reading driver replaces the policy and records
# demos to <eval dir>/demos.npz (feed to a training arm with LIVE_DEMOS=...).
[ -n "${EXPERT:-}" ] && { EXTRA_ARGS+=(--expert "$EXPERT"); OUT_SUFFIX="${OUT_SUFFIX}_expert${EXPERT}"; }
# EXPERT=path (2026-11-09): scripted good/bad paths scored by the training reward.
# PATHPAIRS start/goal pairs per scene (3), PATHKINDS (good,straight,obstacle,offwalk),
# PATHFRAMES=1 saves the generated frame, labels and opacity of every step.
[ -n "${PATHPAIRS:-}" ] && EXTRA_ARGS+=(--path_pairs "$PATHPAIRS")
[ -n "${PATHKINDS:-}" ] && EXTRA_ARGS+=(--path_kinds "$PATHKINDS")
[ "${PATHFRAMES:-0}" = "1" ] && EXTRA_ARGS+=(--path_frames)
# PATHPLANFILE=<json>: drive these hand-picked episodes (path_plan.json format) instead of choosing pairs.
[ -n "${PATHPLANFILE:-}" ] && EXTRA_ARGS+=(--path_plan_file "$PATHPLANFILE")
# PATHPLANONLY=1: write the plan + label grids and stop, to look at the paths before driving them.
[ "${PATHPLANONLY:-0}" = "1" ] && EXTRA_ARGS+=(--path_plan_only)
[ "${KEEPFAILED:-0}" = "1" ] && EXTRA_ARGS+=(--keep_failed_demos)
[ -n "${MEMSNAP:-}" ] && { EXTRA_ARGS+=(--memory_snap_every "$MEMSNAP"); OUT_SUFFIX="${OUT_SUFFIX}_memsnap${MEMSNAP}"; }
# CRASHANY=1 (2026-09-27): score a footprint collision as a crash at ANY throttle. Training
# used crash_requires_motion (throttle >= 0.15), which the strict memory_dino policy learned
# to exploit by creeping on lawn at throttle 0.04 (gnd_AU_180: 68 collision steps per
# episode, outcome TIMEOUT). The rule is adopted from env_config, so it is forced off here.
if [[ "${CRASHANY:-0}" == "1" ]]; then
    FORCEKEYS="${FORCEKEYS:+$FORCEKEYS,}crash_requires_motion"
    OUT_SUFFIX="${OUT_SUFFIX}_crashany"
fi
[ -n "${FORCEKEYS:-}" ] && { EXTRA_ARGS+=(--force_env_keys "$FORCEKEYS"); [[ "${CRASHANY:-0}" == "1" && "$FORCEKEYS" == "crash_requires_motion" ]] || OUT_SUFFIX="${OUT_SUFFIX}_forced"; }
# MAXSTEPSPERM (2026-09-23): checkpoints trained before the budget knob existed have
# no max_steps_per_m in env_config, so eval would keep the fixed 90 steps -- which
# buys 6.3 m at the measured 0.070 m/step cruise, short of the 7.5 m band those arms
# actually reached. Passing it here sizes their episodes the same way training now
# does, so old and new arms are judged on the same task.
if [ -n "${MAXSTEPSPERM:-}" ] && [ "${MAXSTEPSPERM}" != "0" ]; then
    EXTRA_ARGS+=(--max_steps_per_m "$MAXSTEPSPERM" --max_steps_base "${MAXSTEPSBASE:-40}")
    OUT_SUFFIX="${OUT_SUFFIX}_b${MAXSTEPSBASE:-40}p${MAXSTEPSPERM}"
fi
# TESTDIST (2026-09-24): a FIXED-GOAL scene is the same length for every arm, but
# the budget above is sized from the ARM's curriculum reach. On the 8.1 m far corner
# that gave chunk5 (reach 4 m) 104 actions and memDINO_warm (reach 7.5 m) 160 on the
# identical test, and chunk5's 104 cannot cover 8.1 m at any throttle it has shown --
# its TIMEOUT measured the budget, not the policy. Set from the scene table, which
# carries the distance measured with scene_corner_scan.py.
[ -n "${TESTDIST:-}" ] && EXTRA_ARGS+=(--test_goal_dist "$TESTDIST")
[ "${NOADOPT:-0}" = "1" ] && { EXTRA_ARGS+=(--no_adopt); OUT_SUFFIX="${OUT_SUFFIX}_noadopt"; }
# STOCH=1: sampled actions, as in training rollouts. Separate output dir so the
# deterministic cell is not overwritten (2026-09-25).
[ "${STOCH:-0}" = "1" ] && { EXTRA_ARGS+=(--stochastic); OUT_SUFFIX="${OUT_SUFFIX}_stoch"; }

# Reward weights, so the reported `return=` is on TRAINING's scale. They do not
# change a frozen policy's actions, but a return computed with goal_bonus 50
# against training's 1000 is not comparable to anything.
[ -n "${GOALWEIGHT:-}" ]  && EXTRA_ARGS+=(--goal_weight "$GOALWEIGHT")
[ -n "${COLLWEIGHT:-}" ]  && EXTRA_ARGS+=(--collision_weight "$COLLWEIGHT")
[ -n "${GOALBONUS:-}" ]   && EXTRA_ARGS+=(--goal_bonus "$GOALBONUS")
[ -n "${TIMEOUTPEN:-}" ]  && EXTRA_ARGS+=(--timeout_penalty "$TIMEOUTPEN")
[ "${TIMEOUTDIST:-0}" = "1" ] && EXTRA_ARGS+=(--timeout_distance_scaled)
[ -n "${PROXW:-}" ]       && EXTRA_ARGS+=(--proximity_weight "$PROXW")
[ -n "${PROXMARGIN:-}" ]  && EXTRA_ARGS+=(--proximity_margin "$PROXMARGIN")
[ "${PROXDELTA:-0}" = "1" ] && EXTRA_ARGS+=(--proximity_delta)
[ -n "${PROXGND:-}" ] && { EXTRA_ARGS+=(--proximity_ground_weight "$PROXGND" --proximity_ground_margin "${PROXGNDM:-1.2}"); OUT_SUFFIX="${OUT_SUFFIX}_pg${PROXGND}m${PROXGNDM:-1.2}"; }
[ -n "${VOIDCOST:-}" ]    && EXTRA_ARGS+=(--void_cost "$VOIDCOST")
[ -n "${STEPCOST:-}" ]    && EXTRA_ARGS+=(--step_cost "$STEPCOST")

# GOALSUPPORT: reject goals with no reconstruction under them, exactly as
# training does. It existed in check_rewards.sh but NOT here, so every eval
# before 2026-09-03 12:45 sampled goals with the support check OFF (default
# 0.0) while training ran 0.6 -- about 14.5% of eval goals had no world under
# them and could never be reached.
[ -n "${GOALSUPPORT:-}" ] && EXTRA_ARGS+=(--goal_support_radius "$GOALSUPPORT")
[ -n "${GOALSUPPORTFRAC:-}" ] && EXTRA_ARGS+=(--goal_support_min_frac "$GOALSUPPORTFRAC")

# The SEMANTICS MODEL goes in the output name. Two evals of the same policy
# checkpoint under different world models (v21 vs v26) resolved to the same
# directory and the second silently overwrote the first (2026-09-03, jobs
# 464345 -> 464601). Identity = the run directory the checkpoint came from.
if [[ -n "${LIVECKPT:-}" ]]; then
    _semid="$(basename "$(dirname "$LIVECKPT")" | sed 's/^train_semantic_//')_$(basename "$LIVECKPT" .safetensors | sed 's/checkpoint-//; s/epoch-/e/')"
    OUT_SUFFIX="${OUT_SUFFIX}_${_semid}"
fi

# COH / COHTAU / COHTERM: coherence termination, matching training. Without
# these the eval never ends an episode for leaving the reconstructed world,
# while training does -- which is exactly the train/eval mismatch that made
# five evals report ZERO crashes on 2026-09-03.
if [[ -n "${COH:-}" || -n "${COHTERM:-}" ]]; then
    EXTRA_ARGS+=(--coherence_cost_weight "${COH:-0}"
                 --coherence_tau "${COHTAU:-0.4}"
                 --coherence_terminate_tau "${COHTERM:-0}"
                 --coherence_terminate_penalty "${COHPEN:-100}")
    OUT_SUFFIX="${OUT_SUFFIX}_coh${COH:-0}t${COHTAU:-0.4}ct${COHTERM:-0}"
fi
# 2026-09-03 22:20: a radius-1.0 diagnostic OVERWROTE the radius-0.75 eval of
# the same checkpoint because none of these knobs reached the directory name.
# Every knob that changes the task now does.
[ -n "${GOALRADIUS:-}" ] && { EXTRA_ARGS+=(--goal_radius "$GOALRADIUS"); OUT_SUFFIX="${OUT_SUFFIX}_r${GOALRADIUS}"; }
[ -n "${HALT:-}" ] && OUT_SUFFIX="${OUT_SUFFIX}_halt${HALT}"
[ -n "${HALTEPS:-}" ] && OUT_SUFFIX="${OUT_SUFFIX}_he${HALTEPS}"
[ -n "${HALTSCALE:-}" ] && OUT_SUFFIX="${OUT_SUFFIX}_hs${HALTSCALE}"
[ "${SPEEDCOST:-0}" = "1" ] && OUT_SUFFIX="${OUT_SUFFIX}_spd"
[ -n "${COLLPEN:-}" ] && [ "${COLLPEN}" != "1000" ] && OUT_SUFFIX="${OUT_SUFFIX}_cp${COLLPEN}"
[ -n "${SEMW:-}" ]       && EXTRA_ARGS+=(--semantic_weight "$SEMW")
[ -n "${REWSCALE:-}" ]   && EXTRA_ARGS+=(--reward_scale "$REWSCALE")
[ -n "${SMOOTHCOST:-}" ] && EXTRA_ARGS+=(--action_smooth_cost "$SMOOTHCOST")
if [[ "${COLLTERM:-}" != "" ]]; then
    EXTRA_ARGS+=(--collision_terminate_frac "$COLLTERM"
                 --collision_terminate_penalty "${COLLPEN:-20}")
    OUT_SUFFIX="${OUT_SUFFIX}_ct${COLLTERM}"
fi
# Accept MAXSTEPS as well as MAX_STEPS. train_ppo_real.sh spells it MAXSTEPS,
# this script spelled it MAX_STEPS, and nothing warned -- so on 2026-09-03 six
# evals were launched with MAXSTEPS=90 and silently ran the DEFAULT 60 against
# training's 90. The ancestor's headline result (19/20 TIMEOUT) was measured on
# a two-thirds episode budget; several of those episodes had closed 50-64% of
# the distance when the clock ran out.
MAX_STEPS="${MAX_STEPS:-${MAXSTEPS:-}}"
if [[ "${MAX_STEPS:-}" != "" ]]; then
    EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
    OUT_SUFFIX="${OUT_SUFFIX}_s${MAX_STEPS}"
fi
if [[ "${GOAL_XY:-}" != "" ]]; then
    # designed obstacle test: GOAL_XY="x,y" pins the goal off-trajectory
    # (e.g. behind a tree). GOAL_FRAME still caps the spawn range.
    EXTRA_ARGS+=(--goal_xy "$GOAL_XY")
    OUT_SUFFIX="${OUT_SUFFIX}_gxy${GOAL_XY/,/_}"
fi
SCENE=${SCENE:-rugd_trail_00}
# 2026-09-23: twenty evals each burned ~10 min of a GPU before dying inside the
# reconstructor because CLIPS_DIR still pointed at rugd_clips while the campus
# scenes live in campus_clips. Check the file the job will ask for, up front.
# 2026-09-23 (Joana found it): a FIXED goal frame with spawns drawn past that frame
# puts the goal BEHIND a forward-only robot, which then has to spin ~164 deg before
# it can move. Every arm failed quad2_00 episode 1 for this reason and it dragged the
# corner scores down all day. The old, correct runs were named "spawn12-22_goal48".
if [ -n "${GOAL_FRAME:-}" ] && [ -z "${SPAWN_MAX:-}" ] && [ -z "${SPAWNMAX:-}" ]; then
    echo "REFUSED: GOAL_FRAME=$GOAL_FRAME without SPAWN_MAX."
    echo "         Spawns after frame $GOAL_FRAME put the goal behind a forward-only robot."
    echo "         Set SPAWN_MAX to a frame comfortably before the goal, e.g."
    echo "           SPAWN_MAX=$(( GOAL_FRAME > 12 ? GOAL_FRAME - 8 : 1 ))"
    echo "         (pass SPAWN_MAX=$GOAL_FRAME deliberately if you really want U-turns in the test)"
    exit 1
fi
_CLIP="${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}/${SCENE}.mp4"
if [ ! -f "$_CLIP" ]; then
    echo "REFUSED: no clip at $_CLIP"
    echo "         campus scenes need CLIPS_DIR=/scratch/m000204-pm07b/joana/data/campus_clips"
    _alt=/scratch/m000204-pm07b/joana/data/campus_clips/${SCENE}.mp4
    [ -f "$_alt" ] && echo "         (found it at $_alt)"
    exit 1
fi
if [ "$SCENE" != "rugd_trail_00" ]; then
    OUT_SUFFIX="${OUT_SUFFIX}_${SCENE}"    # zero-shot evals get their own dir
fi
# Cached-obs policies must be evaluated on the SAME cache they trained on.
#   OBS_CACHE=ribbon_cache | ribbon_cache_spin ...   NOGATE=1 for ungated runs
LABELS_DIR=/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels
if [[ "${OBS_CACHE:-}" != "" ]]; then
    CACHE_PATHS=$(echo "$OBS_CACHE" | sed 's#[^,]*#/scratch/m000204-pm07b/joana/outputs/&#g')
    EXTRA_ARGS+=(--obs_cache "${CACHE_PATHS}"
                 --trav_path config/traversability_v14.yaml)
    LABELS_DIR=/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14
    if [[ "$OBS_CACHE" == *,* ]]; then
        OUT_SUFFIX="${OUT_SUFFIX}_hybrid"
    else
        OUT_SUFFIX="${OUT_SUFFIX}_${OBS_CACHE#ribbon_cache}"
    fi
fi
[[ "${NOGATE:-0}" == "1" ]] && EXTRA_ARGS+=(--no_alpha_gate)
if [[ -n "${CHUNK:-}" && "${CHUNK}" != "1" ]]; then
    EXTRA_ARGS+=(--action_chunk "$CHUNK")
    OUT_SUFFIX="${OUT_SUFFIX}_chunk${CHUNK}"
fi
# MOTIONFOOT=1: match training's motion-direction footprint rule
if [[ "${MOTIONFOOT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--footprint_along_motion)
    OUT_SUFFIX="${OUT_SUFFIX}_mf"
fi
# FWDONLY=1: match training's forward-only clamp
if [[ "${FWDONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--forward_only)
    OUT_SUFFIX="${OUT_SUFFIX}_fwd"
fi
# LIVE=1: serve live diffusion observations (evals of --live-trained policies).
# ~1.4 s/step: submit with  sbatch --mem=96G --time=03:00:00
if [[ "${LIVE:-0}" == "1" ]]; then
    # TRAV: the traversability table the eval scores with. MUST match the
    # checkpoint's TRAINING table or the eval measures a different task —
    # the default v14 table scores grass 0.75 (walkable!), while the J-arms
    # train with grass 0.0. Default kept for backwards compatibility.
    # 2026-09-20: the default v14 table (grass 0.75, vegetation 0.2) scored a whole day of
    # evals of walkway-trained arms: 18 steps on a lawn without a crash. The eval now ADOPTS
    # the training table from env_config.json when TRAV is not given; TRAV forces a table.
    EXTRA_ARGS+=(--live)
    if [[ -n "${TRAV:-}" ]]; then
        EXTRA_ARGS+=(--trav_path "$TRAV")
        echo "==> eval traversability table FORCED: $TRAV"
    else
        echo "==> eval traversability table: adopted from the checkpoint's env_config.json"
    fi
    if [[ -n "${LIVECKPT:-}" ]]; then
        EXTRA_ARGS+=(--live_ckpt "$LIVECKPT")
    fi
    LABELS_DIR=/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14
    OUT_SUFFIX="${OUT_SUFFIX}_live"
fi
if [[ "${VIDEOS:-}" != "" ]]; then EXTRA_ARGS+=(--videos "$VIDEOS"); fi
echo "==> eval: $RUN_NAME / $(basename "$CKPT") scene=$SCENE spawn_max=${SPAWN_MAX:-default} goal=${GOAL_FRAME:-train(30)}"
# Eval dir names hit the 255-char limit on 2026-09-04 (blind half of the T2
# rerun, 467571): keep the head readable, hash the tail. Same rule as the
# training launcher; the full knob list is in the .out anyway.
OUT_DIR=/scratch/m000204-pm07b/joana/outputs/eval_${RUN_SHORT}_$(basename "$CKPT" .zip)${OUT_SUFFIX}
# EVALTAG (2026-09-19, Joana: "the folders are not organized"): with EVALTAG=A_corners the
# eval lands in outputs/evals/<EVALTAG>/<scene>_<checkpoint>[_goalNN|_gfrA-B]/ instead of one
# long name among hundreds; the training-run knobs are still recorded in metrics.json.
if [ -n "${EVALTAG:-}" ]; then
    _ck=$(basename "$CKPT" .zip | sed 's/ppo_//; s/_steps//')
    _g="${GOAL_FRAME:+_goal${GOAL_FRAME}}${GOALFRAMERANGE:+_gfr${GOALFRAMERANGE/,/-}}"
    OUT_DIR=/scratch/m000204-pm07b/joana/outputs/evals/${EVALTAG}/${SCENE}_${_ck}${_g}
    mkdir -p "$(dirname "$OUT_DIR")"
fi
_base=$(basename "$OUT_DIR")
if [ ${#_base} -gt 200 ]; then
    _h=$(printf '%s' "$_base" | md5sum | cut -c1-8)
    OUT_DIR="$(dirname "$OUT_DIR")/${_base:0:170}_h${_h}"
    echo "==> eval dir name shortened from ${#_base} chars: $(basename "$OUT_DIR")"
fi
echo "==> eval dir: $OUT_DIR"
python scripts/eval_policy.py \
    --checkpoint "$CKPT" \
    --scene "$SCENE" --episodes "${EPISODES:-20}" \
    --clips_dir "${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}" \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir $LABELS_DIR \
    --out_dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"
echo "==> eval done"
