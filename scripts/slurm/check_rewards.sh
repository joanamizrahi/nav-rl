#!/usr/bin/env bash
#SBATCH --job-name=check-rew
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# 2026-09-02: was 01:30:00 / 96G for a job that finishes in 12-15 min and runs
# the same live backend train_ppo_real does on 48G. Over-requesting keeps these
# out of BACKFILL -- the scheduler will slot a small job into a gap ahead of
# higher-priority work only if it provably fits before the next reservation, so
# a 6x time request is the difference between running in a gap and waiting a
# day. Raise --time on the command line for a genuinely long sweep.
# 00:30:00 was SHORTER THAN THE PIPELINE LOAD. Job 463192 (2026-09-03) spent
# its entire allocation on "[LiveDiffusedBackend] loading SEMANTIC pipeline"
# and died having measured nothing -- the load alone takes 30-60 min. It was
# cut to 30 min on 2026-09-02 to win backfill priority; that traded the whole
# result for a queue slot.
#SBATCH --time=2:00:00
# 2026-09-02: was excluding TEN nodes (n04,n06,n13,n14,n17,n21,n24,n26,n30,n31)
# while every other launcher excludes four. Six of seven pending jobs were
# check-rew, barred from n14 and n26 -- nodes this account's training arms were
# running on at that moment. Aligned to the common list. This job asks for 96G
# (more than train's 48G), so if it starts failing on n06/n21/n30/n31, memory
# is why and the exclusion goes back.
#SBATCH --exclude=n04,n13,n17,n18,n24
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-check-rew-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-check-rew-%j.err

# Reward mechanism check: GATE / VOID / IMGVOID measured on real renders.
# Renders ONCE, then sweeps the alpha-gate threshold over identical pixels.
# Knobs: SCENE, TRAV, EPISODES, STEPS, SWEEP, GATETAU, GOALXY, SPAWNFRAME, TAG,
#        LADDER (collision look-ahead distances to test for visibility),
#        COLLAHEAD (second footprint drawn in MAGENTA on every panel),
#        SEMPAL (semantic palette; v26 is 4, v21 was 1 — this was NOT passed
#        until 2026-09-02, so every earlier run decoded a palette-4 model with
#        the palette-1 table and its class ids, and therefore its collision and
#        traversability numbers, were wrong. Geometry — fy, cy, pixel counts,
#        blind zone — is unaffected, so the scene certification still stands),
#        SURVEY (per-scene RGB | diffused sem | splat sem mp4).

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
S=${SCRATCH_ROOT:-/scratch/m000204-pm07b/joana}
cd $S/nav-rl

EXTRA=()
# CLIPS_DIR / POSES_DIR / LABELS_DIR / CLOUDS_DIR (2026-11-05): campus scenes by default; the python defaults are the RUGD paths
EXTRA+=(--clips_dir "${CLIPS_DIR:-$S/data/campus_clips}" --poses_dir "${POSES_DIR:-$S/outputs/poses}" --labels_dir "${LABELS_DIR:-$S/NeoVerse/outputs/sam3_labels_v14}" --clouds_dir "${CLOUDS_DIR:-$S/outputs/scene_clouds/clouds}")
if [ -n "${GOALXY:-}" ]; then
    EXTRA+=(--goal_xy "${GOALXY}")
fi
if [ -n "${SPAWNFRAME:-}" ]; then
    EXTRA+=(--spawn_frame "${SPAWNFRAME}")
fi
if [ -n "${TAG:-}" ]; then
    EXTRA+=(--tag "${TAG}")
fi
# REPLAY=<metrics.json> [REPLAYEPS=5,10] [OUTDIR=...]: re-render an eval's recorded
# poses with the reward map projected into the raster and the diffused frame.
if [ -n "${REPLAY:-}" ]; then
    EXTRA+=(--replay_metrics "${REPLAY}" --replay_episodes "${REPLAYEPS:-}")
    # HISTMODES=cold,walk,same,single: step-0 frame under each history seeding
    [ -n "${HISTMODES:-}" ] && EXTRA+=(--hist_modes "${HISTMODES}")
fi
# COVSWEEP=1 [COVFRAMES=10,20,...] [COVYAWS=0,15,...]: raster alpha vs yaw offset
# from the walk direction at walk frames (no diffusion; minutes).
# STATICSCENE=1: static reconstruction (all frames' Gaussians always rendered)
# OUTDIR: output directory for ANY mode (survey videos of different
# reconstruction modes would otherwise overwrite each other).
[ -n "${OUTDIR:-}" ] && EXTRA+=(--out_dir "${OUTDIR}" --out "${OUTDIR}")
[ "${STATICSCENE:-0}" = "1" ] && EXTRA+=(--static_scene)
# BOXMEM=5 BOXMEMAGG=mean (2026-11-05): the memory column on every step panel, as in training
[ -n "${BOXMEM:-}" ] && EXTRA+=(--collision_box_memory "${BOXMEM}" --collision_box_memory_agg "${BOXMEMAGG:-mean}")
# STATICMOVERS=12,13: person/vehicle Gaussians stay per-frame in static mode
[ -n "${STATICMOVERS:-}" ] && EXTRA+=(--static_movers "${STATICMOVERS}")
if [ "${COVSWEEP:-0}" = "1" ]; then
    EXTRA+=(--cov_sweep)
    # '=' form: a yaw list starting with '-' (e.g. -90,-60,...) is otherwise
    # read by argparse as an option name (470307/470308 died on it)
    [ -n "${COVFRAMES:-}" ] && EXTRA+=(--cov_frames="${COVFRAMES}")
    [ -n "${COVYAWS:-}" ] && EXTRA+=(--cov_yaws="${COVYAWS}")
    [ "${COVPICS:-0}" = "1" ] && EXTRA+=(--cov_pictures)
fi

python scripts/check_rewards.py \
    --scene "${SCENE:-gnd_AUw360}" \
    --trav_path "${TRAV:-config/traversability_v14_walkway.yaml}" \
    --episodes "${EPISODES:-4}" \
    --steps "${STEPS:-8}" \
    --sweep "${SWEEP:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}" \
    --gate_tau "${GATETAU:-0.5}" \
    --sem_palette "${SEMPAL:-4}" \
    --cone_deg "${GOALCONE:-50}" \
    --dist_range "${GOALRANGE:-5,10}" \
    --goal_support_radius "${GOALSUPPORT:-0}" \
    --walk "${WALK:-straight}" \
    --ladder_dists "${LADDER:-0.8,1.0,1.2,1.5,1.8,2.1,2.4}" \
    --collision_look_ahead "${COLLAHEAD:-1.0}" \
    ${SURVEY:+--survey_video} \
    --live_ckpt "${LIVECKPT:-$S/runs/train_semantic_v35b_p6_ppl5/checkpoint-epoch-3.safetensors}" --model_path "$S/NeoVerse/models" \
    "${EXTRA[@]}"

echo "==> reward check done"
