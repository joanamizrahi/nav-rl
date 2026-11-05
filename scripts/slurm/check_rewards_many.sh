#!/usr/bin/env bash
#SBATCH --job-name=check-rew-many
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-check-rew-many-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-check-rew-many-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
# check_rewards.py over a LIST of scenes in one job (2026-09-15, campus scenes):
# the same python call as check_rewards.sh, plus --clips_dir so it reads the
# campus clips folder instead of the RUGD default, and --static_scene by default.
#
#   SCENES="quad1_02 quad1_03 ..." CLIPS_DIR=/scratch/.../data/campus_clips \
#       SURVEY=1 WALK=path sbatch scripts/slurm/check_rewards_many.sh
# Knobs (same meaning as check_rewards.sh): SURVEY, WALK, SEMPAL, LIVECKPT,
# STATICSCENE (default 1), STATICMOVERS, OUTDIR, EPISODES, STEPS,
# SPAWNFRAME (2026-09-16: start the path walk at this recorded frame; with
# EPISODES=1 STEPS=80 SPAWNFRAME=1 the survey video is the whole recorded walk),
# RENDWIN / COVWIN (fusion windows, set them to the training values, 21/21),
# BOXMEM=N (frame memory for the near box; adds the MEMORY column to the survey).
# Per-scene output (camera geometry block, ladder, survey mp4) is separated by
# "=== SCENE <name> ===" lines so certify_scenes.py can read this log.
set -uo pipefail
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
[ -d /users/jmizrahi/.conda/envs/neoverse/bin ] && export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl

: "${SCENES:?set SCENES=\"s1 s2 ...\"}"
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/campus_clips}
LIVECKPT=${LIVECKPT:-/scratch/m000204-pm07b/joana/runs/train_semantic_v26_campus/checkpoint-epoch-10.safetensors}
OUTROOT=${OUTDIR:-/scratch/m000204-pm07b/joana/outputs/check_rew_campus}
mkdir -p "$OUTROOT"
nvidia-smi -L || true

for s in $SCENES; do
    echo "=== SCENE $s ==="
    EXTRA=()
    [ "${STATICSCENE:-1}" = "1" ] && EXTRA+=(--static_scene)
    [ -n "${STATICMOVERS:-}" ] && EXTRA+=(--static_movers "${STATICMOVERS}")
    python scripts/check_rewards.py \
        --scene "$s" \
        --clips_dir "$CLIPS_DIR" \
        --trav_path "${TRAV:-config/traversability_v14_walkway.yaml}" \
        --episodes "${EPISODES:-2}" \
        --steps "${STEPS:-6}" \
        --sem_palette "${SEMPAL:-4}" \
        --walk "${WALK:-path}" \
        --collision_look_ahead "${COLLAHEAD:-1.0}" \
        ${SURVEY:+--survey_video} \
        ${SPAWNFRAME:+--spawn_frame "$SPAWNFRAME"} \
        ${RENDWIN:+--render_window "$RENDWIN"} \
        ${BOXMEM:+--collision_box_memory "$BOXMEM"} \
        ${BOXMEMAGG:+--collision_box_memory_agg "$BOXMEMAGG"} \
        ${ANCHOR:+--anchor_box_every "$ANCHOR"} \
        ${ANCHORDIST:+--anchor_dist "$ANCHORDIST"} \
        ${PATHOVERLAY:+--path_overlay} \
        ${COVWIN:+--coverage_window "$COVWIN"} \
        --live_ckpt "$LIVECKPT" \
        --out_dir "$OUTROOT/$s" --out "$OUTROOT/$s" \
        "${EXTRA[@]}" || echo "!!! FAILED: $s"
done
echo "==> check_rewards_many done"
