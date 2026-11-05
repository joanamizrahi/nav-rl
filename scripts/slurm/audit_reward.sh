#!/usr/bin/env bash
#SBATCH --job-name=audit-reward
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-audit-reward-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-audit-reward-%j.err
set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.zip via --export=CKPT=...}
RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
EXTRA_ARGS=()
[ -n "${SPAWN_MAX:-}" ] && EXTRA_ARGS+=(--spawn_max_frame "$SPAWN_MAX")
[ -n "${GOAL_FRAME:-}" ] && EXTRA_ARGS+=(--goal_frame "$GOAL_FRAME")
echo "==> audit: $RUN_NAME / $(basename "$CKPT")"
python scripts/audit_reward.py \
    --checkpoint "$CKPT" \
    --scene rugd_trail_00 --episodes 5 \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
    --out_dir /scratch/m000204-pm07b/joana/outputs/audit_${RUN_NAME}_$(basename "$CKPT" .zip) \
    "${EXTRA_ARGS[@]}"
echo "==> audit done"
