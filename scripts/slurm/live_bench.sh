#!/usr/bin/env bash
#SBATCH --job-name=live-bench
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-live-bench-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-live-bench-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n21,n24,n26,n31

# Phase-0 decision gate: benchmark the live per-action diffusion render
# (scripts/live_benchmark.py). Prints the s/step table and saves sample
# observation clips per config for visual QA.
#   sbatch --export=ALL scripts/slurm/live_bench.sh
# Knobs: SCENE, FRAMES ("5,9,21"), RES ("560x336,392x224"), STEPS, CKPT.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

CKPT=${CKPT:-/scratch/m000204-pm07b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors}
python scripts/live_benchmark.py \
    --scene "${SCENE:-rugd_trail_00}" \
    --checkpoint "$CKPT" \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --frames "${FRAMES:-5,9,21}" \
    --resolutions "${RES:-560x336,392x224}" \
    --steps "${STEPS:-10}" \
    --out /scratch/m000204-pm07b/joana/outputs/live_bench
echo "==> live benchmark done"
