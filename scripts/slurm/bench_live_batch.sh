#!/usr/bin/env bash
#SBATCH --job-name=bench-batch
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=n04,n13,n17,n18,n21,n24
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-bench-batch-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-bench-batch-%j.err

# Batched live diffusion benchmark: how many robots can one GPU serve?
# Feeds the parallel-training decision (advisor plan, 2026-08-27).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl

python scripts/bench_live_batch.py \
    --scene "${SCENE:-rugd_trail_00}" \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --batches "${BATCHES:-1,2,4,8}" \
    --repeats "${REPEATS:-3}" \
    --height "${HEIGHT:-336}" \
    --width "${WIDTH:-560}" \
    --num_steps "${NSTEPS:-4}" \
    --save_samples "/scratch/m000204-pm07b/joana/outputs/bench_samples/nsteps${NSTEPS:-4}"

echo "==> bench done"
