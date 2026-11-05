#!/usr/bin/env bash
#SBATCH --job-name=replay-trail00
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-replay-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-replay-%j.err
set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
python scripts/replay_real_trajectory.py \
    --scene rugd_trail_00 \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
    --out /scratch/m000204-pm07b/joana/outputs/replay_trail00.mp4
echo "==> replay done"
