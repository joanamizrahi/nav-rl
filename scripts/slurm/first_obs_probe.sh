#!/usr/bin/env bash
#SBATCH --job-name=first-obs
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --exclude=n04,n13,n17,n18,n21,n24
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-first-obs-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-first-obs-%j.err

# First-observation probe: OLD static vs NEW walk-in history, side by side.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl

python scripts/first_obs_probe.py \
    --scene "${SCENE:-rugd_trail_00}" \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --height "${HEIGHT:-224}" \
    --width "${WIDTH:-336}" \
    --out /scratch/m000204-pm07b/joana/outputs/first_obs_probe

echo "==> first-obs done"
