#!/usr/bin/env bash
#SBATCH --job-name=pose-check
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-pose-check-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-pose-check-%j.err

# Pose -> image direction probe (meeting item): does a constructed robot pose
# render an image facing where the pose says? PASS criteria printed by the script.
#   sbatch [--export=SCENE=rugd_trail_00] scripts/slurm/pose_direction_check.sh

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"
SCENE=${SCENE:-rugd_trail_00}
python scripts/pose_direction_check.py --scene "$SCENE" \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
    --model_path /scratch/m000204-pm07b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --out_dir /scratch/m000204-pm07b/joana/outputs/pose_check
echo "==> pose check done"
