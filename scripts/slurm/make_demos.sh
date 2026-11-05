#!/usr/bin/env bash
#SBATCH --job-name=make-demos
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-make-demos-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-make-demos-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
SCENES=$(ls /scratch/m000204-pm07b/joana/outputs/poses/*_poses.npz | xargs -n1 basename | sed "s/_poses.npz//" | grep -v village)
python scripts/make_demo_dataset.py \
    --scenes $SCENES \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels \
    --out /scratch/m000204-pm07b/joana/outputs/demos_v1.npz
echo "==> demos at /scratch/m000204-pm07b/joana/outputs/demos_v1.npz"
