#!/usr/bin/env bash
#SBATCH --job-name=live-demos
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-live-demos-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-live-demos-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n21,n24,n26,n31

# BC demos rendered through the LIVE diffusion backend — the demo observations
# match --live training frame for frame (the old demos_v1.npz was raster obs +
# pre-yaw-fix actions and poisoned cached runs; never reuse it for live).
# Output feeds train_ppo_real.sh via LIVE=1 LIVE_DEMOS=<this npz>.
#   sbatch --export=ALL scripts/slurm/make_live_demos.sh
# Knobs: SCENE (default rugd_trail_00), OUT_NPZ.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

SCENE=${SCENE:-rugd_trail_00}
OUT_NPZ=${OUT_NPZ:-/scratch/m000204-pm07b/joana/outputs/demos_live_${SCENE}.npz}

python scripts/make_demo_dataset.py \
    --scenes "$SCENE" \
    --live \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --out "$OUT_NPZ"
echo "==> live demos: $OUT_NPZ"
