#!/usr/bin/env bash
#SBATCH --job-name=video-steps
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-video-steps-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-video-steps-%j.err
#SBATCH --exclude=n04,n13,n17,n21,n24,n26,n31

# Training-style frames (one generation per policy step) along video_paths.py paths (scripts/video_steps.py).
#   env SCENE=quad2_00 PATHS="a.csv b.csv" sbatch --export=ALL scripts/slurm/video_steps.sh
# Knobs: SCENE, PATHS (space-separated csvs), CKPT, OUT (default outputs/video_steps/<SCENE>).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

SCENE=${SCENE:-quad2_00}
CKPT=${CKPT:-/scratch/m000204-pm07b/joana/runs/train_semantic_v35b_p6_ppl5/checkpoint-epoch-3.safetensors}
OUT=${OUT:-/scratch/m000204-pm07b/joana/outputs/video_steps/${SCENE}}
PATHS=${PATHS:?set PATHS to one or more path csvs}

python scripts/video_steps.py \
    --scene "$SCENE" \
    --clips_dir /scratch/m000204-pm07b/joana/data/campus_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --live_ckpt "$CKPT" \
    --out_dir "$OUT" --paths $PATHS
echo "==> video steps done: $OUT"
