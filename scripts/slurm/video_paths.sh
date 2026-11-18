#!/usr/bin/env bash
#SBATCH --job-name=video-paths
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-video-paths-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-video-paths-%j.err
#SBATCH --exclude=n04,n13,n17,n21,n24,n26,n31

# Camera paths + per-frame coverage C_w for the supplementary video (scripts/video_paths.py). Rasterizer only.
#   env SCENE=quad2_00 LAT="42:+1:9 42:-1:9" TURN="42:+1:180" sbatch --export=ALL scripts/slurm/video_paths.sh
# Knobs: SCENE, LAT, TURN, XY (space-separated lists), OUT (default outputs/video_paths/<SCENE>).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

SCENE=${SCENE:-quad2_00}
OUT=${OUT:-/scratch/m000204-pm07b/joana/outputs/video_paths/${SCENE}}
ARGS=""
[ -n "${LAT:-}" ] && ARGS="$ARGS --lat $LAT"
[ -n "${TURN:-}" ] && ARGS="$ARGS --turn $TURN"
[ -n "${XY:-}" ] && ARGS="$ARGS --xy $XY"

python scripts/video_paths.py \
    --scene "$SCENE" \
    --clips_dir /scratch/m000204-pm07b/joana/data/campus_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --out_dir "$OUT" $ARGS
echo "==> video paths done: $OUT"
