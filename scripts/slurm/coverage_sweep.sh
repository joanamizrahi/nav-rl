#!/usr/bin/env bash
#SBATCH --job-name=cov-sweep
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-cov-sweep-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-cov-sweep-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
# Fusion-window sweep (2026-09-15): coverage / sharpness / raster-vs-generated
# agreement for windows of 81, 21, 9 frames at poses on and off the walk, with
# the generator ON (--diffuse) so the numbers are what the policy would see.
#
#   SCENES="quad1_03 quad2_36" CLIPS_DIR=/scratch/.../data/campus_clips \
#       sbatch scripts/slurm/coverage_sweep.sh
# Knobs: WINDOWS ("81 21 9"), COVWIN (coverage window for the gate, default 21),
# LIVECKPT (semantic checkpoint), OUTDIR, DIFFUSE=0 to run the rasterizer only.
set -uo pipefail
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
[ -d /users/jmizrahi/.conda/envs/neoverse/bin ] && export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl

: "${SCENES:?set SCENES=\"s1 s2 ...\"}"
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/campus_clips}
LIVECKPT=${LIVECKPT:-/scratch/m000204-pm07b/joana/runs/train_semantic_v26_campus/checkpoint-epoch-10.safetensors}
OUTROOT=${OUTDIR:-/scratch/m000204-pm07b/joana/outputs/cov_sweep_campus}
mkdir -p "$OUTROOT"
nvidia-smi -L || true
EXTRA=()
[ "${DIFFUSE:-1}" = "1" ] && EXTRA+=(--diffuse --live_ckpt "$LIVECKPT")
python scripts/coverage_sweep.py \
    --scenes $SCENES \
    --clips_dir "$CLIPS_DIR" \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --out_dir "$OUTROOT" \
    --windows ${WINDOWS:-81 21 9} \
    --coverage_window "${COVWIN:-21}" \
    --sem_palette "${SEMPAL:-4}" \
    "${EXTRA[@]}"
echo "==> coverage_sweep done: $OUTROOT"
