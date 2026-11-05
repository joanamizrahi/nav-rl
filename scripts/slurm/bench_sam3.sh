#!/usr/bin/env bash
#SBATCH --job-name=sam3-bench
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-sam3-bench-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-sam3-bench-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# Cost per step of co-generated semantics vs SAM3 on the generated frame
# (scripts/bench_sam3_per_step.py). Two processes on one GPU, in sequence:
# the world model renders + saves frames in the neoverse env, then SAM3 times
# itself on those frames in the sam3 env (SAM3 is not installed in neoverse,
# and the pipeline leaves ~2 GB of VRAM). One table, one JSON at the end.
#   sbatch scripts/slurm/bench_sam3.sh
# Knobs: SCENE (gnd_AUw360), CKPT (v26 e10), STEPS (20), RES ("560x336").

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PYTHONNOUSERSITE=1
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

CKPT=${CKPT:-/scratch/m000204-pm07b/joana/runs/train_semantic_v26_campus/checkpoint-epoch-10.safetensors}
RES=${RES:-560x336}
OUT=/scratch/m000204-pm07b/joana/outputs/sam3_bench

echo "==> [1/2] render in neoverse env"
/users/jmizrahi/.conda/envs/neoverse/bin/python scripts/bench_sam3_per_step.py --mode render \
    --scene "${SCENE:-gnd_AUw360}" --checkpoint "$CKPT" \
    --width "${RES%x*}" --height "${RES#*x}" --steps "${STEPS:-20}" --out "$OUT"

echo "==> [2/2] SAM3 in sam3 env"
/users/jmizrahi/.conda/envs/sam3/bin/python scripts/bench_sam3_per_step.py --mode sam3 \
    --scene "${SCENE:-gnd_AUw360}" --out "$OUT"
echo "==> sam3 benchmark done"
