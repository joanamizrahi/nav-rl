#!/usr/bin/env bash
#SBATCH --job-name=sweep-reel
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-sweep-reel-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-sweep-reel-%j.err

# Sweep self-coherence reel (scripts/cache_sweep_reel.py). CPU-only.
#   sbatch --export=ALL scripts/slurm/sweep_reel.sh
#   OBS_CACHE=ribbon_cache_spin sbatch --export=ALL scripts/slurm/sweep_reel.sh

set -euo pipefail
module load conda/24.3.0-0
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

SCENE=${SCENE:-rugd_trail_00}
OBS_CACHE=${OBS_CACHE:-ribbon_cache}
python scripts/cache_sweep_reel.py \
    --cache_root "/scratch/m000204-pm07b/joana/outputs/${OBS_CACHE}" \
    --scene "$SCENE" \
    --out "/scratch/m000204-pm07b/joana/outputs/cache_tour/SWEEPREEL_${SCENE}_${OBS_CACHE}.mp4"
echo "==> sweep reel done"
