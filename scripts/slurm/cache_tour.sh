#!/usr/bin/env bash
#SBATCH --job-name=cache-tour
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-cache-tour-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-cache-tour-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# Zigzag tour of a training cache (scripts/cache_tour.py).
#   sbatch --export=ALL scripts/slurm/cache_tour.sh                    # v1 cache
#   OBS_CACHE=ribbon_cache,ribbon_cache_spin sbatch ... (hybrid, needs --mem=96G)
# Knobs: SCENE, OBS_CACHE, AMP, CYCLES, FRAMES, NOGATE=1.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

SCENE=${SCENE:-rugd_trail_00}
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}
OBS_CACHE=${OBS_CACHE:-ribbon_cache}
CACHE_PATHS=$(echo "$OBS_CACHE" | sed 's#[^,]*#/scratch/m000204-pm07b/joana/outputs/&#g')
EXTRA_ARGS=()
[[ "${NOGATE:-0}" == "1" ]] && EXTRA_ARGS+=(--no_alpha_gate)
TAG=${OBS_CACHE//,/+}
if [[ "${STICKY:-}" != "" ]]; then
    EXTRA_ARGS+=(--sweep_sticky "$STICKY")
    TAG="${TAG}_sticky${STICKY}"
fi
if [[ "${BLACK:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--black_invented)
    TAG="${TAG}_black"
fi
if [[ "${FRANGE:-}" != "" ]]; then
    EXTRA_ARGS+=(--frame_range "$FRANGE")
    TAG="${TAG}_f${FRANGE/,/-}"
fi
if [[ "${WIGGLE:-}" != "" ]]; then
    EXTRA_ARGS+=(--wiggle_deg "$WIGGLE")
    TAG="${TAG}_wig${WIGGLE}"
fi

python scripts/cache_tour.py \
    --scene "$SCENE" \
    --clips_dir "$CLIPS_DIR" \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --obs_cache "$CACHE_PATHS" \
    --amplitude "${AMP:-1.2}" --cycles "${CYCLES:-3}" --frames "${FRAMES:-240}" \
    --out "/scratch/m000204-pm07b/joana/outputs/cache_tour/CACHETOUR_${SCENE}_${TAG}.mp4" \
    "${EXTRA_ARGS[@]}"
echo "==> cache tour done"
