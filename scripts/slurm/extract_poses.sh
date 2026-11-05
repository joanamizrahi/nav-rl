#!/usr/bin/env bash
#SBATCH --job-name=extract-poses
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-extract-poses-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-extract-poses-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# Extract real per-clip robot trajectories from the NeoVerse reconstructor
# (TODO: "Extract REAL trajectories ... save poses.npz per clip").
# One reconstruction per clip, no diffusion — a handful of clips fits in 1h.
#
# Output: /scratch/.../outputs/poses/<clip>_poses.npz  — scp these to the Mac
# and feed to validate_reward.py --pose_source npz --poses_npz <file>.

set -euo pipefail

# Adjust if nav-rl is cloned elsewhere on Marlowe.
NAVRL_ROOT=/scratch/m000204-pm07b/joana/nav-rl
# CLIPS_DIR env: point at any clip folder (gnd_clips, scand_clips, ...)
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}
OUT_DIR=${OUT_DIR:-/scratch/m000204-pm07b/joana/outputs/poses}   # env-overridable (2026-09-15): write a trial into its own folder
# CAM_H env: camera mount height in meters (RUGD 0.6; Jackal ZED ~0.5)
# KPRIOR env (2026-09-15): "fx,fy,cx,cy" at the render size -> intrinsics prior
# for every clip (the camera's calibration); unset = the reconstructor guesses.
CAM_H=${CAM_H:-0.6}

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd "$NAVRL_ROOT"
mkdir -p "$OUT_DIR"

# SCENES env (2026-09-15): space-separated scene names -> only those clips;
# unset = every mp4 in CLIPS_DIR (the old behaviour).
if [ -n "${SCENES:-}" ]; then
    VIDEOS=(); for s in $SCENES; do VIDEOS+=("$CLIPS_DIR/$s.mp4"); done
else
    VIDEOS=("$CLIPS_DIR"/*.mp4)
fi
python scripts/extract_poses.py \
    --videos "${VIDEOS[@]}" \
    --output_dir "$OUT_DIR" \
    --reconstructor_path /scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --num_frames 81 --width 560 --height 336 \
    --camera_height_m "$CAM_H" \
    ${KPRIOR:+--intrinsics_prior "$KPRIOR"}

echo "==> done. poses in $OUT_DIR — check the printed step-size / camera-height sanity lines in this log."
