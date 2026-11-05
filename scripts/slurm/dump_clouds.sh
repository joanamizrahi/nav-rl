#!/usr/bin/env bash
#SBATCH --job-name=dump-clouds
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-dump-clouds-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-dump-clouds-%j.err
set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
# Knobs (2026-08-25, proximity-cost prep): SCENES="s1 s2 ...", CLIPS_DIR,
# LABELS_DIR (v14 for anything the proximity cost consumes — its obstacle
# filter uses v14 ids), OUT_DIR (train launcher reads scene_clouds/clouds).
SCENES=${SCENES:-$(ls /scratch/m000204-pm07b/joana/outputs/poses/*_poses.npz | xargs -n1 basename | sed "s/_poses.npz//" | grep -v village)}
python scripts/dump_scene_cloud.py \
    --scenes $SCENES \
    --clips_dir "${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}" \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir "${LABELS_DIR:-/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14}" \
    --out_dir "${OUT_DIR:-/scratch/m000204-pm07b/joana/outputs/scene_clouds/clouds}"
echo "==> clouds in /scratch/m000204-pm07b/joana/outputs/scene_clouds"
