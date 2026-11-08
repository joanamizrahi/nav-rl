#!/usr/bin/env bash
#SBATCH --job-name=scene-cloud
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-scene-cloud-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-scene-cloud-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# Dump a scene's labeled Gaussian cloud (scripts/dump_scene_cloud.py) for the
# measured-semantic-map topdown figure (make_topdown_figure.py --cloud_npz).
#   sbatch --export=ALL scripts/slurm/dump_cloud.sh
# Knobs: SCENES (space-separated), CLIPS_DIR, LABELS_DIR, EXTRA (e.g. "--ply --ply_points 3000000"
# for a SuperSplat .ply of the corner scene; campus scenes need CLIPS_DIR=.../data/campus_clips).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

python scripts/dump_scene_cloud.py \
    --scenes ${SCENES:-rugd_trail_00} \
    --clips_dir "${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}" \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir "${LABELS_DIR:-/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14}" \
    --out_dir /scratch/m000204-pm07b/joana/outputs/scene_clouds ${EXTRA:-}
echo "==> scene cloud done"
