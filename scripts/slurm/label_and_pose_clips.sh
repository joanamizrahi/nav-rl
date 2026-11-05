#!/usr/bin/env bash
#SBATCH --job-name=label-pose
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-label-pose-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-label-pose-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# SAM3 labels (+ v14 remap) and reconstructor poses for NAMED clips only
# (2026-09-07): the six gnd_survey AUw windows had clips but neither labels
# nor poses. Works on a temp folder of symlinks so nothing already labelled
# or posed is touched. After this, dump_clouds.sh gives them a cloud.
#
#   SCENES="gnd_AUw30 gnd_AUw90 ..." CLIPS_DIR=/scratch/.../data/gnd_survey \
#       sbatch scripts/slurm/label_and_pose_clips.sh
set -euo pipefail
: "${SCENES:?set SCENES=\"gnd_a gnd_b ...\"}"
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/gnd_survey}
CAM_H=${CAM_H:-0.6}
NEOVERSE=/scratch/m000204-pm07b/joana/NeoVerse
NAVRL=/scratch/m000204-pm07b/joana/nav-rl
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
TMP=$(mktemp -d /scratch/m000204-pm07b/joana/data/tmp_clips_XXXXXX)
for s in $SCENES; do
    [ -f "$CLIPS_DIR/$s.mp4" ] || { echo "REFUSED: no $CLIPS_DIR/$s.mp4"; exit 3; }
    ln -s "$CLIPS_DIR/$s.mp4" "$TMP/$s.mp4"
    [ -f "$CLIPS_DIR/${s}_odom.npz" ] && ln -s "$CLIPS_DIR/${s}_odom.npz" "$TMP/${s}_odom.npz"
done
echo "==> clips: $SCENES (from $CLIPS_DIR via $TMP)"
cd "$NEOVERSE"
# SAM3 lives in its own conda env (sam3_new_clips.sh uses it); the remap and
# the poses use the neoverse env
for s in $SCENES; do
    echo "==== SAM3 $s ===="
    /users/jmizrahi/.conda/envs/sam3/bin/python sam3_precompute_labels.py --input_path "$TMP/$s.mp4"
done
/users/jmizrahi/.conda/envs/neoverse/bin/python scripts/remap_labels_to_v14.py --dirs outputs/sam3_labels
for s in $SCENES; do
    [ -f "$NEOVERSE/outputs/sam3_labels_v14/$s.npz" ] && echo "labels ok: $s" || echo "MISSING labels: $s"
done
cd "$NAVRL"
python scripts/extract_poses.py \
    --videos "$TMP"/*.mp4 \
    --output_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --reconstructor_path /scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --num_frames 81 --width 560 --height 336 \
    --camera_height_m "$CAM_H"
for s in $SCENES; do
    [ -f "/scratch/m000204-pm07b/joana/outputs/poses/${s}_poses.npz" ] && echo "poses ok: $s" || echo "MISSING poses: $s"
done
rm -rf "$TMP"
echo "==> label + pose done; next: SCENES=\"$SCENES\" CLIPS_DIR=$CLIPS_DIR sbatch scripts/slurm/dump_clouds.sh"
