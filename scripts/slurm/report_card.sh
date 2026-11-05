#!/usr/bin/env bash
#SBATCH --job-name=report-card
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-report-card-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-report-card-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# Reconstruction report card: replay fidelity + coverage-vs-offset per scene.
#   sbatch --export=SCENE=rugd_trail_00 scripts/slurm/report_card.sh
#   sbatch --export=SCENE=gnd_AU_60,CLIPS_DIR=/scratch/.../data/gnd_clips scripts/slurm/report_card.sh
# CLIPS_DIR/POSES_DIR/LABELS_DIR default to the RUGD locations.
# Results: outputs/report_card/<scene>.json (+ printed verdict in this log).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"
SCENE=${SCENE:?set SCENE=... via --export}
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}
POSES_DIR=${POSES_DIR:-/scratch/m000204-pm07b/joana/outputs/poses}
LABELS_DIR=${LABELS_DIR:-/scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels}
python scripts/scene_report_card.py --scene "$SCENE" \
    --clips_dir "$CLIPS_DIR" \
    --poses_dir "$POSES_DIR" \
    --labels_dir "$LABELS_DIR" \
    --model_path /scratch/m000204-pm07b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --out_dir /scratch/m000204-pm07b/joana/outputs/report_card
echo "==> report card done for $SCENE"
