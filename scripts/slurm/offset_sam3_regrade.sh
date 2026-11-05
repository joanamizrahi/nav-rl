#!/usr/bin/env bash
#SBATCH --job-name=offset-sam3
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-offset-sam3-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-offset-sam3-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
# Re-run SAM3 for ONE offset-sweep root and regrade it (2026-09-27).
# Why: offset_sweep.sh keys its SAM3 output by scene name only
# (sam3_labels/offset_frames_<scene>.npz), so the second, third... sweep of the
# same scene (cold / far / xfar, or a re-render with a new model) found "SAM3
# labels exist" and was graded against SAM3 run on ANOTHER sweep's frames.
# This job names the SAM3 output by sweep root + scene, runs SAM3 on the
# GENERATED frames and, new, on the RASTER images too, copies both npz files
# into the sweep folder next to the labels they describe, and regrades.
#
#   SWEEP=/scratch/m000204-pm07b/joana/outputs/offsets_far_v35b SCENES="quad1_01 quad2_00 sequoia1_10" \
#       sbatch scripts/slurm/offset_sam3_regrade.sh
set -uo pipefail
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
export PYTHONNOUSERSITE=1
NEO_ENV=/users/jmizrahi/.conda/envs/neoverse/bin
SAM_ENV=/users/jmizrahi/.conda/envs/sam3/bin
S=/scratch/m000204-pm07b/joana
: "${SWEEP:?set SWEEP=<offset sweep root dir>}"
: "${SCENES:?set SCENES=\"s1 s2 ...\"}"
SEMPAL=${SEMPAL:-6}
TAG=$(basename "$SWEEP")
for s in $SCENES; do
    out="$SWEEP/$s"
    [ -d "$out/frames" ] || { echo "!!! $out/frames missing"; continue; }
    for kind in frames raster; do
        [ -d "$out/$kind" ] || { echo "    $s: no $kind/ dir, skipping"; continue; }
        name="offset_${TAG}_${s}_${kind}"
        ln -sfn "$out/$kind" "$S/NeoVerse/outputs/$name"
        npz="$S/NeoVerse/outputs/sam3_labels/$name.npz"
        if [ -f "$npz" ]; then
            echo "[sam3] $name: exists"
        else
            (export PATH=$SAM_ENV:$PATH; hash -r; cd $S/NeoVerse && python sam3_precompute_labels.py \
                --input_path "outputs/$name" --static_scene --overlay_every 1000) \
                || { echo "!!! SAM3 FAILED: $name"; continue; }
        fi
        cp -f "$npz" "$out/sam3_on_${kind}.npz"   # lives next to the labels it describes
    done
    [ -f "$out/sam3_on_frames.npz" ] || { echo "!!! $s: no SAM3 on generated frames, not regrading"; continue; }
    [ -d "$out/grade" ] && [ ! -d "$out/grade_before_regrade" ] && mv "$out/grade" "$out/grade_before_regrade"
    (export PATH=$NEO_ENV:$PATH; hash -r; cd $S/nav-rl && python scripts/grade_offsets.py --sweep "$out" \
        --sam3 "$out/sam3_on_frames.npz" --sem_palette "$SEMPAL") || echo "!!! grade FAILED: $s"
done
echo "==> offset_sam3_regrade done: $SWEEP"
