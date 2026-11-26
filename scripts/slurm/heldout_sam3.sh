#!/usr/bin/env bash
#SBATCH --job-name=heldout-sam3
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-heldout-sam3-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-heldout-sam3-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
# HELD-OUT SWEEP, SAM 3 BASELINE (2026-11-25, Joana): SAM 3 on the GENERATED frames of a finished held-out sweep
# (same default prompts and v14 remap as the ground-truth labels), then re-grade with --sam3 so graded.csv also says
# how often "segment the generated image" agrees with SAM 3 on the real frame.
# Knobs: SCENES (required), OUTROOT (default outputs/heldout_gaps_v35b).
set -uo pipefail
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
export PYTHONNOUSERSITE=1
NEO_ENV=/users/jmizrahi/.conda/envs/neoverse/bin
SAM_ENV=/users/jmizrahi/.conda/envs/sam3/bin
S=${SCRATCH_ROOT:-/scratch/m000204-pm07b/joana}
: "${SCENES:?set SCENES=\"s1 s2 ...\"}"
OUTROOT=${OUTROOT:-$S/outputs/heldout_gaps_v35b}
_tag=$(basename "$OUTROOT")
for s in $SCENES; do
    out="$OUTROOT/$s"
    [ -f "$out/semantic_labels.npz" ] || { echo "!!! $s: no finished sweep in $out"; continue; }
    ln -sfn "$out/frames" "$S/NeoVerse/outputs/${_tag}_$s"
    sam_npz="$S/NeoVerse/outputs/sam3_labels/${_tag}_$s.npz"
    if [ -f "$sam_npz" ]; then
        echo "[1/2] $s: SAM 3 labels exist"
    else
        (export PATH=$SAM_ENV:$PATH; hash -r; cd $S/NeoVerse && python sam3_precompute_labels.py \
            --input_path "outputs/${_tag}_$s" --static_scene --overlay_every 1000) || { echo "!!! SAM 3 FAILED: $s"; continue; }
    fi
    (export PATH=$NEO_ENV:$PATH; hash -r; cd $S/nav-rl && python scripts/grade_heldout.py --sweep "$out" --sam3 "$sam_npz") \
        | tail -1 || echo "!!! grade FAILED: $s"
done
echo "==> held-out SAM 3 done: $OUTROOT"
