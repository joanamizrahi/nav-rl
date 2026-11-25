#!/usr/bin/env bash
#SBATCH --job-name=heldout-sweep
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-heldout-sweep-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-heldout-sweep-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
# HELD-OUT SWEEP (2026-11-25, Jing's idea): render at recorded cameras from a window of OTHER frames and grade
# against the real frame (scripts/heldout_sweep.py, grade_heldout.py). Stages: [1] render (neoverse env)  [2] grade.
# Knobs: SCENES (required), LIVECKPT, SEMPAL, RENDWIN, COVWIN, SHIFTS, BASE_EVERY (4), BASE_LO (6), BASE_HI (74), OUTROOT.
set -uo pipefail
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
export PYTHONNOUSERSITE=1
NEO_ENV=/users/jmizrahi/.conda/envs/neoverse/bin
S=${SCRATCH_ROOT:-/scratch/m000204-pm07b/joana}
: "${SCENES:?set SCENES=\"s1 s2 ...\"}"
CLIPS=${CLIPS_DIR:-$S/data/campus_clips}
POSES=${POSES_DIR:-$S/outputs/poses}
LABELS=${LABELS_DIR:-$S/NeoVerse/outputs/sam3_labels_v14}
OUTROOT=${OUTROOT:-$S/outputs/heldout_v35b}
LIVECKPT=${LIVECKPT:-$S/runs/train_semantic_v35b_p6_ppl5/checkpoint-epoch-3.safetensors}
SEMPAL=${SEMPAL:-6}
RENDWIN=${RENDWIN:-21}; COVWIN=${COVWIN:-21}
SHIFTS=${SHIFTS:-0,-12,-16,-20,-28,12,16,20,28,36}
mkdir -p "$OUTROOT"
nvidia-smi -L || true
echo "==> scenes: $SCENES | ckpt: $(basename "$LIVECKPT") pal $SEMPAL | win $RENDWIN/$COVWIN | shifts $SHIFTS"
for s in $SCENES; do
    out="$OUTROOT/$s"
    if [ -f "$out/queries.csv" ] && [ -f "$out/semantic_labels.npz" ]; then
        echo "[1/2] $s: renders exist"
    else
        (export PATH=$NEO_ENV:$PATH; hash -r; cd $S/nav-rl && python scripts/heldout_sweep.py --scene "$s" \
            --clips_dir "$CLIPS" --poses_dir "$POSES" --labels_dir "$LABELS" --live_ckpt "$LIVECKPT" \
            --sem_palette "$SEMPAL" --render_window "$RENDWIN" --coverage_window "$COVWIN" --shifts "$SHIFTS" \
            --base_every "${BASE_EVERY:-4}" --base_lo "${BASE_LO:-6}" --base_hi "${BASE_HI:-74}" --out "$out") \
            || { echo "!!! stage1 FAILED: $s"; continue; }
    fi
    (export PATH=$NEO_ENV:$PATH; hash -r; cd $S/nav-rl && python scripts/grade_heldout.py --sweep "$out") \
        || echo "!!! stage2 FAILED: $s"
done
echo "==> held-out sweep done: $OUTROOT"
