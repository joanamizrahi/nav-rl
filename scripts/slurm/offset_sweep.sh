#!/usr/bin/env bash
#SBATCH --job-name=offset-sweep
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-offset-sweep-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-offset-sweep-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
# OFFSET SWEEP (2026-09-18, Joana's experiment): from recorded poses, step
# sideways 0..2 m and turn 0..90 deg, render with the semantic world model,
# run SAM3 on the GENERATED image, and grade generated labels against it and
# against the rasterized hint, by offset and by coverage. No withheld frames.
# Stages per scene: [1] render (neoverse env)  [2] SAM3 on the generated frames
# (sam3 env, inside NeoVerse)  [3] grade (CPU).
# Knobs: SCENES (required), LIVECKPT, SEMPAL, STATICMOVERS, RENDWIN, COVWIN,
#        BASE_EVERY (8), LAT_MAX (2.0) LAT_STEP (0.2), YAW_MAX (90) YAW_STEP (10), OUTROOT.
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
CLIPS=${CLIPS_DIR:-$S/data/campus_clips}
POSES=${POSES_DIR:-$S/outputs/poses}
LABELS=${LABELS_DIR:-$S/NeoVerse/outputs/sam3_labels_v14}
OUTROOT=${OUTROOT:-$S/outputs/offsets}
LIVECKPT=${LIVECKPT:-$S/runs/train_semantic_v33_p6_multi/checkpoint-epoch-9.safetensors}
SEMPAL=${SEMPAL:-6}
RENDWIN=${RENDWIN:-21}; COVWIN=${COVWIN:-21}
mkdir -p "$OUTROOT"
nvidia-smi -L || true
echo "==> scenes: $SCENES | ckpt: $(basename "$LIVECKPT") pal $SEMPAL | movers ${STATICMOVERS:-off} | win $RENDWIN/$COVWIN"
for s in $SCENES; do
    out="$OUTROOT/$s"
    # [1] render
    if [ -f "$out/queries.csv" ] && [ -f "$out/semantic_labels.npz" ]; then
        echo "[1/3] $s: renders exist"
    else
        (export PATH=$NEO_ENV:$PATH; hash -r; cd $S/nav-rl && python scripts/offset_sweep.py --scene "$s" \
            --clips_dir "$CLIPS" --poses_dir "$POSES" --labels_dir "$LABELS" --live_ckpt "$LIVECKPT" \
            --sem_palette "$SEMPAL" --render_window "$RENDWIN" --coverage_window "$COVWIN" \
            ${STATICMOVERS:+--static_movers "$STATICMOVERS"} \
            --base_every "${BASE_EVERY:-8}" --lat_max "${LAT_MAX:-2.0}" --lat_step "${LAT_STEP:-0.2}" \
            --yaw_max "${YAW_MAX:-90}" --yaw_step "${YAW_STEP:-10}" --history "${HISTORY:-gradual}" --out "$out") \
            || { echo "!!! stage1 FAILED: $s"; continue; }
    fi
    # RENDERONLY=1 (2026-11-05): figure renders; skip SAM3 and grading
    [ "${RENDERONLY:-0}" = "1" ] && { echo "[1/3] $s: RENDERONLY, done"; continue; }
    # [2] SAM3 on the generated frames (reads the png directory in file order)
    ln -sfn "$out/frames" "$S/NeoVerse/outputs/offset_frames_$s"
    sam_npz="$S/NeoVerse/outputs/sam3_labels/offset_frames_$s.npz"
    if [ -f "$sam_npz" ]; then
        echo "[2/3] $s: SAM3 labels exist"
    else
        (export PATH=$SAM_ENV:$PATH; hash -r; cd $S/NeoVerse && python sam3_precompute_labels.py \
            --input_path "outputs/offset_frames_$s" --static_scene --overlay_every 1000) \
            || { echo "!!! stage2 FAILED: $s"; continue; }
    fi
    # [3] grade
    (export PATH=$NEO_ENV:$PATH; hash -r; cd $S/nav-rl && python scripts/grade_offsets.py --sweep "$out" \
        --sam3 "$sam_npz" --sem_palette "$SEMPAL") || echo "!!! stage3 FAILED: $s"
done
echo "==> offset sweep done: $OUTROOT"
