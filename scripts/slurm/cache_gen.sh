#!/usr/bin/env bash
#SBATCH --job-name=cache-gen
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-cache-gen-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-cache-gen-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24

# Ribbon-cache generation: renders sweeps (81-pose trajectory files from
# make_ribbon_trajectories.py) through v10 + reader, one inference_semantic
# call per sweep. Each sweep's output dir gets rgb.mp4 + semantic_labels.npz
# + alpha.npz — exactly what CachedDiffusedBackend consumes.
#
#   SCENE   (default rugd_trail_00)
#   SWEEPS  "0-9" or "3" — index range into the manifest's sweep list.
#           GATE: run SWEEPS=0 alone first, eyeball the output, then the rest.
#   RUN_NAME (default train_semantic_v10) — the semantic checkpoint to render with.
#
#   sbatch --export=SCENE=rugd_trail_00,SWEEPS=0 scripts/slurm/cache_gen.sh
#   sbatch --export=SCENE=rugd_trail_00,SWEEPS=1-13 scripts/slurm/cache_gen.sh   (etc.)

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

SCENE=${SCENE:-rugd_trail_00}
SWEEPS=${SWEEPS:?set SWEEPS=<a>-<b> or <a> via --export}
RUN_NAME=${RUN_NAME:-train_semantic_v10}
# TRAJ_TAG: read sweeps from ribbon_traj_<tag>/ (e.g. the spin grid).
TRAJ_DIR=/scratch/m000204-pm07b/joana/outputs/ribbon_traj${TRAJ_TAG:+_$TRAJ_TAG}/${SCENE}
# CACHE_TAG: write to ribbon_cache_<tag>/ instead of the real cache — for
# prompt/settings experiments that must not touch what training reads.
CACHE_DIR=/scratch/m000204-pm07b/joana/outputs/ribbon_cache${CACHE_TAG:+_$CACHE_TAG}/${SCENE}
# PROMPT: override the inpainting text prompt (2026-08-17: off-path sweeps
# dream persistent objects — testing whether a scene-descriptive prompt tames
# the hallucinations in low-alpha headings).
PROMPT=${PROMPT:-}
RUNS=/scratch/m000204-pm07b/joana/runs/${RUN_NAME}
CKPT=$(ls -t "$RUNS"/checkpoint-epoch-*.safetensors 2>/dev/null | head -1 || true)

cd /scratch/m000204-pm07b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"
[ -f "$TRAJ_DIR/manifest.json" ] || { echo "FATAL: no manifest at $TRAJ_DIR — run make_ribbon_trajectories.py first"; exit 1; }
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint under $RUNS"; exit 1; }
grep -q "sweep_manifest" inference_semantic.py || { echo "FATAL: stale checkout (no batch sweep mode — pull NeoVerse)"; exit 1; }

EXTRA_ARGS=()
[ -n "$PROMPT" ] && EXTRA_ARGS+=(--prompt "$PROMPT")
# NUM_FRAMES: render cells longer than the 81-frame default (must be 4k+1).
# The input clip and its label hint are resampled to that length; labels come
# from <scene>_n<N>.npz (scripts/resample_labels.py).
LABELS_NPZ=outputs/sam3_labels_v14/${SCENE}.npz
if [ -n "${NUM_FRAMES:-}" ] && [ "${NUM_FRAMES}" != "81" ]; then
    EXTRA_ARGS+=(--num_frames "$NUM_FRAMES")
    LABELS_NPZ=outputs/sam3_labels_v14/${SCENE}_n${NUM_FRAMES}.npz
    [ -f "$LABELS_NPZ" ] || { echo "FATAL: no $LABELS_NPZ — run scripts/resample_labels.py"; exit 1; }
fi

# CLIPS_DIR: source clip location (default RUGD; gnd_clips etc. for the new
# datasets — their SAM3 v14 labels are keyed by the same scene name).
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/rugd_clips}
# ONE python process for the whole range: the 30GB pipeline loads once and
# every sweep in the range reuses it (was: one load PER sweep — half the cost
# of the whole cache was model loading).
python inference_semantic.py "${EXTRA_ARGS[@]}" \
    --input_path ${CLIPS_DIR}/${SCENE}.mp4 \
    --checkpoint "$CKPT" \
    --output_dir /tmp/unused \
    --sweep_manifest "$TRAJ_DIR/manifest.json" \
    --sweeps "$SWEEPS" \
    --cache_out "$CACHE_DIR" \
    --model_path /scratch/m000204-pm07b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm07b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --semantic_expansion_version 2 --lora_rank 8 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --semantic_labels "$LABELS_NPZ" \
    --num_semantic_classes 14 --semantic_x0_prediction --decode_with_head
# CachedDiffusedBackend reads manifest.json from the CACHE dir (bit us
# 2026-08-17: both smokes died on FileNotFoundError) — keep a copy there.
cp "$TRAJ_DIR/manifest.json" "$CACHE_DIR/manifest.json"
echo "==> cache_gen done for sweeps $SWEEPS of $SCENE"
