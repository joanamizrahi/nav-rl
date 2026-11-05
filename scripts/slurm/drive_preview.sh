#!/usr/bin/env bash
#SBATCH --job-name=drive-prev
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=n04,n13,n17,n18,n21,n24
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-drive-prev-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-drive-prev-%j.err

# Moving-through-the-world diffusion preview: consecutive recorded poses,
# any resolution/steps. HEIGHT/WIDTH multiples of 112. Knobs:
#   HEIGHT WIDTH NSTEPS FRAMES SCENE START TARGET LIVECKPT CLIPS_DIR (campus_clips default; gnd_clips / gnd_survey for gnd scenes)
# Defaults moved to the campus clips and the v35b semantic model on 2026-09-27 (they pointed at rugd + v21).
#   GOALFRAME (goal = pose at this frame) / GOALXY "x,y" — goal marker in
#   both panels + topdown inset + distance HUD (goal-placement design tool)

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl

python scripts/drive_preview.py \
    --scene "${SCENE:-rugd_trail_00}" \
    --clips_dir "${CLIPS_DIR:-/scratch/m000204-pm07b/joana/data/campus_clips}" \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --height "${HEIGHT:-336}" \
    --width "${WIDTH:-560}" \
    --num_steps "${NSTEPS:-4}" \
    --frames "${FRAMES:-40}" \
    --start "${START:-5}" \
    --heading "${HEADING:-tangent}" \
    ${TARGET:+--target_xy "$TARGET"} \
    ${GOALFRAME:+--goal_frame "$GOALFRAME"} \
    ${GOALXY:+--goal_xy "$GOALXY"} \
    ${SPIN:+--spin} \
    ${SPINDEG:+--spin_deg "$SPINDEG"} \
    ${RASTER:+--raster} \
    ${GOALSAMPLE:+--goal_sample "$GOALSAMPLE"} \
    ${GOALAT:+--goal_at="$GOALAT"} \
    ${STRAFE:+--strafe} \
    ${STRAFEM:+--strafe_m "$STRAFEM"} \
    ${SPAWNYAW:+--spawn_yaw_deg "$SPAWNYAW"} \
    ${SPAWNLAT:+--spawn_lat_m "$SPAWNLAT"} \
    ${PANOVIEWS:+--pano_views} \
    ${SEMPAL:+--sem_palette "$SEMPAL"} \
    ${AERIAL:+--aerial "$AERIAL"} \
    --live_ckpt "${LIVECKPT:-/scratch/m000204-pm07b/joana/runs/train_semantic_v35b_p6_ppl5/checkpoint-epoch-3.safetensors}" \
    --out /scratch/m000204-pm07b/joana/outputs/drive_preview${OUTTAG:+_$OUTTAG}

echo "==> drive preview done"
