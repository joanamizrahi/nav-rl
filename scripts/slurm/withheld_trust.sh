#!/usr/bin/env bash
#SBATCH --job-name=withheld-trust
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-withheld-trust-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-withheld-trust-%j.err
#SBATCH --exclude=n04,n13,n17,n18,n24
# TRUST RANGE, end to end (2026-09-16). For each scene: hold frames out, rebuild
# the scene WITHOUT them, render at the withheld poses, grade the generated
# labels against SAM3 on the real withheld frames, and bin by coverage and by
# pose deviation -> A(c), A(d), i.e. tau, tau_term, delta_q from data.
#
#   SCENES="quad1_03 quad2_01" sbatch scripts/slurm/withheld_trust.sh
# Knobs: PATTERNS ("every4 win10": single-frame gaps = high coverage; a window of
# 10 dropped frames = low coverage), LIVECKPT + SEMPAL (semantic model + its
# palette), RENDWIN/COVWIN (must match training: 21/21), KPRIOR (intrinsics
# prior, same as the main poses), CAM_H, TRAVIDS (traversable class ids),
# OUTROOT. Stages are skipped when their outputs already exist, so a failed run
# can be resubmitted without redoing the GPU work.
set -uo pipefail
if command -v module >/dev/null 2>&1; then
    module load conda/24.3.0-0
    module load cuda12.9/toolkit/12.9.1
fi
[ -d /users/jmizrahi/.conda/envs/neoverse/bin ] && export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm07b/joana/nav-rl

: "${SCENES:?set SCENES=\"s1 s2 ...\"}"
S=/scratch/m000204-pm07b/joana
CLIPS=${CLIPS_DIR:-$S/data/campus_clips}
LABELS=${LABELS_DIR:-$S/NeoVerse/outputs/sam3_labels_v14}
WHDIR=${WHDIR:-$S/data/campus_withheld}
WHPOSES=${WHPOSES:-$S/outputs/poses_withheld}
OUTROOT=${OUTROOT:-$S/outputs/trust}
# STATICMOVERS=12,13 (2026-09-18): person/vehicle Gaussians stay per-frame in
# the withheld render, as in training and the walk test. Without it a walking
# person's splats fuse into a trail that paints PERSON over the ground in the
# hint, and the generator copies it (quad1_07 win10/win20, first trust run).
PATTERNS=${PATTERNS:-every4 win10 win20}   # every4 = still surrounded by kept frames (high coverage);
# win10 = ~2 m from the nearest kept frame; win20 = ~4 m, genuinely invented (low coverage end)
LIVECKPT=${LIVECKPT:-$S/runs/train_semantic_v26_campus/checkpoint-epoch-10.safetensors}
SEMPAL=${SEMPAL:-4}
RENDWIN=${RENDWIN:-21}
COVWIN=${COVWIN:-21}
CAM_H=${CAM_H:-0.69}
KPRIOR=${KPRIOR:-322.00,320.00,280.00,168.00}
TRAVIDS=${TRAVIDS:-2,4,6,7,8,9}
mkdir -p "$WHDIR" "$WHPOSES" "$OUTROOT"
nvidia-smi -L || true
echo "==> scenes: $SCENES | patterns: $PATTERNS | ckpt: $(basename "$LIVECKPT") pal $SEMPAL | win $RENDWIN/$COVWIN"

# ---- stage 1 (CPU): withheld subset clips -------------------------------
for s in $SCENES; do
    if ls "$WHDIR/${s}_wh_"*_withheld.json >/dev/null 2>&1; then
        echo "[1/4] $s: subsets exist, skipping"
    else
        python scripts/make_withheld_clips.py --scene "$s" --clips_dir "$CLIPS" \
            --labels_dir "$LABELS" --out "$WHDIR" --patterns $PATTERNS || echo "!!! stage1 FAILED: $s"
    fi
done

# ---- stage 2 (GPU): poses of every subset scene, AT ITS OWN FRAME COUNT ---
# (2026-09-17: posing a 61-frame subset at 81 frames gave 81 poses for 61
# Gaussian source frames -> empty fusion windows -> "At least one Gaussian
# must be present". The subset json carries num_frames; use it per clip, and
# redo any poses file whose count does not match.)
for s in $SCENES; do
    for p in $PATTERNS; do
        v="$WHDIR/${s}_wh_${p}.mp4"; j="$WHDIR/${s}_wh_${p}_withheld.json"
        [ -f "$v" ] && [ -f "$j" ] || { echo "[2/4] missing $v"; continue; }
        nf=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['num_frames'])" "$j")
        pz="$WHPOSES/${s}_wh_${p}_poses.npz"
        if [ -f "$pz" ]; then
            have=$(python -c "import numpy as np,sys; print(len(np.load(sys.argv[1])['positions']))" "$pz")
            if [ "$have" = "$nf" ]; then echo "[2/4] ${s}_wh_${p}: poses exist ($nf frames)"; continue; fi
            echo "[2/4] ${s}_wh_${p}: poses have $have frames, clip has $nf -> redo"; rm -f "$pz"
        fi
        python scripts/extract_poses.py --videos "$v" --output_dir "$WHPOSES" \
            --reconstructor_path $S/NeoVerse/models/NeoVerse/reconstructor.ckpt \
            --num_frames "$nf" --width 560 --height 336 --camera_height_m "$CAM_H" \
            --intrinsics_prior "$KPRIOR" || echo "!!! stage2 FAILED: ${s}_wh_${p}"
    done
done

# ---- stage 3 (GPU): render the withheld poses inside the subset scene ----
# ---- stage 4 (CPU): grade -> A(c), A(d) ---------------------------------
for s in $SCENES; do
    for p in $PATTERNS; do
        wh="$WHDIR/${s}_wh_${p}_withheld.json"
        out="$OUTROOT/${s}_${p}"
        [ -f "$wh" ] || continue
        [ -f "$WHPOSES/${s}_wh_${p}_poses.npz" ] || { echo "[3/4] ${s}_${p}: no poses, skipping"; continue; }
        if [ -f "$out/semantic_labels.npz" ]; then
            echo "[3/4] ${s}_${p}: renders exist"
        else
            python scripts/withheld_render.py --scene "${s}_wh_${p}" --withheld "$wh" \
                --clips_dir "$WHDIR" \
                --poses_dir "$WHPOSES" --labels_dir "$WHDIR" --live_ckpt "$LIVECKPT" \
                --sem_palette "$SEMPAL" --render_window "$RENDWIN" --coverage_window "$COVWIN" \
                --static_scene ${STATICMOVERS:+--static_movers "$STATICMOVERS"} --out "$out" || { echo "!!! stage3 FAILED: ${s}_${p}"; continue; }
        fi
        python scripts/label_acc_vs_coverage.py --pred "$out/semantic_labels.npz" \
            --alpha "$out/alpha.npz" --ref "$out/ref_labels.npz" --hint "$out/hint_labels.npz" \
            --deviation "$out/deviation.csv" --trav "$TRAVIDS" --out "$out/grade" \
            || echo "!!! stage4 FAILED: ${s}_${p}"
    done
done
echo "==> withheld_trust done. Curves: $OUTROOT/<scene>_<pattern>/grade/{curves.png,summary.json}"
