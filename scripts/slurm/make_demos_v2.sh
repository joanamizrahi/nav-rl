#!/usr/bin/env bash
#SBATCH --job-name=demos-v2
#SBATCH --account=marlowe-m000204-pm07b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --exclude=n04,n13,n17,n18,n24
#SBATCH --output=/scratch/m000204-pm07b/joana/slurm-demos-v2-%j.out
#SBATCH --error=/scratch/m000204-pm07b/joana/slurm-demos-v2-%j.err

# BC redemption: regenerate the demonstration dataset on the POST-fix
# convention (fix lives in NavCalibration.from_npz, applied automatically)
# with observations from the cached diffused views (matches training obs).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm07b/joana/nav-rl
echo "commit: $(git log --oneline -1)"

python scripts/make_demo_dataset.py \
    --scenes rugd_trail_00 \
    --clips_dir /scratch/m000204-pm07b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm07b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm07b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --obs_cache /scratch/m000204-pm07b/joana/outputs/ribbon_cache \
    --out /scratch/m000204-pm07b/joana/outputs/demos_v2.npz

echo "==> demos_v2 done"
