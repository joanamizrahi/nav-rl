#!/usr/bin/env bash
# Build the whole stack on a rented GPU box (RunPod H100/A100 80GB, Ubuntu,
# CUDA 12.x). Everything lands on the PERSISTENT volume (/workspace) and the
# absolute paths the code expects are symlinked onto it, so every command we
# use on Marlowe works here unchanged.
#
#   git clone https://github.com/joanamizrahi-png/nav-rl.git
#   bash nav-rl/scripts/cloud_bootstrap.sh
#
set -euo pipefail
ROOT=/workspace/joana
mkdir -p "$ROOT" /scratch/m000204-pm07b
[ -e /scratch/m000204-pm07b/joana ] || ln -s "$ROOT" /scratch/m000204-pm07b/joana
cd "$ROOT"

echo "=== 1/5 repositories"
[ -d nav-rl ]   || git clone https://github.com/joanamizrahi-png/nav-rl.git
[ -d NeoVerse ] || git clone https://github.com/joanamizrahi-png/NeoVerse.git

echo "=== 2/5 conda environment"
if ! command -v conda >/dev/null 2>&1; then
  curl -fsSL -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p /workspace/miniconda3
fi
export PATH=/workspace/miniconda3/bin:$PATH
# conda refuses to create environments until the channel terms are accepted
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda env list | grep -q "^neoverse " || conda create -y -n neoverse python=3.10
conda activate neoverse
pip install -q --upgrade pip
# torch must match the CUDA toolkit on the box, or gsplat refuses to compile
# (RunPod images ship a cu130 torch while nvcc is 12.8 -> CUDA_MISMATCH).
NVCC_CUDA=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo none)
echo "[bootstrap] nvcc CUDA=${NVCC_CUDA:-none}  torch CUDA=${TORCH_CUDA}"
if [ "$TORCH_CUDA" != "12.8" ]; then
  pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
fi
# torchvision is imported by inference_semantic.py but is NOT a torch dependency,
# so it must be installed even when the image already ships a matching torch.
python -c "import torchvision" 2>/dev/null || \
  pip install torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
# Derived from the imports NeoVerse/nav-rl actually make, pinned to the
# versions on Marlowe. The short hand-written list this replaces was missing
# torchvision, torch-scatter, omegaconf, addict, decord, e3nn and more, each
# of which surfaced only after a pipeline load failed.
pip install -q \
  "numpy==1.26.4" "pillow==12.2.0" "accelerate==1.14.0" "addict==2.4.0" \
  "decord==0.6.0" "e3nn==0.6.0" "einops==0.8.2" "evo==1.36.5" "ftfy==6.1.1" \
  "huggingface_hub==1.22.0" "imageio==2.37.3" "imageio-ffmpeg==0.6.0" \
  "jaxtyping==0.3.7" "matplotlib==3.10.9" "modelscope==1.38.1" "moviepy==1.0.3" \
  "omegaconf==2.3.1" "pandas==2.3.3" "peft==0.19.1" "regex==2026.6.28" \
  "requests==2.34.2" "rosbags==0.11.4" "safetensors==0.8.0" "scipy==1.15.3" \
  "sentencepiece==0.2.1" "timm==1.0.27" "tqdm==4.68.4" "transformers==4.57.6" \
  "trimesh==4.12.2" "typing_extensions==4.15.0" "wandb==0.28.0" \
  "opencv-python==4.11.0.86" "deepspeed==0.16.7" pyyaml \
  "stable_baselines3==2.9.0" "gymnasium==1.3.0"
# torch-scatter compiles against torch, like gsplat.
python -c "import torch_scatter" 2>/dev/null || \
  pip install --no-build-isolation torch-scatter
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=${MAX_JOBS:-16}
# --no-build-isolation: otherwise pip builds gsplat in a fresh env and pulls
# its OWN torch (cu130 on RunPod images), which mismatches nvcc 12.8 and fails.
pip install -q setuptools wheel ninja
python -c "import gsplat" 2>/dev/null || \
  pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@8b6319f8335df7de18d4514feb90b60e3941a073"

echo "=== 3/5 public Wan 2.1 weights (~70 GB)"
mkdir -p "$ROOT/NeoVerse/models/NeoVerse"
# One --include per pattern. Passing several patterns after a single
# --include makes them POSITIONAL FILENAMES and silently drops the flag --
# that is how the 28 GB DiT went missing while the run still said "Downloaded".
hf download Wan-AI/Wan2.1-T2V-14B \
  --include "diffusion_pytorch_model*" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" \
  --include "Wan2.1_VAE.pth" \
  --include "google/*" \
  --local-dir "$ROOT/NeoVerse/models/NeoVerse"

echo "=== 4/5 our private bundle"
if [ -f "$ROOT/cloud_bundle.tar" ]; then
  tar -xf "$ROOT/cloud_bundle.tar" -C "$ROOT"
else
  echo "  cloud_bundle.tar not here yet -- transfer it, then re-run this script"
fi

echo "=== 5/5 checks"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print("gpu", p.name, p.total_memory // 2**30, "GB")
try:
    import gsplat; print("gsplat", gsplat.__version__)
except Exception as e:
    print("gsplat MISSING:", e)
PY
for f in NeoVerse/models/NeoVerse/google \
         NeoVerse/models/NeoVerse/reconstructor.ckpt \
         NeoVerse/models/NeoVerse/Wan2.1_VAE.pth \
         NeoVerse/models/NeoVerse/models_t5_umt5-xxl-enc-bf16.pth \
         runs/train_semantic_v26_campus/checkpoint-epoch-10.safetensors \
         outputs/scene_clouds/clouds/gnd_AUw360_cloud.npz; do
  [ -e "$ROOT/$f" ] && echo "ok      $f" || echo "MISSING $f"
done
# The DiT is sharded, so check the count and the total size, not one filename.
python - <<'PY2'
import glob, os
d = os.path.join(os.environ.get("ROOT",""), "NeoVerse/models/NeoVerse")
sh = sorted(glob.glob(os.path.join(d, "diffusion_pytorch_model*.safetensors")))
gb = sum(os.path.getsize(f) for f in sh) / 2**30
print(("ok      " if gb > 20 else "MISSING ") + "DiT shards: %d files, %.1f GB" % (len(sh), gb))
PY2
df -h /workspace | tail -1
echo "==> done. Activate with: export PATH=/workspace/miniconda3/bin:\$PATH && conda activate neoverse"
