#!/bin/bash
# TRELLIS Blackwell Setup Script
# Target: Ubuntu 24.04, Python 3.11, CUDA 12.8, RTX 5080 (compute 10.0)
# Also supports: RTX 3090 (8.6), RTX 4090 (8.9), A100 (8.0)

set -e

# ============================================================
# Parse arguments
# ============================================================
TEMP=$(getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,demo,all -n 'setup.sh' -- "$@")
eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
XFORMERS=false
FLASHATTN=false
DIFFOCTREERAST=false
VOX2SEQ=false
SPCONV=false
MIPGAUSSIAN=false
KAOLIN=false
NVDIFFRAST=false
DEMO=false
ALL=false
ERROR=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --demo) DEMO=true ; shift ;;
        --all) ALL=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "TRELLIS Blackwell Setup (CUDA 12.8 + PyTorch 2.7.0)"
    echo ""
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment (Python 3.11)"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn (prebuilt wheel)"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv-cu126 (forward compat with CUDA 12.8)"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --demo                  Install Gradio demo dependencies"
    echo "  --all                   Install everything for demo (recommended)"
    exit 0
fi

if [ "$ALL" = true ] ; then
    BASIC=true
    XFORMERS=true
    FLASHATTN=true
    SPCONV=true
    KAOLIN=true
    NVDIFFRAST=true
    DIFFOCTREERAST=true
    MIPGAUSSIAN=true
    DEMO=true
fi

# ============================================================
# GPU Detection
# ============================================================
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | xargs)
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
        echo "[GPU] Detected: $GPU_NAME (compute $COMPUTE_CAP, ${VRAM_MB}MB VRAM, driver $DRIVER_VERSION)"
    else
        echo "[GPU] nvidia-smi not found. Please install NVIDIA drivers first."
        exit 1
    fi
}

detect_gpu

# Set CUDA arch list based on detected GPU
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0"
echo "[CUDA] CUDA_HOME=$CUDA_HOME"
echo "[CUDA] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

WORKDIR=$(pwd)

# ============================================================
# Create conda environment
# ============================================================
if [ "$NEW_ENV" = true ] ; then
    echo "[ENV] Creating conda environment: trellis-bw (Python 3.11)"
    conda create -n trellis-bw python=3.11 -y
    echo "[ENV] Activate with: conda activate trellis-bw"
    echo "[ENV] Then re-run this script with desired options."
    exit 0
fi

# ============================================================
# Verify Python and PyTorch
# ============================================================
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "[SYSTEM] Python: $PYTHON_VERSION"

# Check if PyTorch is installed, if not install it
if python -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda or 'none')")
    echo "[SYSTEM] PyTorch: $PYTORCH_VERSION, CUDA: $CUDA_VERSION"
else
    echo "[PYTORCH] Installing PyTorch 2.7.0+cu128..."
    pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda or 'none')")
    echo "[SYSTEM] PyTorch: $PYTORCH_VERSION, CUDA: $CUDA_VERSION"
fi

# ============================================================
# Basic dependencies
# ============================================================
if [ "$BASIC" = true ] ; then
    echo "[BASIC] Installing basic dependencies..."
    pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja
    pip install rembg onnxruntime-gpu==1.24.1
    pip install trimesh open3d==0.19.0 xatlas pyvista pymeshfix igraph transformers
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    echo "[BASIC] Done."
fi

# ============================================================
# Training dependencies
# ============================================================
if [ "$TRAIN" = true ] ; then
    echo "[TRAIN] Installing training dependencies..."
    pip install tensorboard pandas lpips
    pip uninstall -y pillow
    sudo apt install -y libjpeg-dev
    pip install pillow-simd
    echo "[TRAIN] Done."
fi

# ============================================================
# xformers (must match torch 2.7.x)
# ============================================================
if [ "$XFORMERS" = true ] ; then
    echo "[XFORMERS] Installing xformers 0.0.30 for PyTorch 2.7.x + CUDA 12.8..."
    pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
    echo "[XFORMERS] Done."
fi

# ============================================================
# flash-attn (prebuilt wheel - NO pip source builds)
# ============================================================
if [ "$FLASHATTN" = true ] ; then
    echo "[FLASH-ATTN] Installing flash-attn 2.8.3..."
    echo "[FLASH-ATTN] Attempting pip install (may need prebuilt wheel)..."
    pip install flash-attn==2.8.3 --no-build-isolation 2>/dev/null || {
        echo "[FLASH-ATTN] pip install failed. Please download a prebuilt wheel from:"
        echo "  https://github.com/Dao-AILab/flash-attention/releases"
        echo "  or https://flashattn.dev"
        echo "  Then: pip install flash_attn-2.8.3+cuXXX-cp311-cp311-linux_x86_64.whl"
    }
fi

# ============================================================
# spconv (cu126 - forward compat with CUDA 12.8)
# ============================================================
if [ "$SPCONV" = true ] ; then
    echo "[SPCONV] Installing spconv-cu126 2.3.8 (forward compat with CUDA 12.8)..."
    echo "[SPCONV] NOTE: Do NOT install cumm-cu128 manually - let spconv pull its own version."
    pip install spconv-cu126==2.3.8
    echo "[SPCONV] Done."
fi

# ============================================================
# kaolin (cu124 wheel - forward compat with CUDA 12.8)
# ============================================================
if [ "$KAOLIN" = true ] ; then
    echo "[KAOLIN] Installing kaolin 0.18.0 (cu124 wheel, forward compat)..."
    pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html
    echo "[KAOLIN] Done."
fi

# ============================================================
# nvdiffrast (compile from source)
# ============================================================
if [ "$NVDIFFRAST" = true ] ; then
    echo "[NVDIFFRAST] Installing nvdiffrast from source..."
    mkdir -p /tmp/extensions
    if [ ! -d /tmp/extensions/nvdiffrast ]; then
        git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
    fi
    pip install /tmp/extensions/nvdiffrast --no-build-isolation
    echo "[NVDIFFRAST] Done."
fi

# ============================================================
# diffoctreerast (compile from source)
# ============================================================
if [ "$DIFFOCTREERAST" = true ] ; then
    echo "[DIFFOCTREERAST] Installing diffoctreerast from source..."
    mkdir -p /tmp/extensions
    if [ ! -d /tmp/extensions/diffoctreerast ]; then
        git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
    fi
    pip install /tmp/extensions/diffoctreerast --no-build-isolation
    echo "[DIFFOCTREERAST] Done."
fi

# ============================================================
# mip-splatting (compile from source)
# ============================================================
if [ "$MIPGAUSSIAN" = true ] ; then
    echo "[MIP-GAUSSIAN] Installing diff-gaussian-rasterization from mip-splatting..."
    mkdir -p /tmp/extensions
    if [ ! -d /tmp/extensions/mip-splatting ]; then
        git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
    fi
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation
    echo "[MIP-GAUSSIAN] Done."
fi

# ============================================================
# vox2seq (compile from source)
# ============================================================
if [ "$VOX2SEQ" = true ] ; then
    echo "[VOX2SEQ] Installing vox2seq..."
    mkdir -p /tmp/extensions
    cp -r extensions/vox2seq /tmp/extensions/vox2seq
    pip install /tmp/extensions/vox2seq
    echo "[VOX2SEQ] Done."
fi

# ============================================================
# Gradio demo
# ============================================================
if [ "$DEMO" = true ] ; then
    echo "[DEMO] Installing Gradio demo dependencies..."
    pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
    echo "[DEMO] Done."
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "============================================================"
