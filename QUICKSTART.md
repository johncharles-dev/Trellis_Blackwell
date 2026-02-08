# Trellis_Blackwell — Quick Setup Guide

One-shot setup for rented GPU instances (RTX 5090, 5080, 4090, 3090, A100, etc.).

## Prerequisites

The rented instance should have:
- Ubuntu 22.04+ (24.04 recommended)
- NVIDIA driver 570+ installed (use `nvidia-driver-570-open` for Blackwell GPUs)
- CUDA Toolkit 12.8 installed at `/usr/local/cuda-12.8`
- conda or miniconda installed

> Most cloud GPU providers (RunPod, Vast.ai, Lambda, etc.) come with drivers and CUDA pre-installed. Verify with `nvidia-smi`.

---

## Setup (copy-paste the whole block)

```bash
# 1. Clone the repo
git clone https://github.com/johncharles-dev/Trellis_Blackwell.git
cd Trellis_Blackwell

# 2. Create conda environment
conda create -n trellis-bw python=3.11 -y
conda activate trellis-bw

# 3. Install PyTorch 2.7.0 + CUDA 12.8
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Install xformers (needed as dependency even though disabled at runtime on Blackwell)
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128

# 5. Install flash-attn
pip install flash-attn==2.8.3 --no-build-isolation

# 6. Install spconv (cu126 — forward compat with CUDA 12.8)
pip install spconv-cu126==2.3.8

# 7. Install kaolin (cu126 wheel — forward compat)
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html

# 8. Install basic dependencies
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja
pip install rembg onnxruntime-gpu==1.24.1
pip install trimesh open3d==0.19.0 xatlas pyvista pymeshfix igraph transformers
pip install "huggingface_hub>=0.23,<0.25" "transformers>=4.35.0,<4.50" "pydantic>=2.0,<2.10"
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# 9. Install Gradio UI
pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

# 10. Compile CUDA extensions with Blackwell support
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0"

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# diffoctreerast
pip install git+https://github.com/JeffreyXiang/diffoctreerast.git --no-build-isolation

# diff-gaussian-rasterization (mip-splatting)
mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# 11. Run!
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa python -u app.py --precision auto --host 0.0.0.0
```

The app will be available at `http://<instance-ip>:7860`.

---

## Quick run (after setup is done)

```bash
cd Trellis_Blackwell
conda activate trellis-bw
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa python -u app.py --precision auto --host 0.0.0.0
```

---

## Environment variables explained

| Variable | Purpose | Required on |
|----------|---------|-------------|
| `XFORMERS_DISABLED=1` | Disables xFormers in DINOv2 image encoder, falls back to PyTorch SDPA | Blackwell GPUs (sm_120) |
| `ATTN_BACKEND=sdpa` | Switches Trellis sparse/dense attention to PyTorch SDPA | Blackwell GPUs (sm_120) |
| `SPCONV_ALGO=native` | Skips spconv benchmarking at startup (faster cold start) | Optional, any GPU |

On non-Blackwell GPUs (4090, 3090, A100) where xFormers/flash-attn work natively:
```bash
# xFormers backend (default for non-Blackwell)
ATTN_BACKEND=xformers python -u app.py --precision auto --host 0.0.0.0

# Or flash-attn backend
ATTN_BACKEND=flash_attn python -u app.py --precision auto --host 0.0.0.0
```

---

## Troubleshooting

This section covers every dependency conflict and installation issue encountered across different machines. Issues are grouped by category. All code fixes (dtype mismatches, device errors) are already applied in this repo — just `git pull` to get them. The issues below are about **environment setup** that varies per machine.

---

### A. Dependency version conflicts

These happen because pip resolves to the latest versions by default, which break older packages like gradio 4.44.1.

**A1. `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`**
- **Cause**: pip installed huggingface_hub 0.25+ which removed `HfFolder`. Gradio 4.44.1 needs the old API.
- **Fix**:
```bash
pip install "huggingface_hub>=0.23,<0.25"
```

**A2. `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`**
- **Cause**: transformers 5.x is incompatible with huggingface_hub 0.24.x.
- **Fix**: Pin both together:
```bash
pip install "transformers>=4.35.0,<4.50" "huggingface_hub>=0.23,<0.25"
```

**A3. `TypeError: argument of type 'bool' is not iterable` (Gradio crash)**
- **Cause**: pydantic 2.10+ breaks gradio 4.44.1 internals.
- **Fix**:
```bash
pip install "pydantic>=2.0,<2.10"
```

**A4. `gradio 4.44.1 requires pillow<11.0` but `rembg requires pillow>=12.1.0`**
- **Cause**: Version conflict between gradio and rembg. pip will warn but both work fine together.
- **Fix**: Force install pillow and ignore the warning:
```bash
pip install "pillow>=12.1.0"
```

**A5. Version pin cheat sheet** (install these AFTER everything else to fix conflicts):
```bash
pip install "huggingface_hub>=0.23,<0.25" \
            "transformers>=4.35.0,<4.50" \
            "pydantic>=2.0,<2.10" \
            "pillow>=12.1.0"
```

---

### B. Package not found / cannot install

**B1. `Could not find a version that satisfies the requirement kaolin==0.18.0`**
- **Cause**: kaolin has no cu128 wheel. Must point pip to the cu126 index.
- **Fix**:
```bash
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html
```

**B2. `Could not find spconv-cu128`**
- **Cause**: spconv has no cu128 build. cu126 works via CUDA forward compatibility.
- **Fix**:
```bash
pip install spconv-cu126==2.3.8
```
- **WARNING**: Do NOT manually install `cumm-cu128`. It is incompatible with spconv 2.3.8. Let spconv pull its own cumm version.

**B3. `flash-attn` build fails from source**
- **Cause**: flash-attn PyPI package is source-only, compilation takes 30+ minutes and may fail.
- **Fix**: Use a prebuilt wheel:
```bash
# Try pip first (needs compilation tools)
pip install flash-attn==2.8.3 --no-build-isolation

# If that fails, download prebuilt wheel from:
#   https://github.com/Dao-AILab/flash-attention/releases
#   or https://flashattn.dev
# Then: pip install flash_attn-2.8.3+cuXXX-cp311-cp311-linux_x86_64.whl
```

**B4. `ModuleNotFoundError: No module named 'trellis.representations.mesh.flexicubes'`**
- **Cause**: FlexiCubes directory is missing (was a git submodule, now included directly).
- **Fix**:
```bash
git pull origin main   # flexicubes is now a regular directory in the repo
```

---

### C. CUDA extension build failures

All three CUDA extensions (nvdiffrast, diffoctreerast, diff-gaussian-rasterization) must be compiled from source. Common build failures:

**C1. Build fails with "nvcc not found" or "CUDA_HOME not set"**
- **Cause**: CUDA toolkit not in PATH, or CUDA_HOME not pointing to the right version.
- **Fix**: Set these BEFORE any `pip install` of CUDA extensions:
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

# Verify:
nvcc --version   # Should show CUDA 12.8
```
- **On cloud instances**: CUDA may be at `/usr/local/cuda` (symlink) or `/usr/local/cuda-12.X`. Check with `ls /usr/local/cuda*`.

**C2. Build fails with "unsupported GNU version" or GCC errors**
- **Cause**: GCC version incompatible with CUDA 12.8. CUDA 12.8 supports GCC up to 13.x.
- **Fix**:
```bash
# Check GCC version
gcc --version

# Ubuntu 24.04 ships GCC 13.2 — this works.
# Ubuntu 22.04 ships GCC 11.x — this also works.
# If you have GCC 14+, install an older version:
sudo apt install gcc-13 g++-13
export CC=gcc-13 CXX=g++-13
```

**C3. Extension compiles but produces garbage values / `Tried to allocate 66846724.03 GiB`**
- **Cause**: Extension compiled without Blackwell arch in `TORCH_CUDA_ARCH_LIST`. PTX fallback from older architectures produces garbage on Blackwell.
- **Fix**: Rebuild ALL CUDA extensions with the correct arch list:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

# Rebuild nvdiffrast
pip uninstall nvdiffrast -y
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# Rebuild diffoctreerast
pip uninstall diffoctreerast -y
pip install git+https://github.com/JeffreyXiang/diffoctreerast.git --no-build-isolation

# Rebuild diff-gaussian-rasterization
pip uninstall diff-gaussian-rasterization -y
mkdir -p /tmp/extensions
[ ! -d /tmp/extensions/mip-splatting ] && \
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ \
    --no-build-isolation --force-reinstall
```

**C4. `RuntimeError: Cuda error: 209[cudaFuncGetAttributes...]` from nvdiffrast**
- **Cause**: Same as C3 — nvdiffrast compiled without sm_120 support.
- **Fix**: Same as C3 — rebuild nvdiffrast with correct TORCH_CUDA_ARCH_LIST.

**C5. How to verify CUDA extensions are correctly compiled**
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute: {torch.cuda.get_device_capability(0)}')

# Test each extension
try:
    import nvdiffrast; print('nvdiffrast: OK')
except Exception as e: print(f'nvdiffrast: FAIL - {e}')

try:
    import diffoctreerast; print('diffoctreerast: OK')
except Exception as e: print(f'diffoctreerast: FAIL - {e}')

try:
    import diff_gaussian_rasterization; print('diff_gaussian_rasterization: OK')
except Exception as e: print(f'diff_gaussian_rasterization: FAIL - {e}')

try:
    import spconv; print('spconv: OK')
except Exception as e: print(f'spconv: FAIL - {e}')

try:
    import kaolin; print('kaolin: OK')
except Exception as e: print(f'kaolin: FAIL - {e}')
"
```

---

### D. Runtime CUDA errors (after successful install)

**D1. `CUDA error (flash_fwd_launch_template.h:175): no kernel image is available for execution on the device`**
- **Cause**: xFormers flash-attention has no Blackwell kernels. This is FATAL — kills the Python process.
- **This error has TWO causes that need TWO fixes:**

| Missing env var | What crashes | When it crashes |
|----------------|-------------|-----------------|
| `ATTN_BACKEND=sdpa` | Trellis attention layers | During model inference |
| `XFORMERS_DISABLED=1` | DINOv2 image encoder | On first image upload |

- **Fix**: Always launch with BOTH:
```bash
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa python -u app.py --precision auto --host 0.0.0.0
```
- **How to tell which one is missing**: If the app starts and Gradio loads but crashes when you upload an image → missing `XFORMERS_DISABLED=1`. If it crashes during startup model loading → missing `ATTN_BACKEND=sdpa`.

**D2. `RuntimeError: Expected all tensors to be on the same device (cpu vs cuda:0)`**
- **Cause**: Model swapping bug in pipeline code. Already fixed in this repo.
- **Fix**: `git pull` to get the latest code.

**D3. `RuntimeError: mat1 and mat2 must have the same dtype (Float vs Half)`**
- **Cause**: float16 dtype mismatch in TimestepEmbedder. Already fixed in this repo.
- **Fix**: `git pull` to get the latest code.

**D4. `CUDA out of memory`**
- **Cause**: GPU VRAM insufficient for current tier.
- **Fix**:
```bash
# Force low-memory mode (8GB GPUs)
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa python -u app.py --precision half --vram-tier low --host 0.0.0.0

# Or medium mode (12-16GB GPUs) — this is auto-detected but can be forced:
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa python -u app.py --precision half --vram-tier medium --host 0.0.0.0
```

---

### E. System / driver issues (Blackwell-specific)

**E1. GPU not detected / nvidia-smi fails**
- **Blackwell (RTX 5080/5090)** requires:
  - Open kernel modules: `nvidia-driver-570-open` (NOT the proprietary `nvidia-driver-570`)
  - BIOS settings: "Above 4G Decoding" enabled, "Resizable BAR" enabled, CSM disabled
  - Nouveau blacklisted: create `/etc/modprobe.d/blacklist-nouveau.conf` with:
    ```
    blacklist nouveau
    options nouveau modeset=0
    ```
  - Then: `sudo update-initramfs -u && sudo reboot`

**E2. Display / GUI crashes on Blackwell**
- **Cause**: Wayland display server is incompatible with RTX 50 series.
- **Fix**: Switch to X11:
  - Edit `/etc/gdm3/custom.conf`, set `WaylandEnable=false`
  - Reboot
- **Note**: For headless rented instances (no display), this is not an issue.

**E3. CUDA version mismatch — "CUDA driver version is insufficient"**
- **Cause**: Driver too old for CUDA 12.8.
- **Fix**: Install driver 570+:
```bash
sudo apt install nvidia-driver-570-open   # Blackwell
# or
sudo apt install nvidia-driver-570        # Non-Blackwell (4090, 3090, A100)
```

---

### F. Cloud provider-specific notes

**RunPod / Vast.ai / Lambda Labs:**
- Usually come with drivers + CUDA pre-installed. Check CUDA version with `nvcc --version`.
- If CUDA is at `/usr/local/cuda` instead of `/usr/local/cuda-12.8`, adjust `CUDA_HOME`:
```bash
export CUDA_HOME=/usr/local/cuda   # use the symlink
```
- Conda may not be installed. Install miniconda first:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
```

**Docker containers:**
- If running inside Docker, ensure `--gpus all` is passed to `docker run`.
- CUDA toolkit may need to be installed inside the container even if the host has it.

---

### G. Quick diagnostic commands

Run these to diagnose problems on any machine:

```bash
# 1. Check GPU and driver
nvidia-smi

# 2. Check CUDA toolkit
nvcc --version

# 3. Check Python environment
python --version
pip list | grep -E "torch|xformers|flash|spconv|kaolin|gradio|pydantic|huggingface|transformers|pillow"

# 4. Check GPU compute capability from Python
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Compute: {torch.cuda.get_device_capability(0)}, CUDA: {torch.version.cuda}')"

# 5. Check CUDA extensions (see C5 above for full script)
python -c "import nvdiffrast, diffoctreerast, diff_gaussian_rasterization, spconv, kaolin; print('All extensions OK')"

# 6. Check if xformers is disabled properly
XFORMERS_DISABLED=1 python -c "
import os; os.environ['XFORMERS_DISABLED'] = '1'
import torch
hub_dir = torch.hub.get_dir()
print(f'XFORMERS_DISABLED={os.environ.get(\"XFORMERS_DISABLED\", \"not set\")}')
"
```

---

### H. Version lock reference

These are the exact versions verified to work together. If you hit any unexplained conflict, force-install these:

```bash
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install spconv-cu126==2.3.8
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html
pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
pip install open3d==0.19.0 onnxruntime-gpu==1.24.1
pip install "huggingface_hub>=0.23,<0.25" "transformers>=4.35.0,<4.50" "pydantic>=2.0,<2.10" "pillow>=12.1.0"
```

For full details on every error encountered during development, see [ERRORS_AND_FIXES.md](ERRORS_AND_FIXES.md).

---

## What to verify on a new GPU

After setup, confirm these in the startup logs:

```
[GPU] Detected: NVIDIA GeForce RTX XXXX (compute X.X, XXXXMB VRAM, driver 570.XX)
[SPARSE] Backend: spconv, Attention: sdpa
[VRAMManager] VRAMManager(tier=..., dtype=..., vram=...MB, swapping=...)
[ATTENTION] Using backend: sdpa
xFormers is disabled (SwiGLU)        ← confirms XFORMERS_DISABLED=1 is working
xFormers is disabled (Attention)     ← confirms DINOv2 using SDPA
Running on local URL: http://0.0.0.0:7860
```

Then upload an image and generate a 3D asset. If it completes without CUDA errors, the GPU is fully supported.
