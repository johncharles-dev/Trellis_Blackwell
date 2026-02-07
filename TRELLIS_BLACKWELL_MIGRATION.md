# TRELLIS Blackwell Migration Plan

## Context

This document contains the full analysis and implementation plan for modifying the original Microsoft TRELLIS (D:\models\TRELLIS or wherever cloned on Ubuntu) to support NVIDIA Blackwell GPUs (RTX 5080, compute capability 10.0) with auto VRAM detection and tiered precision.

A reference implementation exists at the "trellis-stable" fork by Igor Aherne (https://github.com/IgorAherne/trellis-stable-projectorz) which partially solves this for the image-to-3D pipeline only. Our goal is a CLEAN implementation covering ALL features: text-to-3D, multi-image, variant generation, local editing, and training.

---

## Why Original TRELLIS Fails on Blackwell

1. **CUDA 11.8 has no Blackwell support** — original pins PyTorch 2.4.0 + CUDA 11.8
2. **Hardcoded float32 everywhere** — wastes VRAM, no float16 option
3. **All models loaded to GPU simultaneously** — needs 18-24GB, no memory management
4. **CUDA extensions compiled for old architectures** — no compute 10.0 in arch list
5. **FlexiCubes uses int64 by default** — wastes 2x memory on index tables
6. **Linux-only build system** — setup.sh is bash, no Windows support

---

## Verified Compatible Stack (Ubuntu 24.04)

```
Ubuntu 24.04 LTS
Python 3.11 (via conda or deadsnakes PPA)
NVIDIA Driver 570+ (MUST use -open variant for Blackwell)
CUDA Toolkit 12.8
GCC 13.2 (Ubuntu default, compatible with CUDA 12.8)
```

### Package Versions — All Verified Compatible

```bash
# Core ML
torch==2.7.0+cu128              # from https://download.pytorch.org/whl/cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128
xformers==0.0.30                # from https://download.pytorch.org/whl/cu128 (MUST match torch 2.7.x)
flash-attn==2.8.3               # prebuilt wheel from flashattn.dev or GitHub releases

# Sparse convolution
spconv-cu126==2.3.8             # NO cu128 exists; cu126 works via CUDA forward compat
# Do NOT manually install cumm-cu128 — let spconv pull its own cumm version

# 3D libraries
kaolin==0.18.0                  # from https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu124.html
                                # No cu128 wheel for torch>=2.6; cu124 works via forward compat
open3d==0.19.0                  # pip wheel only, do NOT build from source on 24.04
trimesh==4.11.1
pyvista==0.46.5

# CUDA extensions (compile from source with CUDA_HOME and TORCH_CUDA_ARCH_LIST set)
nvdiffrast==0.4.0               # git+https://github.com/NVlabs/nvdiffrast.git
diffoctreerast                  # git+https://github.com/JeffreyXiang/diffoctreerast.git
diff_gaussian_rasterization     # from mip-splatting submodule

# Background removal
rembg>=2.0.72                   # requires Python >= 3.11
onnxruntime-gpu==1.24.1         # CUDA 12.x compatible; do NOT install onnxruntime alongside

# Other
pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy
ninja rembg[gpu] xatlas pymeshfix igraph transformers
gradio==4.44.1 gradio_litmodel3d==0.0.1
utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
```

### Known Conflicts / Gotchas

1. **spconv-cu128 does not exist** — use spconv-cu126, works via CUDA forward compat
2. **kaolin cu128 does not exist for torch>=2.6** — use cu124 wheel
3. **flash-attn has NO pip wheels** — only source tarballs on PyPI. Use prebuilt from flashattn.dev or GitHub releases
4. **cumm-cu128 (v0.8.2) is incompatible with spconv 2.3.8** (expects cumm ~0.5-0.6) — do NOT install manually
5. **xformers version MUST match torch** — 0.0.30 = torch 2.7.x, 0.0.34 = torch 2.10.x
6. **open3d fails to build from source on Ubuntu 24.04** — always use pip wheel

### Ubuntu 24.04 + RTX 5080 Setup Gotchas

- MUST use open kernel modules: `nvidia-driver-570-open` (proprietary driver does NOT work with Blackwell)
- Blacklist nouveau: create `/etc/modprobe.d/blacklist-nouveau.conf`
- Use X11, NOT Wayland (Wayland crashes with RTX 50 series)
- BIOS: Enable "Above 4G Decoding" + "Resizable BAR", disable CSM
- Python 3.11 not default on 24.04 (ships 3.12) — use conda

### CUDA Extension Compilation

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0"
# 8.0=A100, 8.6=RTX3090/A6000, 8.9=RTX4090, 10.0=RTX5080 Blackwell
```

---

## Implementation Plan — 6 Layers

### Layer 1: New Install System

Rewrite `setup.sh` to:
- Auto-detect GPU compute capability
- Pin to CUDA 12.8 + PyTorch 2.7.0 + Python 3.11
- Add `install.py` for Windows with pre-compiled .whl files
- Add `requirements.txt` with pinned versions

### Layer 2: VRAMManager (NEW file: `trellis/utils/vram_manager.py`)

Auto-detect VRAM and set behavior tier:
- Tier "high" (>=24GB): float32, all models on GPU, no swapping
- Tier "medium" (12-23GB): float16, model swapping between pipeline stages
- Tier "low" (8-11GB): float16, aggressive swapping, int32 FlexiCubes, CPU rembg

Properties: dtype, use_model_swapping, rembg_provider, etc.
Accept override via CLI: `--precision auto|full|half` and `--vram-tier auto|high|medium|low`

### Layer 3: Dynamic dtype (7 files)

Replace hardcoded `.float()` with dynamic dtype matching:

**File 1: `trellis/pipelines/trellis_image_to_3d.py`** — 3 spots:
- `encode_image()`: `desired_dtype = self.models['image_cond_model'].patch_embed.proj.weight.dtype`
- `sample_sparse_structure()`: `desired_dtype = next(flow_model.parameters()).dtype`
- `sample_slat()`: same pattern

**File 2: `trellis/pipelines/trellis_text_to_3d.py`** — same 3 spots + `run_variant()`

**File 3: `trellis/modules/sparse/norm.py`** — use `weight_dtype` instead of hardcoded float32

**File 4: `trellis/representations/mesh/utils_cube.py`** — preserve input dtype in 3 functions

**File 5: `trellis/utils/postprocessing_utils.py`** — float16 observations on low tier

**File 6: `trellis/utils/render_utils.py`** — dynamic dtype for camera matrices

**File 7: `trellis/pipelines/base.py`** — `.to()` method to propagate dtype to all models

### Layer 4: Model Swapping Engine (base.py + both pipelines)

Add to `trellis/pipelines/base.py`:
```python
def _move_models(self, names, device, empty_cache=True):
    for name in names:
        if name in self.models:
            current = next(self.models[name].parameters()).device
            if str(current) != device:
                self.models[name].to(device)
    if empty_cache and device == 'cpu':
        torch.cuda.empty_cache()

def _move_all_models_to_cpu(self):
    self._move_models(list(self.models.keys()), 'cpu', empty_cache=True)
```

Pipeline execution order (only 1-2 models on GPU at a time):
1. image_cond_model -> GPU -> encode -> CPU
2. sparse_structure_flow + decoder -> GPU -> sample -> CPU
3. slat_flow_model -> GPU -> sample -> CPU
4. Each decoder (mesh/gs/rf) -> GPU -> decode -> CPU (one at a time)

Wrap with: `if self.vram_manager.use_model_swapping:` so high-tier skips this.

### Layer 5: FlexiCubes int32 Fork

Switch .gitmodules to use Igor Aherne's fork: https://github.com/IgorAherne/flexicubes-stable-projectorz

Or apply these changes to the original FlexiCubes:
- All lookup tables: add `dtype=torch.int32`
- `.sum()` calls: add `dtype=torch.int32`
- `torch.arange()` calls: add `dtype=torch.int32`
- Face index returns: `.to(torch.int32)`

### Layer 6: Unified Entry Points

Modify `app.py` and `app_text.py`:
```python
parser.add_argument('--precision', choices=['auto', 'full', 'half'], default='auto')
parser.add_argument('--vram-tier', choices=['auto', 'high', 'medium', 'low'], default='auto')
```

ALL features must work:
- Single image -> 3D (TrellisImageTo3DPipeline.run())
- Multi-image -> 3D (TrellisImageTo3DPipeline.run_multi_image())
- Text -> 3D (TrellisTextTo3DPipeline.run())
- Variant generation (TrellisTextTo3DPipeline.run_variant())
- Mesh/Gaussian/RF output formats
- GLB export
- Training (train.py — already has mixed precision, mostly untouched)

---

## File Change Map (14 files total)

| # | File | Type | Changes |
|---|------|------|---------|
| 1 | `trellis/utils/vram_manager.py` | NEW | VRAM detection + tier logic |
| 2 | `setup.sh` | REWRITE | CUDA 12.8 + GPU detection |
| 3 | `install.py` | NEW | Windows installer with pre-built wheels |
| 4 | `requirements.txt` | NEW | Pinned dependency versions |
| 5 | `trellis/pipelines/base.py` | EDIT | Add VRAMManager, _move_models(), error-tolerant model loading, .to() dtype propagation |
| 6 | `trellis/pipelines/trellis_image_to_3d.py` | EDIT | dtype inference (3 spots) + model swapping in run(), run_multi_image(), decode_slat(), CPU rembg |
| 7 | `trellis/pipelines/trellis_text_to_3d.py` | EDIT | Same dtype + swapping for text pipeline + run_variant() |
| 8 | `.gitmodules` (FlexiCubes) | EDIT | Point to int32 fork |
| 9 | `trellis/representations/mesh/utils_cube.py` | EDIT | Preserve dtype in 3 functions |
| 10 | `trellis/modules/sparse/norm.py` | EDIT | Use weight_dtype pattern |
| 11 | `trellis/utils/postprocessing_utils.py` | EDIT | float16 observations on low tier |
| 12 | `trellis/utils/render_utils.py` | EDIT | Dynamic dtype for camera matrices |
| 13 | `app.py` | EDIT | Add --precision auto / --vram-tier args |
| 14 | `app_text.py` | EDIT | Same args |

---

## Reference: Original TRELLIS Architecture

The pipeline has 6 models:
- image_cond_model (~600MB) — DINOv2 ViT-L14
- sparse_structure_flow_model (~200MB) — 3D Vision Transformer
- sparse_structure_decoder (~100MB) — VAE decoder
- slat_flow_model (~2-4GB) — Sparse Transformer (THE BIG ONE)
- slat_decoder_gs (~500MB) — Gaussian decoder
- slat_decoder_mesh (~500MB) — Mesh decoder (FlexiCubes)
- slat_decoder_rf (~500MB) — Radiance Field decoder (optional)

Two-stage pipeline:
1. Sparse Structure: Encode input -> Flow sample -> Decode to voxel coords
2. Structured Latent (SLAT): Flow sample on voxels -> Decode to mesh/gaussian/RF

Environment variables:
- ATTN_BACKEND: 'flash_attn' (default) or 'xformers'
- SPARSE_BACKEND: 'spconv' (default) or 'torchsparse'
- SPCONV_ALGO: 'native' or 'auto'

---

## Reference: Trellis-Stable Changes (Igor Aherne's fork)

What he changed (for reference, not to copy blindly):
1. float16 support via --precision half flag
2. int32 FlexiCubes fork
3. CPU<->GPU model swapping in trellis_image_to_3d.py
4. CPU-based rembg (CPUExecutionProvider)
5. Pre-compiled Windows wheels for CUDA 12.8
6. FastAPI server for StableProjectorz integration
7. Gradio web interface
8. Dynamic dtype inference (desired_dtype pattern)
9. Image resize before background removal

His fork only covers image-to-3D. We need to apply similar patterns to ALL pipelines.
