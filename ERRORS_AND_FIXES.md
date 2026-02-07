# TRELLIS Blackwell Migration - Errors & Fixes Log

## Environment
- **GPU**: NVIDIA RTX 5080 (Blackwell, compute 12.0, 16GB VRAM)
- **OS**: Ubuntu 24.04, Linux 6.14.0
- **Stack**: Python 3.11, CUDA 12.8, PyTorch 2.7.0+cu128, Driver 570+ (open)
- **Conda env**: `trellis-bw`
- **Launch command**: `XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa python -u app.py --precision auto --host 127.0.0.1`

---

## 1. kaolin==0.18.0 Not Found
**Error**: `Could not find a version that satisfies the requirement kaolin==0.18.0`
**Cause**: Install URL was using `cu124` but kaolin wheels for torch 2.7.0 are at `cu126`.
**Fix**: Changed all 3 files (setup.sh, install.py, requirements.txt) to use:
```
--extra-index-url https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html
```

## 2. nvdiffrast Build Fails
**Error**: Build fails during `pip install` of nvdiffrast from source.
**Cause**: Needed `--no-build-isolation` flag and `CUDA_HOME`/`PATH` set for the CUDA compiler.
**Fix** (in setup.sh):
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
pip install <package> --no-build-isolation
```
Applied to: nvdiffrast, diffoctreerast, mip-splatting (diff-gaussian-rasterization).

## 3. Pillow Version Conflict
**Error**: `gradio 4.44.1 requires pillow<11.0` but `rembg requires pillow>=12.1.0`.
**Cause**: Version constraint conflict between gradio and rembg.
**Fix**: Force install pillow 12.1.0 — gradio works fine despite the warning:
```bash
pip install "pillow>=12.1.0"
```

## 4. HfFolder Import Error (huggingface_hub)
**Error**: `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`
**Cause**: huggingface_hub too new for gradio 4.44.1.
**Fix**:
```bash
pip install "huggingface_hub<0.25"
```

## 5. transformers is_offline_mode Import Error
**Error**: `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`
**Cause**: transformers 5.1.0 incompatible with huggingface_hub 0.24.x.
**Fix**:
```bash
pip install "transformers>=4.35.0,<4.50" "huggingface_hub>=0.23,<0.25"
```
This installs transformers 4.48.3.

## 6. FlexiCubes Module Not Found
**Error**: `ModuleNotFoundError: No module named 'trellis.representations.mesh.flexicubes'`
**Cause**: Git submodule not initialized.
**Fix**:
```bash
cd ~/Trellis_Blackwell && git submodule update --init --recursive
```

## 7. VRAMManager `total_mem` AttributeError
**Error**: `AttributeError: 'torch.cuda._CudaDeviceProperties' has no attribute 'total_mem'`
**Cause**: PyTorch property is `total_memory`, not `total_mem`.
**Fix**: In `trellis/utils/vram_manager.py`, changed `props.total_mem` → `props.total_memory`.

## 8. Gradio TypeError "bool is not iterable"
**Error**: `TypeError: argument of type 'bool' is not iterable`
**Cause**: pydantic version too new for gradio 4.44.1.
**Fix**:
```bash
pip install "pydantic>=2.0,<2.10"
```
Installs pydantic 2.9.2.

## 9. Device Mismatch (cpu vs cuda:0) During Model Swapping
**Error**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`
**Cause**: During model swapping, `self.device` property iterates through all models and returns the wrong device after `_move_all_models_to_cpu()`.
**Fix**: In `trellis_image_to_3d.py` and `trellis_text_to_3d.py`, replaced `self.device` with direct model parameter device lookups:
```python
# encode_image:
model_device = self.models['image_cond_model'].patch_embed.proj.weight.device

# sample_sparse_structure / sample_slat:
param = next(flow_model.parameters())
desired_dtype = param.dtype
model_device = param.device
```

## 10. xformers Crashes on Blackwell (FATAL)
**Error**: `CUDA error (flash_fwd_launch_template.h:175): no kernel image is available for execution on the device` → process killed
**Cause**: xformers 0.0.30 flash-attention Hopper kernels have no binary for Blackwell (sm_120/compute 12.0). The CUDA error is fatal and kills the Python process.
**Fix**: Switch to PyTorch native SDPA (scaled_dot_product_attention):
1. Set env vars: `XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa`
2. Added `sdpa` backend support to 4 sparse attention files:
   - `trellis/modules/sparse/__init__.py` — added 'sdpa' to allowed ATTN values
   - `trellis/modules/sparse/attention/full_attn.py` — added sdpa import + dispatch
   - `trellis/modules/sparse/attention/serialized_attn.py` — added sdpa import + dispatch
   - `trellis/modules/sparse/attention/windowed_attn.py` — added sdpa import + dispatch
3. For equal-length sequences: batch as `[B, H, N, C]` → `sdpa(q, k, v)`
4. For variable-length sequences: process per-sequence in a loop
5. Full attention (`trellis/modules/attention/`) already supported `sdpa` natively.

## 11. TimestepEmbedder dtype Mismatch (Float vs Half)
**Error**: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half`
**Cause**: `timestep_embedding()` produces float32 sinusoidal embeddings, but the MLP weights are float16 after `to_dtype(torch.float16)`.
**Fix**: In `trellis/models/sparse_structure_flow.py`, cast `t_freq` to match MLP weight dtype:
```python
def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_freq = t_freq.to(self.mlp[0].weight.dtype)  # Added this line
    t_emb = self.mlp(t_freq)
    return t_emb
```

## 12. FlexiCubes index_add_ dtype Mismatch
**Error**: `RuntimeError: index_add_(): self (Float) and source (Half) must have the same scalar type`
**Cause**: `vd` and `beta_sum` tensors created as float32 (default), but `beta_group` from model is float16. After first fix, `ue_group` (geometry) was float32 while containers were half.
**Fix**: In `trellis/representations/mesh/flexicubes/flexicubes.py`, keep containers as float32 and cast sources:
```python
beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group.float())
vd = vd.index_add_(0, index=edge_group_to_vd, source=(ue_group * beta_group).float()) / beta_sum
# Same for vd_color:
vd_color = vd_color.index_add_(0, index=edge_group_to_vd, source=(uc_group * beta_group).float()) / beta_sum
```

## 13. Gaussian Rasterizer Expects Float32
**Error**: `RuntimeError: expected scalar type Float but found Half`
**Cause**: `diff_gaussian_rasterization` C++ code requires all float32 inputs, but Gaussian properties stored in float16.
**Fix**: In `trellis/renderers/gaussian_render.py`, cast all Gaussian internal tensors to float32 before rendering:
```python
# At start of render():
if pc._xyz is not None and pc._xyz.dtype != torch.float32:
    pc._xyz = pc._xyz.float()
if pc._features_dc is not None and pc._features_dc.dtype != torch.float32:
    pc._features_dc = pc._features_dc.float()
# ... same for _scaling, _rotation, _opacity
```
Also cast SH features, colors_precomp, and override_color to `.float()`.

## 14. diff-gaussian-rasterization CUDA Kernel Broken on Blackwell
**Error**: `torch.OutOfMemoryError: Tried to allocate 66846724.03 GiB` (garbage value)
**Cause**: diff-gaussian-rasterization was compiled without `TORCH_CUDA_ARCH_LIST` including sm_120. The CUDA kernels produce garbage output on Blackwell when running via PTX fallback.
**Fix**: Rebuild with explicit Blackwell architecture:
```bash
pip uninstall diff-gaussian-rasterization -y
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0" \
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="/usr/local/cuda-12.8/bin:$PATH" \
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ \
    --no-build-isolation --force-reinstall
```

## 15. nvdiffrast CUDA Kernel Broken on Blackwell
**Error**: `RuntimeError: Cuda error: 209[cudaFuncGetAttributes(&attr, (void*)fineRasterKernel);]`
**Cause**: Same as #14 — nvdiffrast compiled without sm_120 support.
**Fix**: Rebuild with Blackwell arch:
```bash
pip uninstall nvdiffrast -y
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0" \
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="/usr/local/cuda-12.8/bin:$PATH" \
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```
Also rebuilt `diffoctreerast` the same way.

## 16. Mesh Renderer Vertices Float Cast
**Error**: Potential dtype mismatch in nvdiffrast (mesh vertices may be half).
**Cause**: Mesh vertices from FlexiCubes are float32 but other attributes may be float16.
**Fix**: In `trellis/renderers/mesh_renderer.py`:
```python
vertices = mesh.vertices.float().unsqueeze(0)
```

---

## Key Lessons for Blackwell GPUs

1. **ALL CUDA extensions must be compiled with `TORCH_CUDA_ARCH_LIST` including `12.0`** (or `10.0` for compute 10.0). PTX fallback from older arches produces garbage on Blackwell. This includes:
   - diff-gaussian-rasterization
   - nvdiffrast
   - diffoctreerast
   - vox2seq (if used)

2. **xformers 0.0.30 is incompatible with Blackwell** — the flash-attention Hopper kernels crash the process. Use `XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa` and implement SDPA support in sparse attention modules.

3. **float16 dtype mismatches are pervasive** when running models in half precision. Key places:
   - TimestepEmbedder: sinusoidal embeddings are always float32 but MLP can be float16
   - FlexiCubes: geometry is float32, model params are float16 → cast sources to `.float()` at `index_add_`
   - Gaussian rendering: C++ rasterizer requires float32 → cast all Gaussian internals before render
   - Mesh rendering: nvdiffrast requires float32 → cast vertices

4. **Model swapping breaks `self.device`**: After moving models to CPU, `self.device` returns stale info. Use `next(model.parameters()).device` instead.

5. **Package version pins are critical**:
   - `huggingface_hub>=0.23,<0.25`
   - `transformers>=4.35.0,<4.50`
   - `pydantic>=2.0,<2.10`
   - `pillow>=12.1.0` (ignore gradio warning)

---

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `setup.sh` | kaolin URL fix, --no-build-isolation, CUDA_HOME |
| `install.py` | kaolin URL fix |
| `requirements.txt` | kaolin URL fix |
| `trellis/utils/vram_manager.py` | total_memory fix |
| `trellis/pipelines/trellis_image_to_3d.py` | model_device fix, model swapping |
| `trellis/pipelines/trellis_text_to_3d.py` | model_device fix, model swapping |
| `trellis/modules/sparse/__init__.py` | sdpa backend support |
| `trellis/modules/sparse/attention/full_attn.py` | sdpa dispatch |
| `trellis/modules/sparse/attention/serialized_attn.py` | sdpa dispatch |
| `trellis/modules/sparse/attention/windowed_attn.py` | sdpa dispatch |
| `trellis/models/sparse_structure_flow.py` | TimestepEmbedder dtype cast |
| `trellis/representations/mesh/flexicubes/flexicubes.py` | index_add_ float cast |
| `trellis/renderers/gaussian_render.py` | Gaussian float32 cast |
| `trellis/renderers/mesh_renderer.py` | Vertices float32 cast |
