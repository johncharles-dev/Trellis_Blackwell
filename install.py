#!/usr/bin/env python3
"""
TRELLIS Blackwell - Cross-platform installer
Target: Python 3.11, CUDA 12.8, PyTorch 2.7.0
Supports: Linux (primary), Windows (experimental)
"""

import os
import sys
import subprocess
import platform
import argparse
import shutil
from pathlib import Path

CUDA_ARCH_LIST = "8.0;8.6;8.9;10.0;12.0"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"
KAOLIN_INDEX = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html"


def run_cmd(cmd: str, desc: str = None, check: bool = True) -> subprocess.CompletedProcess:
    if desc:
        print(f"[INSTALL] {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"[INSTALL] WARNING: Command failed (exit {result.returncode}): {cmd}")
    return result


def detect_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                print(f"[GPU] {parts[0]} (compute {parts[1]}, {parts[2]}MB VRAM, driver {parts[3]})")
                return {
                    "name": parts[0],
                    "compute": parts[1],
                    "vram_mb": int(parts[2]),
                    "driver": parts[3],
                }
    except FileNotFoundError:
        pass
    print("[GPU] nvidia-smi not found. Proceeding without GPU detection.")
    return None


def install_pytorch():
    print("[PYTORCH] Installing PyTorch 2.7.0+cu128...")
    run_cmd(
        f"pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 "
        f"--index-url {PYTORCH_INDEX}",
        "Installing PyTorch"
    )


def install_basic():
    run_cmd("pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja", "Installing basic deps")
    run_cmd("pip install rembg onnxruntime-gpu==1.24.1", "Installing rembg + onnxruntime-gpu")
    run_cmd("pip install trimesh open3d==0.19.0 xatlas pyvista pymeshfix igraph transformers", "Installing 3D libs")
    run_cmd(
        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        "Installing utils3d"
    )
    # Pin versions to avoid dependency conflicts
    run_cmd(
        'pip install "huggingface_hub>=0.23,<0.25" "transformers>=4.35.0,<4.50" "pydantic>=2.0,<2.10" "pillow>=12.1.0"',
        "Pinning dependency versions"
    )


def install_xformers():
    run_cmd(
        f"pip install xformers==0.0.30 --index-url {PYTORCH_INDEX}",
        "Installing xformers 0.0.30"
    )


def install_flash_attn():
    print("[FLASH-ATTN] Attempting flash-attn install...")
    result = run_cmd("pip install flash-attn==2.8.3 --no-build-isolation", check=False)
    if result.returncode != 0:
        print("[FLASH-ATTN] pip install failed. Please install from prebuilt wheel:")
        print("  https://github.com/Dao-AILab/flash-attention/releases")
        print("  or https://flashattn.dev")


def install_spconv():
    print("[SPCONV] NOTE: Using spconv-cu126 (forward compat with CUDA 12.8)")
    print("[SPCONV] Do NOT install cumm-cu128 manually.")
    run_cmd("pip install spconv-cu126==2.3.8", "Installing spconv-cu126")


def install_kaolin():
    run_cmd(
        f"pip install kaolin==0.18.0 -f {KAOLIN_INDEX}",
        "Installing kaolin 0.18.0 (cu124 wheel)"
    )


def install_nvdiffrast():
    run_cmd("pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation", "Installing nvdiffrast")


def install_diffoctreerast():
    run_cmd("pip install git+https://github.com/JeffreyXiang/diffoctreerast.git --no-build-isolation", "Installing diffoctreerast")


def install_mipgaussian():
    tmp = Path("/tmp/extensions/mip-splatting")
    if not tmp.exists():
        run_cmd(f"git clone https://github.com/autonomousvision/mip-splatting.git {tmp}", "Cloning mip-splatting")
    run_cmd(f"pip install {tmp}/submodules/diff-gaussian-rasterization/ --no-build-isolation", "Installing diff-gaussian-rasterization")


def install_demo():
    run_cmd("pip install gradio==4.44.1 gradio_litmodel3d==0.0.1", "Installing Gradio")


def main():
    parser = argparse.ArgumentParser(description="TRELLIS Blackwell Installer")
    parser.add_argument("--all", action="store_true", help="Install everything for demo")
    parser.add_argument("--pytorch", action="store_true", help="Install PyTorch 2.7.0+cu128")
    parser.add_argument("--basic", action="store_true", help="Install basic dependencies")
    parser.add_argument("--xformers", action="store_true", help="Install xformers")
    parser.add_argument("--flash-attn", action="store_true", help="Install flash-attn")
    parser.add_argument("--spconv", action="store_true", help="Install spconv-cu126")
    parser.add_argument("--kaolin", action="store_true", help="Install kaolin")
    parser.add_argument("--nvdiffrast", action="store_true", help="Install nvdiffrast")
    parser.add_argument("--diffoctreerast", action="store_true", help="Install diffoctreerast")
    parser.add_argument("--mipgaussian", action="store_true", help="Install mip-splatting")
    parser.add_argument("--demo", action="store_true", help="Install Gradio demo")
    args = parser.parse_args()

    # Set environment
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    cuda_bin = os.path.join(os.environ["CUDA_HOME"], "bin")
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    print(f"[INSTALL] Platform: {platform.system()} {platform.machine()}")
    print(f"[INSTALL] Python: {sys.version}")
    print(f"[INSTALL] CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"[INSTALL] TORCH_CUDA_ARCH_LIST: {CUDA_ARCH_LIST}")

    gpu = detect_gpu()

    if args.all:
        args.pytorch = True
        args.basic = True
        args.xformers = True
        args.flash_attn = True
        args.spconv = True
        args.kaolin = True
        args.nvdiffrast = True
        args.diffoctreerast = True
        args.mipgaussian = True
        args.demo = True

    if args.pytorch:
        install_pytorch()
    if args.basic:
        install_basic()
    if args.xformers:
        install_xformers()
    if args.flash_attn:
        install_flash_attn()
    if args.spconv:
        install_spconv()
    if args.kaolin:
        install_kaolin()
    if args.nvdiffrast:
        install_nvdiffrast()
    if args.diffoctreerast:
        install_diffoctreerast()
    if args.mipgaussian:
        install_mipgaussian()
    if args.demo:
        install_demo()

    print("\n[INSTALL] Done!")


if __name__ == "__main__":
    main()
