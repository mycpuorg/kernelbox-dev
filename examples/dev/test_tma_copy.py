"""Test TMA descriptor support: TMA copy kernel.
Run with: kbox iterate examples/dev/test_tma_copy.py --once
"""
import torch
import numpy as np
import subprocess
import os
import shutil

TILE_SIZE = 128  # must match kernel's TILE_SIZE

def _find_nvcc():
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    candidate = os.path.join(cuda_home, "bin", "nvcc")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError("nvcc not found on PATH or in CUDA_HOME")

def init_once():
    # Compile TMA kernel to cubin
    nvcc = _find_nvcc()
    arch = subprocess.check_output(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
        shell=True).decode().strip().replace('.', '')
    cubin_path = "/tmp/test_tma_copy.cubin"
    subprocess.check_call([
        nvcc, "-cubin", f"-arch=sm_{arch}", "-std=c++17",
        "-o", cubin_path, "examples/dev/tma_copy.cu"
    ])

    n = 1024
    x = torch.randn(n, device="cuda")
    return {
        "kernel": cubin_path,
        "func_name": "tma_copy",
        "inputs": [x],
        "expected": [x.clone()],  # TMA copy should be identity
        "outputs": 1,
        "grid": (n + TILE_SIZE - 1) // TILE_SIZE,
        "block": 32,
    }

def run(inputs, kernel):
    x = inputs[0]
    n = x.numel()
    grid = (n + TILE_SIZE - 1) // TILE_SIZE

    # Create TMA descriptor for input tensor
    # Shape is 1D: (n,) elements, box shape is (TILE_SIZE,)
    tma_in = kernel.tma_desc(
        index=0,          # input chunk 0
        shape=(n,),       # 1D tensor with n elements
        box_shape=(TILE_SIZE,),  # each TMA tile loads TILE_SIZE elements
        dtype=torch.float32,
    )

    return [kernel(x, grid=grid, block=32, params=[
        tma_in,                      # const CUtensorMap tma_in
        kernel.out_ptr(0),           # float* output
        np.uint32(n),                # unsigned int n
    ])]
