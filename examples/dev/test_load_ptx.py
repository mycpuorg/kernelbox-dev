"""Test loading a pre-compiled .ptx file.
Run with: kbox iterate examples/dev/test_load_ptx.py --once
"""
import torch
import subprocess, os, shutil

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
    nvcc = _find_nvcc()
    arch = subprocess.check_output(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
        shell=True).decode().strip().replace('.', '')
    ptx_path = "/tmp/test_scale_iterate.ptx"
    subprocess.check_call([
        nvcc, "-ptx", f"-arch=sm_{arch}", "-o", ptx_path,
        "examples/kernels/scale.cu"
    ])
    n = 4096
    x = torch.randn(n, device="cuda")
    return {
        "kernel": ptx_path,
        "inputs": [x],
        "expected": [x * 2.5],
        "outputs": 1,
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
