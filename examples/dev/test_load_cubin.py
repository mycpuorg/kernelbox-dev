"""Test loading a pre-compiled .cubin file.
Run with: kbox iterate examples/dev/test_load_cubin.py --once
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
    # Compile scale.cu to cubin at test time
    nvcc = _find_nvcc()
    arch = subprocess.check_output(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
        shell=True).decode().strip().replace('.', '')
    cubin_path = "/tmp/test_scale_iterate.cubin"
    subprocess.check_call([
        nvcc, "-cubin", f"-arch=sm_{arch}", "-o", cubin_path,
        "examples/kernels/scale.cu"
    ])
    n = 4096
    x = torch.randn(n, device="cuda")
    return {
        "kernel": cubin_path,
        "func_name": "scale",  # cubin doesn't have source to detect from
        "inputs": [x],
        "expected": [x * 2.5],
        "outputs": 1,
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
