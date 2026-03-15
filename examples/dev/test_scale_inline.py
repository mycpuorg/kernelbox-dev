"""Test file with inline kernel source. Run with: kbox iterate examples/dev/test_scale_inline.py

Demonstrates embedding CUDA kernel source directly in the .py test file,
so you can iterate on the kernel without a separate .cu file.
"""
import torch

KERNEL_CODE = r"""
extern "C" __global__ void scale(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out0[i] = in0[i] * 2.5f;
    }
}
"""

def init_once():
    inputs = torch.randn(4096, device="cuda")
    return {
        "kernel_source": KERNEL_CODE,
        "inputs": [inputs],
        "expected": [inputs * 2.5],
        "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
