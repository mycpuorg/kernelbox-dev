"""Test kernel_mode() with a cached multi-kernel CUDA plan. Run with:
    kbox iterate examples/dev/test_kernel_mode_cuda.py --once
"""
import numpy as np
import torch


ADD_ONE = r"""
extern "C" __global__ void add_one(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] + 1.0f;
}
"""


DOUBLE = r"""
extern "C" __global__ void double_it(
    const float *in0, float *out1, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out1[i] = in0[i] * 2.0f;
}
"""


def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": [ADD_ONE, DOUBLE],
        "inputs": [x],
        "expected": [x + 1.0, x * 2.0],
        "outputs": 2,
        "atol": 1e-5,
    }


def kernel_mode(kernels, scratch_ptr, input_ptrs, output_ptrs, n):
    _ = scratch_ptr
    block = 256
    grid = (n + block - 1) // block
    return [
        {
            "kernel": kernels[0],
            "grid": grid,
            "block": block,
            "params": [input_ptrs[0], output_ptrs[0], np.uint32(n)],
        },
        {
            "kernel": kernels[1],
            "grid": grid,
            "block": block,
            "params": [input_ptrs[0], output_ptrs[1], np.uint32(n)],
            "clear_outputs": False,
        },
    ]
