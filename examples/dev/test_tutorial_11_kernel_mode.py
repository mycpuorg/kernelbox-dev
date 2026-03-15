"""Tutorial 11: Multi-kernel kernel_mode() launch plan.

kernel_mode() gives you full control over a sequence of kernel launches.
You receive kernel handles, buffer pointers, and element count, and return
a list of launch steps. Each step specifies a kernel, grid, block, and params.

Run: kbox iterate examples/dev/test_tutorial_11_kernel_mode.py --once
"""
import torch
import numpy as np

# Step 1: scale by 2.0
SCALE_KERNEL = r"""
extern "C" __global__ void scale2x(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * 2.0f;
}
"""

# Step 2: add 10.0 in-place
ADD_KERNEL = r"""
extern "C" __global__ void add10(
    float *buf, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += 10.0f;
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": [SCALE_KERNEL, ADD_KERNEL],
        "inputs": [x],
        "expected": [x * 2.0 + 10.0],
        "outputs": 1,
    }

def kernel_mode(kernels, input_ptrs, output_ptrs, n):
    block = 256
    grid = (n + block - 1) // block
    return [
        # Step 1: out[0] = in[0] * 2.0
        {
            "kernel": kernels[0],
            "grid": grid,
            "block": block,
            "params": [input_ptrs[0], output_ptrs[0], np.uint32(n)],
        },
        # Step 2: out[0] += 10.0 (in-place on output)
        {
            "kernel": kernels[1],
            "grid": grid,
            "block": block,
            "params": [output_ptrs[0], np.uint32(n)],
            "clear_outputs": False,  # don't zero output between steps!
        },
    ]
