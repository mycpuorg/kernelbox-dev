"""Tutorial 13: Isolated kernel_mode() — sandboxed planning.

When run with --isolated-kernel-benchmark, kernel_mode() executes in a
subprocess with NO GPU access. It receives fake handles and metadata,
returns a plan, and the parent process executes it on the real GPU.

This is how the MCP task service evaluates untrusted code safely.

Run: kbox iterate examples/dev/test_tutorial_13_isolated_kernel_mode.py --once
Run: kbox iterate examples/dev/test_tutorial_13_isolated_kernel_mode.py --once --bench --isolated-kernel-benchmark
"""
import torch
import numpy as np

SQUARE_KERNEL = r"""
extern "C" __global__ void square(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * in0[i];
}
"""

NEGATE_KERNEL = r"""
extern "C" __global__ void negate(
    const float *in0, float *out1, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out1[i] = -in0[i];
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": [SQUARE_KERNEL, NEGATE_KERNEL],
        "inputs": [x],
        "expected": [x * x, -x],
        "outputs": 2,
        "atol": 1e-5,
    }

def kernel_mode(kernels, input_ptrs, output_ptrs, n):
    """This function runs in the isolated subprocess when --isolated-kernel-benchmark is used."""
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
