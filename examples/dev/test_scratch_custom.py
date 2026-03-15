"""Test scratch_ptr introspection: run() receives worker VA for custom params.
Run with: kbox iterate examples/dev/test_scratch_custom.py --once
"""
import torch
import numpy as np

KERNEL_CODE = r"""
extern "C" __global__ void scratch_custom(
    float *scratch,
    const float *in0,
    float *out0,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        scratch[i] = in0[i] * 5.0f;
        __syncthreads();
        out0[i] = scratch[i] - in0[i] * 2.0f;  // = in0[i] * 3.0
    }
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": KERNEL_CODE,
        "kernel_scratch_mib": 1,
        "inputs": [x],
        "expected": [x * 3.0],
    }

def run(inputs, kernel, scratch_ptr):
    """scratch_ptr is auto-detected and passed as the worker-side VA."""
    n = len(inputs[0])
    return [kernel(*inputs, params=[
        np.uint64(scratch_ptr),
        kernel.in_ptr(0),
        kernel.out_ptr(0),
        np.uint32(n),
    ])]
