"""Test file demonstrating custom kernel params. Run with: kbox iterate examples/dev/test_custom_params.py

Uses params=[] to pass arbitrary kernel arguments (pointers + scalars)
instead of the default (in0, ..., out0, ..., n) convention.
"""
import torch
import numpy as np

KERNEL_CODE = r"""
extern "C" __global__ void scaled_add(
    const float *x, const float *y, float *out,
    float a, float b, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + b * y[i];
}
"""

def init_once():
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    a, b = 2.5, -1.0
    return {
        "kernel_source": KERNEL_CODE,
        "inputs": [x, y],
        "expected": [a * x + b * y],
        "outputs": 1,
        "atol": 1e-3,
    }

def run(inputs, kernel):
    x, y = inputs[0], inputs[1]
    n = x.numel()
    return [kernel(x, y, params=[
        kernel.in_ptr(0),     # const float *x
        kernel.in_ptr(1),     # const float *y
        kernel.out_ptr(0),    # float *out
        np.float32(2.5),      # float a
        np.float32(-1.0),     # float b
        np.uint32(n),         # unsigned int n
    ])]
