"""Test extra_params: standard (in0, out0, n) layout + extra scalars.
Run with: kbox iterate examples/dev/test_extra_params.py --once
"""
import torch
import numpy as np

KERNEL_CODE = r"""
extern "C" __global__ void clamp_scale(
    const float *in0, float *out0, unsigned int n,
    float scale, float lo, float hi)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in0[i] * scale;
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out0[i] = v;
    }
}
"""

def init_once():
    n = 4096
    x = torch.randn(n, device="cuda")
    scale, lo, hi = 3.0, -2.0, 2.0
    expected = torch.clamp(x * scale, lo, hi)
    return {
        "kernel_source": KERNEL_CODE,
        "inputs": [x],
        "expected": [expected],
        "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], extra_params=[
        np.float32(3.0),    # float scale
        np.float32(-2.0),   # float lo
        np.float32(2.0),    # float hi
    ])]
