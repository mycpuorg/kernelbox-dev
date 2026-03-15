"""Test custom params with mixed scalar types (float, int, double).
Run with: kbox iterate examples/dev/test_custom_params_mixed.py --once
"""
import torch
import numpy as np

KERNEL_CODE = r"""
extern "C" __global__ void affine_transform(
    const float *x, float *out,
    double scale, float bias, unsigned int n, int negate)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = (float)(scale * (double)x[i]) + bias;
        out[i] = negate ? -val : val;
    }
}
"""

def init_once():
    n = 8192
    x = torch.randn(n, device="cuda")
    scale, bias, negate = 3.14159, -0.5, 1
    expected = -(x.double() * scale).float() - bias  # negate=1
    return {
        "kernel_source": KERNEL_CODE,
        "inputs": [x],
        "expected": [expected],
        "outputs": 1,
        "atol": 1e-3,
    }

def run(inputs, kernel):
    x = inputs[0]
    n = x.numel()
    return [kernel(x, params=[
        kernel.in_ptr(0),         # const float *x
        kernel.out_ptr(0),        # float *out
        np.float64(3.14159),      # double scale  (8 bytes, align 8)
        np.float32(-0.5),         # float bias    (4 bytes, align 4)
        np.uint32(n),             # unsigned int n (4 bytes, align 4)
        np.int32(1),              # int negate    (4 bytes, align 4)
    ])]
