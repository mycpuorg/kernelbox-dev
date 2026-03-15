"""Tutorial 05: Custom kernel parameters.

When your kernel has a non-standard signature (extra scalar arguments,
different parameter order), use params=[] for full control.

Kernel signature: axpby(x, y, out, alpha, beta, n)

Run: kbox iterate examples/dev/test_tutorial_05_custom_params.py --once
"""
import torch
import numpy as np

AXPBY_KERNEL = r"""
extern "C" __global__ void axpby(
    const float *x, const float *y, float *out,
    float alpha, float beta, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + beta * y[i];
    }
}
"""

ALPHA = 2.5
BETA = -0.7

def init_once():
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    return {
        "kernel_source": AXPBY_KERNEL,
        "inputs": [x, y],
        "expected": [ALPHA * x + BETA * y],
        "outputs": 1,
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(inputs[0], inputs[1], params=[
        kernel.in_ptr(0),          # x
        kernel.in_ptr(1),          # y
        kernel.out_ptr(0),         # out
        np.float32(ALPHA),         # alpha
        np.float32(BETA),          # beta
        np.uint32(n),              # n
    ])]
