"""Tutorial 03: Multiple inputs.

A kernel that takes two input buffers and produces one output.
Standard parameter layout: (in0, in1, out0, n).

Run: kbox iterate examples/dev/test_tutorial_03_multi_input.py --once
"""
import torch

LERP_KERNEL = r"""
extern "C" __global__ void lerp_half(
    const float *a, const float *b, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 0.5f * a[i] + 0.5f * b[i];
    }
}
"""

def init_once():
    a = torch.randn(2048, device="cuda")
    b = torch.randn(2048, device="cuda")
    return {
        "kernel_source": LERP_KERNEL,
        "inputs": [a, b],
        "expected": [0.5 * a + 0.5 * b],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
