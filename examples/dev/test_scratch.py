"""Test scratch buffer: kernel receives scratch as first arg, writes to it, reads back.
Run with: kbox iterate examples/dev/test_scratch.py --once
"""
import torch
import numpy as np

KERNEL_CODE = r"""
extern "C" __global__ void scratch_test(
    void *scratch,
    const float *in0,
    float *out0,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Write to scratch, read back, add to input
        float *s = (float *)scratch;
        s[i] = in0[i] * 3.0f;
        __syncthreads();
        out0[i] = s[i] - in0[i];  // should be in0[i] * 2.0
    }
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": KERNEL_CODE,
        "kernel_scratch_mib": 1,  # request scratch
        "inputs": [x],
        "expected": [x * 2.0],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
