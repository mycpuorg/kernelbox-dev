"""Test async launch, sync, and timing APIs.
Run with: kbox iterate examples/dev/test_async_timing.py --once
"""
import torch
import numpy as np

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": r"""
extern "C" __global__ void scale(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * 2.0f;
}
""",
        "inputs": [x],
        "expected": [x * 2.0],
    }

def run(inputs, kernel):
    x = inputs[0]
    n = x.numel()

    # Test 1: async launch + sync
    result = kernel(x, sync=False)
    kernel.sync()

    # Test 2: start_timing + async launch + end_timing (with sync)
    kernel.start_timing()
    result = kernel(x, sync=False)
    elapsed = kernel.end_timing(sync=True)
    assert elapsed > 0, f"Expected positive elapsed time, got {elapsed}"

    # Test 3: start_timing(sync=True) + sync launch + end_timing
    kernel.start_timing(sync=True)
    result = kernel(x)  # sync=True (default)
    elapsed2 = kernel.end_timing(sync=True)
    assert elapsed2 > 0, f"Expected positive elapsed time, got {elapsed2}"

    # Test 4: sync returns elapsed when timing is active
    kernel.start_timing()
    result = kernel(x, sync=False)
    kernel.end_timing(sync=False)  # record event but don't sync
    elapsed3 = kernel.sync()  # sync should return elapsed
    assert elapsed3 > 0, f"Expected positive elapsed from sync(), got {elapsed3}"

    return [result]
