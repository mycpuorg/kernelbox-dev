"""Test benchmark via worker-side timing.
Run with: kbox iterate examples/dev/test_bench.py --once --bench
"""
import torch

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
        "benchmark": True,
        "warmup": 5,
        "iters": 20,
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
