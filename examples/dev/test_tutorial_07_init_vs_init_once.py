"""Tutorial 07: init() vs init_once() — fresh inputs each run.

With init(), inputs are regenerated on every reload. This is useful when
your test data depends on code in the test file that you're editing.

In watch mode, every save regenerates fresh random inputs and re-runs.

Run: kbox iterate examples/dev/test_tutorial_07_init_vs_init_once.py --once
"""
import torch

RELU_KERNEL = r"""
extern "C" __global__ void relu(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out0[i] = fmaxf(in0[i], 0.0f);
    }
}
"""

def init():
    """Called every run — fresh random data each time."""
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": RELU_KERNEL,
        "inputs": [x],
        "expected": [torch.relu(x)],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
