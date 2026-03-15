"""Tutorial 02: Inline CUDA kernel source.

Instead of a .cu file, embed the kernel directly in the test file.
This is handy for quick experiments — editing and saving re-runs immediately.

Run: kbox iterate examples/dev/test_tutorial_02_inline_kernel.py --once
"""
import torch

CLAMP_KERNEL = r"""
extern "C" __global__ void clamp_values(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = in0[i];
        out0[i] = fminf(fmaxf(val, -1.0f), 1.0f);
    }
}
"""

def init_once():
    x = torch.randn(4096, device="cuda") * 3.0  # values in ~[-9, 9]
    return {
        "kernel_source": CLAMP_KERNEL,
        "inputs": [x],
        "expected": [x.clamp(-1.0, 1.0)],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
