"""Tutorial 04: Multiple outputs.

A kernel that reads one input and writes two outputs.
Requires outputs=2 in the state dict so KernelBox allocates two output buffers.

Run: kbox iterate examples/dev/test_tutorial_04_multi_output.py --once
"""
import torch

SPLIT_KERNEL = r"""
extern "C" __global__ void split_pos_neg(
    const float *in0, float *out0, float *out1, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = in0[i];
        out0[i] = fmaxf(val, 0.0f);   // positive part (ReLU)
        out1[i] = fminf(val, 0.0f);   // negative part
    }
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": SPLIT_KERNEL,
        "inputs": [x],
        "expected": [x.clamp(min=0), x.clamp(max=0)],
        "outputs": 2,
    }

def run(inputs, kernel):
    return list(kernel(inputs[0]))
