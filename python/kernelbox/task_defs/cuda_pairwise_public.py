"""Public CUDA task for kernel_mode iteration."""
import torch


ADD_ONE = r"""
extern "C" __global__ void add_one(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] + 1.0f;
}
"""


DOUBLE = r"""
extern "C" __global__ void double_it(
    const float *in0, float *out1, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out1[i] = in0[i] * 2.0f;
}
"""


def init_once():
    x = torch.linspace(-1.0, 1.0, 4096, device="cuda", dtype=torch.float32)
    return {
        "kernel_source": [ADD_ONE, DOUBLE],
        "inputs": [x],
        "expected": [x + 1.0, x * 2.0],
        "outputs": 2,
        "atol": 1e-5,
        "warmup": 1,
        "iters": 5,
    }
