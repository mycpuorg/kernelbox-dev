"""Kernelbox CUDA kernel test for aten.gelu.default (output: gelu_1).

Auto-generated with inline CUDA kernel. Run:
    kbox iterate <this_file>.py --once
    kbox iterate <this_file>.py --once --bench --isolated-kernel-benchmark
"""
import torch
import numpy as np
import kernelbox as kbox


KERNEL_SOURCE = r"""
extern "C" __global__ void gelu_kernel(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}
"""


def init_once():
    inputs, expected = kbox.h5.load_test("examples/pipeline_demo/output/data/gelu_1.h5")
    input_list = [inputs[k] for k in sorted(inputs.keys())
                  if isinstance(inputs[k], torch.Tensor)]
    return {
        "kernel_source": KERNEL_SOURCE,
        "inputs": input_list,
        "expected": expected,
        "outputs": 1,
        "atol": 1e-5,
    }


def kernel_mode(kernels, input_ptrs, output_ptrs, n):
    block = 256
    grid = (n + block - 1) // block
    return [{
        "kernel": kernels[0],
        "grid": grid,
        "block": block,
        "params": [input_ptrs[0], output_ptrs[0], np.uint32(n)],
    }]
