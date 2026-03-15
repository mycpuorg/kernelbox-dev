"""Kernelbox CUDA kernel test for aten.mul.Tensor (output: mul_1).

Auto-generated with inline CUDA kernel. Run:
    kbox iterate <this_file>.py --once
    kbox iterate <this_file>.py --once --bench --isolated-kernel-benchmark
"""
import torch
import numpy as np
import kernelbox as kbox


KERNEL_SOURCE = r"""
extern "C" __global__ void mul_kernel(
    const float *in0, const float *in1, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = a * b;
    }
}
"""


def init_once():
    inputs, expected = kbox.h5.load_test("examples/pipeline_demo/output/data/mul_1.h5")
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
        "params": [input_ptrs[0], input_ptrs[1], output_ptrs[0], np.uint32(n)],
    }]
