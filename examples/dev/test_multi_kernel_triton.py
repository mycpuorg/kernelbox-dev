"""Test multi-kernel Triton sessions. Run with:
    kbox iterate examples/dev/test_multi_kernel_triton.py --once
"""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + 1.0, mask=mask)


@triton.jit
def double_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


def init_once():
    n = 4096
    block = 256
    x = torch.randn(n, device="cuda")
    return {
        "triton_kernel": [add_one_kernel, double_kernel],
        "triton_constexprs": [
            {"BLOCK_SIZE": block},
            {"BLOCK_SIZE": block},
        ],
        "inputs": [x],
        "expected": [x + 1.0, x * 2.0],
        "outputs": 2,
        "atol": 1e-5,
    }


def run(inputs, kernels):
    x = inputs[0]
    n = x.numel()
    grid = (n + 255) // 256
    kernels[0](
        x,
        grid=grid,
        params=[
            kernels[0].in_ptr(0),
            kernels[0].out_ptr(0),
            np.int32(n),
        ],
    )
    outputs = kernels[1](
        x,
        grid=grid,
        params=[
            kernels[1].in_ptr(0),
            kernels[1].out_ptr(1),
            np.int32(n),
        ],
        clear_outputs=False,
    )
    return list(outputs)
