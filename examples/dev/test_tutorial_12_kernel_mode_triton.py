"""Tutorial 12: kernel_mode() with Triton kernels.

Same kernel_mode() pattern but using Triton JIT functions instead of CUDA.

Run: kbox iterate examples/dev/test_tutorial_12_kernel_mode_triton.py --once
"""
import torch
import numpy as np
import triton
import triton.language as tl

@triton.jit
def negate_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, -x, mask=mask)

@triton.jit
def abs_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.abs(x), mask=mask)

BLOCK = 256

def init_once():
    n = 4096
    x = torch.randn(n, device="cuda")
    return {
        "triton_kernel": [negate_kernel, abs_kernel],
        "triton_constexprs": {"BLOCK_SIZE": BLOCK},
        "inputs": [x],
        "expected": [-x, x.abs()],
        "outputs": 2,
        "grid": (n + BLOCK - 1) // BLOCK,
    }

def kernel_mode(kernels, input_ptrs, output_ptrs, n):
    grid = (n + BLOCK - 1) // BLOCK
    return [
        {
            "kernel": kernels[0],
            "grid": grid,
            "block": BLOCK,
            "params": [input_ptrs[0], output_ptrs[0], np.uint32(n)],
        },
        {
            "kernel": kernels[1],
            "grid": grid,
            "block": BLOCK,
            "params": [input_ptrs[0], output_ptrs[1], np.uint32(n)],
            "clear_outputs": False,
        },
    ]
