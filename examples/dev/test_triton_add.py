"""Test Triton kernel support. Run with: kbox iterate examples/dev/test_triton_add.py --once
"""
import torch
import numpy as np
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def init_once():
    n = 4096
    BLOCK = 256
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    return {
        "triton_kernel": add_kernel,
        "triton_constexprs": {"BLOCK_SIZE": BLOCK},
        "inputs": [x, y],
        "expected": [x + y],
        "outputs": 1,
        "grid": (n + BLOCK - 1) // BLOCK,
        "atol": 1e-5,
    }

def run(inputs, kernel):
    x, y = inputs[0], inputs[1]
    n = x.numel()
    return [kernel(x, y, params=[
        kernel.in_ptr(0),     # x_ptr
        kernel.in_ptr(1),     # y_ptr
        kernel.out_ptr(0),    # out_ptr
        np.int32(n),          # n_elements
    ])]
