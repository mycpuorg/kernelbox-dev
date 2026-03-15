"""Test Triton scalar type coercion. Deliberately passes wrong scalar types.
Run with: kbox iterate examples/dev/test_triton_coerce.py --once
"""
import torch
import numpy as np
import triton
import triton.language as tl

@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * scale, mask=mask)

def init_once():
    n = 4096
    BLOCK = 256
    x = torch.randn(n, device="cuda")
    return {
        "triton_kernel": scale_kernel,
        "triton_constexprs": {"BLOCK_SIZE": BLOCK},
        "inputs": [x],
        "expected": [x * 2.5],
        "outputs": 1,
        "grid": (n + BLOCK - 1) // BLOCK,
        "atol": 1e-4,
    }

def run(inputs, kernel):
    x = inputs[0]
    n = x.numel()
    # Deliberately pass wrong types:
    # - n as int64 (kernel expects i32)
    # - scale as float64 (kernel expects fp32, stored as f32)
    # The coercion should fix these automatically
    return [kernel(x, params=[
        kernel.in_ptr(0),     # x_ptr
        kernel.out_ptr(0),    # out_ptr
        np.int64(n),          # n_elements — WRONG: should be i32
        np.float64(2.5),      # scale — WRONG: should be fp32
    ])]
