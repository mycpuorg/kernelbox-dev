"""Test Triton with bare Python int/float params (auto-converted to correct numpy types).
Run with: kbox iterate examples/dev/test_triton_bare_scalars.py --once
"""
import torch
import triton
import triton.language as tl

@triton.jit
def axpb_kernel(x_ptr, out_ptr, n_elements, a, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * a + b, mask=mask)

def init_once():
    n = 4096
    BLOCK = 256
    x = torch.randn(n, device="cuda")
    a, b = 3.0, -1.5
    return {
        "triton_kernel": axpb_kernel,
        "triton_constexprs": {"BLOCK_SIZE": BLOCK},
        "inputs": [x],
        "expected": [x * a + b],
        "outputs": 1,
        "grid": (n + BLOCK - 1) // BLOCK,
        "atol": 1e-4,
    }

def run(inputs, kernel):
    x = inputs[0]
    n = x.numel()
    # Pass bare Python int and float — should be auto-converted
    return [kernel(x, params=[
        kernel.in_ptr(0),
        kernel.out_ptr(0),
        n,          # bare Python int → auto-detected as i32 (fits in 32 bits)
        3.0,        # bare Python float → auto-detected as fp64
        -1.5,       # bare Python float → auto-detected as fp64
    ])]
