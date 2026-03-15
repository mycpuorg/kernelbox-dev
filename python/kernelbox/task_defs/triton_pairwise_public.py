"""Public Triton task for kernel_mode iteration."""
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
    block = 256
    x = torch.linspace(-1.5, 1.5, 4096, device="cuda", dtype=torch.float32)
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
        "warmup": 1,
        "iters": 5,
    }
