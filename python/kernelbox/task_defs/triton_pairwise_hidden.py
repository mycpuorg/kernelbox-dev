"""Hidden Triton task variant for secure kernel_mode evaluation."""
import torch

from kernelbox.task_defs.triton_pairwise_public import add_one_kernel, double_kernel


def init_once():
    block = 256
    x = torch.linspace(-3.0, 3.0, 3072, device="cuda", dtype=torch.float32)
    x = x * 1.25 - 0.75
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
        "iters": 3,
    }
