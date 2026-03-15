"""Iterate version of 04_session: fused_add_mul with two outputs.
Run with: kbox iterate examples/dev/test_fused_add_mul.py --once
"""
import torch

def init_once():
    x = torch.randn(2048, device="cuda")
    y = torch.linspace(-1, 1, 2048, device="cuda")
    return {
        "kernel": "examples/kernels/fused_add_mul.cu",
        "inputs": [x, y],
        "expected": [x + y, x * y],
        "outputs": 2,
        "atol": 1e-5,
    }

def run(inputs, kernel):
    return list(kernel(*inputs))
