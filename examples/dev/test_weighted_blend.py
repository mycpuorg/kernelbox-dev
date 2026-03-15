"""Iterate version of 05_complex: weighted_blend kernel.
Run with: kbox iterate examples/dev/test_weighted_blend.py --once
"""
import torch

def init_once():
    n = 4096
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    w = torch.rand(n, device="cuda")
    blended = w * a + (1 - w) * b
    residual = a - blended
    return {
        "kernel": "examples/kernels/weighted_blend.cu",
        "inputs": [a, b, w],
        "expected": [blended, residual],
        "outputs": 2,
        "atol": 1e-4,
    }

def run(inputs, kernel):
    return list(kernel(*inputs))
