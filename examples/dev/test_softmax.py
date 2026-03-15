"""Iterate version of 08_softmax: softmax_1d with shared memory.
Run with: kbox iterate examples/dev/test_softmax.py --once
"""
import torch

BLOCK = 256

def init_once():
    x = torch.randn(BLOCK, device="cuda")
    return {
        "kernel": "examples/kernels/softmax_1d.cu",
        "inputs": [x],
        "expected": [torch.softmax(x, dim=0)],
        "block": (BLOCK, 1, 1),
        "grid": (1, 1, 1),
        "smem": BLOCK * 4,
        "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
