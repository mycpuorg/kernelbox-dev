"""Iterate version of 01_hello: add_one kernel with sequential input.
Run with: kbox iterate examples/dev/test_hello.py --once
"""
import torch

def init_once():
    x = torch.arange(8, device="cuda", dtype=torch.int32)
    return {
        "kernel": "examples/kernels/add_one.cu",
        "inputs": [x],
        "expected": [x + 1],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
