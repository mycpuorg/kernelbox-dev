"""Tutorial 01: The simplest possible kernel test.

Demonstrates the minimal init_once() + run() pattern.
The kernel adds 1 to every element of an integer array.

Run: kbox iterate examples/dev/test_tutorial_01_basics.py --once
"""
import torch

def init_once():
    x = torch.arange(16, device="cuda", dtype=torch.int32)
    return {
        "kernel": "examples/kernels/add_one.cu",
        "inputs": [x],
        "expected": [x + 1],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
