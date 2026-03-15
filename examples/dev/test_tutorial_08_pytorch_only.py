"""Tutorial 08: Pure PyTorch test (no kernel).

You don't need a CUDA kernel at all. Just define run(inputs) with PyTorch ops.
Useful for testing data pipelines or reference implementations.

Run: kbox iterate examples/dev/test_tutorial_08_pytorch_only.py --once
"""
import torch

def init_once():
    batch, features = 32, 64
    x = torch.randn(batch, features, device="cuda")
    w = torch.randn(features, features, device="cuda")
    b = torch.randn(features, device="cuda")

    expected = torch.relu(x @ w + b)
    return {
        "inputs": [x, w, b],
        "expected": [expected],
    }

def run(inputs):
    x, w, b = inputs
    return [torch.relu(x @ w + b)]
