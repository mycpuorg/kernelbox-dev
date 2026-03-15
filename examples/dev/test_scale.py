"""Test file for scale kernel. Run with: kbox iterate examples/dev/test_scale.py"""
import torch

def init_once():
    inputs = torch.randn(4096, device="cuda")
    return {
        "kernel": "examples/kernels/scale.cu",
        "inputs": [inputs],
        "expected": [inputs * 2.5],
        "benchmark": True,
        "atol": 1e-3,
    }

def init_reload(init_result):
    init_result["atol"] = 5e-4
    return init_result

def run(inputs, kernel):
    return [kernel(inputs[0])]
