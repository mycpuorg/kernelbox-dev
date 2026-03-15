"""Tutorial 10: Built-in benchmarking.

Set benchmark=True in the state dict to auto-run benchmarks after verification.
Or use --bench on the command line.

Run: kbox iterate examples/dev/test_tutorial_10_benchmark.py --once
Run: kbox iterate examples/dev/test_tutorial_10_benchmark.py --once --bench --iters 500
"""
import torch

def init_once():
    n = 1 << 20  # 1M elements
    x = torch.randn(n, device="cuda")
    return {
        "kernel": "examples/kernels/scale.cu",
        "inputs": [x],
        "expected": [x * 2.5],
        "benchmark": True,
        "warmup": 20,
        "iters": 200,
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
