"""Implement scaled dot-product attention to match PyTorch's reference.

    kbox iterate examples/dev/test_attention.py
"""
import torch


def init_once():
    B, H, S, D = 2, 8, 128, 64
    q = torch.randn(B, H, S, D, device="cuda")
    k = torch.randn(B, H, S, D, device="cuda")
    v = torch.randn(B, H, S, D, device="cuda")
    return {
        "inputs": [q, k, v],
        "expected": [torch.nn.functional.scaled_dot_product_attention(q, k, v)],
    }


def run(inputs):
    q, k, v = inputs
    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    weights = torch.softmax(scores, dim=-1)
    return [weights @ v]
