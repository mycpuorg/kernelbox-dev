"""MLP test suite — 5 cases with different shapes, same structure.

    kbox iterate examples/dev/test_mlp_suite.py
"""
import torch


def init_once():
    return {"h5_suite": "examples/data/mlp_cases/"}


def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
