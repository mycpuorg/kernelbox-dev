"""Same MLP test, but loading from .pt instead of .h5.

    kbox iterate examples/dev/test_mlp_pt.py
"""
import torch


def init_once():
    return {"h5": "examples/data/mlp.pt"}


def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
