"""Match a 2-layer MLP: ReLU(x @ W1 + b1) @ W2 + b2.

    kbox iterate examples/dev/test_mlp.py
"""
import torch


def init_once():
    return {"h5": "examples/data/mlp.h5"}


def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
