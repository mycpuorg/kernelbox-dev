"""Match layer normalization from HDF5 reference data.

    kbox iterate examples/dev/test_layernorm.py
"""
import torch


def init_once():
    return {"h5": "examples/data/normalize.h5"}


def run(inputs):
    mean = inputs.x.mean(dim=-1, keepdim=True)
    var  = inputs.x.var(dim=-1, keepdim=True, correction=0)
    return [(inputs.x - mean) / torch.sqrt(var + inputs.eps) * inputs.gamma + inputs.beta]
