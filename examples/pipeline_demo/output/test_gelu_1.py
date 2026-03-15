"""Kernelbox test for aten.gelu.default (output: gelu_1).

Auto-generated from aten graph. Run:
    kbox iterate <this_file>.py --once
"""
import torch


def init_once():
    return {"h5": "data/gelu_1.h5"}


def run(inputs):
    result = torch.ops.aten.gelu.default(inputs["x"])
    return [result]
