"""Kernelbox test for aten.mul.Tensor (output: mul_1).

Auto-generated from aten graph. Run:
    kbox iterate <this_file>.py --once
"""
import torch


def init_once():
    return {"h5": "data/mul_1.h5"}


def run(inputs):
    result = torch.ops.aten.mul.Tensor(inputs["view_1"], inputs["scale"])
    return [result]
