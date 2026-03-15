"""Kernelbox test for aten.add.Tensor (output: add_1).

Auto-generated from aten graph. Run:
    kbox iterate <this_file>.py --once
"""
import torch


def init_once():
    return {"h5": "data/add_1.h5"}


def run(inputs):
    result = torch.ops.aten.add.Tensor(inputs["gelu_1"], inputs["bias"])
    return [result]
