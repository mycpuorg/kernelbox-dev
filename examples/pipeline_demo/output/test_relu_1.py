"""Kernelbox test for aten.relu.default (output: relu_1).

Auto-generated from aten graph. Run:
    kbox iterate <this_file>.py --once
"""
import torch


def init_once():
    return {"h5": "data/relu_1.h5"}


def run(inputs):
    result = torch.ops.aten.relu.default(inputs["mm_1"])
    return [result]
