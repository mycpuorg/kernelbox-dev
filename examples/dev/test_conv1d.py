"""Match a 1D convolution from HDF5 reference data.

    kbox iterate examples/dev/test_conv1d.py
"""
import torch


def init_once():
    return {"h5": "examples/data/conv1d.h5"}


def run(inputs):
    return [torch.nn.functional.conv1d(inputs.signal, inputs.kernel, inputs.bias)]
