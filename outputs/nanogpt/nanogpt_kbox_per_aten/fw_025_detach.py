"""025_model.py:29 [blocks.1] | detach (softmax)


Inputs (4.0 KB total):
  masked_fill_1  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  detach_1  float32[2x2x16x16]  4.0 KB
Ops: _softmax x1, detach x1  (2 ops)

    kbox iterate fw_025_detach.py
"""
import torch


def init_once():
    return {"h5": "data/fw_025_detach.h5"}


def run(inputs):
    _softmax_1 = torch.ops.aten._softmax.default(inputs.masked_fill_1, -1, False)
    detach_1 = torch.ops.aten.detach.default(_softmax_1)
    return [detach_1]
