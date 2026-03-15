"""012_model.py:29 [blocks.0] | detach (softmax)


Inputs (4.0 KB total):
  masked_fill  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  detach  float32[2x2x16x16]  4.0 KB
Ops: _softmax x1, detach x1  (2 ops)

    kbox iterate fw_012_detach.py
"""
import torch


def init_once():
    return {"h5": "data/fw_012_detach.h5"}


def run(inputs):
    _softmax = torch.ops.aten._softmax.default(inputs.masked_fill, -1, False)
    detach = torch.ops.aten.detach.default(_softmax)
    return [detach]
