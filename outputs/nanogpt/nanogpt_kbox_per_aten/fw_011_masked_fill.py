"""011_model.py:28 [blocks.0] | masked_fill (causal mask)


Inputs (5.0 KB total):
  primals_8  float32[1x1x16x16]  1.0 KB
  mul  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  masked_fill  float32[2x2x16x16]  4.0 KB
Ops: alias x1, eq x1, masked_fill x1  (3 ops)

    kbox iterate fw_011_masked_fill.py
"""
import torch


def init_once():
    return {"h5": "data/fw_011_masked_fill.h5"}


def run(inputs):
    alias = torch.ops.aten.alias.default(inputs.primals_8)
    eq = torch.ops.aten.eq.Scalar(alias, 0)
    masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.mul, eq, float('-inf'))
    return [masked_fill]
