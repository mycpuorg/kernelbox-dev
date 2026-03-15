"""024_model.py:28 [blocks.1] | masked_fill (causal mask)


Inputs (5.0 KB total):
  primals_21  float32[1x1x16x16]  1.0 KB
  mul_1  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  masked_fill_1  float32[2x2x16x16]  4.0 KB
Ops: alias x1, eq x1, masked_fill x1  (3 ops)

    kbox iterate fw_024_masked_fill.py
"""
import torch


def init_once():
    return {"h5": "data/fw_024_masked_fill.h5"}


def run(inputs):
    alias_1 = torch.ops.aten.alias.default(inputs.primals_21)
    eq_1 = torch.ops.aten.eq.Scalar(alias_1, 0)
    masked_fill_1 = torch.ops.aten.masked_fill.Scalar(inputs.mul_1, eq_1, float('-inf'))
    return [masked_fill_1]
