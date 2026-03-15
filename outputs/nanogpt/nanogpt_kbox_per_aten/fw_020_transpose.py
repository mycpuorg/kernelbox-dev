"""020_model.py:24 [blocks.1] | transpose (q reshape)


Inputs (4.0 KB total):
  getitem_12  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  transpose_5  float32[2x2x16x16]  4.0 KB
Ops: view x1, transpose x1  (2 ops)

    kbox iterate fw_020_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_020_transpose.h5"}


def run(inputs):
    view_17 = torch.ops.aten.view.default(inputs.getitem_12, [2, 16, 2, 16])
    transpose_5 = torch.ops.aten.transpose.int(view_17, 1, 2)
    return [transpose_5]
