"""008_model.py:25 [blocks.0] | transpose (k reshape)


Inputs (4.0 KB total):
  getitem_4  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  transpose_1  float32[2x2x16x16]  4.0 KB
Ops: view x1, transpose x1  (2 ops)

    kbox iterate fw_008_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_008_transpose.h5"}


def run(inputs):
    view_3 = torch.ops.aten.view.default(inputs.getitem_4, [2, 16, 2, 16])
    transpose_1 = torch.ops.aten.transpose.int(view_3, 1, 2)
    return [transpose_1]
