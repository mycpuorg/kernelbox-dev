"""007_model.py:24 [blocks.0] | transpose (q reshape)


Inputs (4.0 KB total):
  getitem_3  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  transpose  float32[2x2x16x16]  4.0 KB
Ops: view x1, transpose x1  (2 ops)

    kbox iterate fw_007_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_007_transpose.h5"}


def run(inputs):
    view_2 = torch.ops.aten.view.default(inputs.getitem_3, [2, 16, 2, 16])
    transpose = torch.ops.aten.transpose.int(view_2, 1, 2)
    return [transpose]
