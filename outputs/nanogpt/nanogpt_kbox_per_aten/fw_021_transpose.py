"""021_model.py:25 [blocks.1] | transpose (k reshape)


Inputs (4.0 KB total):
  getitem_13  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  transpose_6  float32[2x2x16x16]  4.0 KB
Ops: view x1, transpose x1  (2 ops)

    kbox iterate fw_021_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_021_transpose.h5"}


def run(inputs):
    view_18 = torch.ops.aten.view.default(inputs.getitem_13, [2, 16, 2, 16])
    transpose_6 = torch.ops.aten.transpose.int(view_18, 1, 2)
    return [transpose_6]
