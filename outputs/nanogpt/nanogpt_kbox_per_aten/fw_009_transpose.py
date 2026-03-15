"""009_model.py:26 [blocks.0] | transpose (v reshape)


Inputs (4.0 KB total):
  getitem_5  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  transpose_2  float32[2x2x16x16]  4.0 KB
Ops: view x1, transpose x1  (2 ops)

    kbox iterate fw_009_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_009_transpose.h5"}


def run(inputs):
    view_4 = torch.ops.aten.view.default(inputs.getitem_5, [2, 16, 2, 16])
    transpose_2 = torch.ops.aten.transpose.int(view_4, 1, 2)
    return [transpose_2]
