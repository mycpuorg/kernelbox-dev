"""022_model.py:26 [blocks.1] | transpose (v reshape)


Inputs (4.0 KB total):
  getitem_14  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  transpose_7  float32[2x2x16x16]  4.0 KB
Ops: view x1, transpose x1  (2 ops)

    kbox iterate fw_022_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_022_transpose.h5"}


def run(inputs):
    view_19 = torch.ops.aten.view.default(inputs.getitem_14, [2, 16, 2, 16])
    transpose_7 = torch.ops.aten.transpose.int(view_19, 1, 2)
    return [transpose_7]
