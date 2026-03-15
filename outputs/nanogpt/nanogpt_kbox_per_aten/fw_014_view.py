"""014_model.py:31 [blocks.0] | view (y reshape)


Inputs (4.0 KB total):
  view_7  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  view_8  float32[2x16x32]  4.0 KB
Ops: transpose x1, clone x1, view x1  (3 ops)

    kbox iterate fw_014_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_014_view.h5"}


def run(inputs):
    transpose_4 = torch.ops.aten.transpose.int(inputs.view_7, 1, 2)
    clone_3 = torch.ops.aten.clone.default(transpose_4, memory_format=torch.contiguous_format)
    view_8 = torch.ops.aten.view.default(clone_3, [2, 16, 32])
    return [view_8]
