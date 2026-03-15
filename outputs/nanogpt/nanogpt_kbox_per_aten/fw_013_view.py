"""013_model.py:30 [blocks.0] | view (att @ v)


Inputs (8.0 KB total):
  _softmax  float32[2x2x16x16]  4.0 KB
  transpose_2  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  view_7  float32[2x2x16x16]  4.0 KB
Ops: expand x2, view x1, clone x1, _unsafe_view x1, bmm x1, view x1  (7 ops)

    kbox iterate fw_013_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_013_view.h5"}


def run(inputs):
    expand_2 = torch.ops.aten.expand.default(inputs._softmax, [2, 2, 16, 16])
    expand_3 = torch.ops.aten.expand.default(inputs.transpose_2, [2, 2, 16, 16])
    view_6 = torch.ops.aten.view.default(expand_2, [4, 16, 16])
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format=torch.contiguous_format)
    _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_2, [4, 16, 16])
    bmm_1 = torch.ops.aten.bmm.default(view_6, _unsafe_view_2)
    view_7 = torch.ops.aten.view.default(bmm_1, [2, 2, 16, 16])
    return [view_7]
