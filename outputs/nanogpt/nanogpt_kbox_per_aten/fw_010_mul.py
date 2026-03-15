"""010_model.py:27 [blocks.0] | mul (QK^T scaled)


Inputs (8.0 KB total):
  transpose_1  float32[2x2x16x16]  4.0 KB
  transpose  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  mul  float32[2x2x16x16]  4.0 KB
Ops: transpose x1, expand x2, clone x2, _unsafe_view x2, bmm x1, view x1, mul x1  (10 ops)

    kbox iterate fw_010_mul.py
"""
import torch


def init_once():
    return {"h5": "data/fw_010_mul.h5"}


def run(inputs):
    transpose_3 = torch.ops.aten.transpose.int(inputs.transpose_1, -2, -1)
    expand = torch.ops.aten.expand.default(inputs.transpose, [2, 2, 16, 16])
    expand_1 = torch.ops.aten.expand.default(transpose_3, [2, 2, 16, 16])
    clone = torch.ops.aten.clone.default(expand, memory_format=torch.contiguous_format)
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format=torch.contiguous_format)
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [4, 16, 16])
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_1, [4, 16, 16])
    bmm = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
    view_5 = torch.ops.aten.view.default(bmm, [2, 2, 16, 16])
    mul = torch.ops.aten.mul.Tensor(view_5, 0.25)
    return [mul]
