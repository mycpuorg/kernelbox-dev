"""023_model.py:27 [blocks.1] | mul (QK^T scaled)


Inputs (8.0 KB total):
  transpose_6  float32[2x2x16x16]  4.0 KB
  transpose_5  float32[2x2x16x16]  4.0 KB
Outputs (4.0 KB total):
  mul_1  float32[2x2x16x16]  4.0 KB
Ops: transpose x1, expand x2, clone x2, _unsafe_view x2, bmm x1, view x1, mul x1  (10 ops)

    kbox iterate fw_023_mul.py
"""
import torch


def init_once():
    return {"h5": "data/fw_023_mul.h5"}


def run(inputs):
    transpose_8 = torch.ops.aten.transpose.int(inputs.transpose_6, -2, -1)
    expand_4 = torch.ops.aten.expand.default(inputs.transpose_5, [2, 2, 16, 16])
    expand_5 = torch.ops.aten.expand.default(transpose_8, [2, 2, 16, 16])
    clone_4 = torch.ops.aten.clone.default(expand_4, memory_format=torch.contiguous_format)
    clone_5 = torch.ops.aten.clone.default(expand_5, memory_format=torch.contiguous_format)
    _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_4, [4, 16, 16])
    _unsafe_view_4 = torch.ops.aten._unsafe_view.default(clone_5, [4, 16, 16])
    bmm_2 = torch.ops.aten.bmm.default(_unsafe_view_3, _unsafe_view_4)
    view_20 = torch.ops.aten.view.default(bmm_2, [2, 2, 16, 16])
    mul_1 = torch.ops.aten.mul.Tensor(view_20, 0.25)
    return [mul_1]
