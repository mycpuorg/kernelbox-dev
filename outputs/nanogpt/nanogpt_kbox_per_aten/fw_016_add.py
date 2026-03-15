"""016_model.py:49 [blocks.0] | add (MLP residual, ln_2)


Inputs (36.9 KB total):
  add_1  float32[2x16x32]  4.0 KB
  primals_11  float32[32]  128 B
  primals_12  float32[32]  128 B
  primals_13  float32[128x32]  16.0 KB
  primals_14  float32[128]  512 B
  primals_15  float32[32x128]  16.0 KB
  primals_16  float32[32]  128 B
Outputs (4.0 KB total):
  add_2  float32[2x16x32]  4.0 KB
Ops: native_layer_norm x1, getitem x3, view x4, t x2, addmm x2, gelu x1, add x1  (14 ops)

    kbox iterate fw_016_add.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_016_add.h5"}


def run(inputs):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_1, [32], inputs.primals_11, inputs.primals_12, 1e-05)
    getitem_6 = operator.getitem(native_layer_norm, 0)
    getitem_7 = operator.getitem(native_layer_norm, 1)
    getitem_8 = operator.getitem(native_layer_norm, 2)
    view_11 = torch.ops.aten.view.default(getitem_6, [32, 32])
    t_2 = torch.ops.aten.t.default(inputs.primals_13)
    addmm_2 = torch.ops.aten.addmm.default(inputs.primals_14, view_11, t_2)
    view_12 = torch.ops.aten.view.default(addmm_2, [2, 16, 128])
    gelu = torch.ops.aten.gelu.default(view_12)
    view_13 = torch.ops.aten.view.default(gelu, [32, 128])
    t_3 = torch.ops.aten.t.default(inputs.primals_15)
    addmm_3 = torch.ops.aten.addmm.default(inputs.primals_16, view_13, t_3)
    view_14 = torch.ops.aten.view.default(addmm_3, [2, 16, 32])
    add_2 = torch.ops.aten.add.Tensor(inputs.add_1, view_14)
    return [add_2]
