"""029_model.py:49 [blocks.1] | add (MLP residual, ln_2)


Inputs (36.9 KB total):
  add_3  float32[2x16x32]  4.0 KB
  primals_24  float32[32]  128 B
  primals_25  float32[32]  128 B
  primals_26  float32[128x32]  16.0 KB
  primals_27  float32[128]  512 B
  primals_28  float32[32x128]  16.0 KB
  primals_29  float32[32]  128 B
Outputs (4.0 KB total):
  add_4  float32[2x16x32]  4.0 KB
Ops: native_layer_norm x1, getitem x3, view x4, t x2, addmm x2, gelu x1, add x1  (14 ops)

    kbox iterate fw_029_add.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_029_add.h5"}


def run(inputs):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_3, [32], inputs.primals_24, inputs.primals_25, 1e-05)
    getitem_15 = operator.getitem(native_layer_norm, 0)
    getitem_16 = operator.getitem(native_layer_norm, 1)
    getitem_17 = operator.getitem(native_layer_norm, 2)
    view_26 = torch.ops.aten.view.default(getitem_15, [32, 32])
    t_6 = torch.ops.aten.t.default(inputs.primals_26)
    addmm_6 = torch.ops.aten.addmm.default(inputs.primals_27, view_26, t_6)
    view_27 = torch.ops.aten.view.default(addmm_6, [2, 16, 128])
    gelu_1 = torch.ops.aten.gelu.default(view_27)
    view_28 = torch.ops.aten.view.default(gelu_1, [32, 128])
    t_7 = torch.ops.aten.t.default(inputs.primals_28)
    addmm_7 = torch.ops.aten.addmm.default(inputs.primals_29, view_28, t_7)
    view_29 = torch.ops.aten.view.default(addmm_7, [2, 16, 32])
    add_4 = torch.ops.aten.add.Tensor(inputs.add_3, view_29)
    return [add_4]
