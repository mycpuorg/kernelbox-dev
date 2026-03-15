"""004_model.py:48 [blocks.0] | add (ln_1 residual)


Inputs (8.2 KB total):
  add  float32[2x16x32]  4.0 KB
  primals_4  float32[32]  128 B
  primals_5  float32[32]  128 B
  view_10  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  add_1  float32[2x16x32]  4.0 KB
Ops: native_layer_norm x1, getitem x3, add x1  (5 ops)

    kbox iterate fw_004_add.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_004_add.h5"}


def run(inputs):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add, [32], inputs.primals_4, inputs.primals_5, 1e-05)
    getitem = operator.getitem(native_layer_norm, 0)
    getitem_1 = operator.getitem(native_layer_norm, 1)
    getitem_2 = operator.getitem(native_layer_norm, 2)
    add_1 = torch.ops.aten.add.Tensor(inputs.add, inputs.view_10)
    return [add_1]
