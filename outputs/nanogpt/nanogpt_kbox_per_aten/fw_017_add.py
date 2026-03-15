"""017_model.py:48 [blocks.1] | add (ln_1 residual)


Inputs (8.2 KB total):
  add_2  float32[2x16x32]  4.0 KB
  primals_17  float32[32]  128 B
  primals_18  float32[32]  128 B
  view_25  float32[2x16x32]  4.0 KB
Outputs (4.0 KB total):
  add_3  float32[2x16x32]  4.0 KB
Ops: native_layer_norm x1, getitem x3, add x1  (5 ops)

    kbox iterate fw_017_add.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_017_add.h5"}


def run(inputs):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_2, [32], inputs.primals_17, inputs.primals_18, 1e-05)
    getitem_9 = operator.getitem(native_layer_norm, 0)
    getitem_10 = operator.getitem(native_layer_norm, 1)
    getitem_11 = operator.getitem(native_layer_norm, 2)
    add_3 = torch.ops.aten.add.Tensor(inputs.add_2, inputs.view_25)
    return [add_3]
