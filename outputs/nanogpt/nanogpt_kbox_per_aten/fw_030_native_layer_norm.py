"""030_model.py:71 [ln_f] | native_layer_norm (final LN)


Inputs (4.2 KB total):
  add_4  float32[2x16x32]  4.0 KB
  primals_30  float32[32]  128 B
  primals_31  float32[32]  128 B
Outputs (4.2 KB total):
  getitem_18  float32[2x16x32]  4.0 KB
  getitem_19  float32[2x16x1]  128 B
  getitem_20  float32[2x16x1]  128 B
Ops: native_layer_norm x1, getitem x3  (4 ops)

    kbox iterate fw_030_native_layer_norm.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_030_native_layer_norm.h5"}


def run(inputs):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_4, [32], inputs.primals_30, inputs.primals_31, 1e-05)
    getitem_18 = operator.getitem(native_layer_norm, 0)
    getitem_19 = operator.getitem(native_layer_norm, 1)
    getitem_20 = operator.getitem(native_layer_norm, 2)
    return [getitem_18, getitem_19, getitem_20]
