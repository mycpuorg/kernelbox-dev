"""005_model.py:22 [blocks.0] | view (qkv linear)


Inputs (16.4 KB total):
  getitem  float32[2x16x32]  4.0 KB
  primals_6  float32[96x32]  12.0 KB
  primals_7  float32[96]  384 B
Outputs (12.0 KB total):
  view_1  float32[2x16x96]  12.0 KB
Ops: view x2, t x1, addmm x1  (4 ops)

    kbox iterate fw_005_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_005_view.h5"}


def run(inputs):
    view = torch.ops.aten.view.default(inputs.getitem, [32, 32])
    t = torch.ops.aten.t.default(inputs.primals_6)
    addmm = torch.ops.aten.addmm.default(inputs.primals_7, view, t)
    view_1 = torch.ops.aten.view.default(addmm, [2, 16, 96])
    return [view_1]
