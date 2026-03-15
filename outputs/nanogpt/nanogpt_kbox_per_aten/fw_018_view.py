"""018_model.py:22 [blocks.1] | view (qkv linear)


Inputs (16.4 KB total):
  getitem_9  float32[2x16x32]  4.0 KB
  primals_19  float32[96x32]  12.0 KB
  primals_20  float32[96]  384 B
Outputs (12.0 KB total):
  view_16  float32[2x16x96]  12.0 KB
Ops: view x2, t x1, addmm x1  (4 ops)

    kbox iterate fw_018_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_018_view.h5"}


def run(inputs):
    view_15 = torch.ops.aten.view.default(inputs.getitem_9, [32, 32])
    t_4 = torch.ops.aten.t.default(inputs.primals_19)
    addmm_4 = torch.ops.aten.addmm.default(inputs.primals_20, view_15, t_4)
    view_16 = torch.ops.aten.view.default(addmm_4, [2, 16, 96])
    return [view_16]
