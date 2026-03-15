"""028_model.py:32 [blocks.1] | view (c_proj)


Inputs (8.1 KB total):
  view_23  float32[2x16x32]  4.0 KB
  primals_22  float32[32x32]  4.0 KB
  primals_23  float32[32]  128 B
Outputs (4.0 KB total):
  view_25  float32[2x16x32]  4.0 KB
Ops: view x2, t x1, addmm x1  (4 ops)

    kbox iterate fw_028_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_028_view.h5"}


def run(inputs):
    view_24 = torch.ops.aten.view.default(inputs.view_23, [32, 32])
    t_5 = torch.ops.aten.t.default(inputs.primals_22)
    addmm_5 = torch.ops.aten.addmm.default(inputs.primals_23, view_24, t_5)
    view_25 = torch.ops.aten.view.default(addmm_5, [2, 16, 32])
    return [view_25]
