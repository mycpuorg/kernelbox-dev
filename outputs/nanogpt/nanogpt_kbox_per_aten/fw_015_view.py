"""015_model.py:32 [blocks.0] | view (c_proj)


Inputs (8.1 KB total):
  view_8  float32[2x16x32]  4.0 KB
  primals_9  float32[32x32]  4.0 KB
  primals_10  float32[32]  128 B
Outputs (4.0 KB total):
  view_10  float32[2x16x32]  4.0 KB
Ops: view x2, t x1, addmm x1  (4 ops)

    kbox iterate fw_015_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_015_view.h5"}


def run(inputs):
    view_9 = torch.ops.aten.view.default(inputs.view_8, [32, 32])
    t_1 = torch.ops.aten.t.default(inputs.primals_9)
    addmm_1 = torch.ops.aten.addmm.default(inputs.primals_10, view_9, t_1)
    view_10 = torch.ops.aten.view.default(addmm_1, [2, 16, 32])
    return [view_10]
