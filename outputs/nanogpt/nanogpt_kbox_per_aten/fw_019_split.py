"""019_model.py:23 [blocks.1] | split (q,k,v)


Inputs (12.0 KB total):
  view_16  float32[2x16x96]  12.0 KB
Outputs (12.0 KB total):
  getitem_12  float32[2x16x32]  4.0 KB
  getitem_13  float32[2x16x32]  4.0 KB
  getitem_14  float32[2x16x32]  4.0 KB
Ops: split x1, getitem x3  (4 ops)

    kbox iterate fw_019_split.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_019_split.h5"}


def run(inputs):
    split_1 = torch.ops.aten.split.Tensor(inputs.view_16, 32, 2)
    getitem_12 = operator.getitem(split_1, 0)
    getitem_13 = operator.getitem(split_1, 1)
    getitem_14 = operator.getitem(split_1, 2)
    return [getitem_12, getitem_13, getitem_14]
