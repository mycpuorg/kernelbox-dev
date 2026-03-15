"""000_model.py:67 | arange


Inputs (0 B total):
  (none)
Outputs (128 B total):
  arange  int64[16]  128 B
Ops: arange x1  (1 ops)

    kbox iterate fw_000_arange.py
"""
import torch


def init_once():
    return {"h5": "data/fw_000_arange.h5"}


def run(inputs):
    arange = torch.ops.aten.arange.start(0, 16, dtype=torch.int64, device="cuda", pin_memory=False)
    return [arange]
