"""001_model.py:68 [wte] | embedding


Inputs (8.2 KB total):
  primals_2  float32[64x32]  8.0 KB
  primals_1  int64[2x16]  256 B
Outputs (4.0 KB total):
  embedding  float32[2x16x32]  4.0 KB
Ops: embedding x1  (1 ops)

    kbox iterate fw_001_embedding.py
"""
import torch


def init_once():
    return {"h5": "data/fw_001_embedding.h5"}


def run(inputs):
    embedding = torch.ops.aten.embedding.default(inputs.primals_2, inputs.primals_1)
    return [embedding]
