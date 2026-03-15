"""002_model.py:68 [wpe] | embedding


Inputs (2.1 KB total):
  primals_3  float32[16x32]  2.0 KB
  arange  int64[16]  128 B
Outputs (2.0 KB total):
  embedding_1  float32[16x32]  2.0 KB
Ops: embedding x1  (1 ops)

    kbox iterate fw_002_embedding.py
"""
import torch


def init_once():
    return {"h5": "data/fw_002_embedding.h5"}


def run(inputs):
    embedding_1 = torch.ops.aten.embedding.default(inputs.primals_3, inputs.arange)
    return [embedding_1]
