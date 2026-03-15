# NanoGPT Per-Aten-Op KernelBox Tests

Per-op kernelbox test files covering the **complete forward pass** of nanoGPT, decomposed into 32 source-line groups. Each file wraps one group of `torch.ops.aten.*` calls with real tensor data, ready to run with `kbox iterate`.

## Model Config

| Parameter | Value |
|-----------|-------|
| batch | 2 |
| seq_len (T) | 16 |
| n_embd (C) | 32 |
| n_head | 2 |
| head_dim | 16 |
| n_layer | 2 |
| vocab_size | 64 |
| mlp_hidden | 128 |

## Running a Test

```bash
# Single run
kbox iterate outputs/nanogpt/nanogpt_kbox_per_aten/fw_003_add.py --once

# With benchmarking
kbox iterate outputs/nanogpt/nanogpt_kbox_per_aten/fw_010_mul.py --once --bench

# Watch mode (re-runs on save)
kbox iterate outputs/nanogpt/nanogpt_kbox_per_aten/fw_016_add.py
```

## Regenerating from Scratch

```bash
uv run python3 tools/gen_nanogpt_kbox.py [--output-dir DIR]
```

This runs the full nanoGPT forward pass on CPU with `torch.manual_seed(42)`, captures all intermediate tensors, and generates all 32 `.py` + `.h5` files.

## Forward Pass Map

The 32 ops trace through: token embedding, position embedding, 2 transformer blocks (each with layer norm, multi-head attention, MLP), final layer norm, and logit projection.

```
                        ┌─────────────────── Block 0 ───────────────────┐
                        │                                               │
  fw_000  arange        │  fw_004  ln_1 + residual                     │
  fw_001  wte embed     │  fw_005  qkv linear                          │
  fw_002  wpe embed     │  fw_006  split → q, k, v                     │
  fw_003  tok + pos     │  fw_007  q reshape                           │
          │             │  fw_008  k reshape                           │
          └─────────────│  fw_009  v reshape                           │
                        │  fw_010  QK^T / sqrt(d)                      │
                        │  fw_011  causal mask                         │
                        │  fw_012  softmax                             │
                        │  fw_013  att @ v                             │
                        │  fw_014  y reshape                           │
                        │  fw_015  c_proj linear                       │
                        │  fw_016  ln_2 + MLP + residual               │
                        └──────────────────────────────────────────────┘
                        ┌─────────────────── Block 1 ───────────────────┐
                        │                                               │
                        │  fw_017  ln_1 + residual                     │
                        │  fw_018  qkv linear                          │
                        │  fw_019  split → q, k, v                     │
                        │  fw_020  q reshape                           │
                        │  fw_021  k reshape                           │
                        │  fw_022  v reshape                           │
                        │  fw_023  QK^T / sqrt(d)                      │
                        │  fw_024  causal mask                         │
                        │  fw_025  softmax                             │
                        │  fw_026  att @ v                             │
                        │  fw_027  y reshape                           │
                        │  fw_028  c_proj linear                       │
                        │  fw_029  ln_2 + MLP + residual               │
                        └──────────────────────────────────────────────┘

  fw_030  ln_f (final layer norm)
  fw_031  lm_head (logit projection)
```

## Complete File List

| File | Source Line | Module | Description | Ops | Inputs | Outputs |
|------|-----------|--------|-------------|-----|--------|---------|
| `fw_000_arange.py` | model.py:67 | | arange | 1 | (none) | int64[16] |
| `fw_001_embedding.py` | model.py:68 | wte | token embedding | 1 | primals_2 float32[64x32], primals_1 int64[2x16] | float32[2x16x32] |
| `fw_002_embedding.py` | model.py:68 | wpe | position embedding | 1 | primals_3 float32[16x32], arange int64[16] | float32[16x32] |
| `fw_003_add.py` | model.py:68 | | tok + pos | 1 | embedding float32[2x16x32], embedding_1 float32[16x32] | float32[2x16x32] |
| `fw_004_add.py` | model.py:48 | blocks.0 | ln_1 + residual | 5 | add float32[2x16x32], primals_4/5 float32[32], view_10 float32[2x16x32] | float32[2x16x32] |
| `fw_005_view.py` | model.py:22 | blocks.0 | qkv linear | 4 | getitem float32[2x16x32], primals_6 float32[96x32], primals_7 float32[96] | float32[2x16x96] |
| `fw_006_split.py` | model.py:23 | blocks.0 | split q,k,v | 4 | view_1 float32[2x16x96] | 3x float32[2x16x32] |
| `fw_007_transpose.py` | model.py:24 | blocks.0 | q reshape | 2 | getitem_3 float32[2x16x32] | float32[2x2x16x16] |
| `fw_008_transpose.py` | model.py:25 | blocks.0 | k reshape | 2 | getitem_4 float32[2x16x32] | float32[2x2x16x16] |
| `fw_009_transpose.py` | model.py:26 | blocks.0 | v reshape | 2 | getitem_5 float32[2x16x32] | float32[2x2x16x16] |
| `fw_010_mul.py` | model.py:27 | blocks.0 | QK^T scaled | 10 | transpose float32[2x2x16x16], transpose_1 float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_011_masked_fill.py` | model.py:28 | blocks.0 | causal mask | 3 | primals_8 float32[1x1x16x16], mul float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_012_detach.py` | model.py:29 | blocks.0 | softmax | 2 | masked_fill float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_013_view.py` | model.py:30 | blocks.0 | att @ v | 7 | _softmax float32[2x2x16x16], transpose_2 float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_014_view.py` | model.py:31 | blocks.0 | y reshape | 3 | view_7 float32[2x2x16x16] | float32[2x16x32] |
| `fw_015_view.py` | model.py:32 | blocks.0 | c_proj | 4 | view_8 float32[2x16x32], primals_9 float32[32x32], primals_10 float32[32] | float32[2x16x32] |
| `fw_016_add.py` | model.py:49 | blocks.0 | ln_2 + MLP + residual | 14 | add_1 float32[2x16x32], primals_11-16 | float32[2x16x32] |
| `fw_017_add.py` | model.py:48 | blocks.1 | ln_1 + residual | 5 | add_2 float32[2x16x32], primals_17/18 float32[32], view_25 float32[2x16x32] | float32[2x16x32] |
| `fw_018_view.py` | model.py:22 | blocks.1 | qkv linear | 4 | getitem_9 float32[2x16x32], primals_19 float32[96x32], primals_20 float32[96] | float32[2x16x96] |
| `fw_019_split.py` | model.py:23 | blocks.1 | split q,k,v | 4 | view_16 float32[2x16x96] | 3x float32[2x16x32] |
| `fw_020_transpose.py` | model.py:24 | blocks.1 | q reshape | 2 | getitem_12 float32[2x16x32] | float32[2x2x16x16] |
| `fw_021_transpose.py` | model.py:25 | blocks.1 | k reshape | 2 | getitem_13 float32[2x16x32] | float32[2x2x16x16] |
| `fw_022_transpose.py` | model.py:26 | blocks.1 | v reshape | 2 | getitem_14 float32[2x16x32] | float32[2x2x16x16] |
| `fw_023_mul.py` | model.py:27 | blocks.1 | QK^T scaled | 10 | transpose_5 float32[2x2x16x16], transpose_6 float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_024_masked_fill.py` | model.py:28 | blocks.1 | causal mask | 3 | primals_21 float32[1x1x16x16], mul_1 float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_025_detach.py` | model.py:29 | blocks.1 | softmax | 2 | masked_fill_1 float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_026_view.py` | model.py:30 | blocks.1 | att @ v | 7 | _softmax_1 float32[2x2x16x16], transpose_7 float32[2x2x16x16] | float32[2x2x16x16] |
| `fw_027_view.py` | model.py:31 | blocks.1 | y reshape | 3 | view_22 float32[2x2x16x16] | float32[2x16x32] |
| `fw_028_view.py` | model.py:32 | blocks.1 | c_proj | 4 | view_23 float32[2x16x32], primals_22 float32[32x32], primals_23 float32[32] | float32[2x16x32] |
| `fw_029_add.py` | model.py:49 | blocks.1 | ln_2 + MLP + residual | 14 | add_3 float32[2x16x32], primals_24-29 | float32[2x16x32] |
| `fw_030_native_layer_norm.py` | model.py:71 | ln_f | final layer norm | 4 | add_4 float32[2x16x32], primals_30/31 float32[32] | float32[2x16x32], float32[2x16x1] x2 |
| `fw_031__unsafe_view.py` | model.py:72 | lm_head | logit projection | 4 | primals_2 float32[64x32], getitem_18 float32[2x16x32] | float32[2x16x64] |

## Data Flow

Each test file is self-contained — its `.h5` fixture provides all input tensors. The data flows between groups as:

```
fw_000 (arange) ──────────────────────────────> fw_002 (wpe embedding)
fw_001 (wte embedding) ──┐
fw_002 (wpe embedding) ──┴──> fw_003 (add) ──> fw_004 (ln_1 + residual)
                                                  │
                                          getitem ├──> fw_005 (qkv) ──> fw_006 (split)
                                                  │                        │
                                                  │     ┌── q ── fw_007 ──┐
                                                  │     ├── k ── fw_008 ──┤
                                                  │     └── v ── fw_009 ──┤
                                                  │                       │
                                                  │    fw_010 (QK^T) <────┤
                                                  │         │             │
                                                  │    fw_011 (mask)      │
                                                  │    fw_012 (softmax)   │
                                                  │    fw_013 (att@v) <───┘
                                                  │    fw_014 (reshape)
                                                  │    fw_015 (c_proj) ──> fw_004 (view_10 input)
                                                  │
                                             add_1 ──> fw_016 (MLP) ──> add_2
                                                                          │
                                                    [ Block 1: same pattern, fw_017–fw_029 ]
                                                                          │
                                                                     add_4 ──> fw_030 (ln_f)
                                                                                  │
                                                                          getitem_18 ──> fw_031 (logits)
```

## File Structure

```
outputs/nanogpt/nanogpt_kbox_per_aten/
├── fw_000_arange.py          # 32 test files
├── fw_001_embedding.py
├── ...
├── fw_031__unsafe_view.py
└── data/
    ├── fw_000_arange.h5      # 32 h5 fixtures
    ├── fw_001_embedding.h5
    ├── ...
    └── fw_031__unsafe_view.h5
```

## Test File Format

Every file follows the same structure:

```python
"""NNN_model.py:LINE [module] | description

Inputs (size total):
  tensor_name  dtype[shape]  size
Outputs (size total):
  tensor_name  dtype[shape]  size
Ops: op_name x count  (N ops)

    kbox iterate fw_NNN_opname.py
"""
import torch
import operator  # only if getitem is used


def init_once():
    return {"h5": "data/fw_NNN_opname.h5"}


def run(inputs):
    # All ops use torch.ops.aten.op.variant() form
    # Inputs accessed via attribute: inputs.tensor_name
    result = torch.ops.aten.add.Tensor(inputs.embedding, inputs.embedding_1)
    return [result]
```

## Op Coverage Summary

| Op Type | Count | Files |
|---------|-------|-------|
| arange | 1 | fw_000 |
| embedding | 2 | fw_001, fw_002 |
| add (residual) | 5 | fw_003, fw_004, fw_016, fw_017, fw_029 |
| view (linear) | 8 | fw_005, fw_013, fw_014, fw_015, fw_018, fw_026, fw_027, fw_028 |
| split | 2 | fw_006, fw_019 |
| transpose | 6 | fw_007, fw_008, fw_009, fw_020, fw_021, fw_022 |
| mul (scaled dot) | 2 | fw_010, fw_023 |
| masked_fill | 2 | fw_011, fw_024 |
| detach (softmax) | 2 | fw_012, fw_025 |
| native_layer_norm | 1 | fw_030 |
| _unsafe_view (logits) | 1 | fw_031 |
| **Total** | **32** | |
