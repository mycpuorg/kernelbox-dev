You are working inside the cloned repository at the current working directory, cloned from `https://github.com/mycpuorg/kernelbox-dev` (a fork of `ademeure/kernelbox-dev`).

**Do not assume anything. Read source files before writing a single line of code.**

---

## Step 0: Mandatory repo orientation

Read all of the following in full before proceeding:

```
README.md
CLAUDE.md
GUIDE.md
ITERATE.md
PIPELINE.md
ARUN.md
Makefile
pyproject.toml
python/kernelbox/dev.py
src/
tools/
tests/
examples/dev/test_mlp.py
examples/dev/test_mlp_suite.py
examples/dev/test_tutorial_03_multi_input.py
examples/dev/test_tutorial_06_shared_memory.py
examples/dev/test_tutorial_11_kernel_mode.py
examples/dev/test_tutorial_13_isolated_kernel_mode.py
```

Also `ls -R examples/` and read every other file under `examples/dev/`. Derive from the actual source:

1. Exact import paths and function/class signatures for registering a kernelbox test
2. Exact signature of `run()` — what it receives, how `inputs.field_name` attribute access works vs dict access, what it must return
3. How `init_once()` works — what keys it returns (`"h5"` vs `"h5_suite"`), how the harness uses them
4. How `.h5` files are structured — keys, dtypes, how tensors are loaded
5. CLI command to run a kernelbox test and get correctness + performance output
6. What "isolated kernel mode" means mechanically vs regular kernel mode
7. How inline CUDA is specified — string, decorator, file reference
8. How output tensors are declared and returned
9. How correctness checking works — automatic diff vs PyTorch reference, tolerance
10. Any gotchas in `CLAUDE.md`, `GUIDE.md`, or `ITERATE.md`

Do not write code until you have answered all 10 from actual file contents.

---

## What you are generating

You will generate a complete set of kernelbox `.py` test files — one per source-line group — covering the **entire forward pass of nanoGPT** as captured in the `nanogpt.h5` file. The source of truth for the complete op list and all tensor shapes is the commit at:

```
https://github.com/ademeure/torch-read-modify-write/commit/0322ef6444356080b5abe9c9923c90a529c3968d
```

That commit already generated a reference set of files in `outputs/nanogpt/nanogpt_kbox_per_aten/`. You are regenerating and extending that exact set inside the `mycpuorg/kernelbox-dev` repo, following the same file format and conventions exactly, with all 32 ops covered.

---

## The complete op list with shapes

The following is the **complete** list of all 32 source-line groups from the nanoGPT forward pass. Every file you generate must correspond to exactly one of these entries. Tensor shapes are derived from the nanoGPT config: batch=2, seq_len=16 (T), n_embd=32 (C), n_head=2, head_dim=16, n_layer=2, vocab_size=64, mlp_hidden=128.

```
fw_000  model.py:67   arange
        inputs:  (none)
        outputs: arange  int64[16]
        ops:     arange.start(0, 16, dtype=int64, device=cuda:0, pin_memory=False)

fw_001  model.py:68 [wte]  embedding
        inputs:  primals_2  float32[64x32], primals_1  int64[2x16]
        outputs: embedding  float32[2x16x32]
        ops:     embedding.default(primals_2, primals_1)

fw_002  model.py:68 [wpe]  embedding
        inputs:  primals_3  float32[16x32], arange  int64[16]
        outputs: embedding_1  float32[16x32]
        ops:     embedding.default(primals_3, arange)

fw_003  model.py:68   add
        inputs:  embedding  float32[2x16x32], embedding_1  float32[16x32]
        outputs: add  float32[2x16x32]
        ops:     add.Tensor(embedding, embedding_1)

fw_004  model.py:48 [blocks.0]  add (ln_1 residual)
        inputs:  add  float32[2x16x32], primals_4  float32[32], primals_5  float32[32], view_10  float32[2x16x32]
        outputs: add_1  float32[2x16x32]
        ops:     native_layer_norm.default(add, [32], primals_4, primals_5, 1e-05) → getitem x3, add.Tensor(add, view_10)

fw_005  model.py:22 [blocks.0]  view (qkv linear)
        inputs:  getitem  float32[2x16x32], primals_6  float32[96x32], primals_7  float32[96]
        outputs: view_1  float32[2x16x96]
        ops:     view.default(getitem, [32,32]), t.default(primals_6), addmm.default(primals_7, view, t), view.default(addmm, [2,16,96])

fw_006  model.py:23 [blocks.0]  split (q,k,v)
        inputs:  view_1  float32[2x16x96]
        outputs: getitem_3  float32[2x16x32], getitem_4  float32[2x16x32], getitem_5  float32[2x16x32]
        ops:     split.Tensor(view_1, 32, 2) → getitem x3

fw_007  model.py:24 [blocks.0]  transpose (q reshape)
        inputs:  getitem_3  float32[2x16x32]
        outputs: transpose  float32[2x2x16x16]
        ops:     view.default(getitem_3, [2,16,2,16]), transpose.int(view_2, 1, 2)

fw_008  model.py:25 [blocks.0]  transpose (k reshape)
        inputs:  getitem_4  float32[2x16x32]
        outputs: transpose_1  float32[2x2x16x16]
        ops:     view.default(getitem_4, [2,16,2,16]), transpose.int(view_3, 1, 2)

fw_009  model.py:26 [blocks.0]  transpose (v reshape)
        inputs:  getitem_5  float32[2x16x32]
        outputs: transpose_2  float32[2x2x16x16]
        ops:     view.default(getitem_5, [2,16,2,16]), transpose.int(view_4, 1, 2)

fw_010  model.py:27 [blocks.0]  mul (QK^T scaled)
        inputs:  transpose_1  float32[2x2x16x16], transpose  float32[2x2x16x16]
        outputs: mul  float32[2x2x16x16]
        ops:     transpose.int(transpose_1, -2, -1), expand x2, clone x2, _unsafe_view x2, bmm.default → view, mul.Tensor(view_5, 0.25)

fw_011  model.py:28 [blocks.0]  masked_fill (causal mask)
        inputs:  primals_8  float32[1x1x16x16], mul  float32[2x2x16x16]
        outputs: masked_fill  float32[2x2x16x16]
        ops:     alias.default(primals_8), eq.Scalar(alias, 0), masked_fill.Scalar(mul, eq, -inf)

fw_012  model.py:29 [blocks.0]  detach (softmax)
        inputs:  masked_fill  float32[2x2x16x16]
        outputs: detach  float32[2x2x16x16]
        ops:     _softmax.default(masked_fill, -1, False), detach.default(_softmax)

fw_013  model.py:30 [blocks.0]  view (att @ v)
        inputs:  _softmax  float32[2x2x16x16], transpose_2  float32[2x2x16x16]
        outputs: view_7  float32[2x2x16x16]
        ops:     expand x2, view.default(expand_2, [4,16,16]), clone, _unsafe_view, bmm.default → view.default(bmm_1, [2,2,16,16])

fw_014  model.py:31 [blocks.0]  view (y reshape)
        inputs:  view_7  float32[2x2x16x16]
        outputs: view_8  float32[2x16x32]
        ops:     transpose.int(view_7, 1, 2), clone.default(transpose_4, contiguous_format), view.default(clone_3, [2,16,32])

fw_015  model.py:32 [blocks.0]  view (c_proj)
        inputs:  view_8  float32[2x16x32], primals_9  float32[32x32], primals_10  float32[32]
        outputs: view_10  float32[2x16x32]
        ops:     view.default(view_8, [32,32]), t.default(primals_9), addmm.default(primals_10, view_9, t_1), view.default(addmm_1, [2,16,32])

fw_016  model.py:49 [blocks.0]  add (MLP residual, ln_2)
        inputs:  add_1  float32[2x16x32], primals_11  float32[32], primals_12  float32[32], primals_13  float32[128x32], primals_14  float32[128], primals_15  float32[32x128], primals_16  float32[32]
        outputs: add_2  float32[2x16x32]
        ops:     native_layer_norm → view x2, t x2, addmm x2, gelu, view x2, add.Tensor(add_1, view_14) — 14 ops total

fw_017  model.py:48 [blocks.1]  add (ln_1 residual)
        inputs:  add_2  float32[2x16x32], primals_17  float32[32], primals_18  float32[32], view_25  float32[2x16x32]
        outputs: add_3  float32[2x16x32]
        ops:     native_layer_norm.default(add_2, [32], primals_17, primals_18, 1e-05) → getitem x3, add.Tensor(add_2, view_25)

fw_018  model.py:22 [blocks.1]  view (qkv linear)
        inputs:  getitem_9  float32[2x16x32], primals_19  float32[96x32], primals_20  float32[96]
        outputs: view_16  float32[2x16x96]
        ops:     view.default(getitem_9, [32,32]), t.default(primals_19), addmm.default(primals_20, view_15, t_4), view.default(addmm_4, [2,16,96])

fw_019  model.py:23 [blocks.1]  split (q,k,v)
        inputs:  view_16  float32[2x16x96]
        outputs: getitem_12  float32[2x16x32], getitem_13  float32[2x16x32], getitem_14  float32[2x16x32]
        ops:     split.Tensor(view_16, 32, 2) → getitem x3

fw_020  model.py:24 [blocks.1]  transpose (q reshape)
        inputs:  getitem_12  float32[2x16x32]
        outputs: transpose_5  float32[2x2x16x16]
        ops:     view.default(getitem_12, [2,16,2,16]), transpose.int(view_17, 1, 2)

fw_021  model.py:25 [blocks.1]  transpose (k reshape)
        inputs:  getitem_13  float32[2x16x32]
        outputs: transpose_6  float32[2x2x16x16]
        ops:     view.default(getitem_13, [2,16,2,16]), transpose.int(view_18, 1, 2)

fw_022  model.py:26 [blocks.1]  transpose (v reshape)
        inputs:  getitem_14  float32[2x16x32]
        outputs: transpose_7  float32[2x2x16x16]
        ops:     view.default(getitem_14, [2,16,2,16]), transpose.int(view_19, 1, 2)

fw_023  model.py:27 [blocks.1]  mul (QK^T scaled)
        inputs:  transpose_6  float32[2x2x16x16], transpose_5  float32[2x2x16x16]
        outputs: mul_1  float32[2x2x16x16]
        ops:     transpose.int(transpose_6, -2, -1), expand x2, clone x2, _unsafe_view x2, bmm → view, mul.Tensor(view_20, 0.25)

fw_024  model.py:28 [blocks.1]  masked_fill (causal mask)
        inputs:  primals_21  float32[1x1x16x16], mul_1  float32[2x2x16x16]
        outputs: masked_fill_1  float32[2x2x16x16]
        ops:     alias.default(primals_21), eq.Scalar(alias_1, 0), masked_fill.Scalar(mul_1, eq_1, -inf)

fw_025  model.py:29 [blocks.1]  detach (softmax)
        inputs:  masked_fill_1  float32[2x2x16x16]
        outputs: detach_1  float32[2x2x16x16]
        ops:     _softmax.default(masked_fill_1, -1, False), detach.default(_softmax_1)

fw_026  model.py:30 [blocks.1]  view (att @ v)
        inputs:  _softmax_1  float32[2x2x16x16], transpose_7  float32[2x2x16x16]
        outputs: view_22  float32[2x2x16x16]
        ops:     expand x2, view.default(expand_6, [4,16,16]), clone, _unsafe_view, bmm.default → view.default(bmm_3, [2,2,16,16])

fw_027  model.py:31 [blocks.1]  view (y reshape)
        inputs:  view_22  float32[2x2x16x16]
        outputs: view_23  float32[2x16x32]
        ops:     transpose.int(view_22, 1, 2), clone.default(transpose_9, contiguous_format), view.default(clone_7, [2,16,32])

fw_028  model.py:32 [blocks.1]  view (c_proj)
        inputs:  view_23  float32[2x16x32], primals_22  float32[32x32], primals_23  float32[32]
        outputs: view_25  float32[2x16x32]
        ops:     view.default(view_23, [32,32]), t.default(primals_22), addmm.default(primals_23, view_24, t_5), view.default(addmm_5, [2,16,32])

fw_029  model.py:49 [blocks.1]  add (MLP residual, ln_2)
        inputs:  add_3  float32[2x16x32], primals_24  float32[32], primals_25  float32[32], primals_26  float32[128x32], primals_27  float32[128], primals_28  float32[32x128], primals_29  float32[32]
        outputs: add_4  float32[2x16x32]
        ops:     native_layer_norm → view x2, t x2, addmm x2, gelu, view x2, add — 14 ops total

fw_030  model.py:71 [ln_f]  native_layer_norm (final LN)
        inputs:  add_4  float32[2x16x32], primals_30  float32[32], primals_31  float32[32]
        outputs: getitem_18  float32[2x16x32], getitem_19  float32[2x16x1], getitem_20  float32[2x16x1]
        ops:     native_layer_norm.default(add_4, [32], primals_30, primals_31, 1e-05) → getitem x3

fw_031  model.py:72 [lm_head]  _unsafe_view (logits)
        inputs:  primals_2  float32[64x32], getitem_18  float32[2x16x32]
        outputs: _unsafe_view_6  float32[2x16x64]
        ops:     t.default(primals_2), view.default(getitem_18, [32,32]), mm.default(view_30, t_8), _unsafe_view.default(mm, [2,16,64])
```

---

## Step 1: Generate the .h5 fixture files

For each of the 32 ops above, check whether a corresponding `.h5` file already exists in the repo (look under `examples/dev/data/` or any `outputs/` directory). If `nanogpt.h5` exists anywhere on disk, use it as the source — load it with `h5py` and extract the relevant tensors for each op.

If no `nanogpt.h5` exists, generate all fixture data synthetically using the following procedure:

```python
import torch, h5py, numpy as np

torch.manual_seed(42)

# Generate all primary inputs with the shapes listed above
# primals_1: int64[2,16] — token ids in [0, 64)
# primals_2: float32[64,32] — wte weight
# primals_3: float32[16,32] — wpe weight
# primals_4..5: float32[32] — ln_1 weight/bias blocks.0
# primals_6: float32[96,32], primals_7: float32[96] — c_attn blocks.0
# primals_8: float32[1,1,16,16] — causal mask (upper triangular 0/1)
# primals_9: float32[32,32], primals_10: float32[32] — c_proj blocks.0
# primals_11..12: float32[32] — ln_2 weight/bias blocks.0
# primals_13: float32[128,32], primals_14: float32[128] — fc1 blocks.0
# primals_15: float32[32,128], primals_16: float32[32] — fc2 blocks.0
# primals_17..18: float32[32] — ln_1 weight/bias blocks.1
# primals_19: float32[96,32], primals_20: float32[96] — c_attn blocks.1
# primals_21: float32[1,1,16,16] — causal mask blocks.1
# primals_22: float32[32,32], primals_23: float32[32] — c_proj blocks.1
# primals_24..25: float32[32] — ln_2 weight/bias blocks.1
# primals_26: float32[128,32], primals_27: float32[128] — fc1 blocks.1
# primals_28: float32[32,128], primals_29: float32[32] — fc2 blocks.1
# primals_30..31: float32[32] — ln_f weight/bias

# Run the full forward graph once to compute all intermediate tensors
# Save everything to h5 — both primary inputs and all intermediates
# Use the exact key names from the op list above (arange, embedding, add, getitem, view_1, etc.)
```

For each `fw_NNN` file, create `data/fw_NNN_<opname>.h5` containing exactly the tensors listed as inputs for that op, under their exact key names. Follow the same key naming and serialization conventions used in the existing `.h5` files you found in the repo during Step 0.

---

## Step 2: Generate the 32 kernelbox .py files

Output directory: `outputs/nanogpt/nanogpt_kbox_per_aten/`

Each file must follow this **exact template**, derived from the reference files in the commit. Do not invent structure — match this precisely:

```python
"""NNN_model.py:LINE [module] | source_line_comment

Inputs (TOTAL total):
  tensor_name  dtype[shape]  size
  ...
Outputs (TOTAL total):
  tensor_name  dtype[shape]  size
  ...
Ops: op_name x count, ...  (N ops)

    kbox iterate fw_NNN_opname.py
"""
import torch
# import operator   ← only if getitem is used
# import math       ← only if needed


def init_once():
    return {"h5": "data/fw_NNN_opname.h5"}


def run(inputs):
    # ops in execution order, using inputs.field_name attribute access
    # intermediate results as local variables
    # return list of output tensors
    return [output_tensor]
```

Key rules derived from the reference files:
- `init_once()` returns `{"h5": "data/fw_NNN_opname.h5"}` for single-instance ops
- `run(inputs)` receives an object with attribute access: `inputs.tensor_name` not `inputs["tensor_name"]`
- All op calls use the full qualified path: `torch.ops.aten.op.variant(...)` — never `torch.nn.functional`, never abbreviated
- `operator.getitem` is imported and used for tuple outputs (split, native_layer_norm, etc.)
- Return value is always a Python list of tensors
- The docstring header block is required and must match the format above exactly, including the `kbox iterate fw_NNN_opname.py` line at the bottom of the docstring
- No extra imports, no classes, no helper functions — just `init_once()` and `run()`

---

## Step 3: Apply to every op in the list

Generate all 32 files. The complete file list is:

```
outputs/nanogpt/nanogpt_kbox_per_aten/fw_000_arange.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_001_embedding.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_002_embedding.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_003_add.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_004_add.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_005_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_006_split.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_007_transpose.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_008_transpose.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_009_transpose.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_010_mul.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_011_masked_fill.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_012_detach.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_013_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_014_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_015_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_016_add.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_017_add.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_018_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_019_split.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_020_transpose.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_021_transpose.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_022_transpose.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_023_mul.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_024_masked_fill.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_025_detach.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_026_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_027_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_028_view.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_029_add.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_030_native_layer_norm.py
outputs/nanogpt/nanogpt_kbox_per_aten/fw_031__unsafe_view.py
```

Also generate a corresponding `.h5` file for each under:
```
outputs/nanogpt/nanogpt_kbox_per_aten/data/fw_NNN_opname.h5
```

---

## Step 4: Validation

After generating all 32 files:

1. Run `python -m py_compile` on every generated `.py` file. Fix all syntax errors before proceeding.
2. Run `python -c "import ast; ast.parse(open('FILE').read())"` as a backup parse check if needed.
3. For every file, manually trace through the `run()` body and confirm: every `inputs.field_name` reference matches a key that exists in the corresponding `.h5` file, all intermediate variable names are consistent across dependent ops, and all `operator.getitem` calls reference actual tuple-returning ops.
4. Cross-check producer-consumer consistency: the output tensor name of `fw_NNN` must match the input tensor name used in `fw_NNN+M` that consumes it. Concretely:
   - `fw_001` outputs `embedding`, `fw_003` uses `inputs.embedding` ✓
   - `fw_002` outputs `embedding_1`, `fw_003` uses `inputs.embedding_1` ✓
   - `fw_006` outputs `getitem_3, getitem_4, getitem_5`, `fw_007` uses `inputs.getitem_3` ✓
   - `fw_012` produces `_softmax` (internal) and `detach`, `fw_013` uses `inputs._softmax` ✓
   - `fw_030` outputs `getitem_18`, `fw_031` uses `inputs.getitem_18` ✓
   — verify every such dependency in the chain
5. If a GPU is available, run `kbox iterate outputs/nanogpt/nanogpt_kbox_per_aten/fw_000_arange.py` (or the equivalent harness CLI from your Step 0 reading) on at least `fw_000`, `fw_003`, `fw_006`, `fw_012`, `fw_030`, and `fw_031` as a smoke test. Report correctness pass/fail.

---

## Step 5: Write a generation script for future use

Create `tools/gen_nanogpt_kbox.py` — a standalone script that, given a compiled aten graph file and a `nanogpt.h5` fixture, regenerates the entire `nanogpt_kbox_per_aten/` directory from scratch. It should:

1. Parse the aten graph (as Python source via `ast`) to extract all op calls, their input variable names, output variable names, and source line annotations from comments
2. For each source-line group, emit one `fw_NNN_opname.py` using the template from Step 2
3. Load the `.h5` fixture, extract the relevant tensors for each group, and write per-group `.h5` files
4. Print a summary at the end:

```
Generated 32 kbox files in outputs/nanogpt/nanogpt_kbox_per_aten/
Generated 32 h5 fixtures in outputs/nanogpt/nanogpt_kbox_per_aten/data/
Ops covered: arange, embedding x2, add x5, view x8, split x2, transpose x6, mul x2, masked_fill x2, detach x2, native_layer_norm x2, _unsafe_view x1
```

---

## Hard constraints

- Do not invent kernelbox APIs. Every function call and convention must be verified against the source you read in Step 0.
- `run(inputs)` must use `inputs.field_name` attribute access, not `inputs["field_name"]` dict access — the reference files confirm this.
- `init_once()` must return `{"h5": "data/fw_NNN_opname.h5"}` (single string path, not a list).
- `import operator` must be present if and only if the file uses `operator.getitem`.
- No file should import anything that isn't `torch`, `operator`, or `math`.
- All op calls must use the fully qualified `torch.ops.aten.op.variant(...)` form — this is what the reference files use.
- The docstring format including the `kbox iterate fw_NNN_opname.py` footer line is required in every file.
- Do not skip `fw_000_arange.py` — `arange` has no inputs and no `.h5` inputs block, which is a valid edge case. The `init_once()` still returns an h5 path (the file will just have no input datasets).
- All 32 `.py` files and all 32 `.h5` files must be present when you are done. Do not generate a subset.
