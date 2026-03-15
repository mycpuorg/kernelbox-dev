Here's the plain prompt:

---

You are working on a project called **kernelbox** — a framework for benchmarking and optimizing GPU kernels by loading real tensor inputs from `.h5` files and iterating on correctness and performance.

Your task is to build an end-to-end pipeline with three phases:

---

## Context

Study these files carefully before writing any code:

- `examples/dev/test_mlp.py` — baseline: how a kernelbox test is structured, how PyTorch ops are run
- `examples/dev/test_mlp_suite.py` — how `.h5` inputs are loaded and fed into kernelbox
- `examples/dev/test_tutorial_03_multi_input.py` — how inline CUDA kernels with multiple inputs work
- `examples/dev/test_tutorial_06_shared_memory.py` — more complex kernel patterns (shared memory, etc.)
- `examples/dev/test_tutorial_11_kernel_mode.py` — how multiple kernels are composed via kernel mode
- `examples/dev/test_tutorial_13_isolated_kernel_mode.py` — **preferred pattern**: fully isolated kernel mode with minimal reward hacking risk

Understand the conventions: how inputs are named, how outputs are collected, what `run()` looks like, how `.h5` files map to tensor inputs, how correctness checking works, and how performance is measured.

---

## Phase 1: Generate kernelbox `.py` files from aten op graphs

Given an aten op graph like the following:

```python
def run(input):
    out = {}
    out["view_31"] = torch.ops.aten.view.default(input["tangents_1"], [32, 64])
    out["t_9"] = torch.ops.aten.t.default(out["view_31"])
    out["mm_1"] = torch.ops.aten.mm.default(out["t_9"], input["view_30"])
    out["t_10"] = torch.ops.aten.t.default(out["mm_1"])
    out["t_11"] = torch.ops.aten.t.default(input["t_8"])
    out["mm_2"] = torch.ops.aten.mm.default(out["view_31"], out["t_11"])
    out["view_32"] = torch.ops.aten.view.default(out["mm_2"], [2, 16, 32])
    out["t_12"] = torch.ops.aten.t.default(out["t_10"])
    return [out["t_12"]]
```

Generate kernelbox `.py` test files that:

1. **Per aten op (multi-input, single-output)**: one kernelbox file per op. Each file loads its tensor inputs from the corresponding `.h5` file (following the pattern in `test_mlp_suite.py`), runs the single aten op, and exposes it through the kernelbox `run()` interface.

2. **Per op group (stretch goal)**: one kernelbox file per logical fusion group — either matching a Triton fusion boundary or grouped by source line comment (e.g., `# /model.py:49`). This is lower priority; do per-aten first.

For now, skip strides — shapes are enough. We will add strides info later.

Follow the isolated kernel mode pattern from `test_tutorial_13_isolated_kernel_mode.py` as the target structure wherever possible.

---

## Phase 2: Replace aten ops with inline CUDA kernels

For a given aten op (or small group of ops), replace the PyTorch implementation inside the kernelbox `.py` file with an inline CUDA kernel.

**Scope for MVP**: only ops that are NOT GEMMs and NOT attention. Focus on elementwise and reduction ops such as:
- `aten.gelu`
- `aten.native_layer_norm`
- `aten.add`, `aten.mul`, `aten.relu`, etc.

Example targets from actual compiled model code:

```python
# self.blocks.1.mlp.1 (GELU)
# /model.py:49
blocks1_mlp1_gelu: 'float32[2, 16, 128]' = aten.gelu(blocks1_mlp0_view_1)
```

```python
# self.ln_f (LayerNorm)
# /model.py:71
ln_f_native_layer_norm = aten.native_layer_norm(blocks1_add_1, [32], ln_f_weight, ln_f_bias, 1e-05)
ln_f_getitem: 'float32[2, 16, 32]' = operator.getitem(ln_f_native_layer_norm, 0)
ln_f_getitem_1: 'float32[2, 16, 1]' = operator.getitem(ln_f_native_layer_norm, 1)
ln_f_getitem_2: 'float32[2, 16, 1]' = operator.getitem(ln_f_native_layer_norm, 2)
```

For each such op or group:

1. Allocate output tensors using `torch.empty(...)` with the correct shapes and dtypes
2. Write an inline CUDA kernel that implements the op
3. Wire up inputs and outputs following the multi-input pattern in `test_tutorial_03_multi_input.py`
4. Use isolated kernel mode (`test_tutorial_13_isolated_kernel_mode.py`) as the wrapping structure
5. Iterate via kernelbox until both correctness (matches PyTorch reference) and performance targets are met

For `native_layer_norm` and other ops that return tuples via `operator.getitem`, handle all outputs explicitly.

---

## Phase 3: Inject optimized kernels back into the compiled aten graph

Once a kernel passes correctness and performance checks in kernelbox, integrate it back into the actual compiled aten `.py` file that is injected via `torch.compile`.

The replacement should:

1. Remove the original aten op call(s) for the target op/group
2. Replace with the validated inline CUDA kernel, pre-allocating output tensors via `torch.empty`
3. Preserve all variable names so downstream ops in the graph are unaffected
4. Handle multi-output ops (like `native_layer_norm`) by maintaining all `getitem` outputs with correct names

For MVP, only replace non-GEMM, non-attention ops. Skip ops that involve matmul (`aten.mm`, `aten.bmm`, `aten.addmm`) or attention (`aten.scaled_dot_product_attention`) — leave those as-is.

---

## What to build

1. A script or set of utilities that, given:
   - an aten graph `run()` function (as Python source or AST)
   - a corresponding `.h5` file with real tensor inputs

   ...automatically generates per-op kernelbox `.py` files ready to run

2. A template or utility for wrapping a single aten op as an inline CUDA kernel inside the kernelbox isolated mode structure, with correctness checking against the PyTorch reference

3. A script that takes a validated kernelbox CUDA kernel and patches it back into the full compiled aten `.py` graph file, replacing the original op call

---

## Constraints and preferences

- Follow the exact conventions and APIs used in the existing kernelbox examples — do not invent new abstractions unless necessary
- Default to `test_tutorial_13_isolated_kernel_mode.py` (isolated kernel mode) as the wrapping pattern; only fall back to the non-isolated variant if there is a concrete blocker
- Keep inline CUDA kernels readable — no unnecessary cleverness unless it is needed for performance
- All generated code must be valid Python that can be run directly with the kernelbox test runner
- Do not attempt to handle GEMMs or attention in this pass
