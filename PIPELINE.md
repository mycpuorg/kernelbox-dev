# KernelBox Pipeline

End-to-end pipeline for converting `torch.compile` aten op graphs into optimized CUDA kernels via KernelBox.

## Overview

```
Phase 1                    Phase 2                    Phase 3
aten graph + .h5    →    per-op kernelbox tests    →    CUDA kernel tests    →    patched aten graph
(torch.compile)          (PyTorch reference)            (inline CUDA)              (validated kernels)
```

The pipeline lives in `python/kernelbox/pipeline/` with a CLI at `tools/kbox_pipeline.py`.

## Quick Start

```bash
# Phase 1: Generate per-op test files from an aten graph
python tools/kbox_pipeline.py generate \
    --graph examples/pipeline_demo/graph.py \
    --h5 examples/pipeline_demo/graph_inputs.h5 \
    --output-dir output/

# Phase 2: Add CUDA kernels for all supported ops
python tools/kbox_pipeline.py cuda-all \
    --graph examples/pipeline_demo/graph.py \
    --test-dir output/ \
    --output-dir output_cuda/

# Phase 2 (single op): Generate CUDA kernel test for one op
python tools/kbox_pipeline.py cuda \
    --op gelu --h5 output/data/gelu_1.h5 --output test_gelu_cuda.py

# Iterate on the CUDA kernel until it passes
kbox iterate output_cuda/test_gelu_1_cuda.py --once
kbox iterate output_cuda/test_gelu_1_cuda.py --once --bench --isolated-kernel-benchmark

# Phase 3: Inject validated kernels back into the aten graph
python tools/kbox_pipeline.py inject \
    --graph examples/pipeline_demo/graph.py \
    --kernel gelu_1=output_cuda/test_gelu_1_cuda.py \
    --kernel relu_1=output_cuda/test_relu_1_cuda.py \
    --output patched_graph.py

# List supported ops
python tools/kbox_pipeline.py list-ops
```

## Phase 1: Generate Per-Op Tests

**Input:** An aten graph `run(input)` function + an `.h5` file with real tensor inputs.

```python
# Example aten graph (from torch.compile)
def run(input):
    out = {}
    out["gelu_1"] = torch.ops.aten.gelu.default(input["x"])
    out["add_1"]  = torch.ops.aten.add.Tensor(out["gelu_1"], input["bias"])
    out["mm_1"]   = torch.ops.aten.mm.default(out["add_1"], input["weight"])
    out["relu_1"] = torch.ops.aten.relu.default(out["mm_1"])
    return [out["relu_1"]]
```

**What it does:**

1. Parses the graph via AST to extract each aten op, its inputs, and outputs
2. Loads the `.h5` file and executes the graph to capture all intermediate tensors
3. For each op, saves a per-op `.h5` (inputs + expected output) and generates a kernelbox `.py` test file
4. Skips GEMMs (`mm`, `bmm`, `addmm`), attention, and view-like ops by default

**Output:** Per-op test files ready for `kbox iterate`:

```python
# Generated: test_gelu_1.py
def init_once():
    return {"h5": "data/gelu_1.h5"}

def run(inputs):
    result = torch.ops.aten.gelu.default(inputs["x"])
    return [result]
```

**Python API:**

```python
from kernelbox.pipeline import generate_per_op_tests

generated = generate_per_op_tests(graph_source, "data.h5", "output/")
# Returns: [(AtenOp, test_file_path), ...]
```

## Phase 2: Replace With CUDA Kernels

**Input:** A per-op `.h5` file from Phase 1.

**What it does:** Generates a kernelbox test file with an inline CUDA kernel replacing the aten op. Uses the isolated kernel mode pattern (`kernel_mode()`) so the kernel can be benchmarked safely with `--isolated-kernel-benchmark`.

**Supported ops (15):**

| Category | Ops |
|---|---|
| Activation | `gelu`, `relu`, `silu`, `sigmoid`, `tanh` |
| Unary math | `neg`, `abs`, `exp`, `log`, `rsqrt` |
| Binary elementwise | `add`, `mul`, `sub`, `div` |
| Reduction | `native_layer_norm` |

**Output:** Kernelbox test with inline CUDA + `kernel_mode()`:

```python
# Generated: test_gelu_1_cuda.py
KERNEL_SOURCE = r"""
extern "C" __global__ void gelu_kernel(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}
"""

def init_once():
    inputs, expected = kbox.h5.load_test("data/gelu_1.h5")
    return {
        "kernel_source": KERNEL_SOURCE,
        "inputs": [inputs[k] for k in sorted(inputs.keys()) if ...],
        "expected": expected,
        "outputs": 1,
        "atol": 1e-5,
    }

def kernel_mode(kernels, input_ptrs, output_ptrs, n):
    block = 256
    grid = (n + block - 1) // block
    return [{"kernel": kernels[0], "grid": grid, "block": block,
             "params": [input_ptrs[0], output_ptrs[0], np.uint32(n)]}]
```

**Python API:**

```python
from kernelbox.pipeline import generate_cuda_test, get_cuda_template, list_supported_ops

# Single op
generate_cuda_test(op, "data/gelu_1.h5", "test_gelu_cuda.py")

# Standalone (no AtenOp needed)
source = wrap_with_cuda_kernel("gelu", [input_tensor], [expected_tensor])
```

## Phase 3: Inject Back Into Aten Graph

**Input:** The original aten graph + validated kernelbox test file(s) from Phase 2.

**What it does:**

1. Reads the validated CUDA kernel from the test file
2. Wraps it in a C++ function compatible with `torch.utils.cpp_extension.load_inline`
3. Replaces the targeted aten op line in the graph with a call to the compiled CUDA extension
4. Preserves all variable names so downstream ops are unaffected
5. Skips GEMMs and attention ops

**Output:** Patched graph where selected aten ops are replaced with inline CUDA:

```python
# Before
out["gelu_1"] = torch.ops.aten.gelu.default(input["x"])

# After
out["gelu_1"] = _gelu_kernel_ext.gelu_kernel_forward(input["x"])
```

The compiled CUDA extension is loaded at module level via `load_inline`.

**Python API:**

```python
from kernelbox.pipeline import inject_kernel, generate_patched_graph

# Single op
patched = inject_kernel(graph_source, "gelu_1", "test_gelu_cuda.py")

# Multiple ops
generate_patched_graph(
    graph_source,
    validated_kernels={"gelu_1": "test_gelu_cuda.py", "relu_1": "test_relu_cuda.py"},
    output_path="patched_graph.py",
)
```

## File Layout

```
python/kernelbox/pipeline/
├── __init__.py           # Public API exports
├── graph_parser.py       # AST-based aten graph parser
├── codegen.py            # Phase 1: per-op test + h5 generation
├── cuda_templates.py     # Phase 2: CUDA kernel templates + test generation
└── inject.py             # Phase 3: kernel injection into aten graphs

tools/
└── kbox_pipeline.py      # CLI: generate, cuda, cuda-all, inject, list-ops

examples/pipeline_demo/   # Working end-to-end demo
├── graph.py              # Sample aten graph with gelu, add, mm, relu, mul
├── graph_inputs.h5       # Matching tensor inputs
├── output/               # Phase 1 output
│   ├── test_gelu_1.py
│   ├── test_add_1.py
│   ├── test_relu_1.py
│   ├── test_mul_1.py
│   └── data/             # Per-op h5 files
├── output_cuda/          # Phase 2 output
│   ├── test_gelu_1_cuda.py
│   ├── test_add_1_cuda.py
│   ├── test_relu_1_cuda.py
│   └── test_mul_1_cuda.py
└── patched_graph.py      # Phase 3 output
```

## Design Decisions

- **Isolated kernel mode** (`kernel_mode()`) is the default wrapping pattern, matching `test_tutorial_13_isolated_kernel_mode.py`
- **Per-op `.h5` files** contain both graph-input and intermediate tensors needed by that op, making each test fully self-contained
- **`torch.utils.cpp_extension.load_inline`** is used for Phase 3 injection since it's standard PyTorch and doesn't require kernelbox at inference time
- **GEMMs and attention** are always skipped — these are better served by cuBLAS/cuDNN or specialized kernels like FlashAttention
- **View-like ops** (reshape, transpose, permute, etc.) are skipped by default since they don't involve computation
