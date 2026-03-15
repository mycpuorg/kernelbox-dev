# KernelBox Developer Guide

KernelBox is a CUDA kernel development toolkit built around a single workflow:
**write a kernel, save, see results instantly**. It compiles CUDA (or Triton)
kernels via NVRTC, launches them through a private worker daemon, and verifies
outputs against expected results — all with hot-reload on file save.

This guide covers every feature with real-world examples you can run directly.

---

## Quick Start

```bash
# Install dependencies and build
uv sync
make tools

# Run a test in watch mode (hot-reloads on save)
kbox iterate examples/dev/test_scale.py

# Run once and exit
kbox iterate examples/dev/test_scale.py --once

# Run once with benchmarking
kbox iterate examples/dev/test_scale.py --once --bench
```

---

## How It Works

KernelBox uses a two-process architecture:

```
Your test file (.py)
    |
    v
Python process (kernelbox)
    - Loads test file, calls init_once() or init()
    - Compiles .cu source to cubin via NVRTC
    - Allocates GPU memory via CUDA VMM
    - Sends cubin + memory file descriptors to worker
    |
    | Unix socket + SCM_RIGHTS fd passing
    v
C worker daemon (kbox_worker_daemon)
    - Imports VMM chunks (zero-copy, same GPU memory)
    - Caches cubins by hash (avoids reloading)
    - Launches kernels on GPU
    - Returns timing and status
```

The key insight is **zero-copy sharing**: both processes map the same physical
GPU memory. The Python side writes inputs, the worker launches the kernel, and
the Python side reads outputs — no copies at any point. This gives ~7us IPC
overhead per kernel launch.

---

## Writing Test Files

Every test file defines two things: **state** (what to run) and **execution**
(how to run it).

### Minimal Example

```python
# test_add_one.py
import torch

def init_once():
    x = torch.arange(8, device="cuda", dtype=torch.int32)
    return {
        "kernel": "kernels/add_one.cu",
        "inputs": [x],
        "expected": [x + 1],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
```

Run it:
```bash
kbox iterate test_add_one.py --once
```

Output:
```
[kbox] Loading test_add_one.py...
[kbox] Inputs:   torch.int32[8]
[kbox] Expected: torch.int32[8]
[kbox] Ran in 312.5ms
[kbox]   output[0]: [1, 2, 3, 4, 5, 6, 7, 8] int32 (8 elements)
[kbox] PASS: output[0]: PASS (max_err=0.00e+00)
```

### State Dictionary Keys

The dict returned by `init_once()` or `init()` controls everything:

**Kernel source** (exactly one required):
| Key | Type | Description |
|-----|------|-------------|
| `kernel` | str | Path to `.cu`, `.cubin`, or `.ptx` file |
| `kernel_source` | str/bytes | Inline CUDA source or raw cubin/PTX |
| `triton_kernel` | function | A `@triton.jit` decorated function |

**Data** (at least one required):
| Key | Type | Description |
|-----|------|-------------|
| `inputs` | list | Input tensors |
| `expected` | list | Expected output tensors |
| `h5` | str | Path to `.h5` or `.pt` file (auto-splits inputs/expected) |
| `h5_suite` | str | Path to directory of `.h5`/`.pt` files |
| `suite` | TestSuite | Pre-loaded test suite |

**Launch configuration** (all optional):
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `grid` | int/tuple | auto | Grid dimensions |
| `block` | int/tuple | 256 | Block dimensions |
| `smem` | int | 0 | Shared memory bytes |
| `outputs` | int/list | 1 | Output count or per-output specs |
| `out_dtype` | dtype | same as input | Output dtype override |

**Parameters** (optional):
| Key | Type | Description |
|-----|------|-------------|
| `params` | list | Full custom kernel parameter list |
| `extra_params` | list | Extra scalars appended after standard layout |
| `kernel_scratch_mib` | int | Scratch buffer allocation in MiB |

**Behavior** (optional):
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `atol` | float | 1e-5 | Absolute tolerance |
| `rtol` | float | 1e-5 | Relative tolerance |
| `benchmark` | bool | False | Auto-benchmark after pass |
| `restore_inputs` | bool | True | Snapshot/restore inputs each run |
| `func_name` | str | auto-detected | Kernel entry point name |
| `triton_constexprs` | dict | {} | Triton constexpr overrides |

### init_once() vs init()

- **`init_once()`** — called once. Inputs persist across hot-reloads. Use when
  you want to keep the same test data while editing the kernel.
- **`init()`** — called before every run. Fresh inputs each time. Use when inputs
  depend on code that changes.

### The run() Function

`run()` receives inputs and optionally a kernel handle:

```python
# Pure PyTorch (no kernel)
def run(inputs):
    return [torch.relu(inputs[0])]

# With kernel handle
def run(inputs, kernel):
    return [kernel(inputs[0])]

# With scratch pointer
def run(inputs, kernel, scratch_ptr):
    return [kernel(inputs[0])]

# Multi-kernel
def run(inputs, kernels):
    out1 = kernels[0](inputs[0])
    out2 = kernels[1](inputs[0])
    return [out1, out2]
```

The return value is always a **list of output tensors**.

### The post() Hook

Optional. Called after verification with the outputs and full state:

```python
def post(outputs, state):
    print(f"Max value: {outputs[0].max().item():.4f}")
```

---

## Inline Kernels

You can embed CUDA source directly in the test file:

```python
SCALE_KERNEL = r"""
extern "C" __global__ void scale(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * 2.5f;
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": SCALE_KERNEL,
        "inputs": [x],
        "expected": [x * 2.5],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
```

Changes to the test file trigger hot-reload, so editing the inline kernel
source and saving will recompile and re-run.

---

## Custom Kernel Parameters

The default parameter layout is `(in0, in1, ..., out0, out1, ..., n)`. For
kernels with different signatures, use `params=[]`:

```python
# Kernel signature: scale_shift(const float *x, float *y, float a, float b, uint n)
KERNEL = r"""
extern "C" __global__ void scale_shift(
    const float *x, float *y, float a, float b, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    a, b = 2.5, -1.0
    return {
        "kernel_source": KERNEL,
        "inputs": [x],
        "expected": [a * x + b],
        "outputs": 1,
    }

def run(inputs, kernel):
    import numpy as np
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0),       # x pointer
        kernel.out_ptr(0),      # y pointer
        np.float32(2.5),        # a
        np.float32(-1.0),       # b
        np.uint32(inputs[0].numel()),  # n
    ])]
```

**Pointer helpers:**
- `kernel.in_ptr(i)` — lazy reference to input buffer `i`
- `kernel.out_ptr(i)` — lazy reference to output buffer `i`
- `kernel.scratch_ptr` — scratch buffer address

These are resolved at launch time to actual device virtual addresses.

---

## Multiple Outputs

```python
KERNEL = r"""
extern "C" __global__ void split(
    const float *in0, float *out0, float *out1, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out0[i] = in0[i] + 1.0f;
        out1[i] = in0[i] * 2.0f;
    }
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": KERNEL,
        "inputs": [x],
        "expected": [x + 1.0, x * 2.0],
        "outputs": 2,
    }

def run(inputs, kernel):
    return list(kernel(inputs[0]))
```

---

## Data-Driven Tests

### Single data file

```python
def init_once():
    return {"h5": "data/mlp.pt"}

def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
```

The `.pt` or `.h5` file contains named tensors. Keys starting with `expected`
are separated as expected outputs; everything else becomes inputs accessible
via `inputs.key_name`.

### Test suites (directory of cases)

```python
def init_once():
    return {"h5_suite": "data/mlp_cases/"}

def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
```

Each `.pt`/`.h5` file in the directory is a separate test case. All cases share
the same `run()` function. Adding a new file to the directory triggers a
hot-reload.

---

## Scratch Buffers

Request GPU scratch memory for intermediate computations:

```python
def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": KERNEL,
        "kernel_scratch_mib": 1,   # 1 MiB scratch
        "inputs": [x],
        "expected": [x * 2.0],
    }

def run(inputs, kernel, scratch_ptr):
    # scratch_ptr is a device VA (int) for 1 MiB of GPU memory
    return [kernel(inputs[0])]
```

---

## Triton Kernels

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols
    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def init_once():
    rows, cols = 128, 256
    x = torch.randn(rows, cols, device="cuda")
    return {
        "triton_kernel": softmax_kernel,
        "triton_constexprs": {"BLOCK_SIZE": 256},
        "inputs": [x],
        "expected": [torch.softmax(x, dim=1)],
        "grid": rows,
        "atol": 1e-4,
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
```

---

## Multi-Kernel Sessions with kernel_mode()

For workloads that chain multiple kernels, define `kernel_mode()` instead of
`run()`. This gives you full control over the launch sequence:

```python
import numpy as np

ADD_ONE = r"""
extern "C" __global__ void add_one(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] + 1.0f;
}
"""

DOUBLE = r"""
extern "C" __global__ void double_it(
    const float *in0, float *out1, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out1[i] = in0[i] * 2.0f;
}
"""

def init_once():
    x = torch.randn(4096, device="cuda")
    return {
        "kernel_source": [ADD_ONE, DOUBLE],
        "inputs": [x],
        "expected": [x + 1.0, x * 2.0],
        "outputs": 2,
        "atol": 1e-5,
    }

def kernel_mode(kernels, scratch_ptr, input_ptrs, output_ptrs, n):
    block = 256
    grid = (n + block - 1) // block
    return [
        {
            "kernel": kernels[0],
            "grid": grid,
            "block": block,
            "params": [input_ptrs[0], output_ptrs[0], np.uint32(n)],
        },
        {
            "kernel": kernels[1],
            "grid": grid,
            "block": block,
            "params": [input_ptrs[0], output_ptrs[1], np.uint32(n)],
            "clear_outputs": False,
        },
    ]
```

### kernel_mode() signature

The framework inspects your function signature and injects only the parameters
you ask for:

```python
def kernel_mode(kernels, input_ptrs, output_ptrs, n):
    ...
def kernel_mode(kernels, input_ptrs, output_ptrs, n, scratch_ptr):
    ...
def kernel_mode(kernels, input_ptrs, output_ptrs, n, inputs_meta, outputs_meta):
    ...
```

**Available parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | handle | Single kernel (if only one) |
| `kernels` | list | All kernel handles |
| `input_ptrs` | list[int] | Input buffer device addresses |
| `output_ptrs` | list[int] | Output buffer device addresses |
| `n` | int | Element count |
| `scratch_ptr` | int | Scratch buffer address |
| `inputs_meta` | list[dict] | Input tensor metadata (dtype, shape, stride, numel) |
| `outputs_meta` | list[dict] | Output tensor metadata |

### Step dict format

Each step in the returned list describes one kernel launch:

```python
{
    "kernel": kernels[0],       # kernel handle (required)
    "grid": 16,                 # int or (gx, gy, gz) tuple
    "block": 256,               # int or (bx, by, bz) tuple
    "params": [...],            # parameter list
    "smem": 0,                  # shared memory bytes
    "clear_outputs": True,      # zero outputs before this launch
    "sync": False,              # synchronize after this launch
}
```

---

## Isolated Kernel Mode (Sandboxed Evaluation)

Run `kernel_mode()` planning in a subprocess with **no GPU access**:

```bash
kbox iterate test_file.py --once --bench --isolated-kernel-benchmark
```

The subprocess receives tensor metadata and kernel handles but cannot touch GPU
memory. It returns a launch plan that the parent process executes. This is used
by the task service for secure evaluation of untrusted `kernel_mode()` code.

---

## MCP Server (AI-Assisted Development)

KernelBox includes an MCP server for integration with AI tools:

```bash
kbox mcp
```

This exposes three tools:

1. **`list_tasks`** — enumerate available benchmark tasks
2. **`get_task_reference`** — get public task code, template, and iterate wrapper
3. **`evaluate_task_kernel_mode`** — submit `kernel_mode()` code for evaluation
   against hidden test data

### Task workflow

1. AI calls `list_tasks` to see available tasks
2. AI calls `get_task_reference("cuda_pairwise")` to get:
   - Public task source (inputs, expected, kernels)
   - `kernel_mode()` template to fill in
   - Iterate wrapper for local development
3. AI writes a `kernel_mode()` implementation
4. AI calls `evaluate_task_kernel_mode("cuda_pairwise", kernel_mode_code="...")`
   to test against hidden data with benchmarking

The evaluation runs in an isolated subprocess — the submitted code never gets
direct GPU access, only the ability to describe a launch plan.

---

## Benchmarking

### CLI flags

```bash
kbox iterate test.py --once --bench                    # default: 10 warmup, 100 iters
kbox iterate test.py --once --bench --warmup 20 --iters 500
kbox iterate test.py --once --bench --benchtime 3.0    # run for 3 seconds
kbox iterate test.py --once --bench --l2-flush 1       # flush L2 before benchmark
kbox iterate test.py --once --bench --l2-flush-per-iter 1  # flush per iteration
```

### In-test benchmarking

Set `"benchmark": True` in the state dict to auto-benchmark after verification:

```python
def init_once():
    return {
        "kernel": "kernel.cu",
        "inputs": [...],
        "expected": [...],
        "benchmark": True,
        "warmup": 10,
        "iters": 200,
    }
```

---

## Debugging Failed Tests

### Diff dumps

When a test fails, dump diagnostics:

```bash
kbox iterate test.py --once --dump debug.h5
kbox iterate test.py --once --dump debug.h5 --dump_min   # failed outputs only
kbox iterate test.py --once --dump debug.h5 --dump_max   # all + inputs + stats
```

The dump file contains per-output groups with `expected`, `actual`, `absdiff`,
`reldiff`, `max_abserr`, and `max_relerr`.

### Tolerance tuning

```bash
kbox iterate test.py --once --atol 1e-3 --rtol 1e-3
```

Or per-test:
```python
def init_once():
    return {
        ...,
        "atol": 1e-3,
        "rtol": 1e-3,
    }
```

---

## Watch Mode

The default mode. KernelBox watches for changes and re-runs:

```bash
kbox iterate test.py          # watches until Ctrl+C
kbox iterate test.py --interval 0.5   # custom poll interval
```

**Auto-watched files:**
- The test `.py` file itself
- Kernel files referenced by `kernel` key
- Data files referenced by `h5` key
- Suite directories referenced by `h5_suite`
- Python modules imported by the test file

Editing any of these triggers a reload. The workflow:

1. Open your kernel in an editor
2. Run `kbox iterate test.py` in a terminal
3. Edit the kernel, save
4. See results in ~5-15ms (compilation) + kernel runtime

---

## CLI Reference

```
kbox iterate <test.py> [options]
kbox <test.py> [options]              # shorthand

Options:
  --once                Run once and exit
  --atol FLOAT          Absolute tolerance (default: 1e-5)
  --rtol FLOAT          Relative tolerance (default: 1e-5)
  --bench               Run benchmark after successful verification
  --warmup INT          Benchmark warmup iterations (default: 10)
  --iters INT           Benchmark iterations (default: 100)
  --benchtime FLOAT     Benchmark for N seconds instead of fixed iterations
  --timeout FLOAT       Worker dispatch timeout in seconds
  --dump PATH           Write diff artifacts on failure
  --dump_min            Dump only failed outputs
  --dump_max            Dump all outputs + inputs + stats
  --interval FLOAT      File poll interval (default: 0.3s)
  --l2-flush INT        L2 flush passes before benchmark
  --l2-flush-per-iter INT  L2 flushes per benchmark iteration
  --l2-dirty            Use dirty (read+write) L2 flush
  --isolated-kernel-benchmark  Run kernel_mode() in isolated subprocess

kbox mcp               Start MCP server
kbox version           Print version
kbox --help            Show help
```
