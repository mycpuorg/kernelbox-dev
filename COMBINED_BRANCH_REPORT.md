# Combined Branch Report: `combined-main-codex`

This document describes the `combined-main-codex` branch, which merges two lines
of work that diverged from `main` at commit `ac6dd0e`:

1. **Build fixes & iterate-only cleanup** (from `main`, commit `b470c93`)
2. **Task service, MCP server, isolated kernel mode** (from `codex/polish-20260313`, commits `d7338bf` + `d839866`)
3. **Isolated planner env fix** (merge-time fix, commit `b5d033c`)

**Test status: 49/49 passing** (`make test-all`)

---

## Changes from main branch (b470c93)

### Build system fixes

- **CUDA_HOME detection** (`Makefile`): Fixed `dirname` errors when `nvcc` is
  not on PATH. The shell expansion now guards against empty `which nvcc` output.
- **libcuda.so link path** (`Makefile`): Added `$(CUDA_HOME)/lib64/stubs` to
  `LDFLAGS`. On many systems (containers, cloud VMs), `libcuda.so` lives in the
  stubs directory rather than `lib64` directly. Without this, `make all` failed
  with `-lcuda: No such file or directory`.
- **sync-python target** (`Makefile`): Now requires `uv` and prints install
  instructions if missing, rather than silently falling back to pip.
- **Python interpreter in tests** (`tests/test_kbox_cli.sh`): Replaced hardcoded
  `.venv/bin/python` with `$KBOX_TEST_PYTHON` env var fallback to `python3`.

### nvcc discovery in test files

Three test files (`test_load_cubin.py`, `test_load_ptx.py`, `test_tma_copy.py`)
called `nvcc` directly, which failed when it wasn't on PATH. Added a
`_find_nvcc()` helper that checks `shutil.which("nvcc")` then falls back to
`$CUDA_HOME/bin/nvcc`.

### 3D grid/block tuple fix

`KernelSession._expand_per_kernel()` in `dev.py` treated any tuple or list as a
per-kernel value list. This broke 3D grid tuples like `(1, 1, 1)` in
single-kernel sessions (interpreted as 3 per-kernel scalar grids).

**Fix**: Only plain `list` values are treated as per-kernel lists. `tuple` values
are kept as single values (e.g., 3D grid/block dimensions). Multi-kernel sessions
should use `[grid_for_k0, grid_for_k1, ...]` (a list) to specify per-kernel grids.

### Direct API removal

Removed all `0*.py` direct Python API examples:
- `01_hello.py`, `02_verify.py`, `03_multi_io.py`, `04_session.py`
- `05_complex.py`, `06_watch.py`, `07_benchmark.py`, `08_softmax.py`

Created iterate-mode replacements for the four that were tested:
- `test_hello.py` — add_one kernel with sequential int32 input
- `test_fused_add_mul.py` — two-output kernel (sum + product)
- `test_weighted_blend.py` — three-input, two-output blend kernel
- `test_softmax.py` — softmax with shared memory and explicit 3D grid/block

### CLAUDE.md

Added `CLAUDE.md` with build commands, architecture overview, two-process model
description, protocol contract details, key source file map, and error recovery
documentation.

---

## Changes from codex/polish-20260313 (d7338bf + d839866)

### Task service (`python/kernelbox/task_service.py`, 218 lines)

A trusted evaluation service that connects the MCP server to kernelbox's iterate
infrastructure. Key functions:

- `list_task_summaries()` — returns metadata for all registered tasks
- `get_task_reference(task_name)` — returns public task source, kernel_mode
  template, and iterate wrapper for local development
- `evaluate_task_kernel_mode(task_name, code_or_path)` — runs a `kernel_mode()`
  submission against hidden test data with isolated planning, returns
  correctness and benchmark results

The service enforces a public/hidden split: users see the public task definition
and write a `kernel_mode()` function, but evaluation runs against hidden test
cases with different data distributions.

### Task registry (`python/kernelbox/tasks.py`, 99 lines)

Allowlisted task definitions. Each `TaskSpec` has:
- `public_module` — visible to the user (input generation, kernel source)
- `hidden_module` — used only during evaluation (different data, suite of cases)
- `template_module` — reference `kernel_mode()` implementation

Current tasks:
- **`cuda_pairwise`** — two CUDA kernels: `add_one` (out=x+1) and `double_it` (out=x*2)
- **`triton_pairwise`** — same operations via Triton JIT kernels

### Task definitions (`python/kernelbox/task_defs/`, 6 files, 180 lines total)

Each task has three modules:
- `*_public.py` — `init_once()` with public inputs/expected, kernel source
- `*_hidden.py` — `init_once()` with hidden `TestSuite` (3 cases: small/medium/large)
- `*_kernel_mode.py` — reference `kernel_mode()` that returns launch steps

### Isolated kernel mode (`python/kernelbox/isolated_kernel_mode.py`, 571 lines)

Runs `kernel_mode()` planning in a sandboxed one-shot subprocess. The subprocess:
- Has no GPU access (`CUDA_VISIBLE_DEVICES=""`, `NVIDIA_VISIBLE_DEVICES=void`)
- Communicates via Unix socket IPC (JSON over length-prefixed messages)
- Receives tensor metadata (shapes, dtypes, n) and kernel descriptors
- Returns a list of launch steps (kernel index, grid, block, params, smem)

The parent process then executes those steps on the real GPU via `KernelSession`.

This prevents `kernel_mode()` submissions from directly touching GPU memory or
running arbitrary CUDA code outside the declared kernel launches.

**Architecture**:
```
Parent (GPU access)                 Subprocess (no GPU)
  │                                    │
  ├─ serialize tensor metadata ──────► │
  │                                    ├─ import kernel_mode()
  │                                    ├─ call kernel_mode(fake_handles, ...)
  │                                    ├─ serialize launch steps
  │  ◄─────────────── steps JSON ──────┤
  ├─ execute steps via KernelSession   │
  └─ benchmark + verify               └─ exit
```

### MCP server (`tools/kbox_mcp.py`, 83 lines)

A Model Context Protocol server exposing three tools:
- `list_tasks` — enumerate available tasks
- `get_task_reference` — get public source + iterate wrapper for local dev
- `evaluate_task_kernel_mode` — submit code for hidden evaluation

Started via `kbox mcp`. Requires the optional `mcp` Python package.

### CLI dispatcher expansion (`tools/kbox.c`, +137 lines)

- Added `kbox mcp` command dispatching to `kbox_mcp.py`
- Added `kbox isolated-kernel-mode` command dispatching
- Improved venv/Python detection: checks `.venv/bin/python`, then `uv run python`,
  then `python3`
- Added `prepend_path_dir()` to ensure the exec directory is on PATH
- Added `join2()`/`join3()` string helpers for path construction

### Iterate mode extensions (`tools/kbox_iterate.py`, `python/kernelbox/iterate.py`)

- Added `--isolated-kernel-benchmark` CLI flag
- `watch_file()` and internal `_run_and_compare()` gained
  `isolated_kernel_benchmark` parameter
- When enabled, `kernel_mode()` planning runs in the isolated subprocess
  instead of in-process

### Tests

- `tests/test_kbox_mcp.py` (160 lines) — end-to-end MCP round-trip test:
  spawns the MCP server, calls all three tools, verifies list/reference/evaluate
- `tests/test_task_service.py` (63 lines) — unit test for task registry and
  module loading helpers
- `test_kbox_cli.sh` additions: `isolated kernel_mode CUDA benchmark` test,
  MCP round-trip test, task service coverage test, MCP help check,
  PATH-less dispatch test, install surface check for new helper scripts

---

## Merge-time fix (b5d033c)

### Isolated planner environment

The isolated subprocess set `HOME=tmpdir`, which prevented Python from finding
user-installed packages in `~/.local/lib/python3.10/site-packages`. This caused
`ModuleNotFoundError: No module named 'numpy'` in non-venv environments.

**Fix**: Preserve the real `HOME` from the parent environment. The subprocess is
still GPU-isolated via `CUDA_VISIBLE_DEVICES=""` etc. Also dropped the `-I`
(isolated mode) Python flag and added `PYTHONPATH` to the environment passthrough
list.

---

## Known issues / audit findings

### Correctness
- `_get_sm_arch()` (`dev.py:110-117`) ignores CUDA driver error codes
- `_reserved[3]` in `worker_config_t` documented as "must be 0" but carries
  cubin hash in RUN requests — misleading comment
- `WORKER_PROTOCOL_VERSION` is defined but never validated on the wire

### Dead code
- `import signal` unused in `dev.py:29`
- `import os` unused in `kbox_mcp.py:3`
- Redundant `import hashlib as _hashlib` in `dev.py:310`

### Documentation gaps
- README repo layout missing new modules (isolated_kernel_mode, task_service,
  task_defs, kbox_mcp, etc.)
- ITERATE.md missing `kernel_mode()` function contract
- ITERATE.md/TESTS.md missing `test_multi_kernel_triton.py` and
  `test_kernel_mode_cuda.py` from examples lists

### Minor
- `sync_stream` in worker daemon ignores `timeout_ms` (marked TODO)
- Unix sockets created in `/tmp` with no `fchmod` (mitigated by unique paths)

---

## File inventory (new or modified vs main)

| File | Lines | Status | Purpose |
|------|------:|--------|---------|
| `python/kernelbox/isolated_kernel_mode.py` | 571 | new | Sandboxed kernel_mode planning subprocess |
| `python/kernelbox/task_service.py` | 218 | new | Trusted task evaluation service |
| `python/kernelbox/tasks.py` | 99 | new | Task registry and spec loading |
| `python/kernelbox/task_defs/*.py` | 180 | new | CUDA + Triton pairwise task definitions |
| `tools/kbox_mcp.py` | 83 | new | MCP server (list/reference/evaluate) |
| `tools/kbox_isolated_kernel_mode.py` | 32 | new | Isolated planner CLI entry point |
| `tests/test_kbox_mcp.py` | 160 | new | MCP end-to-end test |
| `tests/test_task_service.py` | 63 | new | Task service unit test |
| `tools/kbox.c` | +137 | modified | MCP + isolated dispatch, Python detection |
| `tools/kbox_iterate.py` | +27 | modified | --isolated-kernel-benchmark flag |
| `python/kernelbox/iterate.py` | +44 | modified | Isolated benchmark integration |
| `python/kernelbox/dev.py` | +4 | modified | _expand_per_kernel tuple fix |
| `tests/test_kbox_cli.sh` | +27 | modified | New test entries |
| `Makefile` | +4 | modified | Build fixes |

**Total new code**: ~1,400 lines (excluding uv.lock)
