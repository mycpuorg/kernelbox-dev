# KernelBox

KernelBox is now an iterate-only CUDA kernel workflow.

The repo keeps one path:

- Python iterate/dev APIs in `python/kernelbox/dev.py`
- a private per-session C worker daemon in `tools/kbox_worker_daemon.c`
- VMM-backed input/output sharing between Python and the worker
- hot reload for test files, kernel files, and data-backed `.h5` / `.pt` cases

The old `run/test/suite/daemon/full-isolation` toolchain is intentionally gone on this branch.

## Quick Start

```bash
uv sync
make tools

kbox iterate examples/dev/test_scale.py
kbox iterate examples/dev/test_mlp_suite.py --once
kbox examples/dev/test_triton_add.py --once
```

## Supported Iterate Features

- `.cu` kernels compiled through NVRTC
- inline `kernel_source`
- precompiled `.cubin` and `.ptx`
- pure-PyTorch `run(inputs)` files with no kernel at all
- HDF5 / PyTorch data files via `h5`
- suite directories via `h5_suite`
- auto-reload on `.py`, `.cu`, `.h5`, `.pt`, and suite case file changes
- custom `params`, `extra_params`, `scratch_ptr`, and TMA descriptors
- Triton kernels via `triton_kernel`
- async timing, benchmarking, and diff dumps

## Build

`make all` builds:

- `build/tools/kbox`
- `build/tools/kbox_worker_daemon`
- low-level smoke tests: `test_cuda_check`, `test_ipc_fd`

Useful targets:

- `make tools`
- `make test`
- `make test-cli`
- `make test-all`

## Python Package

Core runtime dependencies:

- `torch`
- `cuda-python`
- `numpy`
- `h5py`

Optional:

- `mcp` for `kbox mcp`
- `triton` for Triton iterate examples and Triton sessions

## Repo Layout

```text
python/kernelbox/
  __init__.py
  data_spec.py
  dev.py        kernel engine: compilation, IPC, KernelSession, dev()
  iterate.py    iterate orchestration: watch, reload, test, benchmark
  h5.py
  vmm.py

tools/
  kbox.c
  kbox_iterate.py
  kbox_protocol.h
  kbox_worker_daemon.c

src/
  cuda_check.c/h
  ipc.c/h
  va.h
  vmm.c/h

examples/dev/
  test_*.py      iterate-mode integration examples
  tma_copy.cu

examples/kernels/
  add_one.cu
  fused_add_mul.cu
  saxpy.cu
  scale.cu
  softmax_1d.cu
  square.cu
  weighted_blend.cu

examples/data/
  *.h5 / *.pt
  mlp_cases/
```

## Docs

- [ITERATE.md](ITERATE.md)
- [TESTS.md](TESTS.md)
