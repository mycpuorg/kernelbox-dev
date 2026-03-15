# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Setup
uv sync                  # Install Python dependencies
make all                 # Build everything (tools + C smoke tests)
make tools               # Build just kbox CLI and worker daemon

# Tests
make test                # C smoke tests (test_cuda_check, test_ipc_fd)
make test-cli            # CLI + integration tests (tests/test_kbox_cli.sh)
make test-all            # Both

# Run iterate mode
kbox iterate examples/dev/test_scale.py          # Watch mode (hot-reload on save)
kbox iterate examples/dev/test_scale.py --once    # Single run
kbox examples/dev/test_scale.py --once            # Shorthand
```

Requires CUDA toolkit (`nvcc` on PATH or `CUDA_HOME` set). C code compiles with GCC against `-lcuda -lnvrtc`.

## Architecture

KernelBox is an iterate-only CUDA kernel development toolkit. The old run/test/suite/daemon toolchain has been removed.

### Two-process model

1. **Python side** (`python/kernelbox/`) — compiles kernels, manages GPU memory via VMM, orchestrates test execution and hot-reload
2. **C worker daemon** (`tools/kbox_worker_daemon.c`) — private per-session process that receives compiled cubins and launches kernels on GPU

They communicate over a Unix domain socket using a binary protocol defined in `tools/kbox_protocol.h`. The `worker_config_t` struct (104 bytes, packed as `<24IQ` in Python) is the wire format — any layout change must update both `kbox_protocol.h` and `dev.py:_WORKER_CONFIG_FMT` and bump `WORKER_PROTOCOL_VERSION`.

GPU buffers are shared zero-copy via CUDA VMM: Python exports file descriptors for memory chunks, the worker imports and maps them to device virtual addresses using SCM_RIGHTS fd passing over the socket.

### Key source files

- **`python/kernelbox/dev.py`** (~134KB) — Core engine: `KernelSession` class, NVRTC compilation pipeline, IPC protocol marshaling, cubin caching, Triton kernel support, TMA descriptors
- **`python/kernelbox/iterate.py`** (~54KB) — Watch-mode orchestration: file watching (inotify on Linux), test file reloading, benchmarking, verification, CUDA crash recovery
- **`python/kernelbox/h5.py`** — Data I/O: `TensorDict`, HDF5/PyTorch file loading, test suite management
- **`python/kernelbox/data_spec.py`** — Spec string resolution (`randn`, `zeros`, `seq`, `const:V`, `rand:A:B`, etc.)
- **`python/kernelbox/vmm.py`** — `VMMPool` class for CUDA VMM chunk management and fd export/import
- **`tools/kbox_worker_daemon.c`** (~28KB) — Worker: receives protocol messages, manages VMM pool, caches cubins (FNV-1a hash, 64 entries), launches kernels
- **`tools/kbox.c`** — CLI dispatcher: routes `kbox iterate` to `kbox_iterate.py`, finds it in libexec or source tree
- **`tools/kbox_iterate.py`** — CLI argument parsing, calls `kernelbox.iterate.watch_file()`

### Test file contract (iterate mode)

Test files define `init_once()` or `init()` returning a state dict, plus a `run(inputs)` function. The state dict specifies kernel source (`kernel`, `kernel_source`, or `triton_kernel`), `inputs`, `expected`, launch config (`grid`, `block`, `smem`), and optional features (`params`, `scratch_ptr`, `benchmark`, `atol`/`rtol`). See `ITERATE.md` for the full key reference.

### Caching layers

- Python-side: LRU cache of compiled cubins by MD5 hash (128 entries)
- Worker-side: cubin cache by FNV-1a hash (64 entries), avoids reloading on repeated runs

### Error recovery

Worker timeout (default 0.75s) force-kills and respawns the daemon. CUDA crashes (device-side asserts, illegal addresses) trigger automatic process restart. Input tensors are snapshot/restored by default (`restore_inputs=True`) to prevent data corruption across iterations.
