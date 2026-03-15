# Tests

This repo now tests only the iterate-mode stack.

## Automated

Run everything:

```bash
make test-all
```

### `make test`

Low-level C smoke tests:

- `test_cuda_check`
- `test_ipc_fd`

### `make test-cli`

`tests/test_kbox_cli.sh` covers:

- `kbox --help`
- `kbox version`
- PATH-less dispatch for `kbox iterate --help`
- shorthand dispatch: `kbox <test.py> --once`
- unknown-command failure
- every iterate integration example in `examples/dev/test_*.py`
  - except `test_mlp_debug.py`, which is intentionally failing
- watch-mode reload regressions:
  - single `.pt` data file changes trigger rerun
  - `h5_suite` case-file changes trigger rerun
  - imported helper `.py` module changes trigger rerun
  - `h5_suite` case file added triggers rerun with updated count
  - inline `kernel_source` changes trigger rerun
- MCP task-service round trip:
  - task listing
  - public reference retrieval
  - hidden isolated evaluation
- iterate-mode contract tests:
  - scratch buffer (`kernel_scratch_mib`)
  - custom kernel params (`params=[]`)
- install surface: `build/tools/kbox` and `build/tools/kbox_worker_daemon` present

## Manual

These are still useful, but not part of `make test-cli`:

- `examples/dev/test_mlp_debug.py`
  - intentionally fails to exercise `--dump`, `--dump_min`, and `--dump_max`
- ad hoc watch runs against any `examples/dev/test_*.py`

## Relevant Fixtures

Kernel fixtures kept for iterate mode:

- `examples/kernels/add_one.cu`
- `examples/kernels/fused_add_mul.cu`
- `examples/kernels/saxpy.cu`
- `examples/kernels/scale.cu`
- `examples/kernels/softmax_1d.cu`
- `examples/kernels/square.cu`
- `examples/kernels/weighted_blend.cu`
- `examples/dev/tma_copy.cu`

Data fixtures kept for iterate mode:

- `examples/data/mlp.h5`
- `examples/data/mlp.pt`
- `examples/data/mlp_cases/`
- `examples/data/blend.h5`
- `examples/data/blend.pt`
- `examples/data/conv1d.h5`
- `examples/data/conv1d.pt`
- `examples/data/normalize.h5`
- `examples/data/normalize.pt`
