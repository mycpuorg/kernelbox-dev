# `kbox iterate`

`kbox iterate` is the only CLI mode in this repo.

It loads a Python test file, keeps inputs on GPU when possible, hot-reloads code on save, and dispatches kernel launches through a private `kbox_worker_daemon` process.

## CLI

```bash
kbox iterate <test_file.py> [options]
kbox <test_file.py> [options]
```

Common options:

- `--once` run a single iteration and exit
- `--atol`, `--rtol` verification tolerances
- `--bench` run benchmark after a successful verification
- `--warmup`, `--iters`, `--benchtime` benchmark controls
- `--timeout` worker dispatch timeout in seconds
- `--dump`, `--dump_min`, `--dump_max` write diff artifacts on failure
- `--interval` file polling interval when inotify is unavailable
- `--l2-flush`, `--l2-flush-per-iter`, `--l2-dirty` benchmark cache controls

Examples:

```bash
kbox iterate examples/dev/test_scale.py
kbox iterate examples/dev/test_scale.py --once --bench
kbox iterate examples/dev/test_mlp.py --dump debug.h5
kbox iterate examples/dev/test_mlp_suite.py --once
```

## Test File Contract

A test file must define one of:

```python
def init_once():
    ...
```

or

```python
def init():
    ...
```

and a `run()` function:

```python
def run(inputs):
    ...
```

If kernel-backed execution is used, `run()` may request extra injected arguments:

```python
def run(inputs, kernel):
    ...

def run(inputs, kernel, scratch_ptr):
    ...
```

## State Dictionary Keys

`init()` / `init_once()` return a dict. Relevant keys:

- `kernel`: path to `.cu`, `.cubin`, or `.ptx`
- `kernel_source`: inline CUDA source or raw cubin/PTX bytes
- `triton_kernel`: a `@triton.jit` function
- `triton_constexprs`: constexpr overrides for Triton
- `func_name`: explicit kernel entry point name
- `inputs`: input tensors or `TensorDict`
- `expected`: expected outputs
- `outputs`: output count or explicit output specs
- `h5`: path to a single `.h5` / `.pt` test case
- `h5_suite`: path to a directory of `.h5` / `.pt` cases
- `suite`: preloaded `kbox.h5.TestSuite`
- `grid`, `block`, `smem`: launch configuration defaults
- `params`: full custom kernel parameter list
- `extra_params`: appended scalar parameter list
- `benchmark`: enable benchmark output after pass
- `restore_inputs`: restore input snapshots after each run, default `True`
- `kernel_scratch_mib`: extra scratch allocation for `scratch_ptr`
- `atol`, `rtol`: per-test tolerances

At least one of `inputs` + `expected`, `h5`, `h5_suite`, or `suite` must be supplied.

## Auto-Watched Inputs

In watch mode, KernelBox reloads when any of these change:

- the test `.py` file
- imported helper Python files loaded by the test module
- the kernel file referenced by `kernel`
- a single data file referenced by `h5`
- the suite directory referenced by `h5_suite`
- each current suite case file inside `h5_suite`

That means touching or replacing a `.pt` / `.h5` input file, or any file inside a suite directory, triggers a fresh runtime reload and rerun.

## Data Helpers

Single-case data:

```python
def init_once():
    return {"h5": "examples/data/mlp.pt"}
```

Suite data:

```python
def init_once():
    return {"h5_suite": "examples/data/mlp_cases/"}
```

Low-level loading:

```python
import kernelbox as kbox

case = kbox.h5.load("examples/data/mlp.pt")
inputs, expected = kbox.h5.load_test("examples/data/mlp.h5")
suite = kbox.h5.load_tests("examples/data/mlp_cases/")
```

## Examples

All examples use iterate mode (`test_*.py` files):

- `examples/dev/test_hello.py`
- `examples/dev/test_fused_add_mul.py`
- `examples/dev/test_weighted_blend.py`
- `examples/dev/test_softmax.py`
- `examples/dev/test_scale.py`
- `examples/dev/test_scale_inline.py`
- `examples/dev/test_mlp.py`
- `examples/dev/test_mlp_pt.py`
- `examples/dev/test_mlp_suite.py`
- `examples/dev/test_attention.py`
- `examples/dev/test_attention_dict.py`
- `examples/dev/test_blend.py`
- `examples/dev/test_conv1d.py`
- `examples/dev/test_layernorm.py`
- `examples/dev/test_custom_params.py`
- `examples/dev/test_custom_params_mixed.py`
- `examples/dev/test_extra_params.py`
- `examples/dev/test_scratch.py`
- `examples/dev/test_scratch_custom.py`
- `examples/dev/test_async_timing.py`
- `examples/dev/test_bench.py`
- `examples/dev/test_load_cubin.py`
- `examples/dev/test_load_ptx.py`
- `examples/dev/test_tma_copy.py`
- `examples/dev/test_triton_add.py`
- `examples/dev/test_triton_bare_scalars.py`
- `examples/dev/test_triton_coerce.py`
- `examples/dev/test_triton_softmax.py`

Debug/failure example:

- `examples/dev/test_mlp_debug.py` intentionally fails and is meant for `--dump*` debugging flows
