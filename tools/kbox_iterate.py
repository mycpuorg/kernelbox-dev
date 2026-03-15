#!/usr/bin/env python3
"""kbox iterate — Fast iteration on a kernel test file.

Load inputs and expected outputs, then hot-reload your run()
function on every file save.

Usage:
    kbox iterate <test_file.py>

The test file defines hook functions:

    def init_once():                  # called once, inputs persist
        return {"inputs": [...], "expected": [...]}
    # OR
    def init():                       # called every iteration, fresh inputs
        return {"inputs": [...], "expected": [...]}

    def run(inputs):                  # hot-reloaded on save
        return [output_tensor, ...]
"""
import argparse
import sys
import os
from pathlib import Path


def _ensure_kernelbox_importable():
    try:
        import kernelbox  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "python",
        here.parent.parent.parent / "python",
    ]
    for candidate in candidates:
        if (candidate / "kernelbox").is_dir():
            sys.path.insert(0, str(candidate))
            return


_ensure_kernelbox_importable()


def main():
    p = argparse.ArgumentParser(
        prog="kbox iterate",
        description="Fast iteration on a kernel test file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
test file convention:
  def init_once():                  called once, inputs persist across saves
      return {"inputs": [...], "expected": [...]}
  def init():                       called every iteration, fresh inputs
      return {"inputs": [...], "expected": [...]}
  def run(inputs):                  hot-reloaded on every save
      return [output_tensor, ...]

  one of init() or init_once() is required (mutually exclusive).
  you can switch between them across reloads.

  optional hooks:
  def run(inputs, kernel):         gets KernelSession if "kernel" in state
  def post(outputs, state):         called after run+verify

examples:
  kbox iterate test_scale.py
  kbox iterate test_blend.py --atol 1e-3
  kbox iterate test_scale.py --once
  kbox iterate test_scale.py --once --bench --benchtime 2.0
""",
    )
    p.add_argument("test_file", help="Path to test .py file")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--interval", type=float, default=0.3,
                   help="File poll interval in seconds (default: 0.3)")
    dump_group = p.add_mutually_exclusive_group()
    dump_group.add_argument("--dump", metavar="PATH",
                   help="Write HDF5/PT diff on failure (all outputs)")
    dump_group.add_argument("--dump_min", metavar="PATH",
                   help="Write HDF5/PT diff on failure (only failed outputs)")
    dump_group.add_argument("--dump_max", metavar="PATH",
                   help="Write HDF5/PT diff on failure (all outputs + inputs)")
    p.add_argument("--once", action="store_true",
                   help="Run once then exit (no file watching)")
    p.add_argument("--bench", action="store_true",
                   help="Benchmark after each successful verification")
    p.add_argument("--warmup", type=int, default=10,
                   help="Benchmark warmup iterations (default: 10)")
    p.add_argument("--iters", type=int, default=None,
                   help="Benchmark iterations (default: 100, or unlimited with --benchtime)")
    p.add_argument("--benchtime", type=float, default=None,
                   help="Max benchmark time in seconds (default: no limit)")
    p.add_argument("--timeout", type=float, default=None,
                   help="Worker dispatch timeout in seconds (default: 0.75)")
    p.add_argument("--l2-flush", type=int, default=None, metavar="N",
                   help="L2 flush passes before benchmark loop (default: off for run_fn, 1 for kernel)")
    p.add_argument("--l2-flush-per-iter", type=int, default=None, metavar="N",
                   help="L2 flush passes per benchmark iteration (default: off for run_fn, 1 for kernel)")
    p.add_argument("--l2-dirty", action="store_true",
                   help="Use dirty (read+write) L2 flush instead of read-only")
    p.add_argument("--isolated-kernel-benchmark", action="store_true",
                   help="Run kernel_mode() planning in a one-shot isolated subprocess")

    args = p.parse_args()

    if not os.path.isfile(args.test_file):
        print(f"Error: file '{args.test_file}' not found.", file=sys.stderr)
        sys.exit(1)

    dump_path = args.dump or args.dump_min or args.dump_max
    dump_mode = "min" if args.dump_min else "max" if args.dump_max else None

    from kernelbox.iterate import watch_file
    watch_file(args.test_file, atol=args.atol, rtol=args.rtol,
               interval=args.interval, dump=dump_path, dump_mode=dump_mode,
               once=args.once, bench=args.bench, warmup=args.warmup,
               iters=args.iters, benchtime=args.benchtime,
               timeout=args.timeout, l2_flush=args.l2_flush,
               l2_flush_per_iter=args.l2_flush_per_iter,
               l2_dirty=args.l2_dirty,
               isolated_kernel_benchmark=args.isolated_kernel_benchmark)


if __name__ == "__main__":
    main()
