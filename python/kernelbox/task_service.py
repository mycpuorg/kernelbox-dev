"""Trusted task service used by the MCP server."""
from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
import tempfile

import torch

from .isolated_kernel_mode import build_isolated_kernel_mode_plan
from .iterate import (
    _benchmark_run,
    _load_iter_runtime,
    _run_and_compare,
    _run_suite,
)
from .tasks import (
    get_task,
    list_tasks,
    load_module,
    make_iterate_wrapper,
    module_source,
    task_summary,
)


def list_task_summaries():
    return [task_summary(task) for task in list_tasks()]


def get_task_reference(task_name: str):
    task = get_task(task_name)
    return {
        "task": task_summary(task),
        "wrapper_filename": "task_wrapper.py",
        "submission_filename": "submission_kernel_mode.py",
        "public_task_source": module_source(task.public_module),
        "kernel_mode_template": module_source(task.template_module),
        "iterate_wrapper": make_iterate_wrapper(task),
        "iterate_command": (
            "kbox iterate task_wrapper.py --once --bench"
        ),
        "isolated_iterate_command": (
            "kbox iterate task_wrapper.py --once --bench --isolated-kernel-benchmark"
        ),
    }


def _iter_input_tensors(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, torch.Tensor)]
    if isinstance(value, dict):
        return [v for v in value.values() if isinstance(v, torch.Tensor)]
    if hasattr(value, "__dict__"):
        return [v for v in value.__dict__.values() if isinstance(v, torch.Tensor)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [v for v in value if isinstance(v, torch.Tensor)]
    return []


def _candidate_file(kernel_mode_code=None, kernel_mode_path=None):
    if bool(kernel_mode_code) == bool(kernel_mode_path):
        raise ValueError("Specify exactly one of kernel_mode_code or kernel_mode_path")
    if kernel_mode_path is not None:
        path = Path(kernel_mode_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"kernel_mode_path does not exist: {path}")
        return str(path), None

    tmp = tempfile.NamedTemporaryFile(
        prefix="kbox_kernel_mode_", suffix=".py", delete=False)
    try:
        tmp.write(kernel_mode_code.encode("utf-8"))
        tmp.flush()
        return tmp.name, tmp.name
    finally:
        tmp.close()


def _build_isolated_runner(session, kernel_mode_path):
    plan_cache = {}

    def _plan(run_inputs):
        tensor_inputs = _iter_input_tensors(run_inputs)
        norm_inputs, out_specs, n, _, _, _, _ = \
            session._resolve_inputs_and_outputs(tensor_inputs)
        key = (
            tuple((t.data_ptr(), tuple(t.shape), t.dtype) for t in norm_inputs),
            tuple(out_specs),
            n,
        )
        cached = plan_cache.get(key)
        if cached is not None:
            return cached

        norm_inputs, out_specs, n, steps = build_isolated_kernel_mode_plan(
            session, kernel_mode_path, tensor_inputs)
        cached = (norm_inputs, steps)
        plan_cache[key] = cached
        return cached

    def run_fn(run_inputs):
        norm_inputs, steps = _plan(run_inputs)
        return session.run_steps(steps, *norm_inputs)

    def bench_fn(bench_inputs, warmup=10, iters=None, benchtime=None,
                 l2_flush=1, l2_flush_per_iter=1, l2_dirty=False,
                 wall_time=False):
        norm_inputs, steps = _plan(bench_inputs)
        return session.benchmark_steps(
            steps, *norm_inputs, warmup=warmup, iters=iters,
            benchtime=benchtime, l2_flush=l2_flush,
            l2_flush_per_iter=l2_flush_per_iter,
            l2_dirty=l2_dirty, wall_time=wall_time)

    return run_fn, bench_fn


def evaluate_task_kernel_mode(task_name: str, kernel_mode_code: str | None = None,
                              kernel_mode_path: str | None = None,
                              wall_time: bool = False):
    """Run correctness + benchmark for a trusted hidden task."""
    task = get_task(task_name)
    candidate_path, cleanup_path = _candidate_file(
        kernel_mode_code=kernel_mode_code,
        kernel_mode_path=kernel_mode_path)
    hidden_mod = load_module(task.hidden_module)
    hidden_path = Path(hidden_mod.__file__).resolve()
    try:
        rt = _load_iter_runtime(
            str(hidden_path),
            str(hidden_path.parent),
            None,
            default_atol=1e-5,
            default_rtol=1e-5,
            default_bench=True,
            default_wall_time=wall_time,
            require_entrypoint=False,
        )
        session = rt["state"].get("_session")
        if session is None:
            raise RuntimeError("task does not define a kernel-backed session")
        run_fn, bench_fn = _build_isolated_runner(session, candidate_path)

        if rt["suite"] is not None:
            ok, _ = _run_suite(run_fn, rt["suite"], rt["atol"], rt["rtol"],
                               restore_inputs=rt["restore_inputs"])
        else:
            ok, _ = _run_and_compare(
                run_fn, rt["inputs"], rt["expected"], rt["atol"], rt["rtol"],
                restore_inputs=rt["restore_inputs"])
        bench = None
        if ok:
            l2_kw = {}
            if rt["l2_flush"] is not None:
                l2_kw["l2_flush"] = rt["l2_flush"]
            if rt["l2_flush_per_iter"] is not None:
                l2_kw["l2_flush_per_iter"] = rt["l2_flush_per_iter"]
            l2_kw["l2_dirty"] = rt["l2_dirty"]

            if rt["suite"] is not None:
                if rt["bench_suite_per_case"]:
                    per_case = []
                    for name, case_inputs, _expected in rt["suite"].cases:
                        result = bench_fn(
                            case_inputs,
                            warmup=rt["warmup"],
                            iters=rt["iters"],
                            benchtime=rt["benchtime"],
                            wall_time=rt["wall_time"],
                            **l2_kw,
                        )
                        per_case.append({"name": name, "benchmark": result})
                    bench = {"per_case": per_case}
                else:
                    all_inputs = [case_inputs for _name, case_inputs, _expected
                                  in rt["suite"].cases]

                    def suite_run_fn(_unused):
                        for ci in all_inputs:
                            run_fn(ci)

                    bench = _benchmark_run(
                        suite_run_fn, None,
                        warmup=rt["warmup"],
                        iters=rt["iters"],
                        benchtime=rt["benchtime"],
                        wall_time=rt["wall_time"],
                        **l2_kw,
                    )
            else:
                bench = bench_fn(
                    rt["inputs"],
                    warmup=rt["warmup"],
                    iters=rt["iters"],
                    benchtime=rt["benchtime"],
                    wall_time=rt["wall_time"],
                    **l2_kw,
                )

        return {
            "task": task_summary(task),
            "ok": bool(ok),
            "benchmark": bench,
            "suite": rt["suite"] is not None,
            "suite_cases": len(rt["suite"]) if rt["suite"] is not None else None,
        }
    finally:
        if cleanup_path is not None:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass
