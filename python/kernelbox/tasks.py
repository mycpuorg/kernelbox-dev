"""Allowlisted task registry for the MCP benchmark service."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    description: str
    public_module: str
    hidden_module: str
    template_module: str


_TASKS = {
    "cuda_pairwise": TaskSpec(
        name="cuda_pairwise",
        title="CUDA Pairwise Outputs",
        description="Two trusted CUDA kernels write output[0]=x+1 and output[1]=x*2.",
        public_module="kernelbox.task_defs.cuda_pairwise_public",
        hidden_module="kernelbox.task_defs.cuda_pairwise_hidden",
        template_module="kernelbox.task_defs.cuda_pairwise_kernel_mode",
    ),
    "triton_pairwise": TaskSpec(
        name="triton_pairwise",
        title="Triton Pairwise Outputs",
        description="Two trusted Triton kernels write output[0]=x+1 and output[1]=x*2.",
        public_module="kernelbox.task_defs.triton_pairwise_public",
        hidden_module="kernelbox.task_defs.triton_pairwise_hidden",
        template_module="kernelbox.task_defs.triton_pairwise_kernel_mode",
    ),
}


def list_tasks():
    return [_TASKS[name] for name in sorted(_TASKS)]


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown task '{name}'. Known tasks: {known}") from exc


def _module_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise FileNotFoundError(f"Module not found: {module_name}")
    return Path(spec.origin).resolve()


def module_path(module_name: str) -> str:
    return str(_module_path(module_name))


def module_source(module_name: str) -> str:
    return _module_path(module_name).read_text(encoding="utf-8")


def task_summary(task: TaskSpec) -> dict:
    return {
        "name": task.name,
        "title": task.title,
        "description": task.description,
        "public_module": task.public_module,
        "template_module": task.template_module,
    }


def make_iterate_wrapper(task: TaskSpec, submission_module: str = "submission_kernel_mode") -> str:
    """Generate a local iterate wrapper that keeps the task trusted."""
    return f'''"""Generated KernelBox task wrapper for {task.name}."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import {task.public_module} as _task
from {submission_module} import kernel_mode

if hasattr(_task, "init_once"):
    init_once = _task.init_once
if hasattr(_task, "init"):
    init = _task.init
if hasattr(_task, "post"):
    post = _task.post
if hasattr(_task, "watch_files"):
    watch_files = _task.watch_files
'''


def load_module(module_name: str):
    return importlib.import_module(module_name)
