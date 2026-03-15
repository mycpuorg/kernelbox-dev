#!/usr/bin/env python3
"""KernelBox MCP server."""
import os
import sys
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

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:
    if exc.name and exc.name.split(".")[0] == "mcp":
        print(
            "kbox mcp requires the optional 'mcp' dependency. "
            "Install it with 'uv sync' or 'pip install \"kernelbox[mcp]\"'.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise

from kernelbox.task_service import (
    evaluate_task_kernel_mode,
    get_task_reference,
    list_task_summaries,
)


mcp = FastMCP(
    name="KernelBox",
    instructions=(
        "KernelBox exposes allowlisted kernel tasks. "
        "Use list_tasks/get_task_reference for the public fast-iteration path, "
        "and evaluate_task_kernel_mode for the hidden isolated benchmark path."
    ),
)


@mcp.tool(name="list_tasks", structured_output=True,
          description="List the trusted tasks available for iteration and evaluation.")
def list_tasks() -> dict[str, object]:
    return {"tasks": list_task_summaries()}


@mcp.tool(name="get_task_reference", structured_output=True,
          description="Get public reference code and iterate wrapper text for a trusted task.")
def get_task_reference_tool(task_name: str) -> dict[str, object]:
    return get_task_reference(task_name)


@mcp.tool(name="evaluate_task_kernel_mode", structured_output=True,
          description="Evaluate a kernel_mode() implementation on hidden data with isolated planning.")
def evaluate_task_kernel_mode_tool(task_name: str, kernel_mode_code: str | None = None,
                                   kernel_mode_path: str | None = None,
                                   wall_time: bool = False) -> dict[str, object]:
    return evaluate_task_kernel_mode(
        task_name,
        kernel_mode_code=kernel_mode_code,
        kernel_mode_path=kernel_mode_path,
        wall_time=wall_time,
    )


if __name__ == "__main__":
    mcp.run()
