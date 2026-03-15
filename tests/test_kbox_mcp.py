#!/usr/bin/env python3
"""Integration test for the KernelBox MCP task server."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import anyio

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KBOX = ROOT / "build" / "tools" / "kbox"


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


async def _exercise_mcp(kbox_path: Path):
    server = StdioServerParameters(
        command=str(kbox_path),
        args=["mcp"],
        cwd=ROOT,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tool_result = await session.list_tools()
            names = {tool.name for tool in tool_result.tools}
            _require(
                {"list_tasks", "get_task_reference", "evaluate_task_kernel_mode"} <= names,
                f"unexpected MCP tool set: {sorted(names)}",
            )

            listed = await session.call_tool("list_tasks")
            _require(not listed.isError, f"list_tasks failed: {listed}")
            tasks = listed.structuredContent["tasks"]
            by_name = {task["name"]: task for task in tasks}
            _require(
                {"cuda_pairwise", "triton_pairwise"} <= set(by_name),
                f"missing tasks in MCP listing: {sorted(by_name)}",
            )

            cuda_ref_result = await session.call_tool(
                "get_task_reference",
                {"task_name": "cuda_pairwise"},
            )
            _require(not cuda_ref_result.isError, f"get_task_reference failed: {cuda_ref_result}")
            cuda_ref = cuda_ref_result.structuredContent
            _require(
                "submission_kernel_mode" in cuda_ref["iterate_wrapper"],
                "iterate wrapper did not import submission_kernel_mode",
            )
            _require(
                "def kernel_mode" in cuda_ref["kernel_mode_template"],
                "reference kernel_mode template missing function definition",
            )
            _require(
                "--isolated-kernel-benchmark" in cuda_ref["isolated_iterate_command"],
                "isolated iterate command missing new flag",
            )

            cuda_eval = await session.call_tool(
                "evaluate_task_kernel_mode",
                {
                    "task_name": "cuda_pairwise",
                    "kernel_mode_code": cuda_ref["kernel_mode_template"],
                },
            )
            _require(not cuda_eval.isError, f"CUDA task evaluation failed: {cuda_eval}")
            cuda_eval_data = cuda_eval.structuredContent
            _require(cuda_eval_data["ok"], "CUDA hidden task correctness failed")
            _require(cuda_eval_data["suite"], "CUDA hidden task should run as a suite")
            _require(cuda_eval_data["suite_cases"] == 3, "unexpected CUDA suite size")
            _require(
                len(cuda_eval_data["benchmark"]["per_case"]) == 3,
                "CUDA per-case benchmark results missing",
            )

            triton_ref_result = await session.call_tool(
                "get_task_reference",
                {"task_name": "triton_pairwise"},
            )
            _require(not triton_ref_result.isError, f"Triton get_task_reference failed: {triton_ref_result}")
            triton_ref = triton_ref_result.structuredContent

            triton_eval = await session.call_tool(
                "evaluate_task_kernel_mode",
                {
                    "task_name": "triton_pairwise",
                    "kernel_mode_code": triton_ref["kernel_mode_template"],
                },
            )
            _require(not triton_eval.isError, f"Triton task evaluation failed: {triton_eval}")
            triton_eval_data = triton_eval.structuredContent
            _require(triton_eval_data["ok"], "Triton hidden task correctness failed")
            _require(not triton_eval_data["suite"], "Triton hidden task should be single-case")
            _require(
                triton_eval_data["benchmark"] is not None,
                "Triton benchmark result missing",
            )

            return cuda_ref


def _exercise_reference_iterate(kbox_path: Path, reference: dict):
    with tempfile.TemporaryDirectory(prefix="kbox_mcp_ref_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        wrapper_path = tmpdir_path / reference["wrapper_filename"]
        submission_path = tmpdir_path / reference["submission_filename"]
        wrapper_path.write_text(reference["iterate_wrapper"], encoding="utf-8")
        submission_path.write_text(reference["kernel_mode_template"], encoding="utf-8")

        proc = subprocess.run(
            [
                str(kbox_path),
                "iterate",
                str(wrapper_path),
                "--once",
                "--bench",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--isolated-kernel-benchmark",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 77:
            raise SystemExit(77)
        if proc.returncode != 0:
            raise RuntimeError(
                f"iterate wrapper failed (rc={proc.returncode})\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        combined = proc.stdout + proc.stderr
        _require(
            "PASS:" in combined or "Suite:" in combined,
            f"iterate wrapper output missing success marker:\n{combined}",
        )


def main():
    kbox_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_KBOX
    reference = anyio.run(_exercise_mcp, kbox_path)
    _exercise_reference_iterate(kbox_path, reference)


if __name__ == "__main__":
    main()
