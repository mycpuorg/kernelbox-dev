"""Tutorial 14: MCP task evaluation (programmatic).

Shows how to use the task service directly from Python, the same way
the MCP server does it. This is the flow an AI agent follows:
1. List available tasks
2. Get the public reference for a task
3. Submit a kernel_mode() implementation for evaluation

Run: kbox iterate examples/dev/test_tutorial_14_mcp_task.py --once
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

def init_once():
    """Use a simple kernel just to make this a valid iterate test."""
    x = torch.arange(8, device="cuda", dtype=torch.int32)
    return {
        "kernel": "examples/kernels/add_one.cu",
        "inputs": [x],
        "expected": [x + 1],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]

def post(outputs, state):
    """After the basic test passes, exercise the task service API."""
    from kernelbox.task_service import (
        list_task_summaries,
        get_task_reference,
        evaluate_task_kernel_mode,
    )

    # 1. List tasks
    tasks = list_task_summaries()
    print(f"\n  Available tasks: {[t['name'] for t in tasks]}")

    # 2. Get reference for cuda_pairwise
    ref = get_task_reference("cuda_pairwise")
    print(f"  Task: {ref['task']['title']}")
    print(f"  Iterate command: {ref['iterate_command']}")

    # 3. Evaluate with the template kernel_mode
    result = evaluate_task_kernel_mode(
        "cuda_pairwise",
        kernel_mode_code=ref["kernel_mode_template"],
    )
    print(f"  Evaluation OK: {result['ok']}")
    if result.get("benchmark"):
        bm = result["benchmark"]
        print(f"  Benchmark: min={bm['min_ms']:.3f}ms median={bm['median_ms']:.3f}ms")
