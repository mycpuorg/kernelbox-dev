"""Phase 1: Generate per-op kernelbox test files from aten op graphs.

Given an aten graph run() function and a corresponding .h5 file with real
tensor inputs, generates:
  - Per-op .h5 files with inputs and expected outputs
  - Per-op kernelbox .py test files ready to run with kbox iterate
"""

import os
import operator

import torch
import numpy as np

from .graph_parser import parse_aten_graph, AtenGraph, AtenOp


# ── Graph execution ──────────────────────────────────────────────────────

def _run_graph_and_capture(graph, input_tensors):
    """Execute the graph with real inputs and capture all intermediate tensors.

    Returns dict mapping output_name -> tensor for every op in the graph.
    """
    intermediates = {}

    for op in graph.ops:
        resolved_args = []
        for arg in op.args:
            if arg.kind == "input":
                resolved_args.append(input_tensors[arg.key])
            elif arg.kind == "intermediate":
                resolved_args.append(intermediates[arg.key])
            elif arg.kind == "literal":
                resolved_args.append(arg.value)

        if op.op_name == "__getitem__":
            result = operator.getitem(resolved_args[0], resolved_args[1])
        else:
            aten_fn = getattr(torch.ops.aten, op.op_name)
            variant = getattr(aten_fn, op.op_variant)
            result = variant(*resolved_args)

        intermediates[op.output_name] = result

    return intermediates


# ── H5 data saving ──────────────────────────────────────────────────────

_UNSUPPORTED_NP_DTYPES = {torch.bfloat16}


def _tensor_to_numpy(tensor):
    """Convert a torch tensor to numpy, handling unsupported dtypes."""
    t_cpu = tensor.detach().cpu()
    if t_cpu.dtype in _UNSUPPORTED_NP_DTYPES:
        return t_cpu.float().numpy()
    return t_cpu.numpy()


def _save_op_h5(path, tensor_inputs, expected, scalar_attrs=None):
    """Save per-op inputs and expected output(s) to an h5 file."""
    import h5py

    with h5py.File(path, "w") as f:
        for name, tensor in tensor_inputs.items():
            np_data = _tensor_to_numpy(tensor)
            ds = f.create_dataset(name, data=np_data)
            ds.attrs["torch_dtype"] = str(tensor.dtype)

        if len(expected) == 1:
            np_data = _tensor_to_numpy(expected[0])
            ds = f.create_dataset("expected", data=np_data)
            ds.attrs["torch_dtype"] = str(expected[0].dtype)
        else:
            for i, exp in enumerate(expected):
                np_data = _tensor_to_numpy(exp)
                ds = f.create_dataset(f"expected_{i}", data=np_data)
                ds.attrs["torch_dtype"] = str(exp.dtype)

        if scalar_attrs:
            for k, v in scalar_attrs.items():
                f.attrs[k] = v


# ── Test file generation ────────────────────────────────────────────────

def _generate_test_source(op, h5_relative_path, n_expected):
    """Generate the Python source for a per-op kernelbox test file."""
    lines = []
    lines.append(f'"""Kernelbox test for {op.full_op_name} (output: {op.output_name}).')
    lines.append(f'')
    lines.append(f'Auto-generated from aten graph. Run:')
    lines.append(f'    kbox iterate <this_file>.py --once')
    lines.append(f'"""')

    # Imports
    lines.append(f'import torch')
    if op.op_name == "__getitem__":
        lines.append(f'import operator')
    lines.append(f'')
    lines.append(f'')

    # init_once
    lines.append(f'def init_once():')
    lines.append(f'    return {{"h5": "{h5_relative_path}"}}')
    lines.append(f'')
    lines.append(f'')

    # run function
    lines.append(f'def run(inputs):')

    # Build the op call
    call_args = []
    for arg in op.args:
        if arg.kind in ("input", "intermediate"):
            call_args.append(f'inputs["{arg.key}"]')
        elif arg.kind == "literal":
            call_args.append(repr(arg.value))

    args_str = ", ".join(call_args)

    if op.op_name == "__getitem__":
        lines.append(f'    result = operator.getitem({args_str})')
    else:
        lines.append(
            f'    result = torch.ops.aten.{op.op_name}.{op.op_variant}'
            f'({args_str})')

    # Return
    if n_expected > 1:
        lines.append(f'    return list(result)')
    else:
        lines.append(f'    return [result]')

    lines.append(f'')
    return "\n".join(lines)


# ── Main entry point ────────────────────────────────────────────────────

def generate_per_op_tests(graph_source, h5_path, output_dir,
                          skip_gemm=True, skip_attention=True,
                          skip_view_like=True, skip_getitem=True):
    """Generate per-op kernelbox test files from an aten graph.

    Args:
        graph_source: Python source of the run(input) function.
        h5_path: Path to .h5 file with real tensor inputs (use load_graph format).
        output_dir: Directory to write generated files.
        skip_gemm: Skip GEMM ops (mm, bmm, addmm).
        skip_attention: Skip attention ops.
        skip_view_like: Skip view/reshape ops.
        skip_getitem: Skip operator.getitem ops.

    Returns:
        List of (op, test_file_path) tuples for generated tests.
    """
    from .. import h5 as kbox_h5

    graph = parse_aten_graph(graph_source)

    # Load real tensor inputs
    all_data = kbox_h5.load_graph(h5_path)

    input_tensors = {}
    for name in graph.input_names:
        if name in all_data:
            input_tensors[name] = all_data[name]
        else:
            raise KeyError(
                f"Input '{name}' referenced in graph but not found in "
                f"{h5_path}. Available keys: {sorted(all_data.keys())}")

    # Run graph to capture all intermediates
    intermediates = _run_graph_and_capture(graph, input_tensors)

    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    generated = []
    for op in graph.ops:
        if skip_gemm and op.is_gemm:
            continue
        if skip_attention and op.is_attention:
            continue
        if skip_view_like and op.is_view_like:
            continue
        if skip_getitem and op.is_getitem:
            continue

        # Collect tensor inputs for this op
        op_inputs = {}
        scalar_attrs = {}
        for arg in op.args:
            if arg.kind == "input":
                op_inputs[arg.key] = input_tensors[arg.key]
            elif arg.kind == "intermediate":
                op_inputs[arg.key] = intermediates[arg.key]

        # Get expected output
        result = intermediates[op.output_name]
        if isinstance(result, (tuple, list)):
            expected = [t for t in result if isinstance(t, torch.Tensor)]
        else:
            expected = [result]

        # Save h5
        safe_name = op.output_name.replace("/", "_").replace(".", "_")
        h5_name = f"{safe_name}.h5"
        h5_out_path = os.path.join(data_dir, h5_name)
        _save_op_h5(h5_out_path, op_inputs, expected, scalar_attrs or None)

        # Generate test file
        h5_relative = os.path.relpath(h5_out_path, output_dir)
        test_source = _generate_test_source(
            op,
            h5_relative_path=h5_relative,
            n_expected=len(expected),
        )

        test_name = f"test_{safe_name}.py"
        test_path = os.path.join(output_dir, test_name)
        with open(test_path, "w") as f:
            f.write(test_source)

        generated.append((op, test_path))
        print(f"  Generated: {test_name} ({op.full_op_name})")

    return generated
