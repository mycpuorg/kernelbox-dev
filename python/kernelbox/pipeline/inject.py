"""Phase 3: Inject validated CUDA kernels back into the compiled aten graph.

Takes a validated kernelbox CUDA kernel and patches it back into the full
compiled aten .py graph file, replacing the original op call(s).

The replacement uses torch.utils.cpp_extension.load_inline to compile the
CUDA kernel into a callable PyTorch extension.
"""

import re
from typing import List, Optional

from .graph_parser import parse_aten_graph, AtenOp, AtenGraph


# ── Read kernel info from validated test file ────────────────────────────

def _read_kernel_from_test(test_file_path):
    """Read a validated kernelbox test file and extract kernel info.

    Returns dict with: kernel_source, func_name, n_inputs, n_outputs.
    """
    with open(test_file_path) as f:
        source = f.read()

    # Extract KERNEL_SOURCE
    kernel_match = re.search(
        r'(?:KERNEL_SOURCE|KERNEL)\s*=\s*r"""(.*?)"""',
        source, re.DOTALL)
    if not kernel_match:
        raise ValueError(
            f"Could not find KERNEL_SOURCE in {test_file_path}")
    kernel_source = kernel_match.group(1).strip()

    # Extract function name from __global__ void funcname(
    func_match = re.search(
        r'__global__\s+void\s+(\w+)\s*\(', kernel_source)
    func_name = func_match.group(1) if func_match else "kernel"

    # Count pointer params to infer n_inputs/n_outputs
    param_match = re.search(
        r'__global__\s+void\s+\w+\s*\((.*?)\)', kernel_source, re.DOTALL)
    n_const_ptrs = 0
    n_mut_ptrs = 0
    if param_match:
        params_str = param_match.group(1)
        for param in params_str.split(","):
            param = param.strip()
            if "const" in param and "*" in param:
                n_const_ptrs += 1
            elif "*" in param and "const" not in param:
                n_mut_ptrs += 1

    return {
        "kernel_source": kernel_source,
        "func_name": func_name,
        "n_inputs": n_const_ptrs,
        "n_outputs": n_mut_ptrs,
    }


# ── C++ wrapper generation ──────────────────────────────────────────────

def _generate_cpp_wrapper(func_name, kernel_source, n_inputs, n_outputs):
    """Generate a C++ wrapper for load_inline around a raw CUDA kernel.

    Returns (cuda_source, wrapper_func_name).
    """
    wrapper_name = f"{func_name}_forward"

    # Build input args
    input_args = ", ".join(
        f"torch::Tensor in{i}" for i in range(n_inputs))
    ptr_args = ", ".join(
        f"in{i}.data_ptr<float>()" for i in range(n_inputs))
    out_ptrs = ", ".join(
        f"out{i}.data_ptr<float>()" for i in range(n_outputs))

    out_allocs = "\n".join(
        f"    auto out{i} = torch::empty_like(in0);"
        for i in range(n_outputs))
    out_returns = ", ".join(f"out{i}" for i in range(n_outputs))

    if n_outputs == 1:
        ret_type = "torch::Tensor"
        ret_stmt = f"    return out0;"
    else:
        ret_type = "std::vector<torch::Tensor>"
        ret_stmt = f"    return {{{out_returns}}};"

    cuda_source = f"""\
#include <torch/extension.h>
#include <cuda_runtime.h>

{kernel_source}

{ret_type} {wrapper_name}({input_args}) {{
{out_allocs}
    int n = in0.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    {func_name}<<<blocks, threads>>>({ptr_args}, {out_ptrs}, n);
    return {out_returns if n_outputs == 1 else '{' + out_returns + '}'};
}}
"""
    # Fix the return for single output
    if n_outputs == 1:
        cuda_source = cuda_source.replace(
            f"return {out_returns};", "return out0;")

    return cuda_source, wrapper_name


def _generate_layer_norm_cpp_wrapper(func_name, kernel_source, shapes):
    """Generate a specialized C++ wrapper for layer_norm."""
    rows = shapes.get("rows", -1)
    cols = shapes.get("cols", -1)
    eps_val = shapes.get("eps", 1e-5)

    cuda_source = f"""\
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

{kernel_source}

std::vector<torch::Tensor> {func_name}_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias
) {{
    int cols = input.size(-1);
    auto flat = input.reshape({{-1, cols}});
    int rows = flat.size(0);

    auto output = torch::empty_like(flat);
    auto mean_out = torch::empty({{rows}}, input.options());
    auto rstd_out = torch::empty({{rows}}, input.options());

    int threads = std::min(256, cols);
    int smem = 2 * threads * (int)sizeof(float);

    {func_name}<<<rows, threads, smem>>>(
        flat.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), output.data_ptr<float>(),
        mean_out.data_ptr<float>(), rstd_out.data_ptr<float>(),
        rows, cols, {eps_val}f);

    output = output.reshape(input.sizes());
    mean_out = mean_out.reshape({{rows, 1}});
    rstd_out = rstd_out.reshape({{rows, 1}});
    return {{output, mean_out, rstd_out}};
}}
"""
    return cuda_source, f"{func_name}_forward"


# ── Source patching ──────────────────────────────────────────────────────

def inject_kernel(graph_source, target_output_name, test_file_path,
                  shapes=None):
    """Replace an aten op in the graph with a validated CUDA kernel.

    Args:
        graph_source: Original Python source of the full aten graph.
        target_output_name: The out["key"] name of the op to replace.
        test_file_path: Path to the validated kernelbox test file.
        shapes: Optional shape info for ops like layer_norm.

    Returns:
        Modified Python source with the op replaced by inline CUDA.
    """
    kernel_info = _read_kernel_from_test(test_file_path)
    graph = parse_aten_graph(graph_source)

    target_op = graph.get_op(target_output_name)
    if target_op is None:
        raise ValueError(
            f"Op with output '{target_output_name}' not found in graph. "
            f"Available: {[op.output_name for op in graph.ops]}")

    func_name = kernel_info["func_name"]
    n_inputs = kernel_info["n_inputs"]
    n_outputs = kernel_info["n_outputs"]

    # Generate C++ wrapper
    if target_op.op_name == "native_layer_norm":
        cpp_src, wrapper_name = _generate_layer_norm_cpp_wrapper(
            func_name, kernel_info["kernel_source"], shapes or {})
    else:
        cpp_src, wrapper_name = _generate_cpp_wrapper(
            func_name, kernel_info["kernel_source"], n_inputs, n_outputs)

    # Build module-level code block
    ext_var = f"_{func_name}_ext"
    src_var = f"_{func_name.upper()}_CUDA_SRC"

    module_block = []
    module_block.append(f'# ── Inline CUDA: {target_op.full_op_name} ──')
    module_block.append(f'from torch.utils.cpp_extension import load_inline '
                        f'as _load_inline')
    module_block.append(f'{src_var} = r"""')
    module_block.append(cpp_src.rstrip())
    module_block.append(f'"""')
    module_block.append(
        f'{ext_var} = _load_inline("{func_name}_ext", "", '
        f'cuda_sources=[{src_var}], functions=["{wrapper_name}"])')
    module_block.append(f'')

    # Build replacement call
    call_args = []
    for arg in target_op.args:
        if arg.kind == "input":
            call_args.append(f'input["{arg.key}"]')
        elif arg.kind == "intermediate":
            call_args.append(f'out["{arg.key}"]')
        # skip literal args — they're baked into the wrapper

    args_str = ", ".join(call_args)

    if target_op.op_name == "native_layer_norm":
        # layer_norm wrapper takes (input, weight, bias)
        replacement = (
            f'    out["{target_output_name}"] = '
            f'{ext_var}.{wrapper_name}({args_str})')
    elif n_outputs == 1:
        replacement = (
            f'    out["{target_output_name}"] = '
            f'{ext_var}.{wrapper_name}({args_str})')
    else:
        replacement = (
            f'    out["{target_output_name}"] = '
            f'{ext_var}.{wrapper_name}({args_str})')

    # Patch source — match only assignment targets, not argument references
    import re
    target_pattern = re.compile(
        r'^\s*out\[(["\'])' + re.escape(target_output_name) + r'\1\]\s*=')

    source_lines = graph_source.split("\n")
    new_lines = []
    module_block_added = False

    for line in source_lines:
        # Add module block before def run(
        if line.strip().startswith("def run(") and not module_block_added:
            new_lines.extend(module_block)
            module_block_added = True

        # Replace only lines where out["target"] is the assignment target
        if target_pattern.match(line):
            new_lines.append(replacement)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def inject_multiple(graph_source, replacements, shapes_map=None):
    """Replace multiple aten ops in the graph.

    Args:
        graph_source: Original Python source.
        replacements: List of (target_output_name, test_file_path) tuples.
        shapes_map: Optional dict mapping output_name -> shapes dict.

    Returns:
        Modified Python source.
    """
    result = graph_source
    shapes_map = shapes_map or {}
    for target_name, test_path in replacements:
        shapes = shapes_map.get(target_name)
        result = inject_kernel(result, target_name, test_path, shapes)
    return result


def generate_patched_graph(graph_source, validated_kernels, output_path,
                           shapes_map=None):
    """Generate a complete patched aten graph file.

    Main Phase 3 entry point.

    Args:
        graph_source: Original Python source of the aten graph.
        validated_kernels: Dict mapping output_name -> test_file_path.
        output_path: Path to write the patched graph.
        shapes_map: Optional dict mapping output_name -> shapes dict.

    Returns:
        The patched source string.
    """
    graph = parse_aten_graph(graph_source)

    replacements = []
    skipped = []
    for op in graph.ops:
        if op.output_name in validated_kernels:
            replacements.append(
                (op.output_name, validated_kernels[op.output_name]))
        elif op.is_gemm or op.is_attention:
            skipped.append(
                f"  {op.output_name}: {op.full_op_name} (GEMM/attention)")

    if skipped:
        print(f"Skipped {len(skipped)} GEMM/attention ops:")
        for s in skipped:
            print(s)

    result = inject_multiple(
        graph_source,
        replacements,
        shapes_map=shapes_map,
    )

    with open(output_path, "w") as f:
        f.write(result)

    print(f"Wrote patched graph to {output_path}")
    print(f"  Replaced {len(replacements)} ops, "
          f"skipped {len(skipped)} GEMM/attention ops")

    return result
