"""Phase 2: CUDA kernel templates for common aten ops.

Generates kernelbox test files that replace aten ops with inline CUDA kernels,
using the isolated kernel mode pattern from test_tutorial_13.

Scope: elementwise and reduction ops only (no GEMMs, no attention).
"""

from dataclasses import dataclass
from typing import List, Optional

from .graph_parser import AtenOp


# ── CUDA kernel template dataclass ──────────────────────────────────────

@dataclass
class CudaKernelTemplate:
    """A CUDA kernel template for an aten op."""
    kernel_source: str
    func_name: str
    n_inputs: int
    n_outputs: int
    needs_shared_memory: bool = False
    block_size: int = 256


# ── Elementwise template builders ───────────────────────────────────────

def _elementwise_unary(func_name, op_expr, dtype="float"):
    source = (
        f'extern "C" __global__ void {func_name}(\n'
        f'    const {dtype} *in0, {dtype} *out0, unsigned int n\n'
        f') {{\n'
        f'    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        f'    if (i < n) {{\n'
        f'        {dtype} x = in0[i];\n'
        f'        out0[i] = {op_expr};\n'
        f'    }}\n'
        f'}}'
    )
    return CudaKernelTemplate(
        kernel_source=source, func_name=func_name,
        n_inputs=1, n_outputs=1,
    )


def _elementwise_binary(func_name, op_expr, dtype="float"):
    source = (
        f'extern "C" __global__ void {func_name}(\n'
        f'    const {dtype} *in0, const {dtype} *in1, {dtype} *out0, '
        f'unsigned int n\n'
        f') {{\n'
        f'    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        f'    if (i < n) {{\n'
        f'        {dtype} a = in0[i];\n'
        f'        {dtype} b = in1[i];\n'
        f'        out0[i] = {op_expr};\n'
        f'    }}\n'
        f'}}'
    )
    return CudaKernelTemplate(
        kernel_source=source, func_name=func_name,
        n_inputs=2, n_outputs=1,
    )


# ── Op-specific templates ───────────────────────────────────────────────

def gelu_kernel(dtype="float"):
    return _elementwise_unary(
        "gelu_kernel",
        "x * 0.5f * (1.0f + erff(x * 0.7071067811865476f))", dtype)

def relu_kernel(dtype="float"):
    return _elementwise_unary("relu_kernel", "x > 0 ? x : 0", dtype)

def silu_kernel(dtype="float"):
    return _elementwise_unary(
        "silu_kernel", "x / (1.0f + expf(-x))", dtype)

def sigmoid_kernel(dtype="float"):
    return _elementwise_unary(
        "sigmoid_kernel", "1.0f / (1.0f + expf(-x))", dtype)

def tanh_kernel(dtype="float"):
    return _elementwise_unary("tanh_kernel", "tanhf(x)", dtype)

def neg_kernel(dtype="float"):
    return _elementwise_unary("neg_kernel", "-x", dtype)

def abs_kernel(dtype="float"):
    return _elementwise_unary("abs_kernel", "fabsf(x)", dtype)

def exp_kernel(dtype="float"):
    return _elementwise_unary("exp_kernel", "expf(x)", dtype)

def log_kernel(dtype="float"):
    return _elementwise_unary("log_kernel", "logf(x)", dtype)

def rsqrt_kernel(dtype="float"):
    return _elementwise_unary("rsqrt_kernel", "rsqrtf(x)", dtype)

def add_kernel(dtype="float"):
    return _elementwise_binary("add_kernel", "a + b", dtype)

def mul_kernel(dtype="float"):
    return _elementwise_binary("mul_kernel", "a * b", dtype)

def sub_kernel(dtype="float"):
    return _elementwise_binary("sub_kernel", "a - b", dtype)

def div_kernel(dtype="float"):
    return _elementwise_binary("div_kernel", "a / b", dtype)


def native_layer_norm_kernel(dtype="float"):
    """Layer normalization kernel.

    Each block handles one row. Uses shared memory for mean/variance reduction.
    Outputs: normalized, mean, rstd (3 outputs for native_layer_norm).
    """
    source = r'''extern "C" __global__ void layer_norm_kernel(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    float *mean_out,
    float *rstd_out,
    unsigned int rows,
    unsigned int cols,
    float eps
) {
    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sq_sum = sdata + blockDim.x;

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    if (row >= rows) return;

    const float *row_in = input + row * cols;
    float *row_out = output + row * cols;

    // Compute sum and sum-of-squares
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (unsigned int j = tid; j < cols; j += block_size) {
        float val = row_in[j];
        local_sum += val;
        local_sq_sum += val * val;
    }
    s_sum[tid] = local_sum;
    s_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    // Tree reduction
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)cols;
    float var = s_sq_sum[0] / (float)cols - mean * mean;
    float rstd = rsqrtf(var + eps);

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    // Normalize
    for (unsigned int j = tid; j < cols; j += block_size) {
        float val = row_in[j];
        float normed = (val - mean) * rstd;
        row_out[j] = normed * weight[j] + bias[j];
    }
}'''
    return CudaKernelTemplate(
        kernel_source=source, func_name="layer_norm_kernel",
        n_inputs=3, n_outputs=3, needs_shared_memory=True, block_size=256,
    )


# ── Registry ────────────────────────────────────────────────────────────

_KERNEL_REGISTRY = {
    "gelu": gelu_kernel,
    "relu": relu_kernel,
    "silu": silu_kernel,
    "sigmoid": sigmoid_kernel,
    "tanh": tanh_kernel,
    "neg": neg_kernel,
    "abs": abs_kernel,
    "exp": exp_kernel,
    "log": log_kernel,
    "rsqrt": rsqrt_kernel,
    "add": add_kernel,
    "mul": mul_kernel,
    "sub": sub_kernel,
    "div": div_kernel,
    "native_layer_norm": native_layer_norm_kernel,
}


def get_cuda_template(op_name, dtype="float"):
    """Look up a CUDA kernel template for the given aten op name."""
    factory = _KERNEL_REGISTRY.get(op_name)
    if factory is None:
        return None
    return factory(dtype)


def list_supported_ops():
    """Return the list of aten ops with CUDA kernel templates."""
    return sorted(_KERNEL_REGISTRY.keys())


# ── Isolated kernel mode test file generation ───────────────────────────

def _generate_elementwise_test(op, template, h5_path):
    """Generate a kernelbox isolated-mode test for an elementwise op."""
    lines = []
    lines.append(
        f'"""Kernelbox CUDA kernel test for {op.full_op_name} '
        f'(output: {op.output_name}).')
    lines.append(f'')
    lines.append(f'Auto-generated with inline CUDA kernel. Run:')
    lines.append(f'    kbox iterate <this_file>.py --once')
    lines.append(
        f'    kbox iterate <this_file>.py --once --bench '
        f'--isolated-kernel-benchmark')
    lines.append(f'"""')
    lines.append(f'import torch')
    lines.append(f'import numpy as np')
    lines.append(f'import kernelbox as kbox')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'KERNEL_SOURCE = r"""')
    lines.append(template.kernel_source)
    lines.append(f'"""')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'def init_once():')
    lines.append(f'    inputs, expected = kbox.h5.load_test("{h5_path}")')
    lines.append(f'    input_list = [inputs[k] for k in sorted(inputs.keys())')
    lines.append(
        f'                  if isinstance(inputs[k], torch.Tensor)]')
    lines.append(f'    return {{')
    lines.append(f'        "kernel_source": KERNEL_SOURCE,')
    lines.append(f'        "inputs": input_list,')
    lines.append(f'        "expected": expected,')
    lines.append(f'        "outputs": {template.n_outputs},')
    lines.append(f'        "atol": 1e-5,')
    lines.append(f'    }}')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'def kernel_mode(kernels, input_ptrs, output_ptrs, n):')
    lines.append(f'    block = {template.block_size}')
    lines.append(f'    grid = (n + block - 1) // block')

    # Build params list: all input ptrs, all output ptrs, then n
    params = []
    for i in range(template.n_inputs):
        params.append(f'input_ptrs[{i}]')
    for i in range(template.n_outputs):
        params.append(f'output_ptrs[{i}]')
    params.append('np.uint32(n)')

    lines.append(f'    return [{{')
    lines.append(f'        "kernel": kernels[0],')
    lines.append(f'        "grid": grid,')
    lines.append(f'        "block": block,')
    lines.append(f'        "params": [{", ".join(params)}],')
    lines.append(f'    }}]')
    lines.append(f'')
    return "\n".join(lines)


def _generate_layer_norm_test(op, template, h5_path, shapes):
    """Generate a kernelbox test for native_layer_norm."""
    rows = shapes.get("rows", 32)
    cols = shapes.get("cols", 32)
    eps = shapes.get("eps", 1e-5)

    lines = []
    lines.append(
        f'"""Kernelbox CUDA kernel test for {op.full_op_name} '
        f'(output: {op.output_name}).')
    lines.append(f'')
    lines.append(f'Auto-generated with inline CUDA layer norm kernel. Run:')
    lines.append(f'    kbox iterate <this_file>.py --once')
    lines.append(
        f'    kbox iterate <this_file>.py --once --bench '
        f'--isolated-kernel-benchmark')
    lines.append(f'"""')
    lines.append(f'import torch')
    lines.append(f'import numpy as np')
    lines.append(f'import kernelbox as kbox')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'KERNEL_SOURCE = r"""')
    lines.append(template.kernel_source)
    lines.append(f'"""')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'def init_once():')
    lines.append(f'    inputs, expected = kbox.h5.load_test("{h5_path}")')
    lines.append(f'    input_list = [inputs[k] for k in sorted(inputs.keys())')
    lines.append(
        f'                  if isinstance(inputs[k], torch.Tensor)]')
    lines.append(f'    return {{')
    lines.append(f'        "kernel_source": KERNEL_SOURCE,')
    lines.append(f'        "inputs": input_list,')
    lines.append(f'        "expected": expected,')
    lines.append(f'        "outputs": 3,')
    lines.append(f'        "atol": 1e-4,')
    lines.append(f'    }}')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'def kernel_mode(kernels, input_ptrs, output_ptrs, n):')
    lines.append(f'    rows = {rows}')
    lines.append(f'    cols = {cols}')
    lines.append(f'    block = min(256, cols)')
    lines.append(f'    eps = np.float32({eps})')
    lines.append(f'    return [{{')
    lines.append(f'        "kernel": kernels[0],')
    lines.append(f'        "grid": rows,')
    lines.append(f'        "block": block,')
    lines.append(f'        "params": [')
    lines.append(f'            input_ptrs[0],   # input')
    lines.append(f'            input_ptrs[1],   # weight')
    lines.append(f'            input_ptrs[2],   # bias')
    lines.append(f'            output_ptrs[0],  # normalized output')
    lines.append(f'            output_ptrs[1],  # mean')
    lines.append(f'            output_ptrs[2],  # rstd')
    lines.append(f'            np.uint32(rows),')
    lines.append(f'            np.uint32(cols),')
    lines.append(f'            eps,')
    lines.append(f'        ],')
    lines.append(f'        "smem": 2 * block * 4,')
    lines.append(f'    }}]')
    lines.append(f'')
    return "\n".join(lines)


def generate_cuda_test(op, h5_path, output_path, shapes=None):
    """Generate a kernelbox test with inline CUDA kernel for the given op.

    Args:
        op: The AtenOp to generate a kernel for.
        h5_path: Path to the per-op h5 file (from Phase 1).
        output_path: Path to write the generated test file.
        shapes: Optional shape info for ops like layer_norm
                (keys: "rows", "cols", "eps").

    Returns:
        The generated source string, or None if no template exists.
    """
    template = get_cuda_template(op.op_name)
    if template is None:
        return None

    if op.op_name == "native_layer_norm":
        source = _generate_layer_norm_test(
            op, template, h5_path, shapes or {})
    else:
        source = _generate_elementwise_test(op, template, h5_path)

    with open(output_path, "w") as f:
        f.write(source)

    return source


def wrap_with_cuda_kernel(op_name, input_tensors, expected_outputs,
                          dtype="float"):
    """Generate a standalone kernelbox test file source for an op.

    Convenience function for Phase 2 that doesn't require a parsed AtenOp.

    Args:
        op_name: The aten op name (e.g., "gelu", "relu", "add").
        input_tensors: List of input torch.Tensors (for shape info).
        expected_outputs: List of expected output torch.Tensors.
        dtype: CUDA dtype string.

    Returns:
        Python source string for the kernelbox test file, or None.
    """
    template = get_cuda_template(op_name, dtype)
    if template is None:
        return None

    shapes = [list(t.shape) for t in input_tensors]
    n = input_tensors[0].numel()

    lines = []
    lines.append(f'"""Kernelbox CUDA kernel test for aten.{op_name}.')
    lines.append(f'')
    lines.append(f'Run: kbox iterate <this_file>.py --once')
    lines.append(f'"""')
    lines.append(f'import torch')
    lines.append(f'import numpy as np')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'KERNEL_SOURCE = r"""')
    lines.append(template.kernel_source)
    lines.append(f'"""')
    lines.append(f'')
    lines.append(f'')

    # init_once with inline tensors
    lines.append(f'def init_once():')
    for i, t in enumerate(input_tensors):
        s = list(t.shape)
        lines.append(f'    in{i} = torch.randn({s}, device="cuda")')
    lines.append(f'    expected = [torch.ops.aten.{op_name}.default('
                 + ", ".join(f"in{i}" for i in range(len(input_tensors)))
                 + ')]')
    lines.append(f'    return {{')
    lines.append(f'        "kernel_source": KERNEL_SOURCE,')
    lines.append(f'        "inputs": ['
                 + ", ".join(f"in{i}" for i in range(len(input_tensors)))
                 + '],')
    lines.append(f'        "expected": expected,')
    lines.append(f'        "outputs": {len(expected_outputs)},')
    lines.append(f'    }}')
    lines.append(f'')
    lines.append(f'')

    # kernel_mode
    lines.append(f'def kernel_mode(kernels, input_ptrs, output_ptrs, n):')
    lines.append(f'    block = {template.block_size}')
    lines.append(f'    grid = (n + block - 1) // block')
    params = []
    for i in range(template.n_inputs):
        params.append(f'input_ptrs[{i}]')
    for i in range(template.n_outputs):
        params.append(f'output_ptrs[{i}]')
    params.append('np.uint32(n)')
    lines.append(f'    return [{{')
    lines.append(f'        "kernel": kernels[0],')
    lines.append(f'        "grid": grid,')
    lines.append(f'        "block": block,')
    lines.append(f'        "params": [{", ".join(params)}],')
    lines.append(f'    }}]')
    lines.append(f'')
    return "\n".join(lines)
