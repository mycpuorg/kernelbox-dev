"""Pipeline utilities for converting aten op graphs into kernelbox tests.

Phase 1: Parse aten graphs and generate per-op kernelbox test files
Phase 2: Replace aten ops with inline CUDA kernels
Phase 3: Inject validated kernels back into the compiled aten graph
"""

from .graph_parser import parse_aten_graph, AtenOp, AtenGraph, OpArg
from .codegen import generate_per_op_tests
from .cuda_templates import (
    generate_cuda_test,
    wrap_with_cuda_kernel,
    get_cuda_template,
    list_supported_ops,
)
from .inject import inject_kernel, inject_multiple, generate_patched_graph
