#!/usr/bin/env python3
"""KernelBox pipeline: aten graph -> per-op tests -> CUDA kernels -> patched graph.

Usage:

    # Phase 1: Generate per-op test files from an aten graph
    python tools/kbox_pipeline.py generate \\
        --graph graph.py --h5 data.h5 --output-dir output/

    # Phase 2: Add CUDA kernels to a per-op test
    python tools/kbox_pipeline.py cuda \\
        --op gelu --h5 output/data/gelu_1.h5 --output test_gelu_cuda.py

    # Phase 2: Batch — add CUDA kernels for all supported ops
    python tools/kbox_pipeline.py cuda-all \\
        --graph graph.py --test-dir output/ --output-dir output_cuda/

    # Phase 3: Inject validated kernels back into the aten graph
    python tools/kbox_pipeline.py inject \\
        --graph graph.py \\
        --kernel gelu_1=output_cuda/test_gelu_1_cuda.py \\
        --output patched_graph.py

    # List supported ops for Phase 2
    python tools/kbox_pipeline.py list-ops
"""

import argparse
import os
import sys

# Add python/ to path for imports
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))


def cmd_generate(args):
    """Phase 1: Generate per-op kernelbox test files."""
    from kernelbox.pipeline import generate_per_op_tests

    with open(args.graph) as f:
        graph_source = f.read()

    print(f"Generating per-op tests from {args.graph}")
    print(f"  h5: {args.h5}")
    print(f"  output: {args.output_dir}")

    generated = generate_per_op_tests(
        graph_source, args.h5, args.output_dir,
        skip_gemm=not args.include_gemm,
        skip_attention=not args.include_attention,
        skip_view_like=not args.include_views,
    )

    print(f"\nGenerated {len(generated)} test files")
    for op, path in generated:
        print(f"  {os.path.basename(path)}: {op.full_op_name}")


def cmd_cuda(args):
    """Phase 2: Generate CUDA kernel test for a single op."""
    from kernelbox.pipeline.graph_parser import AtenOp
    from kernelbox.pipeline.cuda_templates import generate_cuda_test

    op = AtenOp(
        output_name=args.name or args.op,
        op_name=args.op,
        op_variant="default",
    )

    shapes = {}
    if args.rows:
        shapes["rows"] = args.rows
    if args.cols:
        shapes["cols"] = args.cols
    if args.eps:
        shapes["eps"] = args.eps

    result = generate_cuda_test(op, args.h5, args.output, shapes or None)
    if result is None:
        print(f"No CUDA template available for aten.{args.op}")
        print(f"Supported ops: {', '.join(_list_ops())}")
        sys.exit(1)

    print(f"Generated CUDA kernel test: {args.output}")


def cmd_cuda_all(args):
    """Phase 2 batch: Generate CUDA kernel tests for all supported ops."""
    from kernelbox.pipeline import parse_aten_graph, generate_cuda_test
    from kernelbox.pipeline.cuda_templates import get_cuda_template

    with open(args.graph) as f:
        graph_source = f.read()

    graph = parse_aten_graph(graph_source)
    os.makedirs(args.output_dir, exist_ok=True)

    count = 0
    for op in graph.non_gemm_non_attention_ops():
        template = get_cuda_template(op.op_name)
        if template is None:
            print(f"  Skip {op.output_name}: no template for {op.op_name}")
            continue

        safe_name = op.output_name.replace("/", "_").replace(".", "_")
        h5_path = os.path.join(args.test_dir, "data", f"{safe_name}.h5")

        if not os.path.exists(h5_path):
            print(f"  Skip {op.output_name}: h5 not found at {h5_path}")
            continue

        output_path = os.path.join(
            args.output_dir, f"test_{safe_name}_cuda.py")
        result = generate_cuda_test(op, h5_path, output_path)
        if result:
            print(f"  Generated: test_{safe_name}_cuda.py ({op.op_name})")
            count += 1

    print(f"\nGenerated {count} CUDA kernel tests")


def cmd_inject(args):
    """Phase 3: Inject validated kernels back into the aten graph."""
    from kernelbox.pipeline import generate_patched_graph

    with open(args.graph) as f:
        graph_source = f.read()

    # Parse --kernel name=path pairs
    validated = {}
    for spec in args.kernel:
        if "=" not in spec:
            print(f"Error: --kernel must be name=path, got: {spec}")
            sys.exit(1)
        name, path = spec.split("=", 1)
        validated[name] = path

    print(f"Injecting {len(validated)} validated kernels into {args.graph}")
    generate_patched_graph(graph_source, validated, args.output)


def _list_ops():
    from kernelbox.pipeline.cuda_templates import list_supported_ops
    return list_supported_ops()


def cmd_list_ops(args):
    """List supported aten ops with CUDA kernel templates."""
    ops = _list_ops()
    print(f"Supported aten ops ({len(ops)}):")
    for op in ops:
        print(f"  aten.{op}")


def main():
    parser = argparse.ArgumentParser(
        description="KernelBox pipeline: aten graph -> CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # Phase 1: generate
    p_gen = sub.add_parser("generate",
                           help="Phase 1: Generate per-op test files")
    p_gen.add_argument("--graph", required=True,
                       help="Path to aten graph .py file")
    p_gen.add_argument("--h5", required=True,
                       help="Path to .h5 file with tensor inputs")
    p_gen.add_argument("--output-dir", required=True,
                       help="Output directory for generated files")
    p_gen.add_argument("--include-gemm", action="store_true",
                       help="Include GEMM ops")
    p_gen.add_argument("--include-attention", action="store_true",
                       help="Include attention ops")
    p_gen.add_argument("--include-views", action="store_true",
                       help="Include view/reshape ops")

    # Phase 2: cuda (single op)
    p_cuda = sub.add_parser("cuda",
                            help="Phase 2: Generate CUDA kernel test")
    p_cuda.add_argument("--op", required=True,
                        help="Aten op name (e.g., gelu, relu, add)")
    p_cuda.add_argument("--h5", required=True,
                        help="Path to per-op .h5 file")
    p_cuda.add_argument("--output", required=True,
                        help="Output test file path")
    p_cuda.add_argument("--name", help="Output name (default: op name)")
    p_cuda.add_argument("--rows", type=int, help="Rows (for layer_norm)")
    p_cuda.add_argument("--cols", type=int, help="Cols (for layer_norm)")
    p_cuda.add_argument("--eps", type=float, help="Epsilon (for layer_norm)")

    # Phase 2: cuda-all (batch)
    p_cuda_all = sub.add_parser("cuda-all",
                                help="Phase 2: Batch CUDA kernel tests")
    p_cuda_all.add_argument("--graph", required=True,
                            help="Path to aten graph .py file")
    p_cuda_all.add_argument("--test-dir", required=True,
                            help="Phase 1 output dir (with data/ subdir)")
    p_cuda_all.add_argument("--output-dir", required=True,
                            help="Output directory for CUDA test files")

    # Phase 3: inject
    p_inj = sub.add_parser("inject",
                           help="Phase 3: Inject validated kernels")
    p_inj.add_argument("--graph", required=True,
                       help="Path to original aten graph .py file")
    p_inj.add_argument("--kernel", action="append", default=[],
                       help="name=path pair (repeatable)")
    p_inj.add_argument("--output", required=True,
                       help="Output patched graph path")

    # list-ops
    sub.add_parser("list-ops", help="List supported aten ops")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "cuda": cmd_cuda,
        "cuda-all": cmd_cuda_all,
        "inject": cmd_inject,
        "list-ops": cmd_list_ops,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
