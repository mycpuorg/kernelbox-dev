"""Parse aten op graphs from Python source into structured op representations.

Handles the dict-based format produced by torch.compile's aten graph output::

    def run(input):
        out = {}
        out["view_31"] = torch.ops.aten.view.default(input["tangents_1"], [32, 64])
        out["t_9"] = torch.ops.aten.t.default(out["view_31"])
        ...
        return [out["t_12"]]
"""

import ast
import textwrap
from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass
class OpArg:
    """A single argument to an aten op."""
    kind: str       # "input", "intermediate", "literal"
    key: str = ""   # for input/intermediate: the dict key
    value: Any = None  # for literal: the Python value

    def __repr__(self):
        if self.kind == "literal":
            return f"OpArg(literal={self.value!r})"
        return f"OpArg({self.kind}={self.key!r})"


@dataclass
class AtenOp:
    """A single aten op extracted from the graph."""
    output_name: str
    op_name: str       # e.g., "view", "gelu", "mm"
    op_variant: str    # e.g., "default"
    args: List[OpArg] = field(default_factory=list)
    line_number: int = 0

    @property
    def full_op_name(self):
        return f"aten.{self.op_name}.{self.op_variant}"

    @property
    def torch_op(self):
        return f"torch.ops.aten.{self.op_name}.{self.op_variant}"

    @property
    def is_gemm(self):
        return self.op_name in ("mm", "bmm", "addmm", "matmul", "linear")

    @property
    def is_attention(self):
        return "attention" in self.op_name

    @property
    def is_view_like(self):
        return self.op_name in (
            "view", "reshape", "t", "transpose", "permute", "expand",
            "contiguous", "unsqueeze", "squeeze", "slice", "select",
            "as_strided", "clone",
        )

    @property
    def is_getitem(self):
        return self.op_name == "__getitem__"

    @property
    def tensor_input_args(self):
        """Return args that reference tensors (input or intermediate)."""
        return [a for a in self.args if a.kind in ("input", "intermediate")]

    @property
    def literal_args(self):
        """Return literal (non-tensor) arguments."""
        return [a for a in self.args if a.kind == "literal"]


@dataclass
class AtenGraph:
    """Complete parsed aten op graph."""
    ops: List[AtenOp]
    input_names: List[str]
    output_names: List[str]
    source: str

    def get_op(self, output_name: str) -> Optional[AtenOp]:
        for op in self.ops:
            if op.output_name == output_name:
                return op
        return None

    def get_op_inputs(self, op: AtenOp) -> List[str]:
        """Get the graph input names required (transitively) by this op."""
        needed = set()
        visited = set()
        self._collect_inputs(op, needed, visited)
        return sorted(needed)

    def _collect_inputs(self, op, needed, visited):
        if op.output_name in visited:
            return
        visited.add(op.output_name)
        for arg in op.args:
            if arg.kind == "input":
                needed.add(arg.key)
            elif arg.kind == "intermediate":
                dep_op = self.get_op(arg.key)
                if dep_op:
                    self._collect_inputs(dep_op, needed, visited)

    def non_gemm_non_attention_ops(self):
        """Return ops suitable for CUDA kernel replacement."""
        return [
            op for op in self.ops
            if not op.is_gemm and not op.is_attention
            and not op.is_view_like and not op.is_getitem
        ]


# ── AST visitor ──────────────────────────────────────────────────────────

class _GraphVisitor(ast.NodeVisitor):
    """AST visitor that extracts aten ops from a run() function."""

    def __init__(self):
        self.ops = []
        self.input_names = set()
        self.output_names = []

    def _parse_arg(self, node) -> OpArg:
        """Parse a function call argument into an OpArg."""
        # input["key"]
        if isinstance(node, ast.Subscript):
            val = node.value
            if isinstance(val, ast.Name):
                key = self._get_subscript_key(node)
                if val.id == "input":
                    self.input_names.add(key)
                    return OpArg(kind="input", key=key)
                if val.id == "out":
                    return OpArg(kind="intermediate", key=key)

        # Literal values
        if isinstance(node, ast.Constant):
            return OpArg(kind="literal", value=node.value)
        if isinstance(node, ast.List):
            return OpArg(kind="literal",
                         value=[self._eval_literal(e) for e in node.elts])
        if isinstance(node, ast.Tuple):
            return OpArg(kind="literal",
                         value=tuple(self._eval_literal(e) for e in node.elts))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._eval_literal(node.operand)
            if inner is not None:
                return OpArg(kind="literal", value=-inner)
        if isinstance(node, ast.Name):
            if node.id in ("True", "False", "None"):
                return OpArg(kind="literal",
                             value={"True": True, "False": False,
                                    "None": None}[node.id])

        # Fallback: try to unparse
        try:
            return OpArg(kind="literal", value=ast.unparse(node))
        except Exception:
            return OpArg(kind="literal", value=None)

    def _eval_literal(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_literal(node.operand)
        try:
            return ast.literal_eval(node)
        except Exception:
            return ast.unparse(node)

    def _get_subscript_key(self, node):
        """Extract the key from a subscript node like d["key"]."""
        sl = node.slice
        if isinstance(sl, ast.Constant):
            return str(sl.value)
        # Python 3.8 compat
        if isinstance(sl, ast.Index):
            if isinstance(sl.value, ast.Constant):
                return str(sl.value.value)
        return ast.unparse(sl)

    def _parse_op_call(self, node):
        """Extract (op_name, op_variant) from torch.ops.aten.X.Y(...).

        Returns (op_name, op_variant) or None.
        """
        if not isinstance(node, ast.Call):
            return None
        func = node.func

        # torch.ops.aten.op_name.variant
        if isinstance(func, ast.Attribute):
            parts = []
            cur = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()

            # ["torch", "ops", "aten", op_name, variant]
            if len(parts) >= 5 and parts[:3] == ["torch", "ops", "aten"]:
                return parts[3], parts[4]
            # aten.op_name (without torch.ops prefix)
            if len(parts) >= 2 and parts[0] == "aten":
                variant = parts[2] if len(parts) > 2 else "default"
                return parts[1], variant

        # operator.getitem(...)
        if isinstance(func, ast.Attribute):
            if (isinstance(func.value, ast.Name)
                    and func.value.id == "operator"
                    and func.attr == "getitem"):
                return "__getitem__", "default"

        return None

    def visit_FunctionDef(self, node):
        if node.name == "run":
            for stmt in node.body:
                self._visit_stmt(stmt)
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    self._parse_return(stmt)

    def _visit_stmt(self, stmt):
        if not isinstance(stmt, ast.Assign):
            return
        if len(stmt.targets) != 1:
            return
        target = stmt.targets[0]

        # out["key"] = ...
        if not (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "out"):
            return

        output_name = self._get_subscript_key(target)
        parsed = self._parse_op_call(stmt.value)
        if parsed is None:
            return

        op_name, op_variant = parsed
        args = []
        for arg_node in stmt.value.args:
            args.append(self._parse_arg(arg_node))
        for kw in stmt.value.keywords:
            args.append(self._parse_arg(kw.value))

        self.ops.append(AtenOp(
            output_name=output_name,
            op_name=op_name,
            op_variant=op_variant,
            args=args,
            line_number=stmt.lineno,
        ))

    def _parse_return(self, stmt):
        if isinstance(stmt.value, ast.List):
            for elt in stmt.value.elts:
                if isinstance(elt, ast.Subscript):
                    key = self._get_subscript_key(elt)
                    self.output_names.append(key)


def parse_aten_graph(source: str) -> AtenGraph:
    """Parse an aten op graph from Python source.

    Args:
        source: Python source containing a run(input) function.

    Returns:
        AtenGraph with all ops, input names, and output names.
    """
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    visitor = _GraphVisitor()
    visitor.visit(tree)
    return AtenGraph(
        ops=visitor.ops,
        input_names=sorted(visitor.input_names),
        output_names=visitor.output_names,
        source=source,
    )
