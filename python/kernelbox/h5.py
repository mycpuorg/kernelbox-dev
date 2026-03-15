"""Tensor I/O — load .h5 and .pt files into TensorDicts with zero boilerplate.

Quick start::

    import kernelbox as kbox

    # Load any format — detected by extension
    d = kbox.h5.load("model.h5")    # HDF5
    d = kbox.h5.load("model.pt")    # PyTorch
    y = d.x @ d.weights + d.bias  # attribute access

    # Auto-split into input/expected for iteration
    input, expected = kbox.h5.load_test("mlp.h5")   # or mlp.pt
    input, expected = kbox.h5.load_test("mlp.pt")

    # Dump comparison results (format chosen by extension)
    kbox.h5.dump_diff("debug.h5", expected=[exp], actual=[out])
    kbox.h5.dump_diff("debug.pt", expected=[exp], actual=[out])
"""

import os
import torch

_NP_TO_TORCH = None
_load_cache = {}  # (abs_path, device, keys_tuple) -> (mtime, TensorDict)

_H5_EXTS = {".h5", ".hdf5"}
_PT_EXTS = {".pt", ".pth"}
_ALL_EXTS = _H5_EXTS | _PT_EXTS


class TensorDict(dict):
    """Dict with attribute access — math expressions read naturally.

    Behaves exactly like a normal dict, but keys are also available as
    attributes::

        d = TensorDict(x=tensor_x, w=tensor_w)
        y = d.x @ d.w          # attribute access
        y = d["x"] @ d["w"]    # dict access — same thing

    Tab-completion works in notebooks and REPLs.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __dir__(self):
        return list(self.keys()) + list(super().__dir__())

    def __repr__(self):
        lines = []
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                lines.append(f"  {k}: {v.dtype} {list(v.shape)}")
            else:
                lines.append(f"  {k}: {v!r}")
        return "TensorDict(\n" + "\n".join(lines) + "\n)"


# ── numpy dtype map (lazy, for h5) ──────────────────────────────────────

def _get_np_dtype_map():
    global _NP_TO_TORCH
    if _NP_TO_TORCH is not None:
        return _NP_TO_TORCH
    import numpy as np
    _NP_TO_TORCH = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int8"):    torch.int8,
        np.dtype("int16"):   torch.int16,
        np.dtype("int32"):   torch.int32,
        np.dtype("int64"):   torch.int64,
        np.dtype("uint8"):   torch.uint8,
        np.dtype("bool"):    torch.bool,
    }
    return _NP_TO_TORCH


_TORCH_DTYPE_NAMES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16":  torch.float16,
    "torch.float32":  torch.float32,
    "torch.float64":  torch.float64,
}


def _to_tensor(np_arr, dtype_map, device, torch_dtype_attr=None):
    pt_dtype = dtype_map.get(np_arr.dtype)
    if pt_dtype is not None:
        t = torch.from_numpy(np_arr).to(dtype=pt_dtype, device=device)
    else:
        t = torch.from_numpy(np_arr.astype("float32")).to(device=device)
    if torch_dtype_attr and torch_dtype_attr in _TORCH_DTYPE_NAMES:
        t = t.to(_TORCH_DTYPE_NAMES[torch_dtype_attr])
    return t


# ── load ─────────────────────────────────────────────────────────────────

def load(path, device="cuda", keys=None):
    """Load an ``.h5`` or ``.pt`` file into a :class:`TensorDict`.

    Format is detected by extension:

    * ``.h5`` / ``.hdf5`` — HDF5 (requires ``h5py``).  Groups are
      flattened with ``/`` separators; file-level attributes go under
      ``__attrs__``.
    * ``.pt`` / ``.pth``  — PyTorch (``torch.load``).  Expects a dict
      of tensors.  Nested dicts are flattened the same way.  Non-tensor
      values go under ``__attrs__``.

    Results are cached by ``(path, mtime, device, keys)`` — repeated
    calls with an unchanged file return instantly (same GPU tensors,
    no disk I/O).  This makes hot-reload during ``kbox iterate`` free.

    Args:
        path:    Path to data file.
        device:  Target device (default ``"cuda"``).
        keys:    Load only these dataset names (default: all).

    Returns:
        :class:`TensorDict` — use ``d["x"]`` or ``d.x`` interchangeably.

    Example::

        d = kbox.h5.load("model.h5")  # or model.pt
        y = d.x @ d.weights + d.bias
    """
    abs_path = os.path.abspath(path)
    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in _ALL_EXTS:
        raise ValueError(
            f"Unsupported format '{ext}'. Use: {', '.join(sorted(_ALL_EXTS))}")

    mtime = os.path.getmtime(abs_path)
    cache_key = (abs_path, device, tuple(sorted(keys)) if keys else None)

    cached = _load_cache.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return TensorDict(cached[1])

    if ext in _H5_EXTS:
        result = _load_h5(abs_path, device, keys)
    else:
        result = _load_pt(abs_path, device, keys)

    _load_cache[cache_key] = (mtime, result)
    return TensorDict(result)


# ── h5 backend ───────────────────────────────────────────────────────────

def _load_h5(abs_path, device, keys):
    import h5py

    result = TensorDict()
    attrs = {}
    dtype_map = _get_np_dtype_map()

    with h5py.File(abs_path, "r") as f:
        _load_h5_group(f, result, attrs, dtype_map, device, keys, "")

    if attrs:
        result["__attrs__"] = attrs
    return result


def _load_h5_group(group, result, attrs, dtype_map, device, keys, prefix):
    import h5py

    for k, v in group.items():
        full_key = k if not prefix else f"{prefix}/{k}"
        if isinstance(v, h5py.Group):
            _load_h5_group(v, result, attrs, dtype_map, device, keys,
                           full_key)
        else:
            if keys is not None and full_key not in keys:
                continue
            td = v.attrs.get("torch_dtype", None)
            result[full_key] = _to_tensor(v[:], dtype_map, device, td)

    for k, v in group.attrs.items():
        attrs[f"{prefix}/{k}" if prefix else k] = v


# ── pt backend ───────────────────────────────────────────────────────────

def _load_pt(abs_path, device, keys):
    try:
        raw = torch.load(abs_path, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(abs_path, map_location=device)

    result = TensorDict()
    attrs = {}

    if isinstance(raw, dict):
        _flatten_dict(raw, result, attrs, device, keys, "")
    elif isinstance(raw, (list, tuple)):
        for i, v in enumerate(raw):
            k = str(i)
            if keys is not None and k not in keys:
                continue
            if isinstance(v, torch.Tensor):
                result[k] = v if v.device == torch.device(device) else v.to(device)
            else:
                attrs[k] = v
    elif isinstance(raw, torch.Tensor):
        result["data"] = raw if raw.device == torch.device(device) else raw.to(device)
    else:
        raise ValueError(
            f".pt file must contain a dict, list, or tensor — "
            f"got {type(raw).__name__}")

    if attrs:
        result["__attrs__"] = attrs
    return result


def _flatten_dict(d, result, attrs, device, keys, prefix):
    for k, v in d.items():
        full_key = k if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            _flatten_dict(v, result, attrs, device, keys, full_key)
        elif isinstance(v, torch.Tensor):
            if keys is not None and full_key not in keys:
                continue
            result[full_key] = v if v.device == torch.device(device) else v.to(device)
        else:
            attrs[full_key] = v


# ── load_graph (torch-graph H5 convention) ────────────────────────────────

import re as _re
_SHORT_P = _re.compile(r"^p(\d+)$")
_SHORT_D = _re.compile(r"^d(\d+)$")


def load_graph(path, device="cuda"):
    """Load a torch-graph H5 file into a :class:`TensorDict`.

    Reads all datasets from the ``tensors/`` group, respects the
    ``torch_dtype`` attribute (e.g. bfloat16 stored as float32), and
    expands short H5 names to their FX long forms::

        p6  → primals_6
        d3  → tangents_3

    Both short and long names are accessible in the returned dict.

    Args:
        path:   Path to a torch-graph ``.h5`` file.
        device: Target device (default ``"cuda"``).

    Returns:
        :class:`TensorDict` with all stored tensors on *device*.

    Example::

        t = kbox.h5.load_graph("nanogpt.h5")
        logits = t.primals_1     # by FX name
        logits = t.p1            # by short name — same tensor
    """
    import h5py

    abs_path = os.path.abspath(path)
    mtime = os.path.getmtime(abs_path)
    cache_key = (abs_path, device, "__graph__")

    cached = _load_cache.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return TensorDict(cached[1])

    dtype_map = _get_np_dtype_map()
    result = TensorDict()

    with h5py.File(abs_path, "r") as f:
        tensors_grp = f.get("tensors", f)
        for name, ds in tensors_grp.items():
            if not hasattr(ds, "shape"):
                continue
            td = ds.attrs.get("torch_dtype", None)
            val = _to_tensor(ds[:], dtype_map, device, td)
            result[name] = val
            m = _SHORT_P.match(name)
            if m:
                result[f"primals_{m.group(1)}"] = val
                continue
            m = _SHORT_D.match(name)
            if m:
                result[f"tangents_{m.group(1)}"] = val

    _load_cache[cache_key] = (mtime, result)
    return TensorDict(result)


# ── load_test ────────────────────────────────────────────────────────────

def _parse_expected_index(key, prefix):
    """Extract numeric index from an expected key.

    Returns int index or None if the suffix is non-numeric.

    ``expected``           → 0
    ``expected_0``         → 0
    ``expected_2_blend``   → 2
    ``expected_blend``     → None  (non-numeric)
    """
    if key == prefix:
        return 0
    suffix = key[len(prefix) + 1:]
    head = suffix.split("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def load_test(path, device="cuda", expected_prefix="expected"):
    """Load a data file, auto-splitting input from expected outputs.

    Works with ``.h5``, ``.hdf5``, ``.pt``, and ``.pth`` files.

    Convention:
      - ``expected`` — single expected output.
      - ``expected_0``, ``expected_1``, ... — multiple outputs,
        ordered by the numeric index.
      - ``expected_0_blend``, ``expected_1_residual`` — same, with
        human-readable labels (number determines order, label is ignored).
      - Everything else becomes input tensors in a :class:`TensorDict`.
      - HDF5 attributes / non-tensor .pt values are promoted into
        the input namespace as scalars (``input.eps`` just works).

    Raises ``ValueError`` on:
      - No expected datasets found.
      - Non-numeric suffixes with multiple expected outputs
        (e.g. ``expected_blend`` is ambiguous without a number).
      - Duplicate indices.

    Args:
        path:             Path to data file.
        device:           Target device.
        expected_prefix:  Key prefix for expected outputs.

    Returns:
        ``(input, expected)`` where *input* is a :class:`TensorDict`
        and *expected* is a list of tensors.

    Examples::

        # single output — .h5 or .pt
        input, expected = kbox.h5.load_test("mlp.h5")
        input, expected = kbox.h5.load_test("mlp.pt")

        # multiple outputs (file has expected_0, expected_1)
        input, expected = kbox.h5.load_test("blend.pt")

        def run(input):
            hidden = torch.relu(input.x @ input.w1 + input.b1)
            return [hidden @ input.w2 + input.b2]
    """
    d = load(path, device=device)

    inputs = TensorDict()
    expected_raw = {}

    for k, v in d.items():
        if k == "__attrs__":
            continue
        if k == expected_prefix or k.startswith(expected_prefix + "_"):
            expected_raw[k] = v
        else:
            inputs[k] = v

    if not expected_raw:
        raise ValueError(
            f"No datasets matching '{expected_prefix}' or "
            f"'{expected_prefix}_*' found in {path}")

    if len(expected_raw) == 1:
        expected = list(expected_raw.values())
    else:
        indexed = {}
        bad_keys = []
        for k, v in expected_raw.items():
            idx = _parse_expected_index(k, expected_prefix)
            if idx is None:
                bad_keys.append(k)
                continue
            if idx in indexed:
                raise ValueError(
                    f"Duplicate expected output index {idx}: "
                    f"'{indexed[idx][0]}' and '{k}'")
            indexed[idx] = (k, v)

        if bad_keys:
            raise ValueError(
                f"Multiple expected outputs but these keys have no "
                f"numeric index: {bad_keys}\n"
                f"  Use: {expected_prefix}_0, {expected_prefix}_1, ... "
                f"or {expected_prefix}_0_name, {expected_prefix}_1_name, ...")

        expected = [indexed[i][1] for i in sorted(indexed)]

    if "__attrs__" in d:
        for ak, av in d["__attrs__"].items():
            inputs[ak] = av

    return inputs, expected


def load_inputs(path, input_keys, expected_keys, device="cuda"):
    """Load inputs and expected outputs by explicit key names.

    Works with any supported format (``.h5``, ``.pt``, etc.).

    Returns:
        ``(inputs, expected)`` — both plain lists of tensors.

    Example::

        inputs, expected = kbox.h5.load_inputs(
            "mlp.pt",
            input_keys=["x", "w1", "b1", "w2", "b2"],
            expected_keys=["expected"],
        )
    """
    data = load(path, device=device)
    inputs = [data[k] for k in input_keys]
    expected = [data[k] for k in expected_keys]
    return inputs, expected


# ── dump_diff ────────────────────────────────────────────────────────────

def _build_diff(expected, actual, names, minimal=False):
    """Compute diff data for each output pair. Returns list of dicts."""
    if names is None:
        names = [f"output_{i}" for i in range(len(expected))]

    groups = []
    for name, exp, act in zip(names, expected, actual):
        exp_cpu = exp.detach().float().cpu()
        act_cpu = act.detach().float().cpu()
        absdiff = (exp_cpu - act_cpu).abs()

        tensors = {
            "expected": exp_cpu,
            "actual":   act_cpu,
            "absdiff":  absdiff,
        }
        meta = {
            "max_abserr":  float(absdiff.max()),
            "mean_abserr": float(absdiff.mean()),
            "shape":       list(exp.shape),
            "dtype":       str(exp.dtype),
        }
        if not minimal:
            eps = 1e-8
            reldiff = absdiff / (exp_cpu.abs() + eps)
            tensors["reldiff"] = reldiff
            meta["max_relerr"] = float(reldiff.max())

        groups.append((name, tensors, meta))
    return groups


def dump_diff(path, expected, actual, names=None, inputs=None,
              mode=None, atol=1e-5, rtol=1e-5):
    """Write expected, actual, absdiff, and reldiff per output.

    Format is detected by extension (``.h5`` or ``.pt``).

    Args:
        path:     Output file path.
        expected: List of expected tensors.
        actual:   List of actual output tensors.
        names:    Optional output names (default ``output_0``, ...).
        inputs:   Optional inputs (list, dict, or NamedInputs) for ``max`` mode.
        mode:     ``"min"`` = only failed outputs, ``"max"`` = all outputs + inputs.
                  ``None`` = all outputs (legacy default).
        atol:     Absolute tolerance (for ``min`` filtering).
        rtol:     Relative tolerance (for ``min`` filtering).

    File structure (same in both formats)::

        output_0/
            expected    — reference tensor
            actual      — your result
            absdiff     — |expected - actual|
            reldiff     — |expected - actual| / (|expected| + eps)
            max_abserr  — scalar
            max_relerr  — scalar
            shape       — tensor shape
        output_1/
            ...

    Example::

        kbox.h5.dump_diff("debug.h5", expected=[exp], actual=[out])
        kbox.h5.dump_diff("debug.pt", expected=[exp], actual=[out])
    """
    ext = os.path.splitext(path)[1].lower()
    groups = _build_diff(expected, actual, names, minimal=(mode == "min"))

    # min mode: only keep outputs that fail the allclose check
    if mode == "min":
        filtered = []
        for name, tensors, meta in groups:
            # Match torch.allclose criterion: |a-b| <= atol + rtol * |expected|
            margin = tensors["absdiff"] - atol - rtol * tensors["expected"].abs()
            if margin.max() > 0:  # at least one element fails
                filtered.append((name, tensors, meta))
        groups = filtered

    # max mode: add stats to outputs, prepend input tensors
    if mode == "max":
        for _name, tensors, meta in groups:
            for prefix in ("expected", "actual"):
                if prefix in tensors:
                    stats = _tensor_stats(tensors[prefix])
                    for k, v in stats.items():
                        meta[f"{prefix}_{k}"] = v
        if inputs is not None:
            input_groups = _build_input_groups(inputs)
            groups = input_groups + groups

    if ext in _PT_EXTS:
        _dump_diff_pt(path, groups)
    else:
        _dump_diff_h5(path, groups)


def _tensor_stats(t_cpu):
    """Compute summary stats for a float cpu tensor."""
    return {
        "min": float(t_cpu.min()),
        "max": float(t_cpu.max()),
        "mean": float(t_cpu.mean()),
        "std": float(t_cpu.std()),
        "absmax": float(t_cpu.abs().max()),
        "nan_count": int(t_cpu.isnan().sum()),
        "inf_count": int(t_cpu.isinf().sum()),
    }


def _build_input_groups(inputs):
    """Build dump groups for input tensors (with stats)."""
    import torch
    groups = []

    def _add(name, t):
        t_cpu = t.detach().float().cpu()
        meta = {"shape": list(t.shape), "dtype": str(t.dtype)}
        meta.update(_tensor_stats(t_cpu))
        groups.append((name, {"data": t_cpu}, meta))

    if isinstance(inputs, (list, tuple)):
        for i, t in enumerate(inputs):
            if isinstance(t, torch.Tensor):
                _add(f"input_{i}", t)
    elif isinstance(inputs, dict):
        for key, t in inputs.items():
            if isinstance(t, torch.Tensor):
                _add(f"input_{key}", t)
    elif hasattr(inputs, '_fields') or hasattr(inputs, '__dict__'):
        items = inputs.__dict__.items() if hasattr(inputs, '__dict__') else {}
        for key, t in items:
            if isinstance(t, torch.Tensor):
                _add(f"input_{key}", t)
    return groups


def _dump_diff_h5(path, groups):
    import h5py

    with h5py.File(path, "w") as f:
        for name, tensors, meta in groups:
            g = f.create_group(name)
            for k, v in tensors.items():
                g.create_dataset(k, data=v.numpy())
            for k, v in meta.items():
                g.attrs[k] = v


def _dump_diff_pt(path, groups):
    data = {}
    for name, tensors, meta in groups:
        data[name] = {**tensors, **meta}
    torch.save(data, path)


# ── TestSuite / load_tests ───────────────────────────────────────────────

class TestSuite:
    """A collection of test cases loaded from a directory of data files.

    Each case is an ``(input, expected)`` pair with the same key
    structure.  Used with ``kbox iterate`` via the ``tests`` convention::

        tests = kbox.h5.load_tests("data/mlp_cases/")

        def run(input):
            ...

    Attributes:
        cases:       List of ``(name, input, expected)`` tuples.
        input_keys:  Sorted list of tensor key names (shared by all cases).
        n_expected:  Number of expected output tensors (shared by all cases).
        directory:   Absolute path to the source directory.
    """

    def __init__(self, cases, input_keys, n_expected, directory):
        self.cases = cases
        self.input_keys = input_keys
        self.n_expected = n_expected
        self.directory = directory

    def __len__(self):
        return len(self.cases)

    def __iter__(self):
        for name, inputs, expected in self.cases:
            yield inputs, expected

    def __getitem__(self, idx):
        return self.cases[idx]

    def __repr__(self):
        return (f"TestSuite({len(self.cases)} cases, "
                f"inputs={self.input_keys}, "
                f"n_expected={self.n_expected})")


def load_tests(directory, device="cuda", expected_prefix="expected"):
    """Load all data files from a directory as a :class:`TestSuite`.

    Each ``.h5`` / ``.pt`` file in *directory* is one test case,
    loaded via :func:`load_test`.  All files must share the same
    input key names and number of expected outputs — validated
    file-by-file against the first file's schema.

    Args:
        directory:        Path to directory of data files.
        device:           Target device.
        expected_prefix:  Key prefix for expected outputs.

    Returns:
        :class:`TestSuite` with all cases loaded to GPU.

    Raises:
        ``ValueError`` on empty directory, schema mismatch, or
        any :func:`load_test` error.

    Example::

        tests = kbox.h5.load_tests("data/mlp_cases/")
        # tests.cases[0] = ("case_001.h5", TensorDict(...), [tensor])
        # tests.input_keys = ["b1", "b2", "w1", "w2", "x"]
        # len(tests) = 5

        # In a test file for kbox iterate:
        tests = kbox.h5.load_tests("data/mlp_cases/")
        def run(input):
            hidden = torch.relu(input.x @ input.w1 + input.b1)
            return [hidden @ input.w2 + input.b2]
    """
    dir_path = os.path.abspath(directory)
    if not os.path.isdir(dir_path):
        raise ValueError(f"Not a directory: {directory}")

    files = sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in _ALL_EXTS
    )
    if not files:
        raise ValueError(
            f"No .h5/.hdf5/.pt/.pth files found in {directory}")

    cases = []
    ref_keys = None
    ref_n_exp = None
    ref_file = None

    for fname in files:
        fpath = os.path.join(dir_path, fname)
        inputs, expected = load_test(fpath, device=device,
                                     expected_prefix=expected_prefix)

        cur_keys = sorted(
            k for k, v in inputs.items() if isinstance(v, torch.Tensor))
        cur_n_exp = len(expected)

        if ref_keys is None:
            ref_keys = cur_keys
            ref_n_exp = cur_n_exp
            ref_file = fname
        else:
            if cur_keys != ref_keys:
                missing = set(ref_keys) - set(cur_keys)
                extra = set(cur_keys) - set(ref_keys)
                parts = [f"Input key mismatch in '{fname}' "
                         f"vs reference '{ref_file}'."]
                if missing:
                    parts.append(f"  Missing: {sorted(missing)}")
                if extra:
                    parts.append(f"  Extra:   {sorted(extra)}")
                parts.append(f"  Expected: {ref_keys}")
                raise ValueError("\n".join(parts))

            if cur_n_exp != ref_n_exp:
                raise ValueError(
                    f"Expected output count mismatch in '{fname}': "
                    f"got {cur_n_exp}, reference '{ref_file}' has "
                    f"{ref_n_exp}")

        cases.append((fname, inputs, expected))

    return TestSuite(cases, ref_keys, ref_n_exp, dir_path)
