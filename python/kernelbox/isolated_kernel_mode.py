"""Isolated kernel_mode planning via a one-shot subprocess and socket IPC."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import numpy as np
import torch

from .dev import _TMADescBuilder, _WorkerPtr, _find_worker_daemon, _resolve_dtype


class IsolatedKernelModeError(RuntimeError):
    """Raised when the isolated kernel_mode planner fails."""


def tensor_metadata(tensors, out_specs=None):
    """Return lightweight metadata for input tensors and output specs."""
    input_meta = []
    for t in tensors:
        input_meta.append({
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "numel": int(t.numel()),
            "element_size": int(t.element_size()),
        })

    output_meta = []
    if out_specs is not None:
        for dt, n in out_specs:
            elem_size = int(torch.tensor([], dtype=dt).element_size())
            output_meta.append({
                "dtype": str(dt),
                "shape": [int(n)],
                "stride": [1],
                "numel": int(n),
                "element_size": elem_size,
            })
    return input_meta, output_meta


def _send_json(sock, payload):
    data = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack("<Q", len(data)))
    sock.sendall(data)


def _recv_exact(sock, nbytes):
    chunks = bytearray()
    while len(chunks) < nbytes:
        chunk = sock.recv(nbytes - len(chunks))
        if not chunk:
            raise IsolatedKernelModeError("isolated planner disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_json(sock):
    size = struct.unpack("<Q", _recv_exact(sock, 8))[0]
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


class _IsolatedWorkerPtr(int):
    """Int-like pointer proxy used in the isolated planner process."""

    def __new__(cls, value, kind="raw", index=None, offset=0):
        obj = int.__new__(cls, value)
        obj.kind = kind
        obj.index = index
        obj.offset = offset
        return obj

    def _with_offset(self, delta):
        return _IsolatedWorkerPtr(
            int(self) + delta, self.kind, self.index, self.offset + delta)

    def __add__(self, other):
        if isinstance(other, int):
            return self._with_offset(other)
        return int.__add__(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return self._with_offset(-other)
        return int.__sub__(self, other)


class _IsolatedTMADescBuilder:
    """Serializable stand-in for KernelSession.tma_desc()."""

    __slots__ = (
        "kind", "index", "dtype", "shape", "box_shape", "strides",
        "elem_strides", "swizzle", "l2_promo", "oob_fill", "interleave",
    )

    def __init__(self, kind, index, dtype, shape, box_shape, strides=None,
                 elem_strides=None, swizzle=0, l2_promo=0, oob_fill=0,
                 interleave=0):
        self.kind = kind
        self.index = index
        self.dtype = dtype
        self.shape = tuple(shape)
        self.box_shape = tuple(box_shape)
        self.strides = tuple(strides) if strides is not None else None
        self.elem_strides = (tuple(elem_strides)
                             if elem_strides is not None else None)
        self.swizzle = swizzle
        self.l2_promo = l2_promo
        self.oob_fill = oob_fill
        self.interleave = interleave


class _IsolatedKernelHandle:
    """Serializable stand-in for a kernel handle in the isolated planner."""

    __slots__ = (
        "_index", "func_name", "kind", "default_grid", "default_block",
        "default_smem", "_input_ptrs", "_output_ptrs", "_scratch_ptr",
    )

    def __init__(self, desc, input_ptrs, output_ptrs, scratch_ptr):
        self._index = desc["index"]
        self.func_name = desc.get("func_name")
        self.kind = desc.get("kind")
        self.default_grid = desc.get("default_grid")
        self.default_block = desc.get("default_block")
        self.default_smem = desc.get("default_smem")
        self._input_ptrs = input_ptrs
        self._output_ptrs = output_ptrs
        self._scratch_ptr = scratch_ptr

    def in_ptr(self, index=0):
        return self._input_ptrs[index]

    def out_ptr(self, index=0):
        return self._output_ptrs[index]

    @property
    def scratch_ptr(self):
        return self._scratch_ptr

    def tma_desc(self, index, shape, box_shape, dtype=None, strides=None,
                 elem_strides=None, kind="input", swizzle=0, l2_promo=0,
                 oob_fill=0, interleave=0):
        if dtype is None:
            dtype = torch.float32
        return _IsolatedTMADescBuilder(
            kind=kind, index=index, dtype=dtype, shape=shape,
            box_shape=box_shape, strides=strides,
            elem_strides=elem_strides, swizzle=swizzle,
            l2_promo=l2_promo, oob_fill=oob_fill,
            interleave=interleave,
        )


def _serialize_ptr(ptr):
    return {
        "type": "ptr",
        "kind": ptr.kind,
        "index": ptr.index,
        "offset": ptr.offset,
        "value": int(ptr),
    }


def _serialize_param(param):
    if isinstance(param, _IsolatedWorkerPtr):
        return _serialize_ptr(param)
    if isinstance(param, _IsolatedTMADescBuilder):
        return {
            "type": "tma_desc",
            "kind": param.kind,
            "index": param.index,
            "dtype": str(param.dtype),
            "shape": list(param.shape),
            "box_shape": list(param.box_shape),
            "strides": list(param.strides) if param.strides is not None else None,
            "elem_strides": (list(param.elem_strides)
                              if param.elem_strides is not None else None),
            "swizzle": param.swizzle,
            "l2_promo": param.l2_promo,
            "oob_fill": param.oob_fill,
            "interleave": param.interleave,
        }
    if isinstance(param, np.generic):
        return {
            "type": "np_scalar",
            "dtype": str(param.dtype),
            "value": param.item(),
        }
    if isinstance(param, tuple) and len(param) == 2:
        fmt, value = param
        return {"type": "pack", "fmt": fmt, "value": value}
    if isinstance(param, bytes):
        return {
            "type": "bytes",
            "data": base64.b64encode(param).decode("ascii"),
        }
    if isinstance(param, bool):
        return {"type": "bool", "value": param}
    raise TypeError(
        f"Unsupported isolated param type: {type(param).__name__}")


def _serialize_step(step):
    if isinstance(step, _IsolatedKernelHandle):
        step = {"kernel": step}
    if not isinstance(step, dict):
        raise TypeError("kernel_mode() must return dict steps or kernel handles")

    kernel = step.get("kernel")
    if isinstance(kernel, _IsolatedKernelHandle):
        kernel_index = kernel._index
    elif isinstance(kernel, int):
        kernel_index = kernel
    else:
        raise TypeError("step['kernel'] must be a kernel handle or index")

    payload = {
        "kernel_index": kernel_index,
        "sync": bool(step.get("sync", False)),
    }
    for key in ("n", "grid", "block", "smem", "clear_outputs"):
        if step.get(key) is not None:
            payload[key] = step[key]
    for key in ("params", "extra_params"):
        if step.get(key) is not None:
            payload[key] = [_serialize_param(p) for p in step[key]]
    return payload


def _deserialize_ptr(payload):
    return _WorkerPtr(
        payload.get("value", 0),
        kind=payload["kind"],
        index=payload.get("index"),
        offset=payload.get("offset", 0),
    )


def _deserialize_param(payload):
    kind = payload["type"]
    if kind == "ptr":
        return _deserialize_ptr(payload)
    if kind == "tma_desc":
        return _TMADescBuilder(
            kind=payload["kind"],
            index=payload["index"],
            dtype=_resolve_dtype(payload["dtype"]),
            shape=payload["shape"],
            box_shape=payload["box_shape"],
            strides=payload["strides"],
            elem_strides=payload["elem_strides"],
            swizzle=payload["swizzle"],
            l2_promo=payload["l2_promo"],
            oob_fill=payload["oob_fill"],
            interleave=payload["interleave"],
        )
    if kind == "np_scalar":
        dtype = np.dtype(payload["dtype"])
        return dtype.type(payload["value"])
    if kind == "pack":
        return (payload["fmt"], payload["value"])
    if kind == "bytes":
        return base64.b64decode(payload["data"].encode("ascii"))
    if kind == "bool":
        return bool(payload["value"])
    raise TypeError(f"Unknown isolated param type: {kind}")


def _deserialize_step(step, session):
    result = {
        "kernel": session.kernels[step["kernel_index"]],
        "sync": bool(step.get("sync", False)),
    }
    for key in ("n", "grid", "block", "smem"):
        if key in step:
            result[key] = step[key]
    if "clear_outputs" in step:
        result["clear_outputs"] = bool(step["clear_outputs"])
    for key in ("params", "extra_params"):
        if key in step:
            result[key] = [_deserialize_param(p) for p in step[key]]
    return result


def _kernel_descriptors(session):
    descs = []
    for i, entry in enumerate(session._entries):
        descs.append({
            "index": i,
            "func_name": session._entry_func_name(i),
            "kind": "triton" if entry["triton_mode"] else "cuda",
            "default_grid": entry["default_grid"],
            "default_block": entry["default_block"],
            "default_smem": entry["default_smem"],
        })
    return descs


def _planner_script_path():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "tools" / "kbox_isolated_kernel_mode.py",
        here.parents[3] / "libexec" / "kbox" / "kbox_isolated_kernel_mode.py",
        Path(_find_worker_daemon()).resolve().with_name("kbox_isolated_kernel_mode.py"),
    ]
    found = shutil.which("kbox_isolated_kernel_mode.py")
    if found:
        candidates.append(Path(found).resolve())
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("kbox_isolated_kernel_mode.py not found")


def build_isolated_kernel_mode_plan(session, kernel_mode_path, run_inputs,
                                    out_dtype=None, timeout=10.0):
    """Build a kernel_mode launch plan in a one-shot isolated subprocess."""
    norm_inputs, out_specs, n, _, _, _, _ = \
        session._resolve_inputs_and_outputs(run_inputs, out_dtype=out_dtype)
    session._ensure_persistent_layout(
        list(norm_inputs), out_specs, n,
        scratch_mib=session._user_scratch_mib)
    inputs_meta, outputs_meta = tensor_metadata(norm_inputs, out_specs)

    context = {
        "kernels": _kernel_descriptors(session),
        "scratch_ptr": _serialize_ptr(session.scratch_ptr),
        "input_ptrs": [_serialize_ptr(ptr) for ptr in session.input_ptrs],
        "output_ptrs": [_serialize_ptr(ptr) for ptr in session.output_ptrs],
        "inputs_meta": inputs_meta,
        "outputs_meta": outputs_meta,
        "n": n,
    }

    with tempfile.TemporaryDirectory(prefix="kbox_isolated_") as tmpdir:
        sock_path = os.path.join(tmpdir, "planner.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(sock_path)
        server.listen(1)
        server.settimeout(timeout)

        cmd = [
            sys.executable,
            str(_planner_script_path()),
            "--sock",
            sock_path,
            "--kernel-mode",
            os.path.abspath(kernel_mode_path),
        ]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1" if os.environ.get("VIRTUAL_ENV") else "",
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HSA_VISIBLE_DEVICES": "",
            "HOME": os.environ.get("HOME", tmpdir),
            "TMPDIR": tmpdir,
        }
        for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            env=env,
            cwd=tmpdir,
            text=True,
        )
        try:
            try:
                conn, _ = server.accept()
            except socket.timeout as exc:
                proc.kill()
                stderr = proc.communicate()[1] or ""
                detail = stderr.strip()
                if detail:
                    detail = f"\n{detail}"
                raise IsolatedKernelModeError(
                    f"isolated planner timed out before connecting{detail}"
                ) from exc

            with conn:
                conn.settimeout(timeout)
                _send_json(conn, context)
                try:
                    reply = _recv_json(conn)
                except socket.timeout as exc:
                    proc.kill()
                    stderr = proc.communicate()[1] or ""
                    detail = stderr.strip()
                    if detail:
                        detail = f"\n{detail}"
                    raise IsolatedKernelModeError(
                        f"isolated planner timed out waiting for a reply{detail}"
                    ) from exc

            try:
                stderr = proc.communicate(timeout=timeout)[1] or ""
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                stderr = proc.communicate()[1] or ""
                detail = stderr.strip()
                if detail:
                    detail = f"\n{detail}"
                raise IsolatedKernelModeError(
                    f"isolated planner did not exit cleanly{detail}"
                ) from exc
        except Exception:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
            raise
        finally:
            server.close()

    if proc.returncode != 0:
        raise IsolatedKernelModeError(
            f"isolated planner failed (rc={proc.returncode})\n{stderr.strip()}")
    if reply.get("ok") is not True:
        raise IsolatedKernelModeError(reply.get("error", "isolated planner failed"))
    steps = [_deserialize_step(step, session) for step in reply["steps"]]
    return norm_inputs, out_specs, n, steps


def _load_kernel_mode(path):
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("_kbox_isolated_kernel_mode", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "kernel_mode") or not callable(mod.kernel_mode):
        raise IsolatedKernelModeError(
            f"{path} does not define callable kernel_mode()")
    return mod.kernel_mode


def _prune_kernelbox_imports(kernel_mode_path):
    """Keep stdlib/site-packages imports available, but hide local kernelbox code."""
    kernel_mode_dir = os.path.dirname(os.path.abspath(kernel_mode_path))
    kernelbox_root = Path(__file__).resolve().parents[1]
    pruned = []
    for entry in sys.path:
        abs_entry = os.path.abspath(entry or os.getcwd())
        if abs_entry == kernel_mode_dir:
            pruned.append(entry)
            continue
        if abs_entry == str(kernelbox_root) or abs_entry.startswith(str(kernelbox_root) + os.sep):
            continue
        pruned.append(entry)
    sys.path[:] = pruned
    for name in list(sys.modules):
        if name == "kernelbox" or name.startswith("kernelbox."):
            sys.modules.pop(name, None)


def isolated_planner_main(sock_path, kernel_mode_path):
    """Entry point for the one-shot isolated planner subprocess."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(sock_path)
        request = _recv_json(sock)
        _prune_kernelbox_imports(kernel_mode_path)
        kernel_mode_fn = _load_kernel_mode(kernel_mode_path)

        scratch_ptr = _IsolatedWorkerPtr(
            request["scratch_ptr"]["value"],
            kind=request["scratch_ptr"]["kind"],
            index=request["scratch_ptr"].get("index"),
            offset=request["scratch_ptr"].get("offset", 0),
        )
        input_ptrs = [
            _IsolatedWorkerPtr(ptr["value"], kind=ptr["kind"],
                               index=ptr.get("index"),
                               offset=ptr.get("offset", 0))
            for ptr in request["input_ptrs"]
        ]
        output_ptrs = [
            _IsolatedWorkerPtr(ptr["value"], kind=ptr["kind"],
                               index=ptr.get("index"),
                               offset=ptr.get("offset", 0))
            for ptr in request["output_ptrs"]
        ]
        kernels = [
            _IsolatedKernelHandle(desc, input_ptrs, output_ptrs, scratch_ptr)
            for desc in request["kernels"]
        ]

        kwargs = {}
        sig = inspect.signature(kernel_mode_fn)
        for name, param in sig.parameters.items():
            if name == "kernel":
                kwargs[name] = kernels if len(kernels) > 1 else kernels[0]
            elif name == "kernels":
                kwargs[name] = kernels
            elif name == "scratch_ptr":
                kwargs[name] = scratch_ptr
            elif name == "input_ptrs":
                kwargs[name] = input_ptrs
            elif name == "output_ptrs":
                kwargs[name] = output_ptrs
            elif name in ("input_meta", "inputs_meta"):
                kwargs[name] = request["inputs_meta"]
            elif name in ("output_meta", "outputs_meta"):
                kwargs[name] = request["outputs_meta"]
            elif name == "n":
                kwargs[name] = request["n"]
            elif (param.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                     inspect.Parameter.VAR_KEYWORD)
                    and param.default is inspect.Parameter.empty):
                raise IsolatedKernelModeError(
                    f"Unsupported kernel_mode() parameter: {name}")

        steps = kernel_mode_fn(**kwargs)
        if not isinstance(steps, (list, tuple)):
            raise IsolatedKernelModeError(
                "kernel_mode() must return a list/tuple of step specs")

        _send_json(sock, {
            "ok": True,
            "steps": [_serialize_step(step) for step in steps],
        })
        return 0
    except Exception as exc:
        detail = f"{exc}\n{traceback.format_exc()}"
        try:
            _send_json(sock, {"ok": False, "error": detail})
        except Exception:
            pass
        print(detail, file=sys.stderr, end="")
        return 1
    finally:
        sock.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sock", required=True)
    parser.add_argument("--kernel-mode", required=True)
    args = parser.parse_args(argv)
    raise SystemExit(isolated_planner_main(args.sock, args.kernel_mode))


if __name__ == "__main__":
    main()
