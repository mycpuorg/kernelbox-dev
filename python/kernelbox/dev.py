"""Fast-iteration CUDA kernel development with PyTorch.

Compile, run, verify, and watch CUDA kernels with zero boilerplate.
Kernels are compiled via NVRTC in-process (~5-15ms), launched directly
on PyTorch CUDA tensors, and auto-recompiled on file change.

Quick start::

    import kernelbox as kbox

    # One-liner: compile, run, get output
    out = kbox.dev("add_one.cu", "seq", n=8, dtype="int32")

    # Persistent session with auto-recompile on file edit
    s = kbox.Session("add_one.cu")
    out = s(torch.arange(1024, device="cuda", dtype=torch.int32))

    # Watch mode: re-runs on every file save
    kbox.dev("add_one.cu", "seq", n=1024, ref=lambda x: (x[0]+1,), watch=True)
"""

import atexit
import collections
import hashlib
import inspect
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid

import torch


_DTYPE_MAP = {
    "float32": torch.float32, "float": torch.float32, "f32": torch.float32,
    "float64": torch.float64, "double": torch.float64, "f64": torch.float64,
    "float16": torch.float16, "half": torch.float16, "f16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "int8": torch.int8, "i8": torch.int8,
    "int16": torch.int16, "i16": torch.int16,
    "int32": torch.int32, "int": torch.int32, "i32": torch.int32,
    "int64": torch.int64, "long": torch.int64, "i64": torch.int64,
    "uint8": torch.uint8, "u8": torch.uint8,
    "uint32": torch.int32,
    "bool": torch.bool,
}

# ── Compilation ─────────────────────────────────────────────────────────

_compile_cache = collections.OrderedDict()
_COMPILE_CACHE_MAX = 128


def _detect_func_name(source):
    m = re.search(r'extern\s+"C"\s+__global__\s+void\s+(\w+)', source)
    if m:
        return m.group(1)
    m = re.search(r'__global__\s+void\s+(\w+)', source)
    if m:
        return m.group(1)
    return None


def _cubin_entry_names(data):
    """Extract kernel entry point names from a cubin (ELF) binary.

    Parses section headers for ``.text.<name>`` sections — each is a kernel.
    Returns list of kernel name strings (empty if parse fails).
    """
    if len(data) < 64 or data[:4] != b'\x7fELF':
        return []
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shentsize = struct.unpack_from('<H', data, 58)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]
    if e_shstrndx >= e_shnum or e_shoff == 0:
        return []
    # Read shstrtab section header to locate the string table
    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    if shstr_hdr + 40 > len(data):
        return []
    shstr_offset = struct.unpack_from('<Q', data, shstr_hdr + 24)[0]
    shstr_size = struct.unpack_from('<Q', data, shstr_hdr + 32)[0]
    if shstr_offset + shstr_size > len(data):
        return []
    strtab = data[shstr_offset:shstr_offset + shstr_size]
    names = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if off + 4 > len(data):
            break
        sh_name_idx = struct.unpack_from('<I', data, off)[0]
        if sh_name_idx >= len(strtab):
            continue
        nul = strtab.find(b'\x00', sh_name_idx)
        if nul < 0:
            continue
        name = strtab[sh_name_idx:nul].decode('ascii', errors='replace')
        if name.startswith('.text.'):
            names.append(name[6:])
    return names


def _get_sm_arch():
    from cuda.bindings import driver as cu
    err, dev = cu.cuCtxGetDevice()
    err, major = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    err, minor = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    return f"sm_{major}{minor}"


def _detect_cuda_include():
    """Find the CUDA toolkit include directory for NVRTC."""
    import glob as _glob
    for pattern in ["/usr/local/cuda/include",
                    "/usr/local/cuda-*/include"]:
        for path in sorted(_glob.glob(pattern), reverse=True):
            if os.path.isfile(os.path.join(path, "cuda_fp16.h")):
                return f"-I{path}"
    for env in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(env)
        if val and os.path.isfile(os.path.join(val, "include", "cuda_fp16.h")):
            return f"-I{val}/include"
    return None


def _make_define_header(defines):
    """Build a #define preamble from a dict of {name: value}.

    Float/double values are emitted with full precision to preserve
    bit-exact round-trip through the preprocessor.
    """
    if not defines:
        return ""
    lines = []
    for k, v in defines.items():
        lines.append(f"#define {k} {_format_define_value(v)}")
    return "\n".join(lines) + "\n"


def _format_define_value(v):
    """Format a Python value for a C #define, preserving float precision.

    Floats are emitted as C99 hex float literals (``0x1.4p+1f``) so the
    preprocessor preserves the exact bit pattern.  Values that don't
    round-trip through float32 are emitted as double hex literals
    (no ``f`` suffix).
    """
    import math
    import struct as _struct

    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if math.isinf(v):
            return "INFINITY" if v > 0 else "(-INFINITY)"
        if math.isnan(v):
            return "NAN"
        # Check if value round-trips through float32
        f32_bytes = _struct.pack("f", v)
        f32_val = _struct.unpack("f", f32_bytes)[0]
        if f32_val == v:
            return float.hex(v) + "f"
        return float.hex(v)
    if isinstance(v, int):
        if v > 0x7FFFFFFF or v < -0x80000000:
            return f"{v}LL"
        return str(v)
    # Strings, type names, etc. — pass through as-is.
    return str(v)


def _cache_put(key, value):
    """Insert into compile cache with LRU eviction."""
    _compile_cache[key] = value
    if len(_compile_cache) > _COMPILE_CACHE_MAX:
        _compile_cache.popitem(last=False)


def _nvrtc_compile(source, func_name, defines=None):
    """Compile CUDA source via NVRTC. Returns (code_bytes, is_cubin_bool)."""
    from cuda.bindings import nvrtc
    source = _make_define_header(defines) + source
    src = source.encode("utf-8")

    err, prog = nvrtc.nvrtcCreateProgram(src, b"kernel.cu", 0, [], [])
    opts = [f"--gpu-architecture={_get_sm_arch()}".encode()]
    cuda_inc = _detect_cuda_include()
    if cuda_inc:
        opts.append(cuda_inc.encode())

    (rc,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if rc != 0:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b"\0" * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(f"Compile error in {func_name}:\n{log.decode()}")

    # Prefer cubin (native binary), fall back to PTX
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    if err == 0 and cubin_size > 0:
        cubin = b"\0" * cubin_size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        nvrtc.nvrtcDestroyProgram(prog)
        return bytes(cubin), True

    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b"\0" * ptx_size
    nvrtc.nvrtcGetPTX(prog, ptx)
    nvrtc.nvrtcDestroyProgram(prog)
    return bytes(ptx), False



def _compile_to_cubin(source, func_name, defines=None):
    """Compile CUDA source via NVRTC. Returns (cubin_bytes, is_cubin_bool)."""
    h = hashlib.md5((_make_define_header(defines) + source).encode()).hexdigest()
    key = (h, func_name, "cubin")
    if key in _compile_cache:
        _compile_cache.move_to_end(key)
        return _compile_cache[key]

    result = _nvrtc_compile(source, func_name, defines)
    _cache_put(key, result)
    return result


# ── Worker daemon helpers ──────────────────────────────────────────────

def _find_worker_daemon():
    """Find the kbox_worker_daemon binary."""
    candidates = [
        'build/tools/kbox_worker_daemon',
        'tools/kbox_worker_daemon',
        os.path.join(os.path.dirname(__file__), '../../build/tools/kbox_worker_daemon'),
        os.path.join(os.path.dirname(__file__), '../../tools/kbox_worker_daemon'),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    found = shutil.which('kbox_worker_daemon')
    if found:
        return found
    raise FileNotFoundError(
        "kbox_worker_daemon not found. Build it with: make tools/kbox_worker_daemon"
    )


# PyTorch dtype → protocol dtype code (from kbox_protocol.h)
_DTYPE_CODE = {
    torch.int32: 0,     # DTYPE_UINT32
    torch.float32: 1,   # DTYPE_FLOAT32
    torch.float16: 2,   # DTYPE_FLOAT16
    torch.float64: 3,   # DTYPE_FLOAT64
    torch.bfloat16: 4,  # DTYPE_BFLOAT16
    torch.int8: 5,      # DTYPE_INT8
    torch.uint8: 6,     # DTYPE_UINT8
    torch.int16: 7,     # DTYPE_INT16
    torch.int64: 8,     # DTYPE_INT64
    torch.bool: 9,      # DTYPE_BOOL
}

# Protocol dtype code → element size in bytes
_DTYPE_ELEM_SIZE = {0: 4, 1: 4, 2: 2, 3: 8, 4: 2, 5: 1, 6: 1, 7: 2, 8: 8, 9: 1}

# worker_config_t struct format — must match kbox_protocol.h exactly.
# Any layout change must update both sides and bump WORKER_PROTOCOL_VERSION.
#   0: dtype           uint32  (DTYPE_UINT32=0, FLOAT32=1, FLOAT16=2, FLOAT64=3)
#   1: n               uint32  element count (0xDEAD = shutdown sentinel)
#   2: is_cubin        uint32  1=cubin, 0=PTX
#   3: func_name_len   uint32  bytes of function name following this struct
#   4: kernel_data_len uint32  bytes of kernel binary following func name
#   5: num_inputs      uint32
#   6: num_outputs     uint32
#   7: timeout_ms      uint32  0=default
#   8-10: grid[3]      uint32  (0,0,0 = let worker auto-compute)
#  11-13: block[3]     uint32  (0,0,0 = let worker use defaults)
#  14: smem_bytes      uint32
#  15-17: _reserved[3] uint32  (must be 0)
#  18: request_type    uint32  WORKER_REQ_SETUP/RUN/RELEASE/...
#  19: flags           uint32  WORKER_FLAG_* bitmask
#  20: param_buffer_len uint32 if >0, param buffer follows func_name+cubin in RUN
#  21: scratch_mib     uint32  scratch buffer size in MiB (0 = L2 flush only)
#  22: scratch_zero_offset uint32 byte offset from scratch_ptr to start zeroing
#  23: scratch_zero_bytes  uint32 number of bytes to zero before launch (0 = skip)
#  24: chunk_size      size_t  (uint64) VMM chunk size in bytes
_WORKER_CONFIG_FMT = '<24IQ'
assert struct.calcsize(_WORKER_CONFIG_FMT) == 104, \
    "worker_config_t layout mismatch — update kbox_protocol.h"
_WORKER_SHUTDOWN_N = 0xDEAD
_WORKER_REQ_SETUP = 1
_WORKER_REQ_RUN = 2
_WORKER_REQ_RELEASE = 3
_WORKER_REQ_SYNC = 4
_WORKER_REQ_START_TIMING = 5
_WORKER_REQ_END_TIMING = 6
_WORKER_REQ_L2_FLUSH = 7
_WORKER_REQ_NOOP = 8

import hashlib as _hashlib

def _fnv1a(data):
    """FNV-1a 64-bit hash, matching the worker's C implementation."""
    # Use a fast C-implemented hash and truncate to 64 bits.
    # We only need consistency within a single session (Python ↔ worker),
    # so we compute FNV-1a on the worker side and use a fast hash here
    # that we send to the worker for lookup.
    # Actually — we need the SAME hash. Use struct to do FNV-1a via memoryview.
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

# Sender-side LRU cache: tracks cubin hashes the worker has seen.
# Must be smaller than worker's CACHE_SIZE (64) to guarantee the worker
# still has the cubin cached when we skip sending it.
_SENDER_CACHE_SIZE = 32

class _SenderCache:
    """LRU cache of cubin hashes known to be in the worker cache."""
    __slots__ = ('_order',)
    def __init__(self):
        self._order = []  # most recently used at the end

    def contains(self, h):
        try:
            self._order.remove(h)
            self._order.append(h)
            return True
        except ValueError:
            return False

    def insert(self, h):
        try:
            self._order.remove(h)
        except ValueError:
            pass
        self._order.append(h)
        if len(self._order) > _SENDER_CACHE_SIZE:
            self._order.pop(0)

    def clear(self):
        self._order.clear()

_WORKER_FLAG_NO_MEMSET = 1 << 0
_WORKER_FLAG_NO_EVENTS = 1 << 1
_WORKER_FLAG_NO_HEALTH = 1 << 2
_WORKER_FLAG_KERNEL_PARAMS = 1 << 3
_WORKER_FLAG_NO_SYNC = 1 << 4
_WORKER_FLAG_SYNC = 1 << 5
_WORKER_FLAG_PASS_SCRATCH = 1 << 6

_L2_FLUSH_CLEAN = 1 << 8
_L2_FLUSH_CLEAN_ONLY = 1 << 9


def _ipc_send_all(sock, data):
    """Send all bytes over a socket."""
    mv = memoryview(data)
    sent = 0
    while sent < len(mv):
        n = sock.send(mv[sent:])
        if n == 0:
            raise ConnectionError("Socket closed during send")
        sent += n


def _ipc_recv_all(sock, nbytes):
    """Receive exactly nbytes from a socket."""
    buf = bytearray(nbytes)
    mv = memoryview(buf)
    received = 0
    while received < nbytes:
        n = sock.recv_into(mv[received:])
        if n == 0:
            raise ConnectionError("Socket closed during recv")
        received += n
    return bytes(buf)


def _ipc_send_fd(sock, fd):
    """Send a file descriptor over SCM_RIGHTS (matches C ipc_send_fd)."""
    sock.sendmsg(
        [b'F'],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack('i', fd))],
    )


# ── Kernel launch ───────────────────────────────────────────────────────


# ── L2 cache flush ─────────────────────────────────────────────────────

_L2_FLUSH_SIZE = 256 * 1024 * 1024  # 256 MiB
_l2_flush_buf = None


def _get_l2_flush_buf():
    global _l2_flush_buf
    if _l2_flush_buf is None:
        _l2_flush_buf = torch.empty(_L2_FLUSH_SIZE // 4, dtype=torch.int32,
                                    device="cuda")
    return _l2_flush_buf


def _l2_flush(count=1, dirty=False):
    """Flush L2 cache by touching a large buffer (async, no sync).

    Args:
        count:  Number of flush passes.
        dirty:  If True, read+write (keepdim=True forces write-back).
                If False, read-only (keepdim=False, scalar output).
    """
    buf = _get_l2_flush_buf()
    for _ in range(count):
        if dirty:
            torch.max(buf, dim=0, keepdim=True)
        else:
            torch.max(buf)


# ── Terminal output ─────────────────────────────────────────────────────

_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _log(msg, style=""):
    sys.stderr.write(f"{style}[kbox] {msg}{_RESET}\n")
    sys.stderr.flush()


def _preview(tensor, limit=8):
    flat = tensor.detach().cpu().flatten()
    n = flat.numel()
    if n <= limit:
        vals = flat.tolist()
    else:
        vals = flat[:limit].tolist()
    fmt = ".4f" if tensor.is_floating_point() else ""
    parts = [f"{v:{fmt}}" for v in vals]
    s = "[" + ", ".join(parts)
    if n > limit:
        s += ", ..."
    s += "]"
    dt = str(tensor.dtype).replace("torch.", "")
    shape = list(tensor.shape)
    if len(shape) > 1:
        s += f" {dt} {shape}"
    else:
        s += f" {dt} ({n} elements)"
    return s


# ── Spec parsing ────────────────────────────────────────────────────────

def _parse_n_value(s):
    """Parse an element count like '1024', '4k', '1M'."""
    s = s.strip()
    if s.endswith("k") or s.endswith("K"):
        return int(s[:-1]) * 1024
    if s.endswith("m") or s.endswith("M"):
        return int(s[:-1]) * 1024 * 1024
    return int(s)


def _parse_spec_string(spec_str, default_dtype, default_n):
    """Parse 'randn;dtype=float16;n=2048' into (spec, dtype, n).

    Per-buffer overrides use the same semicolon syntax as ``kbox run``::

        "randn"                     -> ("randn", default_dtype, default_n)
        "randn;dtype=float16"       -> ("randn", float16,       default_n)
        "rand:0:1;dtype=float;n=4k" -> ("rand:0:1", float32,   4096)
    """
    parts = spec_str.split(";")
    spec = parts[0]
    buf_dtype = default_dtype
    buf_n = default_n
    for part in parts[1:]:
        part = part.strip()
        if part.startswith("dtype="):
            buf_dtype = _resolve_dtype(part[6:])
        elif part.startswith("n="):
            buf_n = _parse_n_value(part[2:])
    return spec, buf_dtype, buf_n


def _resolve_inputs(inputs, n, dtype):
    """Turn a mix of tensors, spec strings, and lists into CUDA tensors.

    Spec strings support per-buffer overrides::

        "randn"                  uses global dtype and n
        "randn;dtype=float16"    overrides dtype for this buffer
        "rand;n=2048"            overrides element count for this buffer
        "seq;dtype=int32;n=4k"   overrides both
    """
    from .data_spec import from_spec
    resolved = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            if not inp.is_cuda:
                inp = inp.cuda()
            resolved.append(inp)
        elif isinstance(inp, str):
            spec, buf_dt, buf_n = _parse_spec_string(inp, dtype, n)
            resolved.append(from_spec(spec, dtype=buf_dt, n=buf_n, device="cuda"))
        elif isinstance(inp, (list, tuple)):
            resolved.append(torch.tensor(inp, dtype=dtype, device="cuda"))
        else:
            resolved.append(torch.tensor([inp], dtype=dtype, device="cuda").expand(n))
    return resolved


def _resolve_dtype(dtype):
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        if dtype in _DTYPE_MAP:
            return _DTYPE_MAP[dtype]
        raise ValueError(f"Unknown dtype '{dtype}'. Valid: {', '.join(_DTYPE_MAP)}")
    return dtype


# ── Output resolution ───────────────────────────────────────────────────

def _parse_output_spec(spec_str, default_dtype, default_n):
    """Parse an output spec string into (dtype, n).

    Examples::

        "float32"       -> (torch.float32, default_n)
        "float16;n=1"   -> (torch.float16, 1)
        "int32;n=4k"    -> (torch.int32, 4096)
    """
    parts = spec_str.split(";")
    dtype_str = parts[0].strip()
    dt = _resolve_dtype(dtype_str) if dtype_str else default_dtype
    on = default_n
    for part in parts[1:]:
        part = part.strip()
        if part.startswith("n="):
            on = _parse_n_value(part[2:])
    return dt, on


class _ParamRef:
    """Lazy reference to a worker-side VMM virtual address, resolved at launch time.

    Used in ``params=[]`` to refer to input/output buffer pointers without
    knowing the worker-side VA upfront (it's resolved when the param buffer
    is built, after SETUP has completed).
    """
    __slots__ = ('kind', 'index')
    def __init__(self, kind, index):
        self.kind = kind    # 'input' or 'output'
        self.index = index


class _WorkerPtr(int):
    """Int-like worker VA that preserves pointer provenance across re-SETUPs."""

    def __new__(cls, value, kind='raw', index=None, offset=0):
        obj = int.__new__(cls, value)
        obj.kind = kind
        obj.index = index
        obj.offset = offset
        return obj

    def _with_offset(self, delta):
        return _WorkerPtr(int(self) + delta, self.kind, self.index,
                          self.offset + delta)

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

    def __repr__(self):
        base = int.__repr__(self)
        if self.kind == 'raw':
            return f"_WorkerPtr({base})"
        return (f"_WorkerPtr({base}, kind={self.kind!r}, "
                f"index={self.index!r}, offset={self.offset!r})")


class _TMADesc:
    """128-byte CUtensorMap struct for TMA kernel parameters.

    Created by ``KernelSession.tma_desc()``. Packed into the param buffer
    with 128-byte alignment as required by cuLaunchKernel.

    Can be constructed with raw bytes (pre-built) or as a lazy builder that
    resolves worker VAs during param buffer construction.
    """
    __slots__ = ('data',)
    def __init__(self, data: bytes):
        assert len(data) == 128, f"TMA descriptor must be 128 bytes, got {len(data)}"
        self.data = data


# Torch dtype → CUtensorMapDataType enum value
_TORCH_DTYPE_TO_TMA = {
    torch.float32: 7,   # CU_TENSOR_MAP_DATA_TYPE_FLOAT32
    torch.float16: 6,   # CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    torch.bfloat16: 9,  # CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    torch.float64: 8,   # CU_TENSOR_MAP_DATA_TYPE_FLOAT64
    torch.int32: 4,     # CU_TENSOR_MAP_DATA_TYPE_INT32
    torch.uint8: 0,     # CU_TENSOR_MAP_DATA_TYPE_UINT8
    torch.int8: 12,     # CU_TENSOR_MAP_DATA_TYPE_INT8 (unofficial, maps to UINT8 + sign)
}


class _TMADescBuilder:
    """Lazy TMA descriptor that resolves worker VAs at param buffer build time.

    Created by ``KernelSession.tma_desc()``. Converted to ``_TMADesc``
    in ``_build_param_buffer`` when worker VAs are available.
    """
    __slots__ = ('kind', 'index', 'dtype', 'rank', 'shape', 'strides',
                 'box_shape', 'elem_strides', 'swizzle', 'l2_promo',
                 'oob_fill', 'interleave')

    def __init__(self, kind, index, dtype, shape, box_shape,
                 strides=None, elem_strides=None,
                 swizzle=0, l2_promo=0, oob_fill=0, interleave=0):
        self.kind = kind      # 'input' or 'output'
        self.index = index
        self.dtype = dtype
        self.rank = len(shape)
        self.shape = tuple(shape)
        self.strides = tuple(strides) if strides is not None else None
        self.box_shape = tuple(box_shape)
        self.elem_strides = tuple(elem_strides) if elem_strides is not None else None
        self.swizzle = swizzle
        self.l2_promo = l2_promo
        self.oob_fill = oob_fill
        self.interleave = interleave

    def resolve(self, worker_va):
        """Create the actual CUtensorMap using the resolved worker VA."""
        return _create_tma_desc(
            global_address=worker_va,
            dtype=self.dtype,
            shape=self.shape,
            strides=self.strides,
            box_shape=self.box_shape,
            elem_strides=self.elem_strides,
            swizzle=self.swizzle,
            l2_promo=self.l2_promo,
            oob_fill=self.oob_fill,
            interleave=self.interleave,
        )


def _create_tma_desc(global_address, dtype, shape, box_shape,
                     strides=None, elem_strides=None,
                     swizzle=0, l2_promo=0, oob_fill=0, interleave=0):
    """Create a 128-byte CUtensorMap TMA descriptor.

    Args:
        global_address: Device pointer to the tensor data.
        dtype: torch.dtype of the tensor elements.
        shape: Tuple of dimension sizes (innermost-first / col-major order).
        box_shape: Tuple of TMA box dimensions (tile shape, innermost-first).
        strides: Tuple of strides in bytes (rank-1 entries, innermost-first).
                 If None, assumes contiguous row-major layout.
        elem_strides: Tuple of element strides per dim (default all 1s).
        swizzle: CU_TENSOR_MAP_SWIZZLE_* enum value (0=NONE, 1=32B, 2=64B, 3=128B).
        l2_promo: CU_TENSOR_MAP_L2_PROMOTION_* enum value.
        oob_fill: CU_TENSOR_MAP_FLOAT_OOB_FILL_* enum value.
        interleave: CU_TENSOR_MAP_INTERLEAVE_* enum value.

    Returns:
        _TMADesc with 128 raw bytes.
    """
    from cuda.bindings import driver as cu

    rank = len(shape)
    tma_dtype = _TORCH_DTYPE_TO_TMA.get(dtype)
    if tma_dtype is None:
        raise ValueError(f"Unsupported dtype for TMA: {dtype}")
    tma_data_type = cu.CUtensorMapDataType(tma_dtype)

    elem_size = torch.tensor([], dtype=dtype).element_size()

    # Default strides: contiguous layout → cumulative product in bytes.
    # Note: CUDA requires at least 1 stride entry even for rank=1.
    if strides is None:
        strides_bytes = []
        stride = elem_size
        for i in range(rank - 1):
            stride *= shape[i]
            strides_bytes.append(stride)
        # CUDA requires >= 1 stride entry; for 1D use element size
        if not strides_bytes:
            strides_bytes.append(elem_size)
        strides = tuple(strides_bytes)

    if elem_strides is None:
        elem_strides = (1,) * rank

    global_dim = [cu.cuuint64_t(s) for s in shape]
    global_strides = [cu.cuuint64_t(s) for s in strides]
    box_dim = [cu.cuuint32_t(s) for s in box_shape]
    e_strides = [cu.cuuint32_t(s) for s in elem_strides]

    err, tmap = cu.cuTensorMapEncodeTiled(
        tma_data_type,
        rank,
        global_address,
        global_dim,
        global_strides,
        box_dim,
        e_strides,
        cu.CUtensorMapInterleave(interleave),
        cu.CUtensorMapSwizzle(swizzle),
        cu.CUtensorMapL2promotion(l2_promo),
        cu.CUtensorMapFloatOOBfill(oob_fill),
    )
    if err != 0:
        try:
            err_name = cu.CUresult(err).name
        except Exception:
            err_name = str(err)
        raise RuntimeError(
            f"cuTensorMapEncodeTiled failed: {err_name}\n"
            f"  dtype={dtype}, rank={rank}, addr={hex(global_address)}\n"
            f"  shape={shape}, strides={strides}\n"
            f"  box_shape={box_shape}, elem_strides={elem_strides}\n"
            f"  swizzle={swizzle}, l2_promo={l2_promo}, oob_fill={oob_fill}")

    # Extract raw 128 bytes from the opaque field (16 x uint64)
    raw = b''
    for qword in tmap.opaque:
        raw += struct.pack('<Q', int(qword))
    assert len(raw) == 128
    return _TMADesc(raw)


# Triton signature type → numpy dtype for scalar coercion
_TRITON_SCALAR_TO_NUMPY = {
    "i1": "uint8", "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
    "u1": "uint8", "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
    "fp16": "float16", "fp32": "float32", "f32": "float32", "fp64": "float64",
    "bf16": "float16",  # bf16 scalars passed as float16 storage
}


def _coerce_triton_params(params, param_types):
    """Coerce scalar params to match Triton signature types.

    Triton signatures specify exact types (i32, i64, fp32, etc.). When the user
    passes a numpy scalar of the wrong width (e.g. np.int32 for an i64 param),
    silently coerce it. This prevents subtle param buffer layout corruption.

    Args:
        params: List of param values (modified in-place is avoided; returns new list).
        param_types: List of Triton type strings (e.g. ["*fp32", "*fp32", "i32"]).

    Returns:
        New params list with scalars coerced to correct types.
    """
    import numpy as np
    if param_types is None or len(param_types) == 0:
        return params

    result = list(params)
    for i, (p, ty) in enumerate(zip(result, param_types)):
        # Skip pointer params and _ParamRef — they're already correct
        if isinstance(p, (_ParamRef, _WorkerPtr, _TMADesc)) or (
                isinstance(ty, str) and ty.startswith('*')):
            continue
        if isinstance(ty, str) and ty == 'nvTmaDesc':
            continue

        np_dtype = _TRITON_SCALAR_TO_NUMPY.get(ty)
        if np_dtype is None:
            continue

        target_dtype = np.dtype(np_dtype)

        if isinstance(p, np.generic):
            if p.dtype != target_dtype:
                old_type = p.dtype
                result[i] = target_dtype.type(p)
                _log(f"Coerced params[{i}] from {old_type} to {target_dtype} "
                     f"(Triton signature expects {ty})", _DIM)
        elif isinstance(p, (int, float)):
            # Auto-convert bare Python scalars to the correct numpy type
            result[i] = target_dtype.type(p)
        elif isinstance(p, bool):
            result[i] = np.uint8(1 if p else 0)

    return result


def _resolve_worker_ptr_value(ptr, worker_vas, num_inputs, scratch_va=0):
    """Resolve a _WorkerPtr against the current worker VAs."""
    if ptr.kind == 'input':
        return worker_vas[ptr.index] + ptr.offset
    if ptr.kind == 'output':
        return worker_vas[num_inputs + ptr.index] + ptr.offset
    if ptr.kind == 'scratch':
        return scratch_va + ptr.offset
    return int(ptr)


def _build_param_buffer(params, worker_vas, num_inputs, ptr_align=8, scratch_va=0):
    """Pack kernel parameters into a byte buffer for CU_LAUNCH_PARAM_BUFFER_POINTER.

    Uses natural alignment: each parameter aligned to min(size, 8).
    This matches CUDA's kernel parameter packing rules.

    Args:
        params: List of parameter values. Supported types:
            - _ParamRef: resolves to 8-byte CUdeviceptr (worker VA)
            - numpy scalar (np.uint32, np.float32, np.int64, ...): sized by dtype
            - (format_char, value) tuple: struct.pack style ('I', 42)
            - bytes: raw data injected verbatim (no alignment padding)
            - 0-dim torch.Tensor: auto-converted to numpy scalar
            - _TMADesc: 128-byte CUtensorMap struct (128-byte aligned)
            - bool: converted to np.uint8(0 or 1)
        worker_vas: Tuple of worker-side CUdeviceptr values from SETUP.
        num_inputs: Number of input chunks (output VAs start at this offset).
        ptr_align: Alignment for pointer params (8 for CUDA, 1 for Triton
                   which uses .align 1 on pointer params in PTX).

    Returns:
        bytes: Packed parameter buffer ready for cuLaunchKernel extra[].
    """
    import numpy as np
    buf = bytearray()
    for i, p in enumerate(params):
        if isinstance(p, _ParamRef):
            if p.kind == 'input':
                va = worker_vas[p.index]
            else:
                va = worker_vas[num_inputs + p.index]
            pad = (ptr_align - len(buf) % ptr_align) % ptr_align if ptr_align > 1 else 0
            buf.extend(b'\x00' * pad)
            buf.extend(struct.pack('<Q', va))
        elif isinstance(p, _WorkerPtr):
            va = _resolve_worker_ptr_value(
                p, worker_vas, num_inputs, scratch_va=scratch_va)
            pad = (ptr_align - len(buf) % ptr_align) % ptr_align if ptr_align > 1 else 0
            buf.extend(b'\x00' * pad)
            buf.extend(struct.pack('<Q', va))
        elif isinstance(p, _TMADescBuilder):
            if p.kind == 'input':
                va = worker_vas[p.index]
            else:
                va = worker_vas[num_inputs + p.index]
            desc = p.resolve(va)
            pad = (128 - len(buf) % 128) % 128
            buf.extend(b'\x00' * pad)
            buf.extend(desc.data)
        elif isinstance(p, _TMADesc):
            pad = (128 - len(buf) % 128) % 128
            buf.extend(b'\x00' * pad)
            buf.extend(p.data)
        elif isinstance(p, np.generic):
            size = p.dtype.itemsize
            align = min(size, 8)
            pad = (align - len(buf) % align) % align if align > 1 else 0
            buf.extend(b'\x00' * pad)
            buf.extend(p.tobytes())
        elif isinstance(p, tuple) and len(p) == 2:
            fmt, val = p
            size = struct.calcsize(fmt)
            align = min(size, 8)
            pad = (align - len(buf) % align) % align if align > 1 else 0
            buf.extend(b'\x00' * pad)
            buf.extend(struct.pack(f'<{fmt}', val))
        elif isinstance(p, bytes):
            buf.extend(p)
        elif isinstance(p, bool):
            buf.extend(np.uint8(1 if p else 0).tobytes())
        elif isinstance(p, torch.Tensor) and p.ndim == 0:
            val = p.cpu().numpy()
            size = val.dtype.itemsize
            align = min(size, 8)
            pad = (align - len(buf) % align) % align if align > 1 else 0
            buf.extend(b'\x00' * pad)
            buf.extend(val.tobytes())
        elif isinstance(p, (int, float)):
            raise TypeError(
                f"params[{i}]: bare {type(p).__name__} is ambiguous — "
                f"use np.uint32({p}), np.float32({p}), etc.")
        else:
            raise TypeError(f"params[{i}]: unsupported type {type(p)}")
    return bytes(buf)


def _has_tma_params(params):
    """Check if any param is a TMA descriptor (requires kernelParams launch)."""
    return any(isinstance(p, (_TMADesc, _TMADescBuilder)) for p in params)


def _build_kernel_params_buffer(params, worker_vas, num_inputs, scratch_va=0):
    """Build param buffer with offset table for kernelParams launch.

    Format: [u32 num_params][u32 offset_0]...[u32 offset_{n-1}][param data...]

    The worker reads the header, then builds void *args[num_params] where
    args[i] = &data[offsets[i]]. This supports TMA descriptors (128-byte
    aligned structs) which cannot be passed via CU_LAUNCH_PARAM_BUFFER_POINTER.

    Uses the same alignment rules as _build_param_buffer for layout of the
    data section, but records each param's offset within it.
    """
    import numpy as np
    data = bytearray()
    offsets = []
    for i, p in enumerate(params):
        if isinstance(p, _ParamRef):
            if p.kind == 'input':
                va = worker_vas[p.index]
            else:
                va = worker_vas[num_inputs + p.index]
            pad = (8 - len(data) % 8) % 8
            data.extend(b'\x00' * pad)
            offsets.append(len(data))
            data.extend(struct.pack('<Q', va))
        elif isinstance(p, _WorkerPtr):
            va = _resolve_worker_ptr_value(
                p, worker_vas, num_inputs, scratch_va=scratch_va)
            pad = (8 - len(data) % 8) % 8
            data.extend(b'\x00' * pad)
            offsets.append(len(data))
            data.extend(struct.pack('<Q', va))
        elif isinstance(p, _TMADescBuilder):
            if p.kind == 'input':
                va = worker_vas[p.index]
            else:
                va = worker_vas[num_inputs + p.index]
            desc = p.resolve(va)
            # 128-byte alignment for TMA
            pad = (128 - len(data) % 128) % 128
            data.extend(b'\x00' * pad)
            offsets.append(len(data))
            data.extend(desc.data)
        elif isinstance(p, _TMADesc):
            pad = (128 - len(data) % 128) % 128
            data.extend(b'\x00' * pad)
            offsets.append(len(data))
            data.extend(p.data)
        elif isinstance(p, np.generic):
            size = p.dtype.itemsize
            align = min(size, 8)
            pad = (align - len(data) % align) % align if align > 1 else 0
            data.extend(b'\x00' * pad)
            offsets.append(len(data))
            data.extend(p.tobytes())
        elif isinstance(p, tuple) and len(p) == 2:
            fmt, val = p
            size = struct.calcsize(fmt)
            align = min(size, 8)
            pad = (align - len(data) % align) % align if align > 1 else 0
            data.extend(b'\x00' * pad)
            offsets.append(len(data))
            data.extend(struct.pack(f'<{fmt}', val))
        elif isinstance(p, bytes):
            offsets.append(len(data))
            data.extend(p)
        elif isinstance(p, bool):
            offsets.append(len(data))
            data.extend(np.uint8(1 if p else 0).tobytes())
        elif isinstance(p, torch.Tensor) and p.ndim == 0:
            val = p.cpu().numpy()
            size = val.dtype.itemsize
            align = min(size, 8)
            pad = (align - len(data) % align) % align if align > 1 else 0
            data.extend(b'\x00' * pad)
            offsets.append(len(data))
            data.extend(val.tobytes())
        elif isinstance(p, (int, float)):
            raise TypeError(
                f"params[{i}]: bare {type(p).__name__} is ambiguous — "
                f"use np.uint32({p}), np.float32({p}), etc.")
        else:
            raise TypeError(f"params[{i}]: unsupported type {type(p)}")

    # Build header: [u32 num_params][u32 offset_0]...[u32 offset_{n-1}]
    n_params = len(offsets)
    header = struct.pack('<I', n_params)
    for off in offsets:
        header += struct.pack('<I', off)
    return header + bytes(data)


class _CUDAPtr:
    """Wraps a raw CUDA device pointer for torch.as_tensor via __cuda_array_interface__."""
    __slots__ = ('__cuda_array_interface__',)
    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {
            'shape': shape, 'typestr': typestr, 'data': (ptr, False), 'version': 3,
        }

_TORCH_DTYPE_TO_TYPESTR = {
    torch.float32: '<f4', torch.float16: '<f2', torch.float64: '<f8',
    torch.int32: '<i4', torch.int16: '<i2', torch.int64: '<i8',
    torch.int8: '<i1', torch.uint8: '<u1', torch.bool: '|b1',
}

def _vmm_to_tensor(ptr, n, dtype):
    """Create a PyTorch CUDA tensor pointing directly at a VMM device pointer (zero-copy)."""
    typestr = _TORCH_DTYPE_TO_TYPESTR.get(dtype)
    if typestr is not None:
        return torch.as_tensor(_CUDAPtr(ptr, (n,), typestr), device='cuda')
    # Fallback for dtypes without __cuda_array_interface__ support (e.g. bfloat16):
    # allocate empty tensor and DtoD copy from VMM pointer
    from . import vmm
    elem_sz = _DTYPE_ELEM_SIZE.get(_DTYPE_CODE.get(dtype, 1), 4)
    t = torch.empty(n, dtype=dtype, device='cuda')
    vmm.memcpy_dtod(t.data_ptr(), ptr, n * elem_sz)
    return t


def _resolve_outputs(outputs, default_dtype, default_n):
    """Resolve output specification into a list of (dtype, n) pairs.

    Accepts:
        int             Count — all outputs share default_dtype and default_n.
        list of str     Per-output specs: "float32", "float16;n=1", etc.
        list of dtype   Per-output torch dtypes (all use default_n).
    """
    if isinstance(outputs, int):
        return [(default_dtype, default_n)] * outputs

    result = []
    for spec in outputs:
        if isinstance(spec, torch.dtype):
            result.append((spec, default_n))
        elif isinstance(spec, str):
            result.append(_parse_output_spec(spec, default_dtype, default_n))
        else:
            raise ValueError(
                f"Invalid output spec: {spec!r}\n"
                f"  Expected: int, dtype string, or \"dtype;n=N\".\n"
                f"  Examples: \"float32\", \"float16;n=1\", 2")
    return result


# ── Verification ────────────────────────────────────────────────────────

def _verify(outputs, ref, inputs, atol=1e-5, rtol=1e-5):
    """Verify outputs against a reference. Returns (passed, message).

    Both ``outputs`` and ``ref`` (expected) can be lists/tuples or dicts
    of tensors.  When both are dicts, comparison is done by matching keys.
    """
    if callable(ref):
        expected = ref(inputs)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        elif isinstance(expected, dict):
            pass  # keep as dict
        elif not isinstance(expected, (list, tuple)):
            expected = (expected,)
    elif isinstance(ref, torch.Tensor):
        expected = (ref,)
    elif isinstance(ref, dict):
        expected = ref
    elif isinstance(ref, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in ref):
        expected = ref
    else:
        return True, "no reference"

    # Build (label, out, exp) pairs — works for both list and dict
    if isinstance(expected, dict):
        if not isinstance(outputs, dict):
            return False, "expected is a dict but run() returned a list/tuple"
        pairs = [(k, outputs[k], expected[k]) for k in expected]
    elif isinstance(outputs, dict):
        return False, "run() returned a dict but expected is a list/tuple"
    else:
        pairs = [(str(i), out, exp) for i, (out, exp) in
                 enumerate(zip(outputs, expected))]

    all_ok = True
    msgs = []
    for label, out, exp in pairs:
        if not isinstance(exp, torch.Tensor):
            exp = torch.tensor(exp, dtype=out.dtype, device=out.device)
        if exp.device != out.device:
            exp = exp.to(out.device)
        if exp.dtype != out.dtype:
            exp = exp.to(out.dtype)
        out_f, exp_f = out.float(), exp.float()
        diff = (out_f - exp_f).abs()
        max_err = diff.max().item()
        ok = torch.allclose(out_f, exp_f, atol=atol, rtol=rtol)
        tag = "PASS" if ok else "FAIL"
        msgs.append(f"output[{label}]: {tag} (max_err={max_err:.2e})")
        if not ok:
            all_ok = False
            flat = diff.flatten()
            worst_idx = flat.argmax().item()
            unrav = []
            idx = worst_idx
            for s in reversed(diff.shape):
                unrav.append(idx % s)
                idx //= s
            unrav.reverse()
            msgs.append(
                f"  worst@{list(unrav)}: got={out.flatten()[worst_idx].item():.6e}, "
                f"expected={exp.flatten()[worst_idx].item():.6e}")
            needed_atol = max_err * 1.01
            msgs.append(f"  would pass with atol={needed_atol:.2e}")
    return all_ok, "; ".join(msgs)


# ── KernelSession ───────────────────────────────────────────────────────


class _KernelHandle:
    """Callable view of a single kernel within a shared KernelSession."""

    __slots__ = ('_session', '_index')

    def __init__(self, session, index):
        self._session = session
        self._index = index

    def __call__(self, *inputs, **kw):
        return self._session._call_kernel(self._index, *inputs, **kw)

    @property
    def func_name(self):
        return self._session._entry_func_name(self._index)

    @property
    def compile_ms(self):
        return self._session._entry_value(self._index, 'compile_ms')

    @property
    def scratch_va(self):
        return self._session.scratch_va

    @property
    def scratch_ptr(self):
        return self._session.scratch_ptr

    def in_ptr(self, index=0):
        return self._session.in_ptr(index)

    def out_ptr(self, index=0):
        return self._session.out_ptr(index)

    def tma_desc(self, *args, **kwargs):
        return self._session.tma_desc(*args, **kwargs)

    def sync(self):
        return self._session.sync()

    def start_timing(self, sync=False):
        return self._session.start_timing(sync=sync)

    def end_timing(self, sync=True):
        return self._session.end_timing(sync=sync)

    def l2_flush(self, *args, **kwargs):
        return self._session.l2_flush(*args, **kwargs)

    def zero_scratch(self):
        return self._session.zero_scratch()


class KernelSession:
    """Fast-iteration kernel development session.

    Compiles a CUDA kernel once and caches the result.  On each call,
    checks if the source file changed and recompiles automatically.

    Args:
        kernel_path: Path to .cu source file.
        func_name:   Kernel function name (auto-detected if None).
        outputs:     Number of output tensors the kernel produces.

    Example::

        s = KernelSession("add_one.cu")
        out = s(torch.arange(8, device="cuda", dtype=torch.int32))
        # tensor([1, 2, 3, 4, 5, 6, 7, 8])

        # Edit add_one.cu, then:
        out = s(torch.arange(8, device="cuda", dtype=torch.int32))
        # Automatically recompiles and runs the new version.
    """

    def __init__(self, kernel_path=None, func_name=None, outputs=1, defines=None,
                 timeout=0.75, grid=None, block=None, smem=None,
                 out_dtype=None, kernel_source=None,
                 triton_fn=None, triton_constexprs=None,
                 kernel_scratch_mib=0):
        """Create a session. Compiles the kernel immediately.

        Args:
            kernel_path:   Path to .cu source file.
            func_name:     Override auto-detected function name.
            outputs:       Output specification:
                           - int: number of outputs (all same dtype/n as inputs)
                           - list of specs: per-output control

                           Spec strings: "float32", "float16;n=1", "int32;n=4k"
            defines:       Dict of preprocessor defines (e.g. {"T": "float"}).
            timeout:       Worker dispatch timeout in seconds (default 0.75).
                           On timeout, the worker is killed and respawned.
            grid:          Default grid dims (int or tuple). None = auto.
            block:         Default block dims (int or tuple). None = auto (256).
            smem:          Default dynamic shared memory bytes. None = 0.
            out_dtype:     Default output dtype override (see __call__).
            kernel_source: Inline CUDA source string.
            triton_fn:     A @triton.jit decorated function.
            triton_constexprs: Dict of constexpr param values for Triton compilation.

            Exactly one of kernel_path, kernel_source, or triton_fn must be provided.
        """
        sources = sum(x is not None for x in [kernel_path, kernel_source, triton_fn])
        if sources != 1:
            raise ValueError(
                "Exactly one of kernel_path, kernel_source, or triton_fn must be provided")

        self._raw_outputs = outputs
        self._launch_ms = 0.0
        self._timeout = timeout
        self._user_scratch_mib = kernel_scratch_mib
        self._kernel_scratch_mib = kernel_scratch_mib
        self._scratch_va = 0  # worker-side scratch VA, set during SETUP

        # Worker daemon state
        self._worker_proc = None
        self._worker_sock_path = None
        self._sock = None  # persistent connection to worker
        self._send_queue = bytearray()  # queued request bytes
        self._recv_queue = []  # list of recv_extra sizes per queued request
        self._sender_cache = _SenderCache()  # LRU of cubin hashes worker has
        self._last_cubin_hash = None  # hash of last cubin sent (for g_last_func fast path)
        self._cached_out_tensors = None  # cached output from _run_persistent
        self._cached_out_specs = None
        self._persistent_setup = False  # True if SETUP sent to current worker
        self._persistent_scratch_mib = 0  # scratch_mib used in last SETUP
        self._persistent_num_inputs = 0
        self._persistent_num_outputs = 0
        self._persistent_input_ptrs = ()
        self._worker_vas = ()

        # VMM state
        self._vmm_device = None
        self._vmm_granularity = 0
        self._vmm_chunks = []  # list of (handle, size, mapped_ptr)
        self._vmm_chunk_size = 0

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU available.\n"
                "  kernelbox requires a CUDA-capable GPU and PyTorch with CUDA support.\n"
                "  Install: pip install torch")
        torch.cuda.init()
        torch.empty(1, device="cuda")

        self._entries = self._make_kernel_entries(
            kernel_path=kernel_path,
            kernel_source=kernel_source,
            triton_fn=triton_fn,
            func_name=func_name,
            defines=defines,
            grid=grid,
            block=block,
            smem=smem,
            out_dtype=out_dtype,
            triton_constexprs=triton_constexprs,
        )
        self._kernel_handles = tuple(
            _KernelHandle(self, i) for i in range(len(self._entries)))
        self._active_kernel_index = 0
        self._load_active_kernel_state(0)

        for entry in self._entries:
            if entry['kernel_path'] is not None and not os.path.isfile(entry['kernel_path']):
                raise FileNotFoundError(f"Kernel file not found: {entry['kernel_path']}")

        # Compile CUDA kernels eagerly; Triton still waits for runtime types.
        for i, entry in enumerate(self._entries):
            self._select_kernel(i)
            if not self._triton_mode:
                self._recompile_cubin()
        self._select_kernel(0)
        atexit.register(self._stop_worker)

    _ENTRY_FIELDS = (
        'kernel_path', 'kernel_source', 'func_name_override', 'defines',
        'source_hash', 'cubin', 'cubin_hash', 'cubin_id', 'is_cubin',
        'detected_name', 'compile_ms', 'triton_fn', 'triton_constexprs',
        'triton_mode', 'triton_signature', 'triton_param_types',
        'triton_global_scratch_size', 'triton_profile_scratch_size',
        'triton_scratch_offset', 'triton_num_warps', 'triton_shared',
        'kernel_scratch_mib',
        'default_grid', 'default_block', 'default_smem', 'default_out_dtype',
    )

    @staticmethod
    def _expand_per_kernel(value, count, name):
        # Only plain lists are treated as per-kernel lists.
        # Tuples are kept as single values (e.g. 3D grid/block dims).
        if isinstance(value, list):
            if len(value) != count:
                raise ValueError(
                    f"{name} list length mismatch: expected {count}, got {len(value)}")
            return list(value)
        return [value] * count

    def _make_kernel_entries(self, kernel_path=None, kernel_source=None, triton_fn=None,
                             func_name=None, defines=None, grid=None, block=None,
                             smem=None, out_dtype=None, triton_constexprs=None):
        sources = {
            'kernel_path': kernel_path,
            'kernel_source': kernel_source,
            'triton_fn': triton_fn,
        }
        active = [(name, value) for name, value in sources.items() if value is not None]
        source_name, source_value = active[0]
        count = len(source_value) if isinstance(source_value, (list, tuple)) else 1

        kernel_paths = self._expand_per_kernel(kernel_path, count, 'kernel_path')
        kernel_sources = self._expand_per_kernel(kernel_source, count, 'kernel_source')
        triton_fns = self._expand_per_kernel(triton_fn, count, 'triton_fn')
        func_names = self._expand_per_kernel(func_name, count, 'func_name')
        defines_list = self._expand_per_kernel(defines, count, 'defines')
        grids = self._expand_per_kernel(grid, count, 'grid')
        blocks = self._expand_per_kernel(block, count, 'block')
        smems = self._expand_per_kernel(smem, count, 'smem')
        out_dtypes = self._expand_per_kernel(out_dtype, count, 'out_dtype')
        constexprs = self._expand_per_kernel(
            triton_constexprs or {}, count, 'triton_constexprs')

        entries = []
        for i in range(count):
            path = kernel_paths[i]
            if path is not None:
                path = os.path.abspath(path)
            entries.append({
                'kernel_path': path,
                'kernel_source': kernel_sources[i],
                'func_name_override': func_names[i],
                'defines': defines_list[i],
                'source_hash': None,
                'cubin': None,
                'cubin_hash': None,
                'cubin_id': None,
                'is_cubin': False,
                'detected_name': None,
                'compile_ms': 0.0,
                'triton_fn': triton_fns[i],
                'triton_constexprs': constexprs[i] or {},
                'triton_mode': triton_fns[i] is not None,
                'triton_signature': None,
                'triton_param_types': None,
                'triton_global_scratch_size': 0,
                'triton_profile_scratch_size': 0,
                'triton_scratch_offset': 0,
                'triton_num_warps': 0,
                'triton_shared': 0,
                'kernel_scratch_mib': self._user_scratch_mib,
                'default_grid': grids[i],
                'default_block': blocks[i],
                'default_smem': smems[i],
                'default_out_dtype': out_dtypes[i],
            })
        return entries

    def _entry_value(self, index, key):
        if index == self._active_kernel_index:
            attr = key if key.startswith('_') else f"_{key}" if hasattr(self, f"_{key}") else key
            if hasattr(self, attr):
                return getattr(self, attr)
        return self._entries[index][key]

    def _entry_func_name(self, index):
        entry = self._entries[index]
        if index == self._active_kernel_index:
            return self.func_name
        return entry['func_name_override'] or entry['detected_name']

    def _save_active_kernel_state(self):
        entry = self._entries[self._active_kernel_index]
        entry['kernel_path'] = self.kernel_path
        entry['kernel_source'] = self._kernel_source
        entry['func_name_override'] = self._func_name_override
        entry['defines'] = self._defines
        entry['source_hash'] = self._source_hash
        entry['cubin'] = self._cubin
        entry['cubin_hash'] = self._cubin_hash
        entry['cubin_id'] = getattr(self, '_cubin_id', None)
        entry['is_cubin'] = self._is_cubin
        entry['detected_name'] = self._detected_name
        entry['compile_ms'] = self._compile_ms
        entry['triton_fn'] = self._triton_fn
        entry['triton_constexprs'] = self._triton_constexprs
        entry['triton_mode'] = self._triton_mode
        entry['triton_signature'] = self._triton_signature
        entry['triton_param_types'] = getattr(self, '_triton_param_types', None)
        entry['triton_global_scratch_size'] = self._triton_global_scratch_size
        entry['triton_profile_scratch_size'] = self._triton_profile_scratch_size
        entry['triton_scratch_offset'] = self._triton_scratch_offset
        entry['triton_num_warps'] = getattr(self, '_triton_num_warps', 0)
        entry['triton_shared'] = self._triton_shared
        entry['kernel_scratch_mib'] = self._kernel_scratch_mib
        entry['default_grid'] = self.default_grid
        entry['default_block'] = self.default_block
        entry['default_smem'] = self.default_smem
        entry['default_out_dtype'] = self.default_out_dtype

    def _load_active_kernel_state(self, index):
        entry = self._entries[index]
        self.kernel_path = entry['kernel_path']
        self._kernel_source = entry['kernel_source']
        self._func_name_override = entry['func_name_override']
        self._defines = entry['defines']
        self._source_hash = entry['source_hash']
        self._cubin = entry['cubin']
        self._cubin_hash = entry['cubin_hash']
        self._cubin_id = entry.get('cubin_id')
        self._is_cubin = entry['is_cubin']
        self._detected_name = entry['detected_name']
        self._compile_ms = entry['compile_ms']
        self._triton_fn = entry['triton_fn']
        self._triton_constexprs = entry['triton_constexprs']
        self._triton_mode = entry['triton_mode']
        self._triton_signature = entry['triton_signature']
        self._triton_param_types = entry['triton_param_types']
        self._triton_global_scratch_size = entry['triton_global_scratch_size']
        self._triton_profile_scratch_size = entry['triton_profile_scratch_size']
        self._triton_scratch_offset = entry['triton_scratch_offset']
        self._triton_num_warps = entry['triton_num_warps']
        self._triton_shared = entry['triton_shared']
        self._kernel_scratch_mib = entry.get(
            'kernel_scratch_mib', self._user_scratch_mib)
        self.default_grid = entry['default_grid']
        self.default_block = entry['default_block']
        self.default_smem = entry['default_smem']
        self.default_out_dtype = entry['default_out_dtype']
        self._active_kernel_index = index

    def _select_kernel(self, index):
        if index == self._active_kernel_index:
            return
        self._save_active_kernel_state()
        self._load_active_kernel_state(index)

    @property
    def kernel_count(self):
        return len(self._entries)

    @property
    def kernels(self):
        return list(self._kernel_handles)

    @property
    def kernel_paths(self):
        return [entry['kernel_path'] for entry in self._entries
                if entry['kernel_path'] is not None]

    @property
    def func_name(self):
        return self._func_name_override or self._detected_name

    @property
    def num_outputs(self):
        if isinstance(self._raw_outputs, int):
            return self._raw_outputs
        return len(self._raw_outputs)

    def _make_worker_ptr(self, kind, index=None):
        if kind == 'scratch':
            value = self._scratch_va
        elif kind == 'input':
            value = self._worker_vas[index] if index < len(self._worker_vas) else 0
        elif kind == 'output':
            offset = self._persistent_num_inputs + index
            value = self._worker_vas[offset] if offset < len(self._worker_vas) else 0
        else:
            value = 0
        return _WorkerPtr(value, kind=kind, index=index)

    @property
    def scratch_ptr(self):
        return self._make_worker_ptr('scratch')

    @property
    def input_ptrs(self):
        return tuple(self._make_worker_ptr('input', i)
                     for i in range(self._persistent_num_inputs))

    @property
    def output_ptrs(self):
        count = self._persistent_num_outputs or self.num_outputs
        return tuple(self._make_worker_ptr('output', i) for i in range(count))

    def in_ptr(self, index=0):
        """Reference to worker-side VA for input chunk ``index``.

        Returns a lazy :class:`_ParamRef` for use in ``params=[]``.
        The actual worker VA is resolved at launch time (after SETUP).
        """
        return _ParamRef('input', index)

    def out_ptr(self, index=0):
        """Reference to worker-side VA for output chunk ``index``.

        Returns a lazy :class:`_ParamRef` for use in ``params=[]``.
        The actual worker VA is resolved at launch time (after SETUP).
        """
        return _ParamRef('output', index)

    def tma_desc(self, index, shape, box_shape, dtype=None,
                 strides=None, elem_strides=None, kind='input',
                 swizzle=0, l2_promo=0, oob_fill=0, interleave=0):
        """Create a lazy TMA descriptor for use in ``params=[]``.

        The descriptor is resolved at launch time using the worker-side VA
        for the specified input/output chunk.

        Args:
            index: Input or output chunk index.
            shape: Tensor dimensions (innermost-first, i.e. col-major order).
                   For a row-major [M, N] tensor, pass ``(N, M)``.
            box_shape: TMA tile dimensions (innermost-first).
                       For a row-major [M, N] tensor with 32x64 tiles, pass ``(64, 32)``.
            dtype: Element type (default: inferred from input tensor at call time).
            strides: Byte strides (rank-1 entries, innermost-first).
                     Default: contiguous layout computed from shape and dtype.
            elem_strides: Per-dimension element strides (default: all 1s).
            kind: ``'input'`` or ``'output'`` (default: ``'input'``).
            swizzle: Swizzle mode (0=NONE, 1=32B, 2=64B, 3=128B).
            l2_promo: L2 promotion mode.
            oob_fill: Out-of-bounds fill mode.
            interleave: Interleave mode.

        Returns:
            A lazy descriptor that resolves to a 128-byte CUtensorMap in the param buffer.
        """
        if dtype is None:
            dtype = torch.float32  # Will be overridden if we can infer
        return _TMADescBuilder(
            kind=kind, index=index, dtype=dtype,
            shape=shape, box_shape=box_shape,
            strides=strides, elem_strides=elem_strides,
            swizzle=swizzle, l2_promo=l2_promo,
            oob_fill=oob_fill, interleave=interleave,
        )

    def _call_kernel(self, index, *inputs, **kw):
        self._select_kernel(index)
        return self._launch_active(*inputs, **kw)

    # ── Compilation ──────────────────────────────────────────────────

    def _read_source(self):
        """Read kernel source from file or inline string."""
        if self._kernel_source is not None:
            return self._kernel_source
        return open(self.kernel_path).read()

    def _detect_name_with_fallback(self, source):
        """Detect function name from source, falling back to filename."""
        name = self._func_name_override or _detect_func_name(source)
        if name is None:
            if self.kernel_path is not None:
                name = os.path.splitext(os.path.basename(self.kernel_path))[0]
            else:
                raise ValueError(
                    "Cannot auto-detect kernel function name from inline source.\n"
                    "  Either add: extern \"C\" __global__ void name(...)\n"
                    "  Or pass func_name='your_kernel' explicitly.")
        return name

    def _recompile_cubin(self, input_tensors=None, out_specs=None,
                          params=None):
        """Reload source and compile to cubin bytes. Returns True if recompiled."""
        if self._triton_mode:
            return self._compile_triton(input_tensors, out_specs, params)

        # Check for pre-compiled binary (raw bytes in kernel_source)
        if isinstance(self._kernel_source, bytes):
            h = hashlib.md5(self._kernel_source).hexdigest()
            if h == self._source_hash:
                return False
            self._cubin = self._kernel_source
            self._is_cubin = self._kernel_source[:4] == b'\x7fELF'
            if not self._func_name_override:
                if self._is_cubin:
                    entries = _cubin_entry_names(self._kernel_source)
                    if len(entries) == 1:
                        self._detected_name = entries[0]
                    else:
                        raise ValueError(
                            f"Cannot auto-detect function name from raw cubin"
                            f" ({len(entries)} entry points found).\n"
                            f"  Pass func_name='your_kernel' explicitly.")
                else:
                    m = re.search(rb'\.entry\s+(\w+)', self._kernel_source)
                    if m:
                        self._detected_name = m.group(1).decode()
            self._source_hash = h
            self._compile_ms = 0
            return True

        # Check for .cubin/.ptx file path
        ext = os.path.splitext(self.kernel_path)[1].lower() if self.kernel_path else ''
        if ext in ('.cubin', '.ptx'):
            data = open(self.kernel_path, 'rb').read()
            h = hashlib.md5(data).hexdigest()
            if h == self._source_hash:
                return False
            self._cubin = data
            self._is_cubin = (ext == '.cubin')
            if self._func_name_override:
                self._detected_name = self._func_name_override
            elif ext == '.ptx':
                m = re.search(rb'\.entry\s+(\w+)', data)
                if m:
                    self._detected_name = m.group(1).decode()
                else:
                    self._detected_name = os.path.splitext(
                        os.path.basename(self.kernel_path))[0]
            else:
                self._detected_name = os.path.splitext(
                    os.path.basename(self.kernel_path))[0]
            self._source_hash = h
            self._compile_ms = 0
            _log(f"Loaded {ext[1:].upper()} {os.path.basename(self.kernel_path)}: "
                 f"func={self._detected_name}, {len(data)} bytes", _CYAN)
            return True

        # Standard .cu path — compile via NVRTC
        source = self._read_source()
        h = hashlib.md5(source.encode()).hexdigest()
        if h == self._source_hash:
            return False

        self._detected_name = self._detect_name_with_fallback(source)

        t0 = time.monotonic()
        self._cubin, self._is_cubin = _compile_to_cubin(
            source, self._detected_name, defines=self._defines)
        self._compile_ms = (time.monotonic() - t0) * 1000
        self._source_hash = h
        return True

    def _compile_triton(self, input_tensors=None, out_specs=None,
                         params=None):
        """Compile a @triton.jit function to cubin. Returns True if recompiled."""
        if input_tensors is None and self._cubin is None:
            return False  # Can't compile without input types; defer

        fn = self._triton_fn
        source = inspect.getsource(fn.fn)
        constexpr_str = str(sorted(self._triton_constexprs.items()))

        # Build signature from input/output tensor types + param scalar types
        if input_tensors is not None and out_specs is not None:
            self._triton_signature = self._build_triton_signature(
                fn, input_tensors, out_specs, params)
        if self._triton_signature is None:
            raise RuntimeError(
                "Triton compilation requires input tensors for type inference")

        # Include signature in hash — scalar types affect compiled layout
        sig_str = str(sorted(self._triton_signature.items()))
        h = hashlib.md5((source + constexpr_str + sig_str).encode()).hexdigest()
        if h == self._source_hash:
            return False

        import triton
        from triton.compiler import ASTSource

        t0 = time.monotonic()
        src = ASTSource(
            fn=fn,
            signature=self._triton_signature,
            constexprs=self._triton_constexprs,
        )
        compiled = triton.compile(src)
        self._compile_ms = (time.monotonic() - t0) * 1000

        self._cubin = compiled.kernel  # raw cubin bytes
        self._is_cubin = True
        self._detected_name = compiled.name
        self._source_hash = h

        # Store ordered list of runtime param types for validation/coercion
        self._triton_param_types = list(self._triton_signature.values())

        # Extract launch config from metadata
        meta = compiled.metadata
        warp_size = getattr(meta, 'num_threads_per_warp', 32)
        self._triton_num_warps = meta.num_warps
        self._triton_shared = meta.shared
        self._triton_global_scratch_size = getattr(meta, 'global_scratch_size', 0)
        self._triton_profile_scratch_size = getattr(meta, 'profile_scratch_size', 0)
        triton_scratch_bytes = self._triton_global_scratch_size + self._triton_profile_scratch_size
        if triton_scratch_bytes > 0:
            triton_scratch_mib = (triton_scratch_bytes + (1024*1024 - 1)) // (1024*1024)
            self._kernel_scratch_mib = self._user_scratch_mib + triton_scratch_mib
            self._triton_scratch_offset = self._user_scratch_mib * 1024 * 1024
        # Auto-set block and shared memory from Triton metadata
        self.default_block = meta.num_warps * warp_size
        self.default_smem = meta.shared

        _log(f"Compiled Triton kernel '{self._detected_name}': "
             f"{meta.num_warps} warps, {meta.shared}B smem, "
             f"{self._compile_ms:.0f}ms", _CYAN)
        return True

    def _build_triton_signature(self, fn, input_tensors, out_specs,
                                params=None):
        """Build Triton ASTSource signature dict from function params + runtime types.

        When ``params`` is provided, scalar types are inferred from actual values
        (numpy dtype or Python type). This ensures the compiled cubin's param
        layout matches what the user passes at launch time.
        """
        import numpy as np
        all_params = list(inspect.signature(fn.fn).parameters.items())

        # Identify constexpr params from annotations + explicit dict
        constexpr_names = set(self._triton_constexprs.keys())
        for name, param in all_params:
            ann = param.annotation
            if ann is not inspect.Parameter.empty and 'constexpr' in str(ann):
                constexpr_names.add(name)

        runtime_params = [(n, p) for n, p in all_params if n not in constexpr_names]

        # Map torch dtypes to Triton pointer type strings
        _dt_to_triton = {
            torch.float32: "*fp32", torch.float16: "*fp16",
            torch.float64: "*fp64", torch.bfloat16: "*bf16",
            torch.int32: "*i32", torch.int64: "*i64",
            torch.int8: "*i8", torch.uint8: "*u8",
            torch.bool: "*i1",
        }

        # Map numpy dtypes to Triton scalar type strings
        _np_to_triton_scalar = {
            np.dtype('int8'): "i8", np.dtype('int16'): "i16",
            np.dtype('int32'): "i32", np.dtype('int64'): "i64",
            np.dtype('uint8'): "u8", np.dtype('uint16'): "u16",
            np.dtype('uint32'): "u32", np.dtype('uint64'): "u64",
            np.dtype('float16'): "fp16", np.dtype('float32'): "fp32",
            np.dtype('float64'): "fp64",
        }

        sig = {}

        if params is not None:
            # When params are explicit, use _ParamRef types to determine
            # pointer dtypes in the correct order (handles output-before-input
            # signatures like Triton softmax).
            for i, (name, _) in enumerate(runtime_params):
                if i >= len(params):
                    sig[name] = "i32"
                    continue
                p = params[i]
                if isinstance(p, (_ParamRef, _WorkerPtr)):
                    if p.kind == 'input':
                        dt = input_tensors[p.index].dtype
                        sig[name] = _dt_to_triton.get(dt, "*fp32")
                    elif p.kind == 'output':
                        dt = out_specs[p.index][0]
                        sig[name] = _dt_to_triton.get(dt, "*fp32")
                    else:
                        sig[name] = "*i8"
                else:
                    # Scalar param — infer type from value
                    if isinstance(p, np.generic):
                        sig[name] = _np_to_triton_scalar.get(p.dtype, "i32")
                    elif isinstance(p, float):
                        sig[name] = "fp64"
                    elif isinstance(p, int):
                        sig[name] = "i64" if abs(p) > 0x7FFFFFFF else "i32"
                    elif isinstance(p, bool):
                        sig[name] = "i1"
                    else:
                        sig[name] = "i32"
            return sig

        # Default: assume inputs first, then outputs, then scalars
        n_ptrs = len(input_tensors) + len(out_specs)
        ptr_idx = 0
        for name, _ in runtime_params:
            if ptr_idx < len(input_tensors):
                dt = input_tensors[ptr_idx].dtype
                sig[name] = _dt_to_triton.get(dt, "*fp32")
                ptr_idx += 1
            elif ptr_idx < n_ptrs:
                out_idx = ptr_idx - len(input_tensors)
                dt = out_specs[out_idx][0]
                sig[name] = _dt_to_triton.get(dt, "*fp32")
                ptr_idx += 1
            else:
                sig[name] = "i32"

        return sig

    def update_triton_fn(self, fn, constexprs=None):
        """Update the @triton.jit function (e.g. after module reload)."""
        self._triton_fn = fn
        if constexprs is not None:
            self._triton_constexprs = constexprs
        self._source_hash = None  # force recompile
        self._save_active_kernel_state()

    def update_triton_fns(self, fns, constexprs=None):
        """Update every Triton kernel in a multi-kernel session."""
        if not isinstance(fns, (list, tuple)):
            raise TypeError("update_triton_fns() requires a list/tuple")
        constexprs = constexprs or [{}] * len(fns)
        constexpr_list = self._expand_per_kernel(
            constexprs, len(self._entries), 'triton_constexprs')
        if len(fns) != len(self._entries):
            raise ValueError("update_triton_fns() length mismatch")
        for i, fn in enumerate(fns):
            self._select_kernel(i)
            self._triton_fn = fn
            self._triton_constexprs = constexpr_list[i] or {}
            self._source_hash = None
            self._save_active_kernel_state()
        self._select_kernel(0)

    def update_source(self, source):
        """Update inline kernel source. Triggers recompilation on next call."""
        self._kernel_source = source
        self._source_hash = None
        self._save_active_kernel_state()

    def update_sources(self, sources):
        """Update every inline CUDA source in a multi-kernel session."""
        if not isinstance(sources, (list, tuple)):
            raise TypeError("update_sources() requires a list/tuple")
        if len(sources) != len(self._entries):
            raise ValueError("update_sources() length mismatch")
        for i, source in enumerate(sources):
            self._select_kernel(i)
            self._kernel_source = source
            self._source_hash = None
            self._save_active_kernel_state()
        self._select_kernel(0)

    def update_defaults(self, grid=None, block=None, smem=None, out_dtype=None):
        """Update default launch config for one or many kernels."""
        grids = self._expand_per_kernel(grid, len(self._entries), 'grid')
        blocks = self._expand_per_kernel(block, len(self._entries), 'block')
        smems = self._expand_per_kernel(smem, len(self._entries), 'smem')
        out_dtypes = self._expand_per_kernel(
            out_dtype, len(self._entries), 'out_dtype')
        for i in range(len(self._entries)):
            self._select_kernel(i)
            self.default_grid = grids[i]
            self.default_block = blocks[i]
            self.default_smem = smems[i]
            self.default_out_dtype = out_dtypes[i]
            self._save_active_kernel_state()
        self._select_kernel(0)

    def _resolve_inputs_and_outputs(self, inputs, n=None, out_dtype=None,
                                    grid=None, block=None, smem=None,
                                    params=None, extra_params=None):
        """Normalize launch inputs/config and resolve output specs."""
        if params is not None and extra_params is not None:
            raise ValueError("Cannot specify both params and extra_params")

        if grid is None:
            grid = self.default_grid
        if block is None:
            block = self.default_block
        if smem is None:
            smem = self.default_smem or 0
        if out_dtype is None:
            out_dtype = self.default_out_dtype

        inputs_list = list(inputs)
        for i, t in enumerate(inputs_list):
            if isinstance(t, torch.Tensor) and not t.is_cuda:
                _log(f"Warning: input[{i}] is on CPU, moving to CUDA", _YELLOW)
                inputs_list[i] = t.cuda()
        inputs = tuple(inputs_list)

        if n is None:
            n = inputs[0].numel() if inputs else 1024

        default_dt = inputs[0].dtype if inputs else torch.float32
        if isinstance(self._raw_outputs, int) and out_dtype is not None:
            if isinstance(out_dtype, (list, tuple)):
                out_specs = [(_resolve_dtype(d), n) for d in out_dtype]
            else:
                d = _resolve_dtype(out_dtype)
                out_specs = [(d, n)] * self.num_outputs
        else:
            out_specs = _resolve_outputs(self._raw_outputs, default_dt, n)

        if extra_params is not None:
            import numpy as np
            params = []
            for i in range(len(inputs)):
                params.append(_ParamRef('input', i))
            for i in range(self.num_outputs):
                params.append(_ParamRef('output', i))
            params.append(np.uint32(n))
            params.extend(extra_params)

        return inputs, out_specs, n, grid, block, smem, params

    def _normalize_step_spec(self, step, default_n, default_clear_outputs):
        """Normalize a kernel_mode step spec into a launch dict."""
        if isinstance(step, _KernelHandle):
            step = {'kernel': step}
        if not isinstance(step, dict):
            raise TypeError(
                "kernel_mode() steps must be dicts or kernel handles")
        if 'kernel' not in step:
            raise ValueError("kernel_mode() step is missing 'kernel'")

        kernel = step['kernel']
        if isinstance(kernel, _KernelHandle):
            if kernel._session is not self:
                raise ValueError("kernel_mode() step uses a kernel from another session")
            kernel_index = kernel._index
        elif isinstance(kernel, int):
            kernel_index = kernel
        else:
            raise TypeError(
                "kernel_mode() step 'kernel' must be a handle or kernel index")

        if kernel_index < 0 or kernel_index >= self.kernel_count:
            raise IndexError(f"kernel index out of range: {kernel_index}")
        if step.get('params') is not None and step.get('extra_params') is not None:
            raise ValueError("kernel_mode() step cannot specify both params and extra_params")

        return {
            'kernel_index': kernel_index,
            'n': step.get('n', default_n),
            'grid': step.get('grid'),
            'block': step.get('block'),
            'smem': step.get('smem'),
            'params': step.get('params'),
            'extra_params': step.get('extra_params'),
            'sync': bool(step.get('sync', False)),
            'clear_outputs': bool(step.get('clear_outputs', default_clear_outputs)),
        }

    def _current_outputs(self, out_specs):
        """Wrap the current persistent output mappings as tensors."""
        if (self._cached_out_tensors is not None
                and self._cached_out_specs == out_specs):
            return self._cached_out_tensors

        chunks = self._vmm_chunks
        num_inputs = self._persistent_num_inputs
        outs = []
        for i, (dt, on) in enumerate(out_specs):
            ptr = chunks[num_inputs + i][2]
            outs.append(_vmm_to_tensor(ptr, on, dt))
        result = tuple(outs)
        self._cached_out_tensors = result
        self._cached_out_specs = out_specs
        return result

    def _clear_output_cache(self):
        """Drop cached tensor wrappers for worker-owned output mappings."""
        self._cached_out_tensors = None
        self._cached_out_specs = None

    def _invalidate_persistent_state(self):
        """Drop Python-side state derived from persistent worker mappings."""
        self._persistent_setup = False
        self._persistent_scratch_mib = 0
        self._persistent_num_inputs = 0
        self._persistent_num_outputs = 0
        self._persistent_input_ptrs = ()
        self._worker_vas = ()
        self._scratch_va = 0
        self._clear_output_cache()

    def _invalidate_worker_cache(self):
        """Forget which cubins/functions the worker is assumed to cache."""
        self._sender_cache.clear()
        self._last_cubin_hash = None

    def _close_connection(self):
        """Close the persistent worker socket and drop unsent requests."""
        self._send_queue.clear()
        self._recv_queue.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _handle_worker_status_error(self, status, message):
        """Reset cached state after a non-zero worker status."""
        self._invalidate_persistent_state()
        self._invalidate_worker_cache()
        self._close_connection()
        raise RuntimeError(f"{message} (status={status})")

    def _check_worker_results(self, results, message):
        """Raise if any queued worker request failed."""
        for result in results:
            if result[0] != 0:
                self._handle_worker_status_error(result[0], message)
        return results

    # ── Worker daemon lifecycle ──────────────────────────────────────

    def _start_worker(self):
        """Spawn a private kbox_worker_daemon subprocess."""
        daemon_bin = _find_worker_daemon()
        sock_path = f"/tmp/kbox_session_{os.getpid()}_{uuid.uuid4().hex[:12]}.sock"
        # Clean up stale socket
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        self._worker_proc = subprocess.Popen(
            [daemon_bin, '--sock', sock_path, '--idle-timeout', '3600'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._worker_sock_path = sock_path

        # Wait for socket to appear (up to 5s)
        for _ in range(100):
            if os.path.exists(sock_path):
                return
            time.sleep(0.05)
        stderr = self._worker_proc.stderr.read().decode()
        truncated = f"{stderr[:500]}..." if len(stderr) > 500 else stderr
        raise RuntimeError(
            f"Worker daemon failed to create socket at {sock_path}\n"
            f"  stderr: {truncated}")

    def _stop_worker(self, force=False):
        """Send shutdown sentinel and clean up worker daemon.

        Args:
            force: If True, skip graceful shutdown and SIGKILL immediately.
                   Used after timeouts when the worker is likely hung.
        """
        if self._worker_proc is None:
            return
        try:
            if self._worker_proc.poll() is None:
                if force:
                    self._worker_proc.kill()
                    self._worker_proc.wait()
                else:
                    # Try graceful shutdown via sentinel on persistent socket
                    try:
                        cfg = struct.pack(_WORKER_CONFIG_FMT,
                                          0,  # dtype
                                          _WORKER_SHUTDOWN_N,  # n
                                          0, 0, 0,  # is_cubin, func_name_len, kernel_data_len
                                          0, 0, 0,  # num_inputs, num_outputs, timeout_ms
                                          0, 0, 0,  # grid
                                          0, 0, 0,  # block
                                          0, 0, 0, 0,  # smem, _reserved[3]
                                          0, 0,  # request_type, flags
                                          0, 0,  # param_buffer_len, scratch_mib
                                          0, 0,  # scratch_zero_offset, scratch_zero_bytes
                                          0)  # chunk_size
                        sock = self._sock or socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        if not self._sock:
                            sock.settimeout(2.0)
                            sock.connect(self._worker_sock_path)
                        _ipc_send_all(sock, cfg)
                        sock.close()
                        self._sock = None
                    except Exception as e:
                        _log(f"Warning: graceful shutdown failed: {type(e).__name__}: {e}", _DIM)
                    self._worker_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _log("Warning: worker shutdown timed out, killing", _DIM)
            self._worker_proc.kill()
            self._worker_proc.wait()
        except Exception as e:
            _log(f"Warning: worker cleanup error: {type(e).__name__}: {e}", _DIM)
        finally:
            if self._worker_sock_path and os.path.exists(self._worker_sock_path):
                try:
                    os.unlink(self._worker_sock_path)
                except OSError:
                    pass
            self._close_connection()
            self._worker_proc = None
            self._worker_sock_path = None
            self._invalidate_persistent_state()
            self._invalidate_worker_cache()

    def _ensure_worker(self):
        """Ensure worker daemon is alive, respawning if needed."""
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            if self._worker_sock_path and os.path.exists(self._worker_sock_path):
                return
        # Worker dead or never started — (re)spawn
        if self._worker_proc is not None:
            force = self._worker_proc.poll() is None
            self._stop_worker(force=force)
        self._start_worker()

    def __del__(self):
        try:
            self._stop_worker()
            self._free_vmm_chunks()
        except (OSError, TypeError, AttributeError, ImportError):
            # TypeError/AttributeError: globals may be None during shutdown
            # ImportError: sys.meta_path is None during shutdown
            # OSError: worker process or IPC socket already gone
            pass

    # ── VMM chunk management ─────────────────────────────────────────

    def _get_vmm_device(self):
        """Get CUDA device handle for VMM operations."""
        if self._vmm_device is not None:
            return self._vmm_device
        from cuda.bindings import driver as cu
        err, device = cu.cuCtxGetDevice()
        from . import vmm
        vmm._check(err, "cuCtxGetDevice")
        self._vmm_device = device
        self._vmm_granularity = vmm.get_granularity(device)
        return device

    def _ensure_vmm_chunks(self, count, chunk_size):
        """Allocate or reuse VMM chunks. Returns list of (handle, size, mapped_ptr)."""
        if (len(self._vmm_chunks) == count
                and self._vmm_chunk_size == chunk_size):
            return self._vmm_chunks

        self._free_vmm_chunks()

        from cuda.bindings import driver as cu
        from . import vmm

        device = self._get_vmm_device()
        prop = vmm._alloc_prop(device)

        chunks = []
        try:
            for _ in range(count):
                err, handle = cu.cuMemCreate(chunk_size, prop, 0)
                vmm._check(err, "cuMemCreate")
                ptr = vmm._map_handle(device, handle, chunk_size,
                                      self._vmm_granularity)
                chunks.append((handle, chunk_size, ptr))
        except Exception as e:
            # Clean up partially allocated chunks before re-raising
            for h, sz, p in chunks:
                if p:
                    cu.cuMemUnmap(p, sz)
                    cu.cuMemAddressFree(p, sz)
                cu.cuMemRelease(h)
            raise RuntimeError(
                f"GPU memory allocation failed ({count} x {chunk_size} bytes): "
                f"{type(e).__name__}: {e}\n"
                f"  Check GPU memory with: nvidia-smi") from e

        self._vmm_chunks = chunks
        self._vmm_chunk_size = chunk_size
        return chunks

    def _free_vmm_chunks(self):
        """Release all VMM chunks."""
        if not self._vmm_chunks:
            self._clear_output_cache()
            return
        from cuda.bindings import driver as cu
        for handle, size, ptr in self._vmm_chunks:
            if ptr:
                cu.cuMemUnmap(ptr, size)
                cu.cuMemAddressFree(ptr, size)
            cu.cuMemRelease(handle)
        self._vmm_chunks = []
        self._vmm_chunk_size = 0
        self._clear_output_cache()

    # ── Worker dispatch ──────────────────────────────────────────────

    def _compute_vmm_layout(self, input_tensors, out_specs):
        """Compute VMM chunk layout for inputs + outputs. Returns (chunks, chunk_size, num_inputs, num_outputs)."""
        from . import vmm

        num_inputs = len(input_tensors)
        num_outputs = len(out_specs)
        total = num_inputs + num_outputs

        device = self._get_vmm_device()
        buf_sizes = []
        for t in input_tensors:
            buf_sizes.append(t.nelement() * t.element_size())
        for dt, on in out_specs:
            elem_sz = _DTYPE_ELEM_SIZE.get(_DTYPE_CODE.get(dt, 1), 4)
            buf_sizes.append(on * elem_sz)

        max_buf = max(buf_sizes) if buf_sizes else 4096
        chunk_size = vmm.round_up(max(max_buf, self._vmm_granularity),
                                  self._vmm_granularity)
        chunks = self._ensure_vmm_chunks(total, chunk_size)
        return chunks, chunk_size, num_inputs, num_outputs

    def _export_vmm_fds(self, chunks):
        """Export VMM handles as POSIX fds. Returns list of fd ints."""
        from cuda.bindings import driver as cu
        from . import vmm
        fds = []
        for handle, _size, _ptr in chunks:
            err, fd = cu.cuMemExportToShareableHandle(
                handle,
                cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
            vmm._check(err, "cuMemExportToShareableHandle")
            fds.append(int(fd))
        return fds

    @staticmethod
    def _close_fds(fds):
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass

    @staticmethod
    def _pack_grid_block(grid, block):
        if isinstance(block, int):
            bx, by, bz = block, 1, 1
        elif block:
            bx = block[0]
            by = block[1] if len(block) > 1 else 1
            bz = block[2] if len(block) > 2 else 1
        else:
            bx, by, bz = 0, 0, 0
        if isinstance(grid, int):
            gx, gy, gz = grid, 1, 1
        elif grid:
            gx = grid[0]
            gy = grid[1] if len(grid) > 1 else 1
            gz = grid[2] if len(grid) > 2 else 1
        else:
            gx, gy, gz = 0, 0, 0
        return gx, gy, gz, bx, by, bz

    def _ensure_connection(self):
        """Ensure persistent socket connection to worker."""
        if self._sock is not None:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._close_connection()
            else:
                return
        self._ensure_worker()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(self._worker_sock_path)
        self._sock = sock

    _QUEUE_FLUSH_THRESHOLD = 262144  # auto-flush when send queue exceeds this

    def _queue_request(self, cfg_bytes, extra_data=None, recv_extra=0):
        """Queue a request for batched sending. No FD passing support.
        Auto-flushes if queue exceeds threshold to avoid socket buffer deadlock.
        Returns results from an internal auto-flush, if one happened."""
        auto_results = []
        self._send_queue.extend(cfg_bytes)
        if extra_data:
            for data in extra_data:
                self._send_queue.extend(data)
        self._recv_queue.append(recv_extra)
        # Auto-flush to prevent deadlock: if send queue is too large,
        # the worker's response buffer may fill up while we're still sending.
        if len(self._send_queue) > self._QUEUE_FLUSH_THRESHOLD:
            auto_results = self.flush()
        return auto_results

    def flush(self):
        """Send all queued requests in one write, read all responses back.
        Returns list of (status, elapsed_ms) tuples, with extra bytes where requested."""
        if not self._recv_queue:
            return []
        self._ensure_connection()
        sock = self._sock
        send_buf = bytes(self._send_queue)
        recv_sizes = list(self._recv_queue)
        self._send_queue.clear()
        self._recv_queue.clear()
        try:
            _ipc_send_all(sock, send_buf)
            n_reqs = len(recv_sizes)
            has_extra = any(r > 0 for r in recv_sizes)
            if not has_extra:
                # Fast path: read all responses in one syscall
                all_resp = _ipc_recv_all(sock, n_reqs * 8)
                results = []
                for i in range(n_reqs):
                    status, elapsed_ms = struct.unpack_from('<If', all_resp, i * 8)
                    results.append((status, elapsed_ms))
                return results
            results = []
            for recv_extra in recv_sizes:
                resp = _ipc_recv_all(sock, 8)
                status, elapsed_ms = struct.unpack('<If', resp)
                if recv_extra > 0:
                    extra = _ipc_recv_all(sock, recv_extra)
                    results.append((status, elapsed_ms, extra))
                else:
                    results.append((status, elapsed_ms))
            return results
        except socket.timeout:
            _log(f"Worker timed out after {self._timeout:.1f}s, killing and respawning", _YELLOW)
            self._stop_worker(force=True)
            raise WorkerTimeoutError(
                f"Kernel execution timed out after {self._timeout:.1f}s. "
                f"Worker has been killed and will respawn on next call.")
        except OSError:
            self._invalidate_persistent_state()
            self._close_connection()
            raise

    def _worker_send_recv(self, cfg_bytes, extra_data=None, fds=None,
                          recv_extra=0):
        """Send config + optional data + optional fds on persistent socket, receive status.

        Returns (status, elapsed_ms) or (status, elapsed_ms, extra_bytes) if recv_extra > 0.
        Raises WorkerTimeoutError on socket timeout.
        """
        # Flush any queued requests first
        if self._recv_queue:
            self._check_worker_results(
                self.flush(), "Worker request failed while draining the queue")
        unsent_fds = list(fds) if fds else []
        self._ensure_connection()
        sock = self._sock
        try:
            _ipc_send_all(sock, cfg_bytes)
            if extra_data:
                for data in extra_data:
                    _ipc_send_all(sock, data)
            for fd in unsent_fds:
                _ipc_send_fd(sock, fd)
                os.close(fd)
            unsent_fds = []
            resp = _ipc_recv_all(sock, 8)
            status, elapsed_ms = struct.unpack('<If', resp)
            if recv_extra > 0:
                extra = _ipc_recv_all(sock, recv_extra)
                return status, elapsed_ms, extra
            return status, elapsed_ms
        except socket.timeout:
            self._close_fds(unsent_fds)
            _log(f"Worker timed out after {self._timeout:.1f}s, killing and respawning", _YELLOW)
            self._stop_worker(force=True)
            raise WorkerTimeoutError(
                f"Kernel execution timed out after {self._timeout:.1f}s. "
                f"Worker has been killed and will respawn on next call.")
        except OSError:
            self._close_fds(unsent_fds)
            self._invalidate_persistent_state()
            self._close_connection()
            raise
        except Exception:
            self._close_fds(unsent_fds)
            raise

    def _setup_persistent(self, input_tensors, out_specs, n, scratch_mib=None):
        """Send SETUP request: export VMM fds, worker imports and keeps them mapped."""
        from . import vmm

        if scratch_mib is None:
            scratch_mib = self._kernel_scratch_mib

        chunks, chunk_size, num_inputs, num_outputs = \
            self._compute_vmm_layout(input_tensors, out_specs)

        # DtoD copy input tensors into VMM chunks
        for i, t in enumerate(input_tensors):
            nbytes = t.nelement() * t.element_size()
            vmm.memcpy_dtod(chunks[i][2], t.data_ptr(), nbytes)

        fds = self._export_vmm_fds(chunks)

        if input_tensors:
            primary_dtype = input_tensors[0].dtype
        else:
            primary_dtype = out_specs[0][0] if out_specs else torch.float32
        dtype_code = _DTYPE_CODE.get(primary_dtype, 1)

        cfg = struct.pack(_WORKER_CONFIG_FMT,
                          dtype_code, n,
                          0, 0, 0,  # is_cubin, func_name_len, kernel_data_len
                          num_inputs, num_outputs, 0,  # timeout_ms
                          0, 0, 0,  # grid
                          0, 0, 0,  # block
                          0, 0, 0, 0,  # smem, _reserved[3]
                          _WORKER_REQ_SETUP, 0,  # request_type, flags
                          0, scratch_mib,  # param_buffer_len, scratch_mib
                          0, 0,  # scratch_zero_offset, scratch_zero_bytes
                          chunk_size)

        total = num_inputs + num_outputs
        va_bytes = total * 8 + 8  # input/output VAs + scratch VA
        status, _, va_data = self._worker_send_recv(cfg, fds=fds,
                                                     recv_extra=va_bytes)
        if status != 0:
            self._handle_worker_status_error(status, "Worker SETUP failed")
        # Parse worker-side VAs (for building param buffers) + scratch VA
        all_vas = struct.unpack(f'<{total + 1}Q', va_data)
        worker_vas = all_vas[:total]
        self._scratch_va = all_vas[total]
        self._persistent_setup = True
        self._persistent_scratch_mib = scratch_mib
        self._persistent_num_inputs = num_inputs
        self._persistent_num_outputs = num_outputs
        self._persistent_input_ptrs = tuple(t.data_ptr() for t in input_tensors)
        self._worker_vas = worker_vas  # tuple of CUdeviceptr (uint64)

    def _ensure_persistent_layout(self, input_tensors, out_specs, n, scratch_mib=None):
        """Ensure worker mappings exist for the given I/O layout."""
        from . import vmm

        if scratch_mib is None:
            scratch_mib = self._kernel_scratch_mib

        self._ensure_worker()
        if (self._persistent_setup
                and scratch_mib > self._persistent_scratch_mib):
            self._release_persistent()

        if not self._persistent_setup:
            self._setup_persistent(input_tensors, out_specs, n,
                                   scratch_mib=scratch_mib)
            return

        input_ptrs = tuple(t.data_ptr() for t in input_tensors)
        if input_ptrs == self._persistent_input_ptrs:
            return

        ni, no = len(input_tensors), len(out_specs)
        fits = (ni == self._persistent_num_inputs
                and no == self._persistent_num_outputs
                and all(t.nelement() * t.element_size() <= self._vmm_chunk_size
                        for t in input_tensors))
        if not fits:
            self._release_persistent()
            self._setup_persistent(input_tensors, out_specs, n,
                                   scratch_mib=scratch_mib)
            return

        for i, t in enumerate(input_tensors):
            vmm.memcpy_dtod(self._vmm_chunks[i][2],
                            t.data_ptr(),
                            t.nelement() * t.element_size())
        self._persistent_input_ptrs = input_ptrs

    def _run_persistent(self, out_specs, n, grid=None, block=None, smem=0,
                        params=None, sync=True, clear_outputs=True):
        """Send RUN request: cubin only, worker uses existing persistent mappings.

        Args:
            params: If provided, builds a custom param buffer for cuLaunchKernel.
                    When None, uses legacy arg layout (in0, ..., out0, ..., n).
            sync: If False, launch without waiting (WORKER_FLAG_NO_SYNC).

        Returns tuple of output tensors.
        """
        from . import vmm

        if not self._persistent_setup:
            raise RuntimeError("Cannot RUN without prior SETUP")

        func_name_bytes = self.func_name.encode('utf-8')
        cubin_data = self._cubin

        # For Triton, apply compiled metadata defaults (set during _compile_triton,
        # which runs after __call__ resolves session defaults but before this point)
        if self._triton_mode:
            if block is None or block == 0:
                block = self._triton_num_warps * 32
            if smem is None or smem == 0:
                smem = self._triton_shared

        gx, gy, gz, bx, by, bz = self._pack_grid_block(grid, block)

        primary_dtype = out_specs[0][0] if out_specs else torch.float32
        dtype_code = _DTYPE_CODE.get(primary_dtype, 1)

        # Worker handles memset; skip events and health check for lower overhead
        flags = _WORKER_FLAG_NO_EVENTS | _WORKER_FLAG_NO_HEALTH
        if not clear_outputs:
            flags |= _WORKER_FLAG_NO_MEMSET
        if not sync:
            flags |= _WORKER_FLAG_NO_SYNC

        # Pass scratch as first kernel arg in legacy mode when requested
        if self._kernel_scratch_mib > 0 and params is None:
            flags |= _WORKER_FLAG_PASS_SCRATCH

        # Build param buffer if custom params provided
        param_buf = b''
        use_kernel_params = False
        if params is not None:
            if self._triton_mode:
                # Coerce scalar params to match Triton signature types
                triton_types = getattr(self, '_triton_param_types', None)
                if triton_types:
                    params = _coerce_triton_params(params, triton_types)
                # Triton PTX uses .align 1 on pointer params → tight packing
                param_buf = _build_param_buffer(
                    params, self._worker_vas, self._persistent_num_inputs,
                    ptr_align=1, scratch_va=self._scratch_va)
                # Append 2 scratch pointers (global + profile scratch)
                global_va = 0
                profile_va = 0
                if self._triton_global_scratch_size > 0:
                    global_va = self._scratch_va + self._triton_scratch_offset
                if self._triton_profile_scratch_size > 0:
                    profile_va = (self._scratch_va + self._triton_scratch_offset
                                  + self._triton_global_scratch_size)
                param_buf += struct.pack('<QQ', global_va, profile_va)
            elif _has_tma_params(params):
                # TMA descriptors require kernelParams (void**) launch —
                # CU_LAUNCH_PARAM_BUFFER_POINTER doesn't work with TMA.
                param_buf = _build_kernel_params_buffer(
                    params, self._worker_vas, self._persistent_num_inputs,
                    scratch_va=self._scratch_va)
                use_kernel_params = True
            else:
                param_buf = _build_param_buffer(
                    params, self._worker_vas, self._persistent_num_inputs,
                    scratch_va=self._scratch_va)
            # Auto-compute grid from n when not explicitly specified (worker
            # can't derive grid from the param buffer since n is opaque)
            if gx == 0:
                block_x = bx if bx > 0 else 256
                gx = (n + block_x - 1) // block_x
                if gy == 0: gy = 1
                if gz == 0: gz = 1
                if bx == 0: bx = 256; by = 1; bz = 1
        if use_kernel_params:
            flags |= _WORKER_FLAG_KERNEL_PARAMS

        # Compute Triton scratch zeroing range
        triton_scratch_bytes = (self._triton_global_scratch_size
                                + self._triton_profile_scratch_size)
        scratch_zero_offset = self._triton_scratch_offset if triton_scratch_bytes > 0 else 0
        scratch_zero_bytes = triton_scratch_bytes

        # Determine whether to send full cubin or just hash
        # Compute hash lazily — only when cubin object changes
        if self._cubin_hash is None or id(cubin_data) != getattr(self, '_cubin_id', None):
            self._cubin_hash = _fnv1a(cubin_data)
            self._cubin_id = id(cubin_data)
        cubin_hash = self._cubin_hash
        if cubin_hash == self._last_cubin_hash:
            # Fastest path: worker uses g_last_func, no func_name or cubin needed
            send_func_name = b''
            send_cubin = b''
            hash_lo, hash_hi = 0, 0
        elif self._sender_cache.contains(cubin_hash):
            # Hash-only path: worker looks up by hash in its cache
            send_func_name = func_name_bytes
            send_cubin = b''
            hash_lo = cubin_hash & 0xFFFFFFFF
            hash_hi = (cubin_hash >> 32) & 0xFFFFFFFF
        else:
            # Full path: send cubin, worker caches it
            send_func_name = func_name_bytes
            send_cubin = cubin_data
            hash_lo, hash_hi = 0, 0
            self._sender_cache.insert(cubin_hash)

        self._last_cubin_hash = cubin_hash

        cfg = struct.pack(_WORKER_CONFIG_FMT,
                          dtype_code, n,
                          1 if self._is_cubin else 0,
                          len(send_func_name), len(send_cubin),
                          self._persistent_num_inputs,
                          self._persistent_num_outputs,
                          0,  # timeout_ms
                          gx, gy, gz, bx, by, bz,
                          smem or 0, hash_lo, hash_hi, 0,  # smem, _reserved[3]
                          _WORKER_REQ_RUN, flags,
                          len(param_buf), self._kernel_scratch_mib,
                          scratch_zero_offset, scratch_zero_bytes,
                          self._vmm_chunk_size)

        extra = []
        if send_func_name:
            extra.append(send_func_name)
        if send_cubin:
            extra.append(send_cubin)
        if param_buf:
            extra.append(param_buf)

        if not sync:
            # Queue async kernel launch — will be flushed on next sync/end_timing
            queued_results = self._queue_request(cfg, extra_data=extra)
            self._check_worker_results(
                queued_results, "Kernel execution failed in worker")
            status, elapsed_ms = 0, 0.0
        else:
            status, elapsed_ms = self._worker_send_recv(cfg, extra_data=extra)

        if status != 0:
            self._handle_worker_status_error(
                status, "Kernel execution failed in worker")

        self._launch_ms = elapsed_ms

        return self._current_outputs(out_specs)

    def _release_persistent(self):
        """Send RELEASE request to worker and free Python-side VMM chunks."""
        if not self._persistent_setup:
            return
        try:
            cfg = struct.pack(_WORKER_CONFIG_FMT,
                              0, 0, 0, 0, 0,  # dtype, n, is_cubin, func_name_len, kernel_data_len
                              0, 0, 0,  # num_inputs, num_outputs, timeout_ms
                              0, 0, 0, 0, 0, 0,  # grid, block
                              0, 0, 0, 0,  # smem, _reserved[3]
                              _WORKER_REQ_RELEASE, 0,  # request_type, flags
                              0, 0,  # param_buffer_len, scratch_mib
                              0, 0,  # scratch_zero_offset, scratch_zero_bytes
                              0)  # chunk_size
            self._worker_send_recv(cfg)
        except Exception as e:
            _log(f"Warning: RELEASE failed: {e}", _DIM)
        self._invalidate_persistent_state()
        self._free_vmm_chunks()

    def _dispatch_to_worker(self, input_tensors, out_specs, n,
                            grid=None, block=None, smem=0,
                            params=None, sync=True, clear_outputs=True):
        """Dispatch kernel execution to worker daemon via persistent VMM IPC.

        Uses SETUP/RUN protocol: VMM mappings are kept alive across calls.
        Re-SETUPs automatically when input/output layout changes.

        Args:
            params: Custom kernel parameter list (see __call__ docstring).
            sync: If False, launch without waiting for completion.

        Returns tuple of output tensors.
        """
        self._recompile_cubin(input_tensors=input_tensors,
                              out_specs=out_specs, params=params)
        self._ensure_persistent_layout(input_tensors, out_specs, n)
        return self._run_persistent(out_specs, n, grid, block, smem,
                                    params=params, sync=sync,
                                    clear_outputs=clear_outputs)

    # ── Public API ───────────────────────────────────────────────────

    def _launch_active(self, *inputs, n=None, grid=None, block=None, smem=None,
                       out_dtype=None, params=None, extra_params=None,
                       sync=True, clear_outputs=True):
        """Launch the kernel with the given input tensors.

        Kernel runs in an isolated worker daemon process for crash safety.

        Args:
            *inputs:   CUDA tensors (one per kernel input pointer).
            n:         Element count (default: first input's numel()).
            grid:      Grid dims — int or tuple (None = session default or auto).
            block:     Block dims — int or tuple (None = session default or 256).
            smem:      Dynamic shared memory bytes (None = session default or 0).
            out_dtype: Override output dtype(s) when outputs was set as int.
                       Ignored when outputs is a list of specs.
            params:    Fully custom kernel parameter list (requires persistent mode).
                       When provided, the kernel arg layout is entirely user-controlled.
                       Mutually exclusive with ``extra_params``.
            extra_params:
                       Extra parameters appended after the standard layout
                       ``(in0, ..., out0, ..., n, <extra_params>)``.
                       Use this when your kernel has the standard convention plus
                       additional scalars. Mutually exclusive with ``params``.

                       Supported types for both params/extra_params:
                       - ``kernel.in_ptr(i)``: worker VA for input chunk i
                       - ``kernel.out_ptr(i)``: worker VA for output chunk i
                       - ``np.uint32(val)``, ``np.float32(val)``, etc.: numpy scalars
                       - ``('I', val)``: struct.pack-style (format_char, value) tuple
            sync:      If True (default), wait for kernel completion before returning.
                       If False, launch asynchronously — use sync() to wait later.
                       Outputs are only valid after sync when sync=False.
            clear_outputs:
                       If True (default), zero session output buffers before launch.
                       Set False to preserve previous outputs across back-to-back
                       launches in the same session.

        Returns:
            Single output tensor, or tuple of output tensors if outputs > 1.
        """
        inputs, out_specs, n, grid, block, smem, params = \
            self._resolve_inputs_and_outputs(
                inputs, n=n, out_dtype=out_dtype, grid=grid, block=block,
                smem=smem, params=params, extra_params=extra_params)

        t0 = time.monotonic()
        outs = self._dispatch_to_worker(
            list(inputs), out_specs, n, grid, block, smem,
            params=params, sync=sync, clear_outputs=clear_outputs)
        self._launch_ms = (time.monotonic() - t0) * 1000

        return outs[0] if len(outs) == 1 else outs

    def __call__(self, *inputs, n=None, grid=None, block=None, smem=None,
                 out_dtype=None, params=None, extra_params=None, sync=True,
                 clear_outputs=True):
        if self.kernel_count != 1:
            raise RuntimeError(
                "This session contains multiple kernels. Use session.kernels[i](...) "
                "or session.run_steps(...).")
        return self._launch_active(
            *inputs, n=n, grid=grid, block=block, smem=smem,
            out_dtype=out_dtype, params=params, extra_params=extra_params,
            sync=sync, clear_outputs=clear_outputs)

    def _prepare_step_sequence(self, steps, inputs, n=None, out_dtype=None):
        """Normalize a multi-kernel launch plan and compile its kernels."""
        inputs, out_specs, default_n, _, _, _, _ = self._resolve_inputs_and_outputs(
            inputs, n=n, out_dtype=out_dtype)
        if not steps:
            raise ValueError("kernel_mode() must return at least one step")

        normalized = []
        max_scratch_mib = self._user_scratch_mib
        for i, step in enumerate(steps):
            info = self._normalize_step_spec(
                step, default_n=default_n, default_clear_outputs=(i == 0))
            self._select_kernel(info['kernel_index'])
            params = info['params']
            if info['extra_params'] is not None:
                _, _, _, _, _, _, params = self._resolve_inputs_and_outputs(
                    inputs, n=info['n'], out_dtype=out_dtype,
                    params=None, extra_params=info['extra_params'])
                info['params'] = params
                info['extra_params'] = None
            self._recompile_cubin(input_tensors=list(inputs),
                                  out_specs=out_specs, params=params)
            max_scratch_mib = max(max_scratch_mib, self._kernel_scratch_mib)
            normalized.append(info)

        return inputs, out_specs, normalized, max_scratch_mib

    def run_steps(self, steps, *inputs, n=None, out_dtype=None, sync=True):
        """Execute a precomputed sequence of kernel launches."""
        inputs, out_specs, normalized, max_scratch_mib = \
            self._prepare_step_sequence(steps, inputs, n=n, out_dtype=out_dtype)
        self._ensure_persistent_layout(list(inputs), out_specs, normalized[0]['n'],
                                       scratch_mib=max_scratch_mib)

        result = None
        for i, step in enumerate(normalized):
            self._select_kernel(step['kernel_index'])
            step_sync = step['sync'] or (sync and i == len(normalized) - 1)
            result = self._run_persistent(
                out_specs, step['n'], grid=step['grid'], block=step['block'],
                smem=step['smem'], params=step['params'], sync=step_sync,
                clear_outputs=step['clear_outputs'])

        if sync and normalized and not normalized[-1]['sync']:
            self.sync()
            result = self._current_outputs(out_specs)
        elif result is None:
            result = self._current_outputs(out_specs)
        return result[0] if len(result) == 1 else result

    def benchmark_steps(self, steps, *inputs, warmup=10, iters=100, n=None,
                        out_dtype=None, l2_flush=1, l2_flush_per_iter=1,
                        l2_dirty=False, wall_time=False, benchtime=None):
        """Benchmark a precomputed multi-kernel launch plan."""
        if iters is None and benchtime is None:
            iters = 100
        inputs, out_specs, normalized, max_scratch_mib = \
            self._prepare_step_sequence(steps, inputs, n=n, out_dtype=out_dtype)
        self._ensure_persistent_layout(list(inputs), out_specs, normalized[0]['n'],
                                       scratch_mib=max_scratch_mib)

        def _launch_once():
            for step in normalized:
                self._select_kernel(step['kernel_index'])
                self._run_persistent(
                    out_specs, step['n'], grid=step['grid'], block=step['block'],
                    smem=step['smem'], params=step['params'], sync=False,
                    clear_outputs=step['clear_outputs'])

        for _ in range(warmup):
            _launch_once()
            self.sync()

        if l2_flush:
            self.l2_flush(count=l2_flush)

        gpu_times = []
        wall_times = []
        t_wall_start = time.monotonic()
        i = 0
        while True:
            if l2_flush_per_iter:
                self.l2_flush(count=l2_flush_per_iter)
            if wall_time:
                self.sync()
                t0 = time.monotonic()
            self.start_timing()
            _launch_once()
            elapsed = self.end_timing(sync=True)
            gpu_times.append(elapsed)
            if wall_time:
                wall_times.append((time.monotonic() - t0) * 1000)
            i += 1
            if iters is not None and i >= iters:
                break
            if benchtime is not None and (time.monotonic() - t_wall_start) >= benchtime:
                break

        return _format_bench_result(gpu_times, wall_times, wall_time)

    def _make_lightweight_cfg(self, request_type, flags=0, timeout_ms=0):
        """Build a lightweight config packet (no cubin/params)."""
        return struct.pack(_WORKER_CONFIG_FMT,
                           0, 0, 0, 0, 0, 0, 0,  # dtype..num_outputs
                           timeout_ms,
                           0, 0, 0, 0, 0, 0,  # grid, block
                           0, 0, 0, 0,  # smem, _reserved[3]
                           request_type, flags,
                           0, 0,  # param_buffer_len, scratch_mib
                           0, 0,  # scratch_zero_offset, scratch_zero_bytes
                           0)  # chunk_size

    def _send_lightweight_req(self, request_type, flags=0, timeout_ms=0):
        """Send a lightweight worker request (no cubin/params). Returns (status, elapsed_ms)."""
        return self._worker_send_recv(
            self._make_lightweight_cfg(request_type, flags, timeout_ms))

    def _queue_lightweight_req(self, request_type, flags=0, timeout_ms=0):
        """Queue a lightweight worker request for batched sending."""
        return self._queue_request(
            self._make_lightweight_cfg(request_type, flags, timeout_ms))

    def sync(self):
        """Sync worker stream. Auto-flushes queued requests first.
        Returns elapsed_ms if timing was active."""
        # Queue the sync request, then flush everything at once
        results = self._queue_lightweight_req(
            _WORKER_REQ_SYNC, timeout_ms=int(self._timeout * 1000))
        results.extend(self.flush())
        if not results:
            return 0.0
        self._check_worker_results(results, "Worker sync failed")
        status, elapsed_ms = results[-1]  # sync is the last queued request
        return elapsed_ms

    def noop(self, payload_bytes=0):
        """Send a no-op request to the worker. Pure IPC round-trip, zero GPU work.
        If payload_bytes > 0, sends that many extra bytes after the config."""
        if payload_bytes <= 0:
            self._send_lightweight_req(_WORKER_REQ_NOOP)
        else:
            cfg = struct.pack(_WORKER_CONFIG_FMT,
                              0, payload_bytes, 0, 0, 0, 0, 0, 0,  # dtype, n=payload, ...
                              0, 0, 0, 0, 0, 0,  # grid, block
                              0, 0, 0, 0,  # smem, _reserved[3]
                              _WORKER_REQ_NOOP, 0,
                              0, 0, 0, 0, 0)  # param_buffer_len..chunk_size
            payload = b'\x00' * payload_bytes
            self._worker_send_recv(cfg, extra_data=[payload])

    def start_timing(self, sync=False):
        """Record start CUDA event on worker stream. If sync=True, drain stream first.
        Queued for batched sending (no response needed)."""
        flags = _WORKER_FLAG_SYNC if sync else 0
        timeout_ms = int(self._timeout * 1000) if sync else 0
        self._check_worker_results(
            self._queue_lightweight_req(_WORKER_REQ_START_TIMING, flags, timeout_ms),
            "Worker start_timing failed")

    def end_timing(self, sync=True):
        """Record end CUDA event on worker stream. Returns elapsed_ms if sync=True.
        If sync=True, auto-flushes the queue."""
        flags = _WORKER_FLAG_SYNC if sync else 0
        timeout_ms = int(self._timeout * 1000) if sync else 0
        if sync:
            # Queue end_timing, flush everything, return elapsed from last response
            results = self._queue_lightweight_req(
                _WORKER_REQ_END_TIMING, flags, timeout_ms)
            results.extend(self.flush())
            if not results:
                return 0.0
            self._check_worker_results(results, "Worker end_timing failed")
            status, elapsed_ms = results[-1]
            return elapsed_ms
        else:
            self._check_worker_results(
                self._queue_lightweight_req(_WORKER_REQ_END_TIMING, flags, timeout_ms),
                "Worker end_timing failed")

    @property
    def scratch_va(self):
        """Worker-side virtual address of the scratch buffer (uint64).

        Returns 0 if SETUP hasn't happened yet (the first __call__ triggers
        SETUP, after which this returns the real worker VA).
        """
        return self._scratch_va

    def l2_flush(self, count=1, value=0, clean=False, clean_only=False):
        """Flush L2 cache using worker-side scratch buffer.

        Args:
            count:      Number of memset passes (each with different pattern,
                        final pass uses `value`).
            value:      u32 memset value for the final pass.
            clean:      After memset, read buffer to leave L2 non-dirty.
            clean_only: Skip memset, only read (clean existing L2 lines).
        """
        flags = 0
        if clean:
            flags |= _L2_FLUSH_CLEAN
        if clean_only:
            flags |= _L2_FLUSH_CLEAN_ONLY
        cfg = struct.pack(_WORKER_CONFIG_FMT,
                          0, count,  # dtype (unused), n = flush_count
                          0, 0, 0,  # is_cubin, func_name_len, kernel_data_len
                          0, 0, 0,  # num_inputs, num_outputs, timeout_ms
                          0, 0, 0,  # grid
                          0, 0, 0,  # block
                          value, 0, 0, 0,  # smem=memset_value, _reserved[3]
                          _WORKER_REQ_L2_FLUSH, flags,
                          0, 0,  # param_buffer_len, scratch_mib
                          0, 0,  # scratch_zero_offset, scratch_zero_bytes
                          0)  # chunk_size
        self._check_worker_results(
            self._queue_request(cfg), "Worker l2_flush failed")

    def zero_scratch(self):
        """Zero the scratch buffer (memset to 0)."""
        self.l2_flush(count=1, value=0)

    def run(self, *inputs, n=None, ref=None, atol=1e-5, rtol=1e-5,
            quiet=False, **kw):
        """Run the kernel once with optional verification and output.

        Returns:
            Output tensor(s).
        """
        if self.kernel_count != 1:
            raise RuntimeError(
                "This session contains multiple kernels. Use session.kernels[i](...) "
                "or session.run_steps(...).")
        recompiled = self._recompile_cubin()
        if recompiled and self._compile_ms > 0 and not quiet:
            _log(f"Compiled {os.path.basename(self.kernel_path)} "
                 f"in {self._compile_ms:.1f}ms", _CYAN)

        result = self(*inputs, n=n, **kw)
        outputs = result if isinstance(result, tuple) else (result,)

        if not quiet:
            nn = n or (inputs[0].numel() if inputs else "?")
            _log(f"Launched {self.func_name}: n={nn}, "
                 f"time={self._launch_ms:.2f}ms", _DIM)
            for i, out in enumerate(outputs):
                _log(f"  output[{i}]: {_preview(out)}", _DIM)

        if ref is not None:
            ok, msg = _verify(outputs, ref, list(inputs), atol=atol, rtol=rtol)
            if not quiet:
                style = _GREEN + _BOLD if ok else _RED + _BOLD
                symbol = "PASS" if ok else "FAIL"
                _log(f"{symbol}: {msg}", style)
            if not ok:
                raise AssertionError(f"Verification failed: {msg}")

        return result

    def watch(self, *inputs, ref=None, interval=0.3, n=None, atol=1e-5,
              rtol=1e-5, **kw):
        """Watch the kernel file and re-run on every change.

        Press Ctrl+C to stop.  Only watches the single kernel file.
        For multi-file watching (test files + kernels), use
        :func:`iterate` or :func:`watch_file` instead.

        Args:
            *inputs:  Input tensors (positional, like ``__call__``).
            ref:      Reference function or expected tensor(s).
            interval: File poll interval in seconds.
            n:        Element count.
        """
        if self.kernel_count != 1 or self.kernel_path is None:
            raise RuntimeError(
                "watch() only supports a single file-backed kernel session")
        _log(f"Watching {os.path.basename(self.kernel_path)} "
             f"(Ctrl+C to stop)", _YELLOW + _BOLD)
        print()
        self._watch_run(inputs, ref=ref, n=n, atol=atol, rtol=rtol, **kw)

        last_mt = os.path.getmtime(self.kernel_path)
        try:
            while True:
                time.sleep(interval)
                try:
                    mt = os.path.getmtime(self.kernel_path)
                except OSError:
                    continue
                if mt != last_mt:
                    last_mt = mt
                    print()
                    _log("File changed, recompiling...", _YELLOW)
                    self._watch_run(inputs, ref=ref, n=n, atol=atol,
                                    rtol=rtol, **kw)
        except KeyboardInterrupt:
            print()
            _log("Stopped.", _DIM)

    def _watch_run(self, inputs, **kw):
        try:
            self.run(*inputs, **kw)
        except Exception as e:
            _log(f"ERROR: {e}", _RED + _BOLD)

    def benchmark(self, *inputs, warmup=10, iters=100, n=None,
                  l2_flush=1, l2_flush_per_iter=1, l2_dirty=False,
                  wall_time=False, benchtime=None, **kw):
        """Benchmark kernel via worker-side CUDA event timing.

        Returns:
            dict with min_ms, max_ms, mean_ms, median_ms, iters (and wall_* if wall_time).
        """
        if self.kernel_count != 1:
            raise RuntimeError(
                "This session contains multiple kernels. Use session.kernels[i](...) "
                "or session.benchmark_steps(...).")
        # Warmup
        for _ in range(warmup):
            self(*inputs, sync=False, n=n, **kw)
        self.sync()

        # Pre-benchmark L2 flush (worker-side)
        if l2_flush:
            self.l2_flush(count=l2_flush)

        gpu_times = []
        wall_times = []
        t_wall_start = time.monotonic()

        i = 0
        while True:
            if l2_flush_per_iter:
                self.l2_flush(count=l2_flush_per_iter)
            if wall_time:
                self.sync()
                t0 = time.monotonic()
            self.start_timing()
            self(*inputs, sync=False, n=n, **kw)
            elapsed = self.end_timing(sync=True)
            gpu_times.append(elapsed)
            if wall_time:
                wall_times.append((time.monotonic() - t0) * 1000)
            i += 1
            if iters is not None and i >= iters:
                break
            if benchtime is not None and (time.monotonic() - t_wall_start) >= benchtime:
                break

        return _format_bench_result(gpu_times, wall_times, wall_time)


# ── Convenience function ────────────────────────────────────────────────

def dev(kernel_path, *inputs, outputs=1, n=None, dtype=None, out_dtype=None,
        ref=None, watch=False, func_name=None, grid=None, block=None,
        atol=1e-5, rtol=1e-5, seed=None, smem=0, defines=None, quiet=False):
    """Compile, run, and optionally verify a CUDA kernel.

    This is the main entry point for fast kernel iteration.
    Inputs can be PyTorch tensors, spec strings, or Python lists.

    Args:
        kernel_path: Path to .cu source file.
        *inputs:     Input data — tensors, spec strings, or Python lists.
        outputs:     Output specification:
                     - int: number of outputs (dtype/n inferred from inputs)
                     - list of specs: per-output control
                       "float32", "float16;n=1", "int32;n=4k"
        n:           Default element count (default: from first input, or 1024).
        dtype:       Default data type — "float32", "int32", or torch.dtype.
        out_dtype:   Shorthand when outputs is int: override output dtype(s).
                     Ignored when outputs is a list of specs.
        ref:         Reference for verification:
                     - callable(inputs) -> tuple of expected tensors
                     - single tensor or list of tensors
        watch:       Re-run on every file save (Ctrl+C to stop).
        func_name:   Override auto-detected function name.
        grid:        Grid dimensions.
        block:       Block dimensions.
        atol:        Absolute tolerance for verification.
        rtol:        Relative tolerance for verification.
        seed:        Random seed for reproducible inputs.
        smem:        Dynamic shared memory bytes.

    Returns:
        Output tensor (single output) or tuple of tensors.

    Input specs:
        "zeros"          All zeros
        "ones"           All ones
        "const:V"        Constant value V
        "seq"            0, 1, 2, 3, ...
        "seq:K"          K, K+1, K+2, ...
        "rand"           Uniform [0, 1)
        "rand:A:B"       Uniform [A, B)
        "randn"          Normal(0, 1)
        "randn:M:S"      Normal(mean=M, std=S)

    Per-input overrides (semicolons):
        Each input spec can override dtype/n for that specific input::

            "randn;dtype=float16"       float16 normals
            "rand:0:1;dtype=float;n=2k" float32, 2048 elements
            "seq;dtype=int32"           int32 sequential

    Output specs (when outputs is a list):
        Each output spec sets dtype and optionally element count::

            "float32"          float32, same n as inputs
            "float16;n=1"      float16, 1 element (e.g., scalar result)
            "int32;n=4k"       int32, 4096 elements

    Examples::

        # === Simple: add 1 to sequential integers ===
        out = dev("add_one.cu", "seq", n=8, dtype="int32")

        # === With verification ===
        dev("add_one.cu", "seq", n=1024, dtype="int32",
            ref=lambda inp: (inp[0] + 1,))

        # === Multiple inputs and outputs (count form) ===
        dev("fused_add_mul.cu", "randn", "randn", outputs=2, n=1024,
            ref=lambda inp: (inp[0]+inp[1], inp[0]*inp[1]))

        # === Per-output specs (list form) ===
        dev("fused_add_mul.cu", "randn", "randn",
            outputs=["float32", "float32"], n=1024)

        # === Mixed precision outputs ===
        dev("convert.cu", "randn;dtype=float16",
            outputs=["float32"])

        # === Reduction: full output + scalar sum ===
        dev("reduce.cu", "randn",
            outputs=["float32", "float32;n=1"])

        # === Different dtypes per input ===
        dev("blend.cu",
            "randn;dtype=float32",      # signal A
            "randn;dtype=float32",      # signal B
            "rand;dtype=float16",       # weights in half precision
            outputs=["float32"])

        # === Watch mode ===
        dev("add_one.cu", "seq", n=8, dtype="int32",
            ref=lambda inp: (inp[0]+1,), watch=True)

        # === Explicit tensors ===
        x = torch.randn(1024, device="cuda", dtype=torch.float16)
        y = torch.randn(1024, device="cuda", dtype=torch.float32)
        dev("mixed.cu", x, y, outputs=["float32"])
    """
    dtype = _resolve_dtype(dtype)

    if seed is not None:
        torch.manual_seed(seed)

    if n is None:
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                n = inp.numel()
                break
        if n is None:
            n = 1024
            _log("Using default n=1024 (override with n= or --n)", _DIM)

    resolved = _resolve_inputs(inputs, n, dtype)

    s = Session(kernel_path, func_name=func_name, outputs=outputs,
                defines=defines)

    if watch:
        s.watch(*resolved, ref=ref, n=n, atol=atol, rtol=rtol,
                grid=grid, block=block, smem=smem, out_dtype=out_dtype)
        return None

    return s.run(*resolved, ref=ref, n=n, atol=atol, rtol=rtol,
                 grid=grid, block=block, smem=smem, out_dtype=out_dtype,
                 quiet=quiet)


# ── Shared helpers (used by both dev.py and iterate.py) ────────────────

class WorkerTimeoutError(RuntimeError):
    """Raised when a worker dispatch times out."""
    pass


def _format_bench_result(gpu_times, wall_times=None, wall_time=False):
    """Format benchmark timing arrays into result dict and log output."""
    if not gpu_times:
        _log("No benchmark iterations completed.", _DIM)
        return {}
    gpu_times.sort()
    result = {
        "min_ms": gpu_times[0],
        "max_ms": gpu_times[-1],
        "mean_ms": sum(gpu_times) / len(gpu_times),
        "median_ms": gpu_times[len(gpu_times) // 2],
        "iters": len(gpu_times),
    }
    if wall_time and wall_times:
        wall_times.sort()
        result.update({
            "wall_min_ms": wall_times[0],
            "wall_max_ms": wall_times[-1],
            "wall_mean_ms": sum(wall_times) / len(wall_times),
            "wall_median_ms": wall_times[len(wall_times) // 2],
        })
    label = "GPU:  " if wall_time else "Benchmark: "
    _log(f"{label}"
         f"min={result['min_ms']:.3f}ms  "
         f"median={result['median_ms']:.3f}ms  "
         f"mean={result['mean_ms']:.3f}ms  "
         f"max={result['max_ms']:.3f}ms  "
         f"({result['iters']} iters)", _CYAN + _BOLD)
    if wall_time and wall_times:
        _log(f"Wall: "
             f"min={result['wall_min_ms']:.3f}ms  "
             f"median={result['wall_median_ms']:.3f}ms  "
             f"mean={result['wall_mean_ms']:.3f}ms  "
             f"max={result['wall_max_ms']:.3f}ms", _CYAN)
    return result


Session = KernelSession
