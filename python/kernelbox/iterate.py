"""Iterate-mode orchestration: watch, reload, test, and benchmark."""

import inspect
import os
import sys
import time
import traceback

import torch

from .dev import (
    KernelSession, WorkerTimeoutError,
    _log, _preview, _verify,
    _l2_flush, _format_bench_result,
    _BOLD, _GREEN, _RED, _YELLOW, _CYAN, _DIM, _RESET,
)
from .isolated_kernel_mode import build_isolated_kernel_mode_plan, tensor_metadata


def _safe_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def _check_changes(watch_files, mtimes):
    changed = []
    for f in watch_files:
        mt = _safe_mtime(f)
        if mt != mtimes.get(f, 0):
            mtimes[f] = mt
            changed.append(f)
    return changed


_ITER_DATA_EXTS = (".h5", ".hdf5", ".pt", ".pth")


def _iter_data_files(directory):
    """Return absolute paths for iterate-supported data files in a directory."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return [
        os.path.abspath(os.path.join(directory, name))
        for name in sorted(names)
        if os.path.splitext(name)[1].lower() in _ITER_DATA_EXTS
    ]


def _iter_suite_signature(directory):
    """Build a stable signature for suite reloads from file names + mtimes."""
    return tuple(
        (os.path.basename(path), _safe_mtime(path))
        for path in _iter_data_files(directory)
    )


class _Watcher:
    """File watcher — inotify on Linux, falls back to mtime polling."""

    def __init__(self, paths, interval=0.3):
        self.paths = [os.path.abspath(p) for p in paths]
        self.interval = interval
        self._inotify_fd = -1
        self._wd_to_dir = {}
        self._dir_to_names = {}
        self._watched_dirs = {
            p for p in self.paths
            if os.path.isdir(p)
        }
        self.mode = "poll"

        if sys.platform == "linux":
            try:
                self._setup_inotify()
                self.mode = "inotify"
            except OSError:
                pass

        if self.mode == "poll":
            self._mtimes = {p: _safe_mtime(p) for p in self.paths}

    def _setup_inotify(self):
        import ctypes, ctypes.util, fcntl
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

        fd = libc.inotify_init()
        if fd < 0:
            raise OSError("inotify_init failed")
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._inotify_fd = fd

        IN_MODIFY = 0x2
        IN_CLOSE_WRITE = 0x8
        IN_ATTRIB = 0x4
        IN_MOVED_TO = 0x80
        IN_CREATE = 0x100
        mask = IN_MODIFY | IN_CLOSE_WRITE | IN_ATTRIB | IN_MOVED_TO | IN_CREATE

        for p in self.paths:
            if os.path.isdir(p):
                self._dir_to_names.setdefault(p, set())
            else:
                d = os.path.dirname(p)
                self._dir_to_names.setdefault(d, set()).add(os.path.basename(p))

        for d in self._dir_to_names:
            wd = libc.inotify_add_watch(fd, d.encode(), mask)
            if wd < 0:
                self.close()
                raise OSError(f"inotify_add_watch failed for {d}")
            self._wd_to_dir[wd] = d

    def wait(self):
        """Block until watched files change. Returns list of changed paths."""
        if self.mode == "inotify":
            return self._wait_inotify()
        return self._wait_poll()

    def _wait_poll(self):
        while True:
            time.sleep(self.interval)
            changed = _check_changes(self.paths, self._mtimes)
            if changed:
                return changed

    def _wait_inotify(self):
        import struct, select
        while True:
            r, _, _ = select.select([self._inotify_fd], [], [], 1.0)
            if not r:
                continue
            buf = os.read(self._inotify_fd, 8192)
            changed = set()
            offset = 0
            while offset < len(buf):
                wd, mask, cookie, name_len = struct.unpack_from("iIII", buf, offset)
                offset += 16
                name = buf[offset:offset + name_len].rstrip(b"\x00").decode(
                    errors="replace")
                offset += name_len
                d = self._wd_to_dir.get(wd)
                if d is None:
                    continue
                if d in self._watched_dirs:
                    changed.add(d)
                if name in self._dir_to_names.get(d, set()):
                    changed.add(os.path.join(d, name))
            if changed:
                time.sleep(0.02)  # debounce — editors may write multiple events
                try:
                    os.read(self._inotify_fd, 8192)
                except BlockingIOError:
                    pass
                return list(changed)

    def close(self):
        if self._inotify_fd >= 0:
            os.close(self._inotify_fd)
            self._inotify_fd = -1


class _IterStats:
    """Tracks iteration timing and pass/fail counts."""

    def __init__(self):
        self.n = 0
        self.n_pass = 0
        self._recent = []  # last N elapsed times
        self._cap = 20

    def record(self, elapsed_ms, passed):
        self.n += 1
        if passed:
            self.n_pass += 1
        self._recent.append(elapsed_ms)
        if len(self._recent) > self._cap:
            self._recent.pop(0)

    def summary(self):
        avg = sum(self._recent) / len(self._recent)
        return (f"#{self.n}  {self.n_pass}/{self.n} passed  "
                f"avg {avg:.1f}ms (last {len(self._recent)})")


def _benchmark_run(run_fn, inputs, warmup=10, iters=None, benchtime=None,
                   wall_time=True, l2_flush=0, l2_flush_per_iter=0,
                   l2_dirty=False):
    """Benchmark run_fn(inputs) using PyTorch CUDA events (fallback path)."""
    if iters is None and benchtime is None:
        iters = 100
    _log("Benchmarking...", _CYAN)

    for _ in range(warmup):
        run_fn(inputs)
        torch.cuda.synchronize()

    if l2_flush:
        _l2_flush(count=l2_flush, dirty=l2_dirty)

    gpu_times = []
    wall_times = []
    t_wall_start = time.monotonic()
    i = 0
    while True:
        if l2_flush_per_iter:
            _l2_flush(count=l2_flush_per_iter, dirty=l2_dirty)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if wall_time:
            torch.cuda.synchronize()
            t0 = time.monotonic()
        start.record()
        run_fn(inputs)
        end.record()
        torch.cuda.synchronize()
        gpu_times.append(start.elapsed_time(end))
        if wall_time:
            wall_times.append((time.monotonic() - t0) * 1000)
        i += 1
        if iters is not None and i >= iters:
            break
        if benchtime is not None and (time.monotonic() - t_wall_start) >= benchtime:
            break

    return _format_bench_result(gpu_times, wall_times, wall_time)


def _benchmark_with_session(session, inputs, warmup=10, iters=None, benchtime=None,
                            l2_flush=1, l2_flush_per_iter=1, l2_dirty=False,
                            wall_time=False):
    """Benchmark via KernelSession's worker-side timing."""
    if not isinstance(inputs, (list, tuple)) or session.kernel_count != 1:
        return None
    if iters is None and benchtime is None:
        iters = 100
    return session.benchmark(*inputs, warmup=warmup, iters=iters,
                             benchtime=benchtime, wall_time=wall_time,
                             l2_flush=l2_flush, l2_flush_per_iter=l2_flush_per_iter,
                             l2_dirty=l2_dirty)


def _restart_process():
    """Restart the current process to recover from CUDA context corruption."""
    os.environ["_KBOX_CUDA_RETRY"] = "1"
    _log("Restarting process...", _YELLOW + _BOLD)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _snapshot_inputs(inputs):
    """Clone all tensors in inputs (list or dict). Returns (snapshot, inputs)."""
    if isinstance(inputs, dict):
        snap = {k: v.clone() for k, v in inputs.items()
                if isinstance(v, torch.Tensor)}
    elif isinstance(inputs, (list, tuple)):
        snap = [t.clone() for t in inputs if isinstance(t, torch.Tensor)]
    else:
        return None
    return snap


def _restore_inputs(inputs, snapshot):
    """Async copy snapshot back into inputs. CPU runs ahead, no sync."""
    if snapshot is None:
        return
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and k in snapshot:
                v.copy_(snapshot[k])
    elif isinstance(inputs, (list, tuple)):
        j = 0
        for t in inputs:
            if isinstance(t, torch.Tensor):
                t.copy_(snapshot[j])
                j += 1


_CUDA_DEAD = False  # set when CUDA context is unrecoverable
_WORKER_TIMEOUT_COUNT = 0  # consecutive worker timeout count


def _is_cuda_error(exc):
    """Check if an exception indicates a corrupted CUDA context."""
    msg = str(exc).lower()
    return ("cuda" in msg and ("illegal" in msg or "device-side assert" in msg
            or "an error was encountered" in msg or "not ready" in msg
            or "context is destroyed" in msg))


def _run_and_compare(fn, inputs, expected, atol, rtol, dump=None, dump_mode=None,
                     stats=None, restore_inputs=True):
    """Call fn(inputs), compare to expected, print results.

    If *dump* is a path string, writes an HDF5 diff file on failure
    (expected vs actual with absdiff/reldiff) for offline debugging.
    """
    global _CUDA_DEAD, _WORKER_TIMEOUT_COUNT
    try:
        snap = _snapshot_inputs(inputs) if restore_inputs else None
        torch.cuda.synchronize()
        t0 = time.monotonic()
        result = fn(inputs)
        torch.cuda.synchronize()
        elapsed = (time.monotonic() - t0) * 1000
        _restore_inputs(inputs, snap)
        _WORKER_TIMEOUT_COUNT = 0  # reset on successful execution

        if isinstance(result, dict):
            pass  # keep as dict
        elif isinstance(result, torch.Tensor):
            result = [result]
        elif not isinstance(result, (list, tuple)):
            result = list(result)

        _log(f"Ran in {elapsed:.1f}ms", _DIM)
        if isinstance(result, dict):
            for key, out in result.items():
                _log(f"  output[{key}]: {_preview(out)}", _DIM)
        else:
            for i, out in enumerate(result):
                _log(f"  output[{i}]: {_preview(out)}", _DIM)

        ok, msg = _verify(result, expected, inputs, atol=atol, rtol=rtol)
        style = _GREEN + _BOLD if ok else _RED + _BOLD
        _log(f"{'PASS' if ok else 'FAIL'}: {msg}", style)

        if stats is not None:
            stats.record(elapsed, ok)
            _log(stats.summary(), _DIM)

        if not ok and dump:
            try:
                from .h5 import dump_diff
                dump_diff(dump, expected=expected, actual=list(result),
                          inputs=inputs, mode=dump_mode, atol=atol, rtol=rtol)
                _log(f"Diff dumped to {dump}", _YELLOW)
            except Exception as de:
                _log(f"Dump failed: {de}", _DIM)
        return ok, result
    except Exception as e:
        if _is_cuda_error(e):
            _CUDA_DEAD = True
            _log(f"CUDA CRASH: {e}", _RED + _BOLD)
            _log("CUDA context is corrupted. Will restart on next file change.",
                 _RED)
        elif isinstance(e, WorkerTimeoutError):
            _WORKER_TIMEOUT_COUNT += 1
            _log(f"TIMEOUT: {e}", _RED + _BOLD)
            if _WORKER_TIMEOUT_COUNT >= 2:
                _log("Timed out twice in a row. Waiting for file change...",
                     _RED)
        else:
            _log(f"ERROR: {e}", _RED + _BOLD)
            import traceback
            traceback.print_exc(file=sys.stderr)
        if stats is not None:
            stats.record(0, False)
        return False, None


def _run_suite(fn, suite, atol, rtol, dump=None, dump_mode=None, stats=None,
               restore_inputs=True):
    """Run fn against every case in a TestSuite, report per-case results."""
    from .h5 import TestSuite

    n_cases = len(suite)
    n_pass = 0
    t_total = 0

    for i, (name, inputs, expected) in enumerate(suite.cases):
        label = f"[{i+1}/{n_cases}] {name}"
        _log(label, _CYAN)

        try:
            snap = _snapshot_inputs(inputs) if restore_inputs else None
            torch.cuda.synchronize()
            t0 = time.monotonic()
            result = fn(inputs)
            torch.cuda.synchronize()
            elapsed = (time.monotonic() - t0) * 1000
            _restore_inputs(inputs, snap)
            t_total += elapsed

            if isinstance(result, dict):
                pass  # keep as dict
            elif isinstance(result, torch.Tensor):
                result = [result]
            elif not isinstance(result, (list, tuple)):
                result = list(result)

            ok, msg = _verify(result, expected, inputs,
                              atol=atol, rtol=rtol)
            style = _GREEN if ok else _RED + _BOLD
            _log(f"  {'PASS' if ok else 'FAIL'} ({elapsed:.1f}ms): {msg}",
                 style)

            if ok:
                n_pass += 1
            elif dump:
                try:
                    from .h5 import dump_diff
                    base, ext = os.path.splitext(dump)
                    case_dump = f"{base}_{os.path.splitext(name)[0]}{ext}"
                    dump_diff(case_dump, expected=expected,
                              actual=list(result), inputs=inputs,
                              mode=dump_mode, atol=atol, rtol=rtol)
                    _log(f"  Diff dumped to {case_dump}", _YELLOW)
                except Exception as de:
                    _log(f"  Dump failed: {de}", _DIM)

        except Exception as e:
            _log(f"  ERROR: {e}", _RED + _BOLD)
            import traceback
            traceback.print_exc(file=sys.stderr)

    all_ok = n_pass == n_cases
    style = _GREEN + _BOLD if all_ok else _RED + _BOLD
    _log(f"Suite: {n_pass}/{n_cases} passed ({t_total:.1f}ms total)", style)

    if stats is not None:
        stats.record(t_total, all_ok)
        _log(stats.summary(), _DIM)

    return all_ok, None


def iterate(fn, inputs, expected, watch=None, atol=1e-5, rtol=1e-5,
            interval=0.3, dump=None, once=False,
            bench=False, warmup=10, iters=None, benchtime=None):
    """Fast iteration loop — the core developer workflow.

    Inputs and expected stay in GPU memory for the entire session.
    Only ``fn`` re-executes on each file change (manager/worker pattern).

    Note: ``inputs`` is a keyword argument (list), unlike ``KernelSession.watch()``
    which takes ``*inputs`` positionally.  This allows passing dict inputs
    for named test data.

    Args:
        fn:        callable(inputs) -> list of output tensors.
                   Or a :class:`KernelSession` (auto-wrapped, kernel file auto-watched).
        inputs:    List of input tensors (loaded once, never re-created).
        expected:  List of expected output tensors.
        watch:     File paths to watch for changes.  Defaults to all ``.cu``
                   files in the current directory.
        atol:      Absolute tolerance.
        rtol:      Relative tolerance.
        interval:  File poll interval in seconds.
        dump:      Path to write an HDF5 diff file on failure
                   (expected/actual/absdiff/reldiff).  ``None`` disables.
        once:      If True, run once then return (no file watching).
        bench:     If True, benchmark after each successful verification.
        warmup:    Benchmark warmup iterations.
        iters:     Benchmark iterations (default: 100 if no benchtime).
        benchtime: Max benchmark time in seconds.

    Example::

        import torch, kernelbox as kbox

        x = torch.randn(1024, device="cuda")

        s = kbox.KernelSession("scale.cu")
        kbox.iterate(s, inputs=[x], expected=[x * 2.5])

    Press Ctrl+C to stop.
    """
    if isinstance(fn, KernelSession):
        session = fn
        if session.kernel_count != 1:
            raise ValueError(
                "iterate(KernelSession) only supports single-kernel sessions. "
                "Wrap multi-kernel sessions in a function or use kernel_mode().")
        watch = list(watch or [])
        for kernel_path in session.kernel_paths or ([session.kernel_path]
                                                    if session.kernel_path else []):
            if kernel_path not in [os.path.abspath(w) for w in watch]:
                watch.append(kernel_path)
        num_out = session.num_outputs
        fn = lambda inputs: _as_list(session(*inputs), num_out)

    if watch is None:
        watch = []

    for f in os.listdir("."):
        if f.endswith((".cu", ".cuh")):
            full = os.path.abspath(f)
            if full not in [os.path.abspath(w) for w in watch]:
                watch.append(full)

    if isinstance(inputs, dict):
        n_in = sum(1 for v in inputs.values() if isinstance(v, torch.Tensor))
    else:
        n_in = len(inputs)
    if isinstance(expected, dict):
        n_exp = sum(1 for v in expected.values() if isinstance(v, torch.Tensor))
    else:
        n_exp = len(expected)

    if not once:
        watcher = _Watcher(watch, interval=interval)
        _log("Iterating (Ctrl+C to stop)", _YELLOW + _BOLD)
        if watch:
            _log(f"Watching: {', '.join(os.path.basename(f) for f in watch)}",
                 _DIM)
        _log(f"Inputs: {n_in} tensor(s), "
             f"Expected: {n_exp} tensor(s)", _DIM)
        _log(f"Watch mode: {watcher.mode}", _DIM)
        if dump:
            _log(f"Dump on failure: {dump}", _DIM)
        print()

    stats = _IterStats()
    ok, _outputs = _run_and_compare(fn, inputs, expected, atol, rtol, dump=dump,
                          stats=stats)
    if bench and ok:
        _benchmark_run(fn, inputs, warmup=warmup, iters=iters,
                       benchtime=benchtime)

    if once:
        return

    try:
        while True:
            changed = watcher.wait()
            print()
            for f in changed:
                _log(f"Changed: {os.path.basename(f)}", _YELLOW)
            ok, _outputs = _run_and_compare(fn, inputs, expected, atol, rtol, dump=dump,
                                  stats=stats)
            if bench and ok:
                _benchmark_run(fn, inputs, warmup=warmup, iters=iters,
                               benchtime=benchtime)
    except KeyboardInterrupt:
        print()
        _log("Stopped.", _DIM)
    finally:
        watcher.close()


def _as_list(result, num_out):
    if num_out == 1:
        return [result]
    return list(result)


# ── File-based watch (kbox iterate) ───────────────────────────────────────

_loaded_module = None  # previous module, cleared on reload
_loaded_submodules = {}  # {module_name: file_path} for modules imported by test file


def _is_user_module(name):
    """Return True if module is user code (not an installed package)."""
    mod = sys.modules.get(name)
    if mod is None:
        return False
    filepath = getattr(mod, "__file__", None)
    if filepath is None:
        return False
    # Installed packages live in site-packages or dist-packages
    filepath = os.path.abspath(filepath)
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker in filepath:
            return False
    return True


def _load_module(path):
    global _loaded_module, _loaded_submodules
    import importlib.util

    # Release old module's references (tensors, sessions, etc.)
    if _loaded_module is not None:
        _loaded_module.__dict__.clear()
    # Evict user modules that were imported by the previous test file
    for name in list(_loaded_submodules):
        mod = sys.modules.pop(name, None)
        if mod is not None:
            mod.__dict__.clear()
    sys.modules.pop("_kbox_test", None)

    before = set(sys.modules.keys())
    spec = importlib.util.spec_from_file_location("_kbox_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("_kbox_test", None)

    # Track new user modules (with file paths) so we can watch + evict on reload
    _loaded_submodules = {}
    for k in sys.modules.keys() - before:
        if _is_user_module(k):
            _loaded_submodules[k] = os.path.abspath(sys.modules[k].__file__)
    _loaded_module = mod
    return mod


def _resolve_iter_kernel_path(kernel_path, test_dir):
    if os.path.isabs(kernel_path):
        return kernel_path
    test_rel = os.path.abspath(os.path.join(test_dir, kernel_path))
    if os.path.exists(test_rel):
        return test_rel
    return os.path.abspath(kernel_path)


def _resolve_iter_kernel_value(value, test_dir):
    if isinstance(value, (list, tuple)):
        return [_resolve_iter_kernel_path(v, test_dir) for v in value]
    return _resolve_iter_kernel_path(value, test_dir)



def _load_iter_runtime(test_path, test_dir, state, default_atol, default_rtol,
                       default_bench, default_wall_time=True,
                       default_timeout=None, default_warmup=10,
                       default_iters=None, default_benchtime=None,
                       default_l2_flush=None, default_l2_flush_per_iter=None,
                       default_l2_dirty=False,
                       isolated_kernel_benchmark=False,
                       require_entrypoint=True):
    """Load the test module and resolve the active iteration contract."""
    from .h5 import TestSuite

    mod = _load_module(test_path)
    suite = None
    inputs = None
    expected = None
    atol = default_atol
    rtol = default_rtol
    bench = default_bench
    wall_time = default_wall_time

    # Detect init mode: init() vs init_once() (mutually exclusive)
    has_init = hasattr(mod, "init") and callable(mod.init)
    has_init_once = hasattr(mod, "init_once") and callable(mod.init_once)
    has_run = hasattr(mod, "run") and callable(mod.run)
    has_kernel_mode = (hasattr(mod, "kernel_mode")
                       and callable(mod.kernel_mode))

    if has_init and has_init_once:
        raise ValueError(
            "test file must define 'init' OR 'init_once', not both.\n"
            "  init()      — called before every run (fresh inputs each time)\n"
            "  init_once() — called once (inputs persist across file changes)")
    if not has_init and not has_init_once:
        raise ValueError(
            "test file must define 'init' or 'init_once'\n"
            "  def init_once(): return {'inputs': [...], 'expected': [...]}\n"
            "  def init(): return {'inputs': [...], 'expected': [...]}\n"
            "  def run(inputs): ...")
    if require_entrypoint and not has_run and not has_kernel_mode:
        raise ValueError(
            "test file must define 'run' or 'kernel_mode'")

    # Removed: init_reload() is no longer supported
    if hasattr(mod, "init_reload"):
        _log("Warning: init_reload() is no longer supported and will be ignored. "
             "Use init() for per-run re-init or init_once() for persistent state.", _YELLOW)

    force_init = has_init  # init() = force every time

    # Determine previous init mode to detect transitions
    prev_mode = None
    if state is not None:
        prev_mode = state.get("_init_mode")

    current_mode = "always" if force_init else "once"

    # Decide whether to call the init function
    if force_init:
        # init() mode: always call init()
        # Release persistent VMM on session if switching from init_once
        if prev_mode == "once" and state is not None:
            session = state.get("_session")
            if isinstance(session, KernelSession):
                session._release_persistent()
        init_state = mod.init()
    else:
        # init_once() mode: only call on first load
        if state is None or prev_mode == "always":
            # First load, or switching from init() → init_once()
            if state is not None:
                session = state.get("_session")
                if isinstance(session, KernelSession):
                    session._release_persistent()
            init_state = mod.init_once()
        else:
            init_state = state

    if not isinstance(init_state, dict):
        raise TypeError("init()/init_once() must return a dict")
    state = init_state
    state["_init_mode"] = current_mode

    def _iter_input_tensors(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, dict):
            return [v for v in value.values() if isinstance(v, torch.Tensor)]
        if hasattr(value, "__dict__"):
            return [v for v in value.__dict__.values() if isinstance(v, torch.Tensor)]
        return list(value)

    def _expand_per_kernel(value, count, name):
        if isinstance(value, (list, tuple)):
            if len(value) != count:
                raise ValueError(
                    f"{name} list length mismatch: expected {count}, got {len(value)}")
            return list(value)
        return [value] * count

    def _session_entry_values(session, key):
        return [entry[key] for entry in session._entries]

    # Validate mutually exclusive kernel sources
    kernel_keys = [k for k in ("kernel", "kernel_source", "triton_kernel") if k in state]
    if len(kernel_keys) > 1:
        raise ValueError(
            f"init()/init_once() must return exactly one of 'kernel', 'kernel_source', "
            f"or 'triton_kernel', got: {kernel_keys}")

    kernel_scratch_mib = state.get("kernel_scratch_mib", 0)

    # Create/reuse KernelSession when "kernel" path is specified
    if "kernel" in state:
        kernel_path = _resolve_iter_kernel_value(state["kernel"], test_dir)
        defines = state.get("defines")
        func_name = state.get("func_name")
        outputs = state.get("outputs", 1)
        timeout = default_timeout or state.get("timeout", 0.75)
        count = len(kernel_path) if isinstance(kernel_path, (list, tuple)) else 1
        kernel_paths = list(kernel_path) if isinstance(kernel_path, (list, tuple)) else [kernel_path]
        defines_list = _expand_per_kernel(defines, count, "defines")
        func_names = _expand_per_kernel(func_name, count, "func_name")

        session = state.get("_session")
        needs_new = (
            not isinstance(session, KernelSession)
            or session.kernel_count != count
            or session.kernel_paths != kernel_paths
            or any(entry["triton_mode"] for entry in session._entries)
            or _session_entry_values(session, "defines") != defines_list
            or _session_entry_values(session, "func_name_override") != func_names
            or session._raw_outputs != outputs
            or session._user_scratch_mib != kernel_scratch_mib
        )
        if needs_new:
            session = KernelSession(kernel_path=kernel_path, func_name=func_name,
                                    outputs=outputs, defines=defines,
                                    timeout=timeout,
                                    grid=state.get("grid"),
                                    block=state.get("block"),
                                    smem=state.get("smem"),
                                    out_dtype=state.get("out_dtype"),
                                    kernel_scratch_mib=kernel_scratch_mib)

        # Update per-launch defaults (no session recreation needed)
        session._timeout = timeout
        session.update_defaults(grid=state.get("grid"),
                                block=state.get("block"),
                                smem=state.get("smem"),
                                out_dtype=state.get("out_dtype"))
        state["_session"] = session

    # Create/reuse KernelSession for inline kernel_source
    elif "kernel_source" in state:
        kernel_source = state["kernel_source"]
        defines = state.get("defines")
        func_name = state.get("func_name")
        outputs = state.get("outputs", 1)
        timeout = default_timeout or state.get("timeout", 0.75)
        count = len(kernel_source) if isinstance(kernel_source, (list, tuple)) else 1
        defines_list = _expand_per_kernel(defines, count, "defines")
        func_names = _expand_per_kernel(func_name, count, "func_name")

        session = state.get("_session")
        needs_new = (
            not isinstance(session, KernelSession)
            or session.kernel_count != count
            or any(entry["kernel_path"] is not None for entry in session._entries)
            or any(entry["triton_mode"] for entry in session._entries)
            or _session_entry_values(session, "defines") != defines_list
            or _session_entry_values(session, "func_name_override") != func_names
            or session._raw_outputs != outputs
            or session._user_scratch_mib != kernel_scratch_mib
        )
        if needs_new:
            # Need a new inline session (first load or switching from file)
            session = KernelSession(kernel_source=kernel_source,
                                    func_name=func_name, outputs=outputs,
                                    defines=defines, timeout=timeout,
                                    grid=state.get("grid"),
                                    block=state.get("block"),
                                    smem=state.get("smem"),
                                    out_dtype=state.get("out_dtype"),
                                    kernel_scratch_mib=kernel_scratch_mib)
        else:
            # Existing inline session — update source (recompiles on hash change)
            if count == 1:
                session.update_source(kernel_source)
            else:
                session.update_sources(kernel_source)

        session._timeout = timeout
        session.update_defaults(grid=state.get("grid"),
                                block=state.get("block"),
                                smem=state.get("smem"),
                                out_dtype=state.get("out_dtype"))
        state["_session"] = session

    # Create/reuse KernelSession for Triton @triton.jit function
    elif "triton_kernel" in state:
        triton_fn = state["triton_kernel"]
        triton_constexprs = state.get("triton_constexprs", {})
        func_name = state.get("func_name")
        outputs = state.get("outputs", 1)
        timeout = default_timeout or state.get("timeout", 0.75)
        count = len(triton_fn) if isinstance(triton_fn, (list, tuple)) else 1
        constexpr_list = _expand_per_kernel(
            triton_constexprs, count, "triton_constexprs")
        func_names = _expand_per_kernel(func_name, count, "func_name")

        session = state.get("_session")
        needs_new = (
            not isinstance(session, KernelSession)
            or session.kernel_count != count
            or not all(entry["triton_mode"] for entry in session._entries)
            or _session_entry_values(session, "func_name_override") != func_names
            or _session_entry_values(session, "triton_constexprs") != constexpr_list
            or session._raw_outputs != outputs
            or session._user_scratch_mib != kernel_scratch_mib
        )
        if needs_new:
            session = KernelSession(triton_fn=triton_fn,
                                    triton_constexprs=triton_constexprs,
                                    func_name=func_name, outputs=outputs,
                                    timeout=timeout,
                                    grid=state.get("grid"),
                                    block=state.get("block"),
                                    smem=state.get("smem"),
                                    out_dtype=state.get("out_dtype"),
                                    kernel_scratch_mib=kernel_scratch_mib)
        else:
            # Existing Triton session — update function (recompiles on source change)
            if count == 1:
                session.update_triton_fn(triton_fn, triton_constexprs)
            else:
                session.update_triton_fns(triton_fn, triton_constexprs)

        session._timeout = timeout
        session.update_defaults(grid=state.get("grid"),
                                block=state.get("block"),
                                smem=state.get("smem"),
                                out_dtype=state.get("out_dtype"))
        state["_session"] = session

    # Auto-load h5/h5_suite string paths (re-load on path or mtime change)
    from . import h5 as _h5
    if "h5" in state and isinstance(state["h5"], str):
        h5_path = _resolve_iter_kernel_path(state["h5"], test_dir)
        h5_mtime = os.path.getmtime(h5_path) if os.path.exists(h5_path) else None
        if (state.get("_h5_path") != h5_path
                or state.get("_h5_mtime") != h5_mtime
                or "inputs" not in state):
            loaded_inputs, loaded_expected = _h5.load_test(h5_path)
            state["inputs"] = loaded_inputs
            state["expected"] = loaded_expected
            state["_h5_path"] = h5_path
            state["_h5_mtime"] = h5_mtime
    if "h5_suite" in state and isinstance(state["h5_suite"], str):
        suite_path = _resolve_iter_kernel_path(state["h5_suite"], test_dir)
        suite_sig = _iter_suite_signature(suite_path)
        if (state.get("_h5_suite_path") != suite_path
                or state.get("_h5_suite_sig") != suite_sig
                or "_h5_suite_files" not in state
                or "suite" not in state):
            state["suite"] = _h5.load_tests(suite_path)
            state["_h5_suite_path"] = suite_path
            state["_h5_suite_sig"] = suite_sig
            state["_h5_suite_files"] = [
                os.path.join(suite_path, name) for name, _mtime in suite_sig
            ]

    suite = state.get("suite")
    if suite is not None and not isinstance(suite, TestSuite):
        raise TypeError("'suite' must be a TestSuite (from kbox.h5.load_tests)")
    if suite is None:
        inputs = state.get("inputs", state.get("input"))
        expected = state.get("expected")
        if inputs is None or expected is None:
            raise ValueError(
                "init()/init_once() must return 'inputs'+'expected', 'h5', 'h5_suite', or 'suite'\n"
                "  return {'inputs': [...], 'expected': [...]}\n"
                "  return {'h5': 'path/to/data.h5'}\n"
                "  return {'h5_suite': 'path/to/cases/'}")
        # Normalize single tensor expected → list (but keep dicts as-is)
        if isinstance(expected, torch.Tensor):
            expected = [expected]
        elif isinstance(expected, dict):
            pass  # keep as dict
    atol = state.get("atol", atol)
    rtol = state.get("rtol", rtol)
    bench = bool(state.get("benchmark", bench))
    wall_time = bool(state.get("wall_time", default_wall_time))
    restore_inputs = bool(state.get("restore_inputs", True))
    bench_suite_per_case = bool(state.get("bench_suite_per_case", False))

    # Benchmark params: state dict overrides CLI defaults
    warmup = state.get("warmup", default_warmup)
    iters = state.get("iters", default_iters)
    benchtime = state.get("benchtime", default_benchtime)

    # L2 flush params: state dict overrides CLI defaults
    l2_flush = state.get("l2_flush", default_l2_flush)
    l2_flush_per_iter = state.get("l2_flush_per_iter", default_l2_flush_per_iter)
    l2_dirty = bool(state.get("l2_dirty", default_l2_dirty))

    run_fn = mod.run if has_run else None
    bench_fn = None

    session = state.get("_session")

    def _ensure_pointer_layout(active_session, run_inputs):
        launch_inputs = _iter_input_tensors(run_inputs)
        norm_inputs, out_specs, n, _, _, _, _ = \
            active_session._resolve_inputs_and_outputs(launch_inputs)
        active_session._ensure_persistent_layout(
            list(norm_inputs), out_specs, n,
            scratch_mib=active_session._user_scratch_mib)
        inputs_meta, outputs_meta = tensor_metadata(norm_inputs, out_specs)
        return norm_inputs, out_specs, n, inputs_meta, outputs_meta

    if has_kernel_mode:
        if not isinstance(session, KernelSession):
            raise ValueError(
                "kernel_mode() requires init()/init_once() to return "
                "'kernel', 'kernel_source', or 'triton_kernel'")
        kernel_mode_fn = mod.kernel_mode
        plan_cache = {}

        def _kernel_mode_key(norm_inputs, out_specs, n):
            return (
                id(kernel_mode_fn),
                tuple((t.data_ptr(), tuple(t.shape), t.dtype) for t in norm_inputs),
                tuple(out_specs),
                n,
            )

        def _build_kernel_mode_plan(run_inputs):
            if isolated_kernel_benchmark:
                norm_inputs, out_specs, n, _, _ = \
                    _ensure_pointer_layout(session, run_inputs)
                key = _kernel_mode_key(norm_inputs, out_specs, n)
                if key in plan_cache:
                    return norm_inputs, plan_cache[key]
                norm_inputs, _out_specs, _n, steps = build_isolated_kernel_mode_plan(
                    session, test_path, norm_inputs)
                plan_cache[key] = list(steps)
                return norm_inputs, plan_cache[key]

            norm_inputs, out_specs, n, inputs_meta, outputs_meta = \
                _ensure_pointer_layout(session, run_inputs)
            key = _kernel_mode_key(norm_inputs, out_specs, n)
            if key in plan_cache:
                return norm_inputs, plan_cache[key]

            sig = inspect.signature(kernel_mode_fn)
            kwargs = {}
            for name, param in sig.parameters.items():
                if name == "kernel":
                    kwargs[name] = (session.kernels if session.kernel_count > 1
                                    else session.kernels[0])
                elif name == "kernels":
                    kwargs[name] = session.kernels
                elif name == "scratch_ptr":
                    kwargs[name] = session.scratch_ptr
                elif name == "input_ptrs":
                    kwargs[name] = session.input_ptrs
                elif name == "output_ptrs":
                    kwargs[name] = session.output_ptrs
                elif name in ("input_meta", "inputs_meta"):
                    kwargs[name] = inputs_meta
                elif name in ("output_meta", "outputs_meta"):
                    kwargs[name] = outputs_meta
                elif name == "n":
                    kwargs[name] = n
                elif (param.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                         inspect.Parameter.VAR_KEYWORD)
                        and param.default is inspect.Parameter.empty):
                    raise ValueError(
                        f"kernel_mode() parameter '{name}' is not supported.\n"
                        "  Supported names: kernel, kernels, scratch_ptr, "
                        "input_ptrs, output_ptrs, input_meta, output_meta, n")

            steps = kernel_mode_fn(**kwargs)
            if not isinstance(steps, (list, tuple)):
                raise TypeError("kernel_mode() must return a list/tuple of step specs")
            plan_cache[key] = list(steps)
            return norm_inputs, plan_cache[key]

        def _run_kernel_mode(run_inputs, _s=session):
            norm_inputs, steps = _build_kernel_mode_plan(run_inputs)
            return _s.run_steps(steps, *norm_inputs)

        def _bench_kernel_mode(bench_inputs, warmup=10, iters=None,
                               benchtime=None, l2_flush=1,
                               l2_flush_per_iter=1, l2_dirty=False,
                               wall_time=False, _s=session):
            norm_inputs, steps = _build_kernel_mode_plan(bench_inputs)
            return _s.benchmark_steps(
                steps, *norm_inputs, warmup=warmup, iters=iters,
                benchtime=benchtime, l2_flush=l2_flush,
                l2_flush_per_iter=l2_flush_per_iter,
                l2_dirty=l2_dirty, wall_time=wall_time)

        run_fn = _run_kernel_mode
        bench_fn = _bench_kernel_mode
    elif has_run:
        run_params = list(inspect.signature(mod.run).parameters.keys())
        wants_kernel = len(run_params) >= 2 and run_params[1] in ("kernel", "kernels")
        wants_scratch = "scratch_ptr" in run_params

        if wants_kernel:
            if not isinstance(session, KernelSession):
                raise ValueError(
                    "run(inputs, kernel) requires init()/init_once() to return "
                    "'kernel', 'kernel_source', or 'triton_kernel'")
            kernel_arg = (session.kernels if (session.kernel_count > 1
                                             or run_params[1] == "kernels")
                          else session)
            if wants_scratch:
                def _run_with_kernel(run_inputs, _run=mod.run, _s=session, _kernel=kernel_arg):
                    _ensure_pointer_layout(_s, run_inputs)
                    return _run(run_inputs, _kernel, scratch_ptr=_s.scratch_ptr)
                run_fn = _run_with_kernel
            else:
                run_fn = lambda run_inputs, _run=mod.run, _kernel=kernel_arg: _run(run_inputs, _kernel)

    post_fn = getattr(mod, "post", None)
    if post_fn is not None and not callable(post_fn):
        post_fn = None

    return {
        "mod": mod,
        "state": state,
        "suite": suite,
        "inputs": inputs,
        "expected": expected,
        "run_fn": run_fn,
        "bench_fn": bench_fn,
        "post_fn": post_fn,
        "atol": atol,
        "rtol": rtol,
        "bench": bench,
        "wall_time": wall_time,
        "restore_inputs": restore_inputs,
        "bench_suite_per_case": bench_suite_per_case,
        "warmup": warmup,
        "iters": iters,
        "benchtime": benchtime,
        "l2_flush": l2_flush,
        "l2_flush_per_iter": l2_flush_per_iter,
        "l2_dirty": l2_dirty,
        "kernel_scratch_mib": kernel_scratch_mib,
    }


def watch_file(test_path, atol=1e-5, rtol=1e-5, interval=0.3, dump=None,
               dump_mode=None, once=False, bench=False, warmup=10, iters=None,
               benchtime=None, timeout=None, l2_flush=None,
               l2_flush_per_iter=None, l2_dirty=False,
               isolated_kernel_benchmark=False):
    """Load a test file once, then iterate on every change.

    Test file hooks (mutually exclusive init):

    - ``init_once()`` — called once, inputs persist across file changes
    - ``init()`` — called before every run (fresh inputs each time)
    - ``run(inputs)``, ``run(inputs, kernel)``, or ``run(inputs, kernels)``
    - ``kernel_mode(...)`` — cached launch plan, replaces ``run`` when present
    - ``post(outputs, state)`` — called after run+verify (optional)

    On ``.py`` change: ``run`` is hot-reloaded. With ``init_once()``, data
    persists. With ``init()``, inputs are regenerated.
    On ``.cu`` change: kernel auto-recompiles, re-runs with existing inputs.
    On ``.h5``/``.pt`` change: data-backed inputs or suite cases reload.

    Args:
        test_path: Path to the test ``.py`` file.
        atol:      Absolute tolerance.
        rtol:      Relative tolerance.
        interval:  File poll interval in seconds.
        dump:      Path for HDF5 diff dump on failure (None = disabled).
        once:      If True, run once then return (no file watching).
        bench:     If True, benchmark after each successful verification.
        warmup:    Benchmark warmup iterations.
        iters:     Benchmark iterations (default: 100 if no benchtime).
        benchtime: Max benchmark time in seconds.
        l2_flush:          L2 flush passes before benchmark loop (None = use defaults).
        l2_flush_per_iter: L2 flush passes per benchmark iteration (None = use defaults).
        l2_dirty:          Use dirty (read+write) L2 flush.
        isolated_kernel_benchmark:
                            Run kernel_mode() planning in a one-shot isolated
                            subprocess with no GPU visibility.
    """
    global _WORKER_TIMEOUT_COUNT
    test_path = os.path.abspath(test_path)
    test_dir = os.path.dirname(test_path)

    _log(f"Loading {os.path.basename(test_path)}...", _CYAN)
    rt = _load_iter_runtime(test_path, test_dir, None, atol, rtol, bench,
                            default_timeout=timeout, default_warmup=warmup,
                            default_iters=iters, default_benchtime=benchtime,
                            default_l2_flush=l2_flush,
                            default_l2_flush_per_iter=l2_flush_per_iter,
                            default_l2_dirty=l2_dirty,
                            isolated_kernel_benchmark=isolated_kernel_benchmark)

    watch = [test_path]
    watch_set = {test_path}

    def _add(path):
        p = os.path.abspath(path)
        if p not in watch_set:
            watch_set.add(p)
            watch.append(p)

    def _collect_watch(mod, state):
        if hasattr(mod, "watch_files"):
            for w in mod.watch_files:
                _add(w if os.path.isabs(w) else os.path.join(test_dir, w))
        if isinstance(state, dict):
            for w in state.get("watch_files", []):
                _add(w if os.path.isabs(w) else os.path.join(test_dir, w))
            if state.get("_h5_path"):
                _add(state["_h5_path"])
            if state.get("_h5_suite_path"):
                _add(state["_h5_suite_path"])
            for w in state.get("_h5_suite_files", []):
                _add(w)
        for val in vars(mod).values():
            if isinstance(val, KernelSession):
                for kernel_path in val.kernel_paths or ([val.kernel_path]
                                                       if val.kernel_path else []):
                    _add(kernel_path)
        # Watch imported helper modules (documented in ITERATE.md)
        for filepath in _loaded_submodules.values():
            _add(filepath)

    _collect_watch(rt["mod"], rt["state"])

    scan_dirs = {test_dir}
    for w in list(watch):
        scan_dirs.add(os.path.dirname(w))
    n_before = len(watch)
    for d in scan_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith((".cu", ".cuh")):
                    _add(os.path.join(d, f))
    n_discovered = len(watch) - n_before
    if n_discovered > 0:
        discovered = [os.path.basename(w) for w in watch[n_before:]]
        _log(f"Auto-discovered: {', '.join(discovered)}", _DIM)

    def _tinfo(t):
        return f"{t.dtype}{list(t.shape)}"

    suite, inputs, expected = rt["suite"], rt["inputs"], rt["expected"]
    if suite is not None:
        _log(f"Suite:    {len(suite)} cases from "
             f"{os.path.basename(suite.directory)}/", _DIM)
        _log(f"Inputs:   {', '.join(suite.input_keys)}", _DIM)
        _log(f"Expected: {suite.n_expected} output(s) per case", _DIM)
    else:
        if isinstance(inputs, dict):
            in_info = ", ".join(
                f"{k}:{_tinfo(v)}" for k, v in inputs.items()
                if isinstance(v, torch.Tensor))
        else:
            in_info = ", ".join(_tinfo(t) for t in inputs)
        if isinstance(expected, dict):
            ex_info = ", ".join(
                f"{k}:{_tinfo(v)}" for k, v in expected.items()
                if isinstance(v, torch.Tensor))
        else:
            ex_info = ", ".join(_tinfo(t) for t in expected)
        _log(f"Inputs:   {in_info}", _DIM)
        _log(f"Expected: {ex_info}", _DIM)

    stats = _IterStats()

    def _run_iter(rt):
        run_fn = rt["run_fn"]
        restore = rt["restore_inputs"]
        suite, inputs, expected = rt["suite"], rt["inputs"], rt["expected"]
        rt_warmup = rt["warmup"]
        rt_iters = rt["iters"]
        rt_benchtime = rt["benchtime"]

        # Build L2 kwargs (only include if set, so defaults propagate)
        l2_kw = {}
        if rt["l2_flush"] is not None:
            l2_kw["l2_flush"] = rt["l2_flush"]
        if rt["l2_flush_per_iter"] is not None:
            l2_kw["l2_flush_per_iter"] = rt["l2_flush_per_iter"]
        l2_kw["l2_dirty"] = rt["l2_dirty"]

        # Zero scratch buffer between kernel calls when requested
        scratch_session = rt["state"].get("_session") if rt["kernel_scratch_mib"] > 0 else None
        if scratch_session is not None:
            try:
                scratch_session.zero_scratch()
            except Exception:
                pass  # scratch not yet allocated (first call)

        if suite is not None:
            ok, outputs = _run_suite(run_fn, suite, rt["atol"], rt["rtol"],
                            dump=dump, dump_mode=dump_mode, stats=stats,
                            restore_inputs=restore)
        else:
            ok, outputs = _run_and_compare(run_fn, inputs, expected, rt["atol"], rt["rtol"],
                                  dump=dump, dump_mode=dump_mode, stats=stats,
                                  restore_inputs=restore)
        if rt.get("post_fn") and outputs is not None:
            try:
                rt["post_fn"](outputs, rt["state"])
            except Exception as e:
                _log(f"post() error: {e}", _RED + _BOLD)
        if rt["bench"] and ok:
            if suite is not None:
                _bench_suite(rt, run_fn, suite, l2_kw,
                             rt_warmup, rt_iters, rt_benchtime)
            else:
                bench_fn = rt["bench_fn"]
                result = bench_fn(inputs, warmup=rt_warmup, iters=rt_iters,
                                  benchtime=rt_benchtime, wall_time=rt["wall_time"],
                                  **l2_kw) if bench_fn else None
                if result is None:
                    _benchmark_run(run_fn, inputs, warmup=rt_warmup,
                                   iters=rt_iters, benchtime=rt_benchtime,
                                   wall_time=rt["wall_time"], **l2_kw)

    def _bench_suite(rt, run_fn, suite, l2_kw, warmup, iters, benchtime):
        """Benchmark a suite — per-case or aggregate (all cases back-to-back)."""
        bench_fn = rt["bench_fn"]
        wall_time = rt["wall_time"]

        if rt["bench_suite_per_case"]:
            # aaabbbccc: benchmark each case individually
            for name, case_inputs, _expected in suite.cases:
                _log(f"Benchmarking {name}...", _CYAN)
                result = None
                if bench_fn is not None:
                    result = bench_fn(case_inputs, warmup=warmup, iters=iters,
                                      benchtime=benchtime, wall_time=wall_time,
                                      **l2_kw)
                if result is None:
                    _benchmark_run(run_fn, case_inputs, warmup=warmup,
                                   iters=iters, benchtime=benchtime,
                                   wall_time=wall_time, **l2_kw)
        else:
            # abcabcabc (default): all cases back-to-back as one "iteration"
            all_inputs = [case_inputs for _name, case_inputs, _expected
                          in suite.cases]

            def suite_run_fn(_unused):
                for ci in all_inputs:
                    run_fn(ci)

            _benchmark_run(suite_run_fn, None, warmup=warmup, iters=iters,
                           benchtime=benchtime, wall_time=wall_time, **l2_kw)

    def _run_with_timeout_retry(rt):
        """Run iteration, retrying once on first timeout (worker already respawned)."""
        _run_iter(rt)
        if _WORKER_TIMEOUT_COUNT == 1:
            _log("Retrying after timeout...", _YELLOW)
            _run_iter(rt)

    def _needs_runtime_reload(changed, rt):
        changed_set = {os.path.abspath(path) for path in changed}
        state = rt["state"]
        if test_path in changed_set:
            return True
        # Reload when imported helper modules change
        if any(p in changed_set for p in _loaded_submodules.values()):
            return True
        h5_path = state.get("_h5_path")
        if h5_path and h5_path in changed_set:
            return True
        suite_path = state.get("_h5_suite_path")
        if suite_path and suite_path in changed_set:
            return True
        suite_files = state.get("_h5_suite_files", [])
        return any(path in changed_set for path in suite_files)

    # Check if this is a restart after CUDA crash
    is_retry = os.environ.pop("_KBOX_CUDA_RETRY", None) == "1"
    if is_retry:
        _log("Restarted after CUDA crash", _YELLOW + _BOLD)

    # Initial run
    if once:
        _run_iter(rt)
    else:
        _run_with_timeout_retry(rt)

    if _CUDA_DEAD and not once:
        if is_retry:
            # Crashed twice in a row — need a file change before retrying
            _log("Crashed again after restart. Waiting for file change...",
                 _RED + _BOLD)
        else:
            # First crash — restart immediately
            _restart_process()

    if once:
        return

    watcher = _Watcher(watch, interval=interval)

    _log(f"Watching: {', '.join(os.path.basename(f) for f in watch)}", _DIM)
    _log(f"Watch mode: {watcher.mode}", _DIM)
    if dump:
        _log(f"Dump on failure: {dump}", _DIM)
    _log("Ctrl+C to stop", _YELLOW + _BOLD)
    print()

    try:
        while True:
            if _CUDA_DEAD:
                _log("Waiting for file change to restart...", _YELLOW)
                watcher.wait()
                _restart_process()

            if _WORKER_TIMEOUT_COUNT >= 2:
                _log("Waiting for file change after repeated timeouts...", _YELLOW)
                watcher.wait()
                _WORKER_TIMEOUT_COUNT = 0

            changed = watcher.wait()

            print()
            for f in changed:
                _log(f"Changed: {os.path.basename(f)}", _YELLOW)

            if _needs_runtime_reload(changed, rt):
                try:
                    t0 = time.monotonic()
                    rt = _load_iter_runtime(test_path, test_dir,
                                            rt["state"], rt["atol"],
                                            rt["rtol"], rt["bench"],
                                            rt["wall_time"],
                                            default_timeout=timeout,
                                            default_warmup=warmup,
                                            default_iters=iters,
                                            default_benchtime=benchtime,
                                            default_l2_flush=l2_flush,
                                            default_l2_flush_per_iter=l2_flush_per_iter,
                                            default_l2_dirty=l2_dirty,
                                            isolated_kernel_benchmark=isolated_kernel_benchmark)
                    t_reload = (time.monotonic() - t0) * 1000
                    n_before = len(watch)
                    _collect_watch(rt["mod"], rt["state"])
                    if len(watch) != n_before:
                        discovered = [os.path.basename(w) for w in watch[n_before:]]
                        _log(f"Watching new paths: {', '.join(discovered)}", _DIM)
                    watcher.close()
                    watcher = _Watcher(watch, interval=interval)
                    _log(f"Reloaded runtime in {t_reload:.1f}ms", _DIM)
                except Exception as e:
                    _log(f"Reload error: {e}", _RED + _BOLD)
                    continue

            _run_with_timeout_retry(rt)
    except KeyboardInterrupt:
        print()
        _log("Stopped.", _DIM)
    finally:
        watcher.close()
