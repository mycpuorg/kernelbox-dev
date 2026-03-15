#!/usr/bin/env python3
"""Microbenchmarks for kernelbox IPC and GPU operation latencies.

Measures wall-clock overhead of individual operations (socket round-trips,
GPU sync, L2 flush, kernel dispatch) and verifies:
  - sync=False calls avoid CPU-GPU stalls (GPU work pipelines correctly)
  - L2 flush actually evicts cache (cold vs warm timing difference)
  - Same tensor inputs reuse VMM mappings without re-SETUP
  - benchmark() per-iteration overhead breakdown

Worker architecture notes:
  - Each operation (sync, start_timing, l2_flush, kernel call) opens a NEW
    Unix socket connection to the worker daemon.
  - l2_flush is ASYNCHRONOUS: GPU work (memset + optional read) is queued
    in-stream and the CPU returns immediately. Stream ordering guarantees
    the flush completes before any subsequent kernel launch on the same stream.
  - sync=False kernel calls: IPC ack is sent as soon as the worker queues the
    launch (before GPU finishes), so the CPU does not stall on GPU execution.
  - start_timing / end_timing record CUDA events in-stream; end_timing(sync=True)
    waits for the event, providing accurate GPU-only elapsed time.

Run with:
    python examples/dev/microbench.py
"""

import time
import torch
import kernelbox as kbox

# ── Timing helpers ────────────────────────────────────────────────────────

def bench_wall(fn, warmup=50, iters=200, label="", print_result=True):
    """Wall-clock microbench. Returns sorted list of times in microseconds."""
    for _ in range(warmup):
        fn()
    times_us = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times_us.append((time.perf_counter_ns() - t0) / 1e3)
    times_us.sort()
    n = len(times_us)
    mn  = times_us[0]
    p10 = times_us[n // 10]
    p50 = times_us[n // 2]
    p90 = times_us[9 * n // 10]
    if print_result:
        print(f"  {label:<52s}  min={mn:6.1f}us  p10={p10:6.1f}us"
              f"  p50={p50:6.1f}us  p90={p90:6.1f}us")
    return times_us


def bench_gpu(run_fn, session, warmup=20, iters=100, label="",
              flush=False, print_result=True):
    """GPU-event benchmark via start_timing/end_timing (no IPC overhead).
    Inputs must already be set up (SETUP done) before calling this."""
    for _ in range(warmup):
        if flush:
            session.l2_flush()
        run_fn()
        session.sync()
    times_ms = []
    for _ in range(iters):
        if flush:
            session.l2_flush()  # synchronous: blocks until GPU flush done
        session.start_timing()
        run_fn()                # sync=False: returns after IPC ack, before GPU done
        elapsed = session.end_timing(sync=True)  # blocks until GPU end event
        times_ms.append(elapsed)
    times_ms.sort()
    n = len(times_ms)
    mn  = times_ms[0]
    p50 = times_ms[n // 2]
    p90 = times_ms[9 * n // 10]
    if print_result:
        tag = " [cold]" if flush else " [warm]"
        print(f"  {label+tag:<52s}  min={mn:.4f}ms  p50={p50:.4f}ms  p90={p90:.4f}ms")
    return times_ms


# ── Kernel source ─────────────────────────────────────────────────────────

SCALE_SRC = r"""
extern "C" __global__ void scale(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * 2.0f;
}
"""

# ── Setup: create session + inputs ONCE ──────────────────────────────────

print("=== Setup (inputs/VMM created once, reused across all benchmarks) ===")
print()

N_SMALL = 4096        # 16 KiB — trivially fast kernel, overhead-dominated
N_LARGE = 1 << 22    # 4M floats = 16 MiB — stresses memory bandwidth + L2

x_small = torch.randn(N_SMALL, device="cuda")
x_large = torch.randn(N_LARGE, device="cuda")
x_alt   = torch.randn(N_SMALL, device="cuda")  # same size, different pointer

# kernel_scratch_mib=0 (default): scale kernel does not need scratch.
# L2 flush uses the worker's own internal 256 MiB scratch (auto-allocated).
# NOTE: setting kernel_scratch_mib > 0 would auto-prepend a 'void *scratch'
# arg to the kernel call, which would break the scale kernel signature.
s = kbox.Session(kernel_source=SCALE_SRC)

# Trigger SETUP explicitly so timing sections don't include first-call overhead.
_ = s(x_small)
s.sync()
print(f"  Session ready, VMM mappings established.")
print()

# ── Section 1: IPC round-trip overhead ────────────────────────────────────

print("=== 1. IPC round-trip latency (socket connect + send + recv) ===")
print("  These involve no GPU work (stream is idle). Pure Python↔worker IPC cost.\n")

# sync() on idle stream: 1 socket round-trip, GPU cuStreamQuery returns instantly.
bench_wall(lambda: s.sync(),
           label="sync() [idle stream]")

# start_timing: records a CUDA event, no GPU wait (stream is idle anyway).
# Followed by end_timing(sync=False) to avoid leaving orphaned timing state.
def _start_end_nosync():
    s.start_timing()
    s.end_timing(sync=False)   # 2 round-trips total, no GPU stall

bench_wall(_start_end_nosync,
           label="start_timing + end_timing(sync=False) [2 round-trips]")

# end_timing(sync=True) waits for the GPU to reach the end event.
# On an idle stream this is instant GPU-side, so measures pure IPC+event cost.
def _start_end_sync():
    s.start_timing()
    s.end_timing(sync=True)    # 2 round-trips + GPU drain (instant on idle)

bench_wall(_start_end_sync,
           label="start_timing + end_timing(sync=True) [idle, 2 rt]")

print()

# ── Section 1b: Socket throughput (bulk sync calls) ──────────────────────

print("=== 1b. Socket throughput: noop() with varying payload ===")
print("  Config header is always 104 bytes. Extra payload tests send throughput.\n")

NOOP_ITERS = 10000
for payload in (0, 1024, 10240, 102400, 1024000):
    t0 = time.perf_counter()
    for _ in range(NOOP_ITERS):
        s.noop(payload_bytes=payload)
    wall_ms = (time.perf_counter() - t0) * 1000
    per_call = wall_ms / NOOP_ITERS * 1000  # us
    total_bytes = (104 + payload + 8) * NOOP_ITERS  # header + payload + response
    bw_mbs = total_bytes / wall_ms / 1000  # MB/s
    label = f"noop({payload:>7d}B payload)"
    print(f"  {label:<30s}  per-call={per_call:7.1f}us  bw={bw_mbs:7.1f} MB/s")

print()

# ── Section 2: L2 flush overhead (asynchronous) ──────────────────────────

print("=== 2. L2 flush overhead ===")
print("  l2_flush is ASYNCHRONOUS: GPU work is queued in-stream and CPU returns immediately.")
print("  Wall time is pure IPC overhead. Stream ordering ensures flush before next kernel.\n")

# Flush using internal 256 MiB scratch (auto-allocated by worker on first use).
bench_wall(lambda: s.l2_flush(count=1),
           label="l2_flush(count=1) [memset, async]")
bench_wall(lambda: s.l2_flush(count=1, clean=True),
           label="l2_flush(count=1, clean=True) [memset+read, async]")
bench_wall(lambda: s.l2_flush(count=1, clean_only=True),
           label="l2_flush(clean_only=True) [read only, async]")
bench_wall(lambda: s.l2_flush(count=2),
           label="l2_flush(count=2) [2x memset, async]")

print()

# ── Section 3: Kernel dispatch latency ────────────────────────────────────

print("=== 3. Kernel dispatch latency (N=4096, trivially fast GPU work) ===")
print("  sync=True: full round-trip + GPU execution + sync.")
print("  sync=False: returns as soon as worker ACKs the launch (before GPU done).\n")

bench_wall(lambda: s(x_small, sync=True),
           label="kernel(sync=True) [IPC + GPU + sync]")
bench_wall(lambda: s(x_small, sync=False),
           label="kernel(sync=False) [IPC only, GPU runs in background]")

print()

# ── Section 4: Pipelining verification ────────────────────────────────────
# If sync=False truly avoids CPU-GPU stalls, then N async launches + 1 sync
# should take approximately:
#   N × IPC_overhead + 1 × max(GPU_time_chain, 0)
# rather than N × (IPC + GPU_time).
#
# For a trivial kernel: GPU time per call ≈ IPC overhead (both ~tens of us).
# With pipelining: CPU sends 16 IPC requests while GPU runs them sequentially.
# Total ≈ 16 × IPC + GPU_time_for_16 (since GPU is pipelined, not stalled).

print("=== 4. Pipelining: N × async launches + sync() ===")
print("  Per-call cost should drop as N grows if GPU runs concurrently with IPC.\n")

for n in (1, 2, 4, 8, 16, 32):
    def _pipeline(n=n):
        for _ in range(n):
            s(x_small, sync=False)
        s.sync()
    times = bench_wall(_pipeline, warmup=30, iters=100,
                       label=f"pipeline({n:2d}×async + sync)", print_result=False)
    p50 = times[len(times) // 2]
    per = p50 / n
    print(f"  pipeline({n:2d}×async + sync):  p50={p50:6.1f}us  per-call={per:5.1f}us")

print()

# ── Section 5: VMM / input reuse ──────────────────────────────────────────
# The first call establishes SETUP (allocates VMM chunks, maps in worker).
# Repeated calls with the SAME tensor avoid DtoD copy (same data_ptr).
# Different tensor of same size only triggers a DtoD copy, not a full re-SETUP.

print("=== 5. VMM / input reuse ===")
print("  Goal: same tensor → no copy, different same-size tensor → DtoD copy only,")
print("  never a full re-SETUP (which would be much more expensive).\n")

_ = s(x_small, sync=True)   # ensure x_small is active input
bench_wall(lambda: s(x_small, sync=True),
           label="same tensor, repeated [no copy]")

_ = s(x_alt, sync=True)     # switch to different tensor → triggers DtoD copy
bench_wall(lambda: s(x_alt, sync=True),
           label="different tensor, same size [DtoD copy]")

_ = s(x_small, sync=True)   # switch back
bench_wall(lambda: s(x_small, sync=True),
           label="back to x_small [DtoD copy]")

print()

# ── Section 6: GPU-side kernel time: warm vs cold cache ───────────────────
# Use start_timing/end_timing to get pure GPU time (removes IPC overhead).
# Large tensor (16 MiB) exceeds typical L2 size → l2_flush should show delta.

print("=== 6. GPU kernel time: warm vs cold L2 cache (N=4M, 16 MiB) ===")
print("  start_timing/end_timing measures GPU time only (IPC overhead excluded).")
print("  l2_flush is async but stream-ordered, so flush completes before kernel runs.\n")

# Trigger SETUP for x_large before the timed section
_ = s(x_large, sync=True)
s.sync()

bench_gpu(lambda: s(x_large, sync=False), session=s, flush=False,
          label="scale 4M floats")
bench_gpu(lambda: s(x_large, sync=False), session=s, flush=True,
          label="scale 4M floats")

print()

# ── Section 7: benchmark() overhead breakdown ─────────────────────────────
# Compare total wall time per iteration for different flush configurations.
# Small tensor so GPU time ≈ 0 — overhead is dominated by IPC round-trips.
# Default benchmark(): per iter = l2_flush(sync) + start_timing + kernel(async)
#                                + end_timing(sync) = 4 IPC round-trips + 2 GPU syncs.

print("=== 7. benchmark() method: wall-time per iteration ===")
print("  Small tensor (N=4096) so GPU work is negligible.")
print("  Shows cost of l2_flush and timing events per iteration.\n")

_ = s(x_small, sync=True)   # ensure x_small is active

configs = [
    dict(l2_flush=0, l2_flush_per_iter=0,
         label="no flush        [start+kernel+end = 3 IPC]"),
    dict(l2_flush=1, l2_flush_per_iter=0,
         label="pre-flush only  [start+kernel+end = 3 IPC]"),
    dict(l2_flush=1, l2_flush_per_iter=1,
         label="flush per iter  [flush+start+kernel+end = 4 IPC]"),
]

for cfg in configs:
    lbl = cfg.pop("label")
    t0  = time.perf_counter()
    r   = s.benchmark(x_small, warmup=20, iters=100, **cfg)
    wall_ms = (time.perf_counter() - t0) * 1000
    per_iter = wall_ms / 100
    print(f"  {lbl:<48s}  wall={wall_ms:6.1f}ms  per_iter={per_iter:5.2f}ms  "
          f"GPU p50={r['median_ms']:.4f}ms")

print()
print("Done.")
