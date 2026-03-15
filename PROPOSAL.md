# KernelBox Pipeline: From `torch.compile` to Custom CUDA in Three Commands

**One-liner:** We built a pipeline that automatically decomposes any PyTorch model's compiled aten graph into individually testable, benchmarkable CUDA kernel targets — then lets you replace them one-by-one with hand-tuned kernels and inject them back.

---

## The Problem

`torch.compile` gives you a monolithic blob of aten ops. You can see the ops, but you can't easily:
- **Isolate** a single op with real tensor data to iterate on
- **Replace** it with a custom CUDA kernel and verify correctness
- **Inject** the validated kernel back into the compiled graph

So kernel optimization stays locked behind framework abstractions, and you're stuck choosing between "use what the compiler gives you" or "rewrite everything from scratch."

## What We Built

A three-phase pipeline on top of [KernelBox](https://github.com/ademeure/kernelbox-dev) that turns any `torch.compile` aten graph into a CUDA kernel development workflow:

**Phase 1 — Decompose:** Parse the aten graph, run it with real inputs, and generate one self-contained kernelbox test file per op (with its own `.h5` fixture data). Each file runs standalone with `kbox iterate`.

**Phase 2 — Replace:** Swap any aten op with an inline CUDA kernel using built-in templates (15 ops: gelu, relu, silu, layer_norm, add, mul, etc.) or write your own. Iterate with hot-reload, correctness checking, and isolated benchmarking.

**Phase 3 — Inject:** Take your validated kernel and patch it back into the original aten graph. The pipeline generates `load_inline` C++ wrappers, preserves all variable names, and leaves GEMMs/attention untouched.

## Demo: Full NanoGPT Forward Pass

We decomposed the complete nanoGPT forward pass (2-layer transformer) into **32 per-op test files** — embedding, attention, MLP, layer norm, logit projection — each with real tensor data and one-command correctness verification. Every op is independently editable and benchmarkable.

```bash
# Decompose any model
python tools/kbox_pipeline.py generate --graph compiled_model.py --h5 data.h5 --output-dir ops/

# Iterate on a single op with hot-reload
kbox iterate ops/test_gelu_1_cuda.py --once --bench

# Inject validated kernels back
python tools/kbox_pipeline.py inject --graph compiled_model.py --kernel gelu_1=ops/test_gelu_1_cuda.py --output patched.py
```

## Why It Matters

This closes the loop between **compiler-generated code** and **hand-optimized kernels**. Instead of choosing one or the other, you get the compiler's output as a starting point, surgically replace the ops that matter, and keep everything else. It's `perf` for GPU kernels — profile, isolate, optimize, reintegrate.
