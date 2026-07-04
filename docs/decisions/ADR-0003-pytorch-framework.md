# ADR-0003: PyTorch as the ML framework

**Status:** Accepted
**Date:** 2026-07-04
**Deciders:** white-dev (project owner)

## Context

The project implements a research model from scratch, on native Windows, on a single consumer
GPU (RTX 3050, 6 GB), with a primary goal of *learning* and a secondary goal of producing
artifacts others can build on. The framework choice determines: available reference
implementations, debugging experience, Windows support, and export paths for the web demo.

## Options considered

### Option A: PyTorch 2.x

| Dimension | Assessment |
|---|---|
| Learning fit | Excellent — define-by-run; debuggable line-by-line with a normal debugger |
| Research ecosystem | Dominant — the overwhelming majority of recent papers ship PyTorch code |
| Windows + consumer CUDA | First-class (cu128 wheels) |
| Performance at our scale | More than enough: AMP/bf16, `torch.compile` when needed |
| Export / serving | ONNX + TorchScript → FastAPI server or in-browser onnxruntime-web |

**Pros:** every candidate research direction has PyTorch baselines to compare against; largest
debugging community; einops/TensorBoard integrate trivially.
**Cons:** eager mode hides some performance intuition that JAX forces you to learn.

### Option B: JAX

**Pros:** functional core (`grad`/`vmap`/`jit`) teaches program transformations deeply.
**Cons:** no supported native-Windows GPU path (WSL2 required); far fewer reference
implementations in under-explored niches — exactly where we need every baseline we can get.

### Option C: TensorFlow / Keras 3

**Pros:** Keras 3 is beginner-friendly; TF.js has a browser story.
**Cons:** research mindshare has moved away; custom from-scratch training is clunkier; not the
ecosystem to invest learning in as of 2026.

## Trade-off analysis

JAX is the only serious alternative and loses on the two constraints that actually bind:
native Windows CUDA support, and reference-implementation availability in under-explored
areas. PyTorch's weaker "forced rigor" is compensated by project rules (from-scratch modules,
shape-explicit code, ablations).

## Consequences

- All model code targets PyTorch ≥ 2.7; bf16 mixed precision by default on the Ampere GPU.
- Web-demo export path is ONNX (or TorchScript), to be verified **early** in Phase 5.
- If a future direction demands JAX (e.g., a TPU grant), that is a new ADR.

## Action items

1. [x] Install the cu128 build and verify `torch.cuda.is_available()` on the dev machine
2. [ ] Phase 3: adopt the bf16/AMP training policy in the training loop
