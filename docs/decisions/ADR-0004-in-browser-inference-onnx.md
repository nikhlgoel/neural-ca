# ADR-0004: In-browser inference via ONNX (export the deterministic step, mask in JS)

**Status:** Accepted
**Date:** 2026-07-04
**Deciders:** nikhlgoel

## Context

Phase 5 ships a live web demo: visitors watch the NCA grow and self-heal, rotate its 3D surface,
and give it tasks (choose a shape, damage it). The model is tiny (~8.3k params), so inference can
run in the visitor's browser — giving zero server cost, no latency, and a site that can't be taken
down. The catch: one NCA step includes a **stochastic fire mask** and **alive-masking**, and
browser ONNX runtimes handle random ops inconsistently.

## Decision

Export only the **deterministic core** of one step to ONNX — perception (fixed conv) + the update
MLP → the per-cell delta. The stochastic fire mask, the residual add, and alive-masking are cheap
element-wise ops done in **JavaScript**. Run the ONNX model with **onnxruntime-web** (WebGPU
backend, WASM fallback) and loop the step in JS. Host as a **static site**.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **A: export deterministic core, mask/RNG in JS** *(chosen)* | Tiny, portable ONNX (**5.9 KB**); no random ops in graph; verified ONNX==PyTorch to ~1e-6 | JS implements the (trivial) masking |
| B: export the whole step incl. randomness | one ONNX call per step | `RandomUniformLike` support in onnxruntime-web is inconsistent → brittle |
| C: reimplement the NCA in a WebGL/WebGPU shader | fastest possible | hand-duplicates the model → drift risk; more code. Keep as a later optimisation |
| D: server-side inference (FastAPI) | no client compute | costs money, adds latency, defeats the free-static-site goal |

## Consequences

- The ONNX graph is trivial and portable; `scripts/export_onnx.py` re-verifies ONNX==PyTorch on
  every export (currently max abs diff ~9.5e-07).
- JS implements only: `fire = rand < fire_rate`, `state += delta * fire`, and the alive max-pool.
- Opset 18, torch.export (dynamo) exporter; dynamic batch/H/W so the browser can choose grid size.
- Live *training* progress (Phase 5) uses a separate small WebSocket channel; **inference stays
  client-side**.

## Action items

1. [x] Export + verify — `neural_ca/eval/export.py`, `scripts/export_onnx.py`
2. [ ] Phase 5: onnxruntime-web step loop + JS masking in `apps/web`
