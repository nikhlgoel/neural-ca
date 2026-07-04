# Technical Design — neural-ca

**Status:** design finalised for **Tier A** (grow + regenerate a single target), architected to
extend to **Tier B** (class-conditional). Companion docs: [PRD](PRD.md) ·
[TECH_STACK](TECH_STACK.md) · [ADRs](decisions/).

## 1. System overview

```text
configs/*.yaml ──▶ train script ──▶ checkpoints/ ──▶ eval script ──▶ metrics report
                      │                   │
                      ▼                   ▼
              TensorBoard (runs/)   export (ONNX / TorchScript)
                                          │
                                          ▼
                              apps/api (inference) ◀── apps/web (demo UI)
```

## 2. Repository architecture *(decided)*

- **src layout** (`src/neural_ca/`): the package must be installed (editable, via `uv sync`) to
  be importable — prevents accidental "works only from repo root" imports and mirrors how
  users consume it.
- **Configs are the single source of truth** for experiments: code defines *mechanisms*,
  YAML defines *choices*. Configs are validated by pydantic schemas at load time, so a typo
  fails loudly before a 10-hour run, not silently during it.
- **Scripts are thin**: argument parsing + config loading + calls into `src/`. All logic lives
  in the package, where it is importable and testable.
- **Planned package layout** (finalised in Phase 2):

  ```text
  src/neural_ca/
    model/      # architecture modules, written from scratch
    data/       # dataset download, preprocessing, loaders
    training/   # loop, optimizer/schedule, checkpointing, AMP policy
    eval/       # metrics and evaluation harness
    utils/      # seeding, device selection, logging helpers
  ```

## 3. Model architecture *(Tier A — Growing NCA)*

The model **is** the per-cell update rule; the grid and its iteration are fixed scaffolding.

**State** — a tensor `(B, C, H, W)`, `C = 16` channels per cell:
- `0–3`: visible **RGBA** (channel 3, alpha, also flags a cell as *alive*)
- `4–15`: **hidden** channels — scratch space for coordination, no prescribed meaning

**One update step**
1. **Perception** *(fixed, no parameters)* — depthwise-convolve each channel with three 3×3
   filters (identity, Sobel-x, Sobel-y) → `3C = 48` channels: each cell sees itself and the
   local gradients around it. Fixed filters keep it a true CA — only the *rule* is learned.
2. **Update rule** *(the only learned part)* — a per-cell MLP as two 1×1 convolutions,
   `48 → 128 → 16` with ReLU between. The **final layer is zero-initialised**, so an untrained
   cell proposes "no change" and training departs from a stable identity.
3. **Residual, stochastic apply** — `state += fire_mask ⊙ Δ`, with `fire_mask` an independent
   Bernoulli(`fire_rate = 0.5`) per cell (asynchronous updates; also a regulariser).
4. **Alive masking** — a cell is alive iff the 3×3 max-pool of its alpha exceeds
   `alive_threshold = 0.1`; cells with no living neighbour are zeroed, so structure grows only
   at living edges.

Iterated `T ~ Uniform[64, 96]` steps per forward pass.

**Parameter budget** — only the MLP has weights: `(48·128 + 128) + (128·16 + 16) ≈ 8.3k`.

**Activation-memory budget** — backprop unrolls all `T` steps, and cost is dominated by the
128-channel MLP hidden activations stored per step: `~ T·B·hidden·H·W`. At the Tier-A defaults
(`T=96, B=8, 72×72`) that is roughly **1–2 GB in fp32**, about half under bf16 — inside the 6 GB
budget with room for gradients and optimiser state. Levers if it gets tight: lower `T`, `batch`,
or `hidden`, or add gradient checkpointing. Verified with real measurements in Phase 3.

**Tier-B seam (class-conditional).** The update module will accept an optional `cond` argument;
a class embedding is concatenated to the perception vector (`48 → 48+E`) or applied as FiLM on
the MLP — so a *single* model can grow a *chosen* target. Unused in Tier A, but wired so Tier B
needs no rewrite.

## 4. Data pipeline *(Tier A)*

**Target** — a single RGBA image on a transparent background, `data.size` (default 40), zero-
padded by `data.pad` (default 16) → a `72×72` grid.

**Builtin targets, no external assets** — to keep the repo license-clean, default targets are
**rendered procedurally** (e.g. `heart`, `circle`) by a generator in `neural_ca/data/`; no
downloaded emoji, no licensing questions. A file path is also accepted
(`data.target: path/to.png`). If we later use Twemoji it is CC-BY 4.0 and gets an attributed row
in [`data/README.md`](../data/README.md) (satisfies PRD NFR3).

**Seed state** — one living cell at the grid centre: alpha and all hidden channels set to 1, RGB
0, everything else 0.

**No train/val split** in the single-target setup — "generalisation" here means *persistence and
regeneration*, evaluated by perturbation (§6), not a held-out set. Tier C (segmentation)
reintroduces standard splits.

## 5. Training *(Tier A)*

**Objective** — pixel-wise MSE between rendered RGBA (`state[:4]`) and the target after `T` steps.

**Sample pool (persistence)** — keep `pool_size = 1024` past end-states. Each batch
(`batch_size = 8`) is drawn from the pool; the single worst (highest-loss) sample is reset to the
clean seed (anchors "grow from scratch"), the rest continue from where they were; results are
written back. The model thus learns to *reach and hold* the target over many steps, not merely to
peak at step `T`.

**Damage (regeneration)** — when `damage = true`, a random circular region is zeroed in a subset
of the batch's best samples before the forward pass, forcing regrowth.

**Stability policy** (what actually makes NCAs train):
- residual updates + zero-init output → start from a stable fixed point;
- **per-parameter gradient normalisation** (`grad /= ‖grad‖ + 1e-8`) — NCA gradients span orders
  of magnitude; this is the standard cure for divergence;
- stochastic firing and randomised `T` as regularisers.

**Optimiser** — Adam, `lr = 2e-3`, step-decay schedule; `~8000` iterations → **minutes** on the
RTX 3050. **Precision** — bf16 autocast on CUDA, fp32 on CPU. **Checkpoint/resume** persists
model + optimiser + pool + RNG to `checkpoints/`. Gradient accumulation is available but
unneeded at this batch size.

## 6. Evaluation *(Tier A)*

**Metrics**
- **Fidelity** — final RGBA MSE, plus a rendered still/GIF for the perceptual check.
- **Persistence** — MSE at `T = 200` and `500` steps (does the pattern hold, or diverge?).
- **Regeneration** — MSE after damaging the converged state and running further steps.

**Ablations** (≥ 3, satisfying PRD G2; 3 seeds where feasible) — change one factor each:
1. hidden channels `C-4 ∈ {8, 12, 16}` — capacity vs. stability;
2. sample pool **on/off** — does persistence training matter?
3. damage augmentation **on/off** — is regeneration learned or emergent?
4. perception: fixed Sobel vs. a learned 3×3 — how much do hand-set filters help?

**Seed policy** — report mean ± range over ≥ 3 seeds for headline numbers; every figure cites the
run directory that produced it (§8). Sanity baseline: the qualitative Growing NCA (Distill 2020)
result on a comparable target.

## 7. Inference & serving *(decided — [ADR-0004](decisions/ADR-0004-in-browser-inference-onnx.md))*

Inference runs **in the browser** — the model is ~8.3k params. We export the *deterministic core*
of one step (perception + update MLP → per-cell delta) to ONNX (opset 18, dynamic batch/H/W); the
stochastic fire mask, residual add, and alive-masking are done in JavaScript. `scripts/export_onnx.py`
verifies the ONNX output matches PyTorch (currently ~1e-6) on every export, and the exported model
is **5.9 KB**. Runtime: onnxruntime-web (WebGPU, WASM fallback); hosting: a static site, no server.
A separate WebSocket streams *live training* progress for the demo, but inference itself is fully
client-side.

## 8. Reproducibility standards *(decided)*

1. One seeding helper used by every entry point (`seed` in config; seeds Python, NumPy, torch).
2. `uv.lock` pins the environment; CI proves a cold machine can rebuild it.
3. Every run directory records: git commit, resolved config, seed, hardware, wall-clock.
4. Any number reported in docs must cite the run directory that produced it.

## 9. Observability *(decided)*

TensorBoard scalars/images per run under `runs/`; devlog entries link to runs. Weights & Biases
may be added later — external service, so it needs an ADR.
