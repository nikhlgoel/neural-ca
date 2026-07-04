# Devlog — 2026-07-04 — Phase 3: implementation & first training run

**Goal:** implement the Growing NCA from scratch and get a first real training run on the RTX 3050.

## Built (from scratch)

- **Model** — [`neural_ca/model/nca.py`](../../src/neural_ca/model/nca.py): fixed Sobel/identity
  perception (depthwise grouped conv, no params), a `48→128→16` 1×1-conv update MLP with
  zero-initialised output, stochastic firing, and alive-masking. **8,336 learned parameters.**
- **Data** — procedural targets (`heart`/`circle`/`square`, no external assets) and the
  single-live-cell seed state.
- **Training** — sample pool (persistence), circular damage (regeneration), random step count,
  bf16 autocast on CUDA, and per-parameter gradient normalisation; driven by `scripts/train.py`.
- **Tests** — 18 total (param count, forward shape, zero-init stability, target/seed shapes,
  pool/damage, and an end-to-end CPU training smoke test).

## First run — `configs/grow_emoji.yaml` (heart), 400 steps

| step | loss |
|---|---|
| 1 | 0.06314 |
| 100 | 0.04327 |
| 200 | 0.03496 |
| 300 | 0.02082 |
| 400 | 0.01808 |

- **Loss fell ~3.5×** and is still dropping — the model is learning to grow the heart.
- **Wall-clock:** 119 s for 400 steps (~0.30 s/step) → a full 8,000-step run ≈ **40 minutes**.
- **Peak VRAM: 1,669 MB (1.67 GB)** — inside the design's 1–2 GB prediction (DESIGN §3), with
  ~4.3 GB headroom on the 6 GB card. No memory tricks needed.

## Issues faced & resolved

| Issue | Resolution |
|---|---|
| `float(loss)` on a grad-tracking tensor raised a UserWarning | log with `loss.detach().item()` |
| CPU pool vs. GPU compute: reordering the pool indices with a CUDA argsort would mix devices | keep pool indices on CPU (`idx[order.cpu()]`), states on GPU |
| Gradient spikes (classic NCA failure mode) | per-parameter grad normalisation `g /= ‖g‖ + 1e-8` — training is stable from step 1 |

## Next steps (finish Phase 3 → Phase 4)

- [x] Visualisation (`neural_ca/eval/render.py` + `scripts/visualize.py`): still PNG, growth GIF,
      and a 3D alpha surface. A 1000-step run reached loss **0.00845** and a clearly recognisable
      heart. (Bug found + fixed: `.numpy()` on a grad tensor → `.detach()` in the renderers.)
- [ ] Checkpoint save/load (model + optimiser + pool + RNG) to `checkpoints/`.
- [ ] Full 8,000-step run; log to TensorBoard under `runs/`.
- [ ] Persistence + regeneration metrics, then the ablation sweep (hidden channels, pool on/off,
      damage on/off) — Phase 4.
