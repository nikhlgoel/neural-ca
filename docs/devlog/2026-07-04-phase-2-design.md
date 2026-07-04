# Devlog — 2026-07-04 — Phase 2: design & scope lock

**Goal:** turn the NCA concept into a concrete, buildable design and lock scope, so Phase 3 is
pure implementation.

## Decisions

- **Scope locked to Tier A** (grow + regenerate a single target), architected so **Tier B**
  (class-conditional) needs no rewrite — the update module will take an optional `cond` argument.
  Rationale in [research/01-nca-deep-dive.md](../research/01-nca-deep-dive.md) §9.
- **Architecture fixed** ([DESIGN.md](../DESIGN.md) §3): 16-channel state (4 RGBA + 12 hidden),
  fixed Sobel/identity perception, a `48→128→16` 1×1-conv update MLP with zero-initialised
  output, stochastic firing, alive-masking. **~8.3k learned parameters.**
- **Config-as-contract**: a typed pydantic schema validates every run's YAML at load time
  (`extra="forbid"` rejects typos before a run starts). First config: `configs/grow_emoji.yaml`.

## Built

- `neural_ca/config.py` — `Config` / `ModelConfig` / `DataConfig` / `TrainConfig` + `load_config`.
- `neural_ca/utils/` — `set_seed` (Python/NumPy/torch) and `resolve_device`, the shared
  reproducibility helpers every entry point will use ([DESIGN.md](../DESIGN.md) §8).
- `neural_ca/{model,data,training,eval}/` — package skeleton, each with a docstring naming what
  Phase 3 fills in.
- Tests: config loads + rejects unknown keys; seeding is reproducible; device resolves.

## Verification

- `uv run pytest` — **7 passed** (3 smoke + 2 config + 2 utils).
- `uv run ruff check .` and `ruff format --check .` — clean.

## Memory-budget sanity (to measure for real in Phase 3)

Update MLP ≈ 8.3k params. Activation memory unrolls `T` steps and is dominated by the 128-channel
MLP hidden layer (`~T·B·hidden·H·W`); at the Tier-A defaults this is ~1–2 GB fp32 (about half
under bf16) — inside the 6 GB budget, with gradient checkpointing as the fallback lever.

## Next steps (Phase 3)

- [ ] Implement perception + update rule in `neural_ca/model/` from scratch, with a per-step
      shape test.
- [ ] Procedural target generator + seed-state builder in `neural_ca/data/`.
- [ ] Training loop in `neural_ca/training/`: sample pool, damage, gradient-normalisation, bf16,
      checkpoints.
- [ ] `scripts/train.py` wiring config → training; first real run on the RTX 3050, and record
      measured VRAM + wall-clock here.
