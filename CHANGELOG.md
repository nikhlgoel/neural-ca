# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added

- Project scaffold: `src/` package layout, smoke tests, scripts, config system skeleton.
- Tooling: uv-managed Python 3.12 environment, ruff (lint + format), pytest, pre-commit hooks, GitHub Actions CI.
- PyTorch environment with automatic wheel selection: CUDA 12.8 on Windows, CPU on Linux/CI.
- Documentation suite: README, PRD & design docs, tech-stack doc, ADRs 0001–0003, devlog, research idea shortlist.
- Research: chose **Neural Cellular Automata** as the model concept; added a two-register deep-dive
  (mechanism, prior work, open gaps, datasets, compute-fit, three-tier scope) in `docs/research/`.
- Named the project **`neural-ca`** (Python import `neural_ca`); renamed the package from the
  `mlmodel` placeholder and aligned package metadata to the GitHub identity.
- Design (Phase 2): finalised the Tier-A Growing NCA architecture, data, training, and evaluation
  plan in `docs/DESIGN.md`; locked scope (Tier A, extensible to Tier B).
- Config system: typed pydantic schema (`neural_ca/config.py`) with fail-fast validation, first
  experiment config `configs/grow_emoji.yaml`, and reproducibility utils (`set_seed`,
  `resolve_device`).
- Package skeleton: `neural_ca/{model,data,training,eval,utils}` subpackages.
- Model (Phase 3): implemented the Growing NCA from scratch (`neural_ca/model/nca.py`, ~8.3k
  params), procedural targets + seed state (`neural_ca/data/`), and the training loop with sample
  pool, damage, and gradient-normalisation (`neural_ca/training/`), plus `scripts/train.py`.
- First training run (heart target): loss 0.063 → 0.018 in 400 steps (~2 min), peak 1.67 GB VRAM.
- Visualisation (`neural_ca/eval/render.py`, `scripts/visualize.py`): render a grown NCA as a
  still PNG, a growth GIF, and a 3D alpha surface. A 1000-step heart run reached loss 0.0085.
- Training upgrades: checkpointing (`neural_ca/training/checkpoint.py`), a late LR schedule, and
  TensorBoard logging (`train --log-dir`).
- ONNX export (`neural_ca/eval/export.py`, `scripts/export_onnx.py`): export the deterministic
  step core and verify it against PyTorch (~1e-6). Exported heart model is 5.9 KB. See ADR-0004.
