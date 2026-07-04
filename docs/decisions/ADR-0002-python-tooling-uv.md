# ADR-0002: uv for the Python toolchain (interpreter, venv, dependencies, lockfile)

**Status:** Accepted
**Date:** 2026-07-04
**Deciders:** white-dev (project owner)

## Context

The project needs: a pinned Python version on Windows, an isolated environment, reproducible
dependency resolution (contributors and CI must get identical environments), and a
low-friction daily workflow. It must also handle PyTorch's unusual packaging — separate wheel
indexes per CUDA version, different builds per platform. Constraint: solo maintainer, so
minimal moving parts wins.

## Options considered

### Option A: uv

| Dimension | Assessment |
|---|---|
| Complexity | Low — one binary does python-install + venv + lock + run |
| Speed | Excellent — Rust resolver, seconds even for torch-sized graphs |
| Reproducibility | `uv.lock` is a real cross-platform lockfile, first-class |
| PyTorch index handling | Native: `[tool.uv.sources]` + per-platform markers |
| Maturity (2026) | High — de-facto standard for new Python projects |

**Pros:** single tool; lockfile covers per-platform torch builds; manages Python itself (no
admin installs); already present on the dev machine.
**Cons:** younger than pip/conda; some older tutorials assume pip.

### Option B: conda / mamba

**Pros:** ubiquitous in older ML tutorials; can manage non-Python binaries.
**Cons:** slow solves; `environment.yml` is weaker than a true lockfile; Anaconda channel
licensing confusion; heavyweight. Decisive: PyTorch deprecated its official conda packages —
pip wheels are the maintained distribution path, and modern wheels bundle the CUDA runtime,
which removes conda's main historical advantage.

### Option C: pip + venv + requirements.txt (+ pip-tools)

**Pros:** zero new concepts.
**Cons:** no Python version management; per-platform torch pinning is manual and error-prone;
multi-file requirements churn; slow.

### Option D: Poetry

**Pros:** mature lockfile workflow.
**Cons:** historically poor fit for torch's custom indexes; doesn't manage interpreters;
slower; ecosystem momentum has moved to uv.

## Trade-off analysis

The decisive requirements are (1) reproducible cross-platform torch — CUDA locally, CPU in CI,
from the *same* lockfile — and (2) minimal moving parts for a solo maintainer. Only uv
satisfies both natively.

## Consequences

- `uv.lock` is committed; `uv sync` is the entire setup story for any contributor or CI runner.
- Python 3.12 pinned via `.python-version` (best compiled-wheel coverage; 3.13/3.14 still lag
  for some CUDA-adjacent libraries).
- We depend on Astral maintaining uv (low risk; escape hatch: `uv export` → requirements.txt).

## Action items

1. [x] Pin Python 3.12 via `.python-version`
2. [x] Configure per-platform torch indexes in `pyproject.toml`
3. [x] Commit `uv.lock`; CI installs via `uv sync`
