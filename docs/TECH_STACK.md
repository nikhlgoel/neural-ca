# Tech Stack

Every technology in this project, what it does, and why it was chosen. Deep trade-off analyses
live in [decisions/](decisions/) (ADRs); this file is the inventory.

## Decided (Phase 0)

| Layer | Choice | Why (short) | Deep dive |
|---|---|---|---|
| Language | Python 3.12 | ML lingua franca; 3.12 has the best compiled-wheel coverage across the ecosystem today (3.13/3.14 still lag for some CUDA-adjacent libs) | [ADR-0002](decisions/ADR-0002-python-tooling-uv.md) |
| Env & packaging | **uv** | One tool for Python install + venv + lockfile + task running; resolves torch-sized graphs in seconds; `uv.lock` gives contributor-proof reproducibility | [ADR-0002](decisions/ADR-0002-python-tooling-uv.md) |
| ML framework | **PyTorch 2.x** | Research standard — nearly every paper we might build on ships PyTorch code; define-by-run is ideal for from-scratch learning | [ADR-0003](decisions/ADR-0003-pytorch-framework.md) |
| CUDA build selection | uv index markers | Windows → cu128 wheels (RTX 3050, driver 610); Linux/CI → CPU wheels; same lockfile serves both | `pyproject.toml` `[tool.uv.sources]` |
| Tensor ergonomics | einops | Shape-explicit tensor ops read like the math; pedagogically better than `.view/.permute` chains | — |
| Config | YAML + pydantic v2 | Declarative experiments, validated at load time; Hydra rejected as over-engineered for a solo repo | — |
| Experiment tracking | TensorBoard | Local, free, no account; W&B optional later (would need an ADR — external service) | — |
| Lint + format | ruff | One fast tool replacing black + isort + flake8 | — |
| Tests | pytest | Standard; smoke tests now, per-module unit tests in Phase 3 | — |
| Git hygiene | pre-commit | Auto lint/format on commit + blocks files > 1 MB (keeps data/weights out of git) | — |
| CI | GitHub Actions | Free for public repos; runs the same three quality gates as local dev | `.github/workflows/ci.yml` |
| Docs | Markdown in-repo (PRD, design, ADRs, devlog, changelog) | Docs version with the code they describe; zero infrastructure | [ADR-0001](decisions/ADR-0001-record-architecture-decisions.md) |
| License | Apache-2.0 + NOTICE | Permissive (useful to others) with explicit attribution requirement and patent grant | README §License |

## Planned (direction decided; finalised after the model concept is fixed)

| Layer | Direction | Notes |
|---|---|---|
| Inference service | FastAPI + exported model (ONNX or TorchScript) | Contract designed in Phase 5 |
| Demo frontend | React + Vite + TypeScript + Tailwind in `apps/web` | If the concept allows, **in-browser inference** via onnxruntime-web → free static hosting (GitHub Pages), no server to pay for or keep alive |
| Scratch demos | Gradio | Throwaway UIs during research only — never the shipped product |
| Dataset / weights hosting | Hugging Face Hub or GitHub Releases | ADR at Phase 6 |

## Hardware profile (the budget everything must fit)

| Resource | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop — 6 GB VRAM, Ampere (sm_86), bf16 ✓ |
| Driver | 610.47 (CUDA 12.x-compatible) |
| OS | Windows 11 |
| Disk | ~340 GB free on the project drive |
| Implication | Sweet spot ≤ ~50 M params with AdamW + bf16; gradient accumulation instead of large batches; datasets that fit on local disk |
