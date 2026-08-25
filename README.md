# neural-ca

> A from-scratch machine-learning research project: pick a genuinely under-explored model
> concept, research it thoroughly, implement and train it end-to-end on consumer hardware,
> ship it behind a web demo — and document every decision along the way.

**Status: Phase 2 done → Phase 3 (implementation). Concept: [Neural Cellular Automata](docs/research/01-nca-deep-dive.md).**
Tiny models where every pixel runs one shared, learned, local rule — and global structure
(growing images, self-repair, ultra-light segmentation) emerges. The day-by-day record lives in
[docs/devlog](docs/devlog/); the candidate analysis and NCA deep dive in
[docs/research](docs/research/).

**Project site:** [neural-ca on GitHub Pages](https://nikhlgoel.github.io/neural-ca/)

## Why this project exists

1. **Learn by building.** Implement a model completely from scratch — architecture, training
   loop, evaluation, serving — rather than fine-tuning something off the shelf.
2. **Work where the crowd isn't.** Target a model family that is under-explored, so the code,
   results, and checkpoints are useful to other people — not the 10,000th MNIST CNN.
3. **Show the work.** Decisions, failures, and fixes are recorded as they happen
   (devlog + ADRs + PRD). The process is the proof of authorship.

## Quick start

The only prerequisites are [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/) —
uv installs the right Python and all pinned dependencies automatically.

```bash
git clone <repo-url>
cd neural-ca
uv sync                                # .venv + locked deps (CUDA torch on Windows, CPU on Linux)
uv run python scripts/env_report.py    # sanity-check Python / PyTorch / GPU
uv run pytest                          # smoke tests
```

Contributing? Also run `uv run pre-commit install` once — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository map

```text
configs/          # YAML experiment configs — the single source of truth for runs
data/             # datasets: gitignored, fetched by scripts (see data/README.md)
docs/
  PRD.md          # product requirements — what we're building and why
  DESIGN.md       # technical design — how it's built
  TECH_STACK.md   # stack inventory and rationale
  decisions/      # ADRs — one file per hard-to-reverse decision
  devlog/         # dated work journal — the project's paper trail
  research/       # literature notes and idea analysis
notebooks/        # exploratory only — results graduate to src/
scripts/          # runnable entry points (env report; later: train / eval / export)
src/neural_ca/    # the Python package (the NCA implementation lives here)
tests/            # pytest suite
```

Planned once the model exists: `apps/api` (inference service) and `apps/web` (demo frontend)
— direction documented in [docs/TECH_STACK.md](docs/TECH_STACK.md).

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Repo scaffold, tooling, environment, docs skeleton | ✅ 2026-07-04 |
| 1 | Model concept chosen (NCA) + deep literature research | ✅ 2026-07-04 |
| 2 | PRD + design finalised; scope locked (Tier A) | ✅ 2026-07-04 |
| 3 | Model implemented from scratch + training pipeline | 🔄 in progress |
| 4 | Experiments, ablations, evaluation report | ⏳ |
| 5 | Inference API + web demo frontend | ⏳ |
| 6 | Public release: polish, checkpoints, tech report | ⏳ |

## Documentation

| Doc | Purpose |
|---|---|
| [PRD](docs/PRD.md) | Goals, users, success metrics, risks, milestones |
| [Design](docs/DESIGN.md) | Architecture: model, data, training, serving, reproducibility |
| [Tech stack](docs/TECH_STACK.md) | Every tool used and why |
| [Decisions](docs/decisions/) | ADRs with explicit trade-off analysis |
| [Devlog](docs/devlog/) | Dated journal: what was done, what broke, how it was fixed |
| [Research](docs/research/) | Idea shortlist and literature deep dives |

## License

[Apache-2.0](LICENSE) — use it freely with attribution (see [NOTICE](NOTICE)).
