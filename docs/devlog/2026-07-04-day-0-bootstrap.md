# Devlog — 2026-07-04 — Day 0: project bootstrap

**Goal:** stand up a complete, professional project skeleton — environment, tooling, CI,
documentation system — and research candidate model concepts, so that all future work is
building the model, not fighting infrastructure.

## Environment audit (starting point)

| Item | Found |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, 6 GB VRAM, driver 610.47 |
| Python | Only the Windows Store shim on PATH (3.13.13) |
| Tooling already present | uv 0.11.7, git 2.53, Node 25.8, gh CLI |
| Disk | ~340 GB free on W: |

## What was done

1. **Decided the stack** — uv + Python 3.12 + PyTorch cu128, ruff/pytest/pre-commit, GitHub
   Actions, markdown docs-in-repo, Apache-2.0. Trade-offs recorded in
   [ADR-0002](../decisions/ADR-0002-python-tooling-uv.md) and
   [ADR-0003](../decisions/ADR-0003-pytorch-framework.md); inventory in
   [TECH_STACK.md](../TECH_STACK.md).
2. **Built the scaffold** — src-layout package (`mlmodel`, placeholder name), smoke tests,
   `scripts/env_report.py`, config skeleton, data/notebook policies, CI workflow, full docs
   suite (README, PRD, DESIGN, CONTRIBUTING, CoC, CHANGELOG, LICENSE/NOTICE).
3. **Created the environment** — `uv sync` installed uv-managed CPython 3.12.13 and 52 locked
   packages, including the CUDA build of PyTorch selected automatically per-platform
   (`[tool.uv.sources]` markers: Windows→cu128, Linux/CI→CPU).
4. **Researched model concepts** — 2025–26 literature sweep across five candidate families;
   analysis and scoring in [research/00-idea-shortlist.md](../research/00-idea-shortlist.md).
   Recommendation: Neural Cellular Automata, with Tiny Recursive Models as strong #2.

## Verification (all green)

```text
python : 3.12.13          torch : 2.11.0+cu128
cuda   : yes — RTX 3050 6GB Laptop (6.0 GiB, sm_86)   bf16 : True
pytest : 3 passed in 4.40s
ruff   : format clean, all lint checks passed
```

## Issues faced & how they were resolved

| Issue | Resolution |
|---|---|
| System Python is the Store shim (and 3.13, where some CUDA-adjacent wheels still lag) | Ignored system Python entirely; uv downloads and pins its own CPython 3.12.13 (`.python-version`) — contributors get the same |
| PyTorch CUDA wheel is 2.6 GiB | Accepted (one-time, ~5 min); CI avoids it via CPU wheels from the same lockfile |
| uv warning: "Failed to hardlink files; falling back to full copy" | uv's cache lives on C: while the project is on W: — cross-drive hardlinks are impossible, so uv copies. Cosmetic; can be silenced with `UV_LINK_MODE=copy` |
| Guessed pre-commit hook versions were stale | `uv run pre-commit autoupdate` corrected them (hooks v5→v6, ruff v0.12.2→v0.15.20) — lesson: pin via autoupdate, don't hand-pick |
| Global git identity (`nikhlgoel`) differs from project email (white.dev.sc@…) | Left as-is for now; **decide before publishing** (fix: `git config user.name/user.email` locally in this repo) |

## Method note

Every decision here was made and reviewed by the team; this devlog and the ADRs record the
reasoning, not just the output, so the whole project history stays auditable.

## Update — concept chosen: Neural Cellular Automata

Owner reviewed the shortlist and **chose Neural Cellular Automata** (with the option to
understand it more deeply first). Actions taken:

- Ran a second, focused literature sweep on NCA (mechanism, Med-NCA/M3D-NCA baselines,
  open problems, browser-inference precedent) and wrote the deep-dive
  [research/01-nca-deep-dive.md](../research/01-nca-deep-dive.md) — written in two registers
  (plain-language boxes + technical detail) since learning is the primary goal.
- Recorded the decision + rationale in the shortlist's Decision section; updated the PRD
  problem statement to the NCA concept.
- Proposed a three-tier scope (A: reproduce growing/regenerating NCA; B: class-conditional
  NCA — the novel twist; C: lightweight medical segmentation stretch). Recommendation:
  commit to A, aim for B.

**Key finding that validated the choice:** an emoji-scale NCA trains in *minutes* on the 3050
and the model is tiny enough to run entirely in the browser — so both "many ablations" and
"zero-cost live demo" are realistic.

## Next steps

- [ ] Phase 2: lock the scope tier (A/B/C) and fill the model/data/training sections of
      [DESIGN.md](../DESIGN.md)
- [ ] Phase 3 kickoff: implement the growing NCA from scratch (`src/neural_ca/model/`), with the
      perception step, update MLP, alive-masking, and the sample pool
- [x] Rename repo/package from the `mlmodel` placeholder → **`neural-ca`** (import `neural_ca`)
- [x] Git identity settled — `nikhlgoel <goelnikhil.com@gmail.com>` (GitHub) is correct; the
      separate `white.dev.sc@gmail.com` is a different account, not the project's git identity
- [ ] Create the GitHub repo and push — **owner's task** (the owner does all committing/pushing)

## Update — project named `neural-ca`

- Owner chose the repo name **`neural-ca`** (Python import `neural_ca`). Renamed the package
  `src/mlmodel` → `src/neural_ca` and updated every reference (pyproject, tests, README,
  DESIGN, PRD, NOTICE); package-metadata author set to the GitHub identity.
- Workflow set: **the owner makes all git commits and pushes.** Recorded in
  [CONTRIBUTING.md](../../CONTRIBUTING.md).
- Verified after rename: `uv lock` re-resolved (ml-model → neural-ca), `uv sync` rebuilt the
  package, `uv run pytest` 3 passed (now importing `neural_ca`), ruff check + format clean.
