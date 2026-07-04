# PRD — neural-ca

| | |
|---|---|
| **Status** | Draft — **concept + Tier-A scope locked** (2026-07-04); design finalised in [DESIGN.md](DESIGN.md) |
| **Owner** | nikhlgoel |
| **Last updated** | 2026-07-04 |

## 1. Problem statement

Small, well-understood, reproducible models in under-explored corners of ML are rare: most
public work clusters around a few crowded families (LLM fine-tunes, standard CNNs/ViTs on
standard benchmarks). This project builds one genuinely under-explored model **from scratch**,
documents the entire process, and ships it in a form others can actually use — code,
checkpoints, and an interactive web demo.

**Chosen concept: Neural Cellular Automata (NCA)** — see the deep dive in
[research/01-nca-deep-dive.md](research/01-nca-deep-dive.md). NCAs are extremely small
(~10k–100k param) models where every pixel runs one shared, learned, local update rule;
global structure (growing images, self-repair, lightweight segmentation) emerges from local
interaction. They are under-explored, genuinely useful (e.g. Med-NCA segments medical images
at ~1/500 the size of a U-Net), and ideal for our hardware and for a live in-browser demo.
The specific gap we target — making a *single* NCA grow a *chosen* (class-conditional) target
— is set with the scope tier in Phase 2.

## 2. Users & use cases

| User | Use case |
|---|---|
| Project owner | Learn ML engineering end-to-end; a reusable base for further research; portfolio evidence for university & remote-job applications |
| Researchers / students | Reference implementation + ablations in an under-documented area |
| Demo visitors | Interact with the model in the browser without installing anything |
| **[TBD — Phase 1]** | Domain-specific users of the chosen model |

## 3. Goals & success metrics

**Learning goals (fixed):**

- **G1** — Implement model, training loop, and evaluation from scratch (no black-box model
  libraries); verified by mapping code line-by-line to the underlying math.
- **G2** — Run ≥ 3 controlled ablations with documented conclusions.
- **G3** — Full reproducibility: any reported number regenerable from `(commit, config, seed)`.

**Product goals (fixed):**

- **G4** — Public repo: CI green, docs complete, tagged v1.0 release including a checkpoint.
- **G5** — Web demo usable by a non-technical visitor in under 30 seconds.

**Model goals (Tier A, locked 2026-07-04):**

- **G6** — a Growing NCA reconstructs a 40×40 target to low RGBA MSE and **persists** to ≥ 500
  steps without diverging.
- **G7** — it **regenerates** the target after localised damage.
- Stretch: **Tier B** — one class-conditional NCA grows a *chosen* target; **Tier C** —
  lightweight medical segmentation vs. a small U-Net.

Scope tiers are defined in [research/01-nca-deep-dive.md](research/01-nca-deep-dive.md) §9; we
commit to **Tier A** and build so Tier B needs no rewrite.

## 4. Non-goals

- Competing with frontier-scale models or chasing leaderboard SOTA.
- Production-grade serving infrastructure — the demo is a demo.
- Guaranteed novel-paper-level results: a **well-documented negative or confirmatory result
  counts as success** (that's how research works).

## 5. Requirements

### Functional

- **FR1** — Single-command environment setup (`uv sync`).
- **FR2** — Single-command training run driven by a config file.
- **FR3** — Deterministic evaluation script producing a metrics report.
- **FR4** — Export path for inference (ONNX / TorchScript — **[TBD]**).
- **FR5** — Web frontend for interactive use (**interaction design [TBD — Phase 1]**).

### Non-functional

- **NFR1** — Trains to reported results on 1× RTX 3050 Laptop (6 GB VRAM), ≤ 48 h per run.
- **NFR2** — Every experiment logged (TensorBoard) and journaled (devlog).
- **NFR3** — Apache-2.0-compatible dependencies only; dataset licenses must permit
  redistribution of derived weights.

## 6. Milestones

Mirrors the roadmap in [README](../README.md) (Phases 0–6). Dates are set at the end of
Phase 1, when scope is known.

## 7. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Chosen concept too compute-hungry | Stalls Phases 3–4 | Feasibility check in Phase 1 against the 6 GB VRAM budget; a scoped-down fallback defined upfront |
| Under-explored ⇒ thin literature/baselines | Hard to judge results | Deep-dive doc collects every comparable number *before* implementation starts |
| Solo-project scope creep | Never ships | PRD non-goals + phase gates; demo scoped to one core interaction |
| Reproducibility drift | Results not trustworthy | Locked deps (`uv.lock`), seeded runs, configs-in-git discipline |

## 8. Open questions

- [ ] Which model concept? → decided in [research/00-idea-shortlist.md](research/00-idea-shortlist.md)
- [ ] Final project/repo/package name (follows the concept)
- [ ] Checkpoint hosting: GitHub Releases vs. Hugging Face Hub (ADR at Phase 6)
