# Research 00 — Model Concept Shortlist

**Date:** 2026-07-04 · **Status:** candidates researched, decision pending
**Method:** web survey of 2025–2026 literature, filtered against the criteria below.

## Selection criteria

A candidate must score well on all five:

| # | Criterion | Why it matters here |
|---|---|---|
| C1 | **Genuinely under-explored** | The point is to work where the crowd isn't — results stay useful and visible |
| C2 | **Trains on 1× RTX 3050 (6 GB)** | Real ablations need *many* runs, not one hero run |
| C3 | **Useful beyond the exercise** | Others can use the code/checkpoints; owner can build further research on it |
| C4 | **Teaches core ML engineering** | Architecture, training dynamics, evaluation — from scratch |
| C5 | **Demoable in a web frontend** | The project ships as something a visitor can touch |

## Comparison at a glance

| Candidate | C1 niche | C2 6 GB fit | C3 useful | C4 learning | C5 demo | Overall |
|---|---|---|---|---|---|---|
| 1. Neural Cellular Automata | ★★★ | ★★★ (even CPU-ok) | ★★★ | ★★★ | ★★★ (best-in-class) | **Recommended** |
| 2. Tiny Recursive Models | ★★★ (new field, 2025–) | ★★☆ (scoped tasks) | ★★☆ | ★★★ | ★★☆ | Strong #2 |
| 3. CfC / liquid time-series nets | ★★☆ | ★★★ | ★★★ (health/IoT) | ★★☆ | ★☆☆ | Solid |
| 4. Tiny discrete-diffusion LM | ★☆☆ (now trendy) | ★★☆ | ★★☆ | ★★★ | ★★★ | Educational |

---

## 1. Neural Cellular Automata (NCA) — recommended

**What it is.** A grid of cells (e.g., pixels) that all run the *same* tiny learned update rule
(~10k–100k parameters), repeatedly. Local communication only — yet global structure emerges:
images that grow from a single seed, self-repair after damage, textures, even segmentation.
Trained end-to-end by backpropagating through the unrolled steps
([Growing NCA, Distill 2020](https://distill.pub/2020/growing-ca/)).

**Why it's under-explored.** The 2025 MICCAI community literally ran an
[educational tutorial](https://openreview.net/forum?id=8bRJLOn42Z) because "NCA remains a
niche research field." Active but tiny: a
[biology-and-beyond survey (Sep 2025)](https://arxiv.org/abs/2509.11131),
[NCA at native pixel resolution (Jun 2025)](https://arxiv.org/html/2506.22899v2),
[Neural Particle Automata (Jan 2026)](https://arxiv.org/pdf/2601.16096),
[Differentiable Logic CA (Jun 2025)](https://arxiv.org/pdf/2506.04912).

**Why it's useful.** Med-NCA-line models segment medical images with ~1/1000 the parameters
of a UNet; [OctreeNCA (2025)](https://arxiv.org/pdf/2508.06993) segments 184-megapixel
pathology images **on consumer hardware**;
[weakly-supervised white-blood-cell segmentation (Jan 2026)](https://www.researchgate.net/publication/399375123)
is current. Tiny, robust, edge-deployable models are a real need.

**Compute fit.** Ideal — minutes-to-hours per run on the 3050; dozens of ablations are realistic.

**Demo story.** The best of any candidate: a live canvas where visitors watch a pattern grow,
erase parts of it, and see it regenerate. The model is small enough to run **entirely in the
browser** (onnxruntime-web / WebGL) → free static hosting, no server.

**Risks.** Small community = fewer ready answers when stuck; benchmarks less standardized —
mitigated by the deep-dive collecting all comparable numbers before implementation.

## 2. Tiny Recursive Models (TRM) — strong #2

**What it is.** A ~5–7 M-param network that *loops over its own latent state*, refining a
candidate answer iteratively — [TRM (Oct 2025)](https://arxiv.org/abs/2510.04871) reached
44.6% on ARC-AGI-1, beating far larger LLMs on hard puzzles (Sudoku-Extreme, Maze-Hard).

**Why it's under-explored.** The field is months old and moving:
[test-time adaptation of TRMs (Nov 2025)](https://arxiv.org/pdf/2511.02886),
[TRM-as-policy-improvement theory (Nov 2025)](https://arxiv.org/pdf/2511.16886),
[Tab-TRM for insurance tabular data (Jan 2026)](https://arxiv.org/html/2601.07675v1),
[Mamba-2 hybrid TRM (Feb 2026)](https://arxiv.org/pdf/2602.12078),
[edge compression of recursive reasoners (Jun 2026)](https://arxiv.org/pdf/2606.26488),
[ARC Prize 2025 report](https://arxiv.org/pdf/2601.10904). Applying TRM to a *new domain*
(Tab-TRM's pattern) is a proven, publishable move.

**Compute fit.** Feasible if scoped to Sudoku/Maze-class tasks or a custom domain (full
ARC-scale training exceeds the budget). Runs take hours-to-days — fewer ablations possible.

**Demo story.** Good: enter a puzzle, watch the network's answer improve over recursion steps.

**Risks.** Hot enough that others may publish your idea first; training is famously
tuning-sensitive (EMA, deep supervision, halting).

## 3. Closed-form Continuous-time nets (CfC / "liquid networks") — solid

**What it is.** RNNs whose hidden state evolves in *continuous time*
([CfC, arXiv 2021 / Nature MI 2022](https://arxiv.org/abs/2106.13898)) — they consume
irregularly-sampled sequences natively (no imputation) and are tiny (thousands of params),
with closed-form updates 10¹–10⁵× faster than neural ODEs.

**Why under-explored / useful.** Small dedicated community; real uses in healthcare and
edge sensing: [patient digital twins (2023)](https://arxiv.org/pdf/2307.04772),
[on-device arousal learning (Apr 2026)](https://arxiv.org/pdf/2604.10815),
[uncertainty-aware liquid nets (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0960077925001432).
Project shape: rigorous benchmark vs GRU-D/transformers on irregular medical vitals + a
monitoring-dashboard demo.

**Risks.** Least flashy demo; the project reads "benchmark study" more than "new model."

## 4. Tiny discrete-diffusion language model — educational

**What it is.** Generate text by iteratively *unmasking* tokens in parallel instead of
left-to-right — train a ~10–30 M-param one from scratch on TinyStories; the demo (watching
text denoise) is mesmerizing.

**Why it dropped to #4.** The niche went mainstream in 2025–26:
[LLaDA 8B](https://openreview.net/forum?id=KnqiC0znVF) matches LLaMA3-8B, there are
[scaling-law studies (Dec 2025)](https://arxiv.org/html/2512.10858), and block diffusion is
already "standard for production" — it fails criterion C1, though masked diffusion is
reported *stronger than autoregressive at small scale*, which keeps a tiny open replication
valuable as a learning artifact.

## Wildcards (viable, not shortlisted)

- **Hyperdimensional computing / VSA** — genuinely obscure, CPU-only friendly, active niche
  ([ACM survey](https://dl.acm.org/doi/full/10.1145/3538531),
  [BiHDTrans, Sep 2025](https://arxiv.org/pdf/2509.24425),
  [2026 Springer collection CFP](https://communities.springernature.com/posts/call-for-papers-hyperdimensional-computing-and-vector-symbolic-architectures)) —
  but teaches non-deep-learning skills, off the project's main learning goal.
- **Spiking neural networks** — big in neuromorphic hardware circles, low-compute trainable
  (surrogate gradients), but tooling friction is high and demos are indirect.
- **Differentiable logic CA** — brand new (Google, 2025); could merge with the NCA track later.

## Recommendation

**Neural Cellular Automata**, with TRM as the fallback if the deep dive surfaces a blocker.
NCA is the only candidate scoring top marks on *all five* criteria, and the only one where the
6 GB budget is an advantage rather than a constraint (the whole field runs on small hardware,
so our results are directly comparable to published work).

## Decision

**Chosen: Neural Cellular Automata (candidate #1).** — decided 2026-07-04 by the project owner.

Rationale: it was the only candidate scoring top marks on all five criteria, and the deep dive
([01-nca-deep-dive.md](01-nca-deep-dive.md)) confirmed the practical case — a working model is
reachable within days on the 6 GB budget, there are strong published baselines (Med-NCA) to
measure against, a clear under-explored gap to target (class-conditional growth), and a
best-in-class in-browser demo. Tiny Recursive Models remains the documented fallback if a
blocker appears.

Next: finalise the scope tier (A/B/C) and fill the model sections of the PRD and DESIGN docs
in Phase 2.
