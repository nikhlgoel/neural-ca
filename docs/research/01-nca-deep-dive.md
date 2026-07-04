# Research 01 — Neural Cellular Automata (NCA): Deep Dive

**Date:** 2026-07-04 · **Status:** chosen concept — research in progress
**Companion:** [00-idea-shortlist.md](00-idea-shortlist.md) (why NCA was chosen over the alternatives)

This is the "understand it before building it" document. It has two registers:
**"In simple terms"** boxes explain each idea plainly; the surrounding text goes deep. If you
only read the boxes, you'll still understand what we're building and why.

---

## 1. What is an NCA, and what is it *for*?

> **In simple terms.** Imagine a grid of tiny robots, one per pixel. Every robot is identical
> and a bit short-sighted: it can only see its immediate neighbours. Each tick, every robot
> looks around and updates its own colour using the *same* little rulebook. Run this for a
> while and — astonishingly — the whole grid can grow a picture from a single dot, and if you
> erase part of it, it grows back. An NCA is that shared rulebook, **learned** by a neural
> network instead of written by hand.

**The purpose.** NCAs are a model of *self-organization*: complex global structure emerging
from simple, purely local rules. That's interesting for three reasons that map onto real uses:

1. **They're astonishingly small.** The entire model is one cell's update rule — often
   **~10k–100k parameters**. Med-NCA does medical image segmentation with **~13k parameters
   and ~50 kB of storage**, yet beats a classic U-Net by 2–3% Dice on hippocampus and prostate
   segmentation while being **~500× smaller**
   ([Med-NCA, arXiv 2302.03473](https://arxiv.org/pdf/2302.03473)).
2. **They're robust and self-repairing.** Because behaviour is local and trained under
   perturbation, NCAs recover from damage and noise — the 3D variant M3D-NCA even runs on a
   **Raspberry Pi 4 (2 GB)** and ships built-in quality control from cell-state variance
   ([M3D-NCA, MICCAI 2023](https://arxiv.org/abs/2309.02954)).
3. **They connect to big ideas cheaply.** They're a bridge to morphogenesis/biology, to
   diffusion-style iterative generation, and even to reasoning
   ([NCA for ARC-AGI, 2025](https://arxiv.org/html/2506.15746v1)) — but you can study them on
   a laptop GPU.

> **Why this fits our project.** Under-explored (the 2025 MICCAI community ran a
> [beginner tutorial](https://openreview.net/forum?id=8bRJLOn42Z) precisely because NCA "remains
> a niche field"), genuinely useful (tiny robust segmentation/edge models), a superb browser
> demo, and — uniquely among our candidates — the 6 GB VRAM budget is *comfortable*, so we can
> run the many ablations that real learning requires.

## 2. How it works (the mechanism)

> **In simple terms.** Each pixel stores a few numbers: its visible colour (RGBA) plus some
> extra "hidden notes" only it and its neighbours use to coordinate. One step = three moves:
> **(a) look around** (what are my neighbours doing?), **(b) think** (a tiny 2-layer network
> decides how I should change), **(c) nudge myself** by that amount. Repeat 50–100 times.

**State.** The grid is a tensor of shape `(C, H, W)`. The first 4 channels are RGBA (channel 3,
alpha, doubles as "am I a living cell?"). The remaining `C-4` are **hidden channels** — scratch
space for coordination. Typical `C = 16`.

**One update step** (the classic [Growing NCA, Distill 2020](https://distill.pub/2020/growing-ca/)
recipe):

1. **Perception.** Convolve each channel with fixed Sobel-x, Sobel-y, and identity filters →
   a `3C`-channel "what's around me and which way things change" vector per cell. (Fixed
   filters keep it a true CA: perception is hard-wired; only the *rule* is learned.)
2. **Update rule.** A tiny per-cell MLP (1×1 convs): `3C → 128 → C`, ReLU between. Output is a
   **delta** added to the current state (a residual update — the cell nudges itself).
3. **Stochastic update.** Each cell applies its update only with probability ~0.5, so cells
   don't march in lockstep (models asynchronous biology; also a regularizer).
4. **Alive masking.** Cells with no living neighbour (max-pooled alpha < 0.1) are forced to
   zero — structure only grows at the edges of what's already alive.

Do this `T` times (T random in e.g. `[64, 96]` during training), then read RGBA off the first
4 channels. **The whole model is steps 1–2's ~10k weights.** Everything else is fixed rules.

> **The one non-obvious trick — the "pool".** How do you teach it to be *stable* and
> *self-healing*, not just to hit the target once at step 64 and then explode at step 200?
> You keep a pool of past end-states, and each batch you start *from those* (not always from
> the clean seed), sometimes damaging them first. So the network is constantly asked "here's a
> half-built or damaged pattern — fix it and hold it steady." That's what turns a one-shot
> generator into a persistent, regenerating organism. (Details in §5.)

## 3. Family tree — what people build with NCAs

| Variant | What it does | Why it matters to us |
|---|---|---|
| **Growing NCA** ([Distill 2020](https://distill.pub/2020/growing-ca/)) | Grow a target image from one seed; regenerate after damage | The canonical starting point; our Phase-3 baseline |
| **Self-classifying NCA** ([Distill 2020](https://distill.pub/2020/selforg-mnist/)) | Cells reach *consensus* on which MNIST digit they form | Shows NCAs can do discriminative tasks, not just generation |
| **Texture NCA** | Synthesize/parametrize textures | Great, cheap demo material |
| **Med-NCA / M3D-NCA** ([2302.03473](https://arxiv.org/pdf/2302.03473), [2309.02954](https://arxiv.org/abs/2309.02954)) | Lightweight 2D/3D medical segmentation, +quality control | The "genuinely useful" endgame; strong published baselines |
| **From Cells to Pixels** ([2506.22899](https://arxiv.org/abs/2506.22899)) | NCA at native high resolution | Attacks the resolution/scaling limit |
| **Differentiable Logic CA** ([2506.04912](https://arxiv.org/pdf/2506.04912)) | Discrete logic-gate rules (Game-of-Life → patterns) | A spicier, very-under-explored offshoot |
| **NCA for ARC-AGI** ([2506.15746](https://arxiv.org/html/2506.15746v1)) | Self-organization applied to abstract reasoning | Shows the frontier ambition |

## 4. Open problems (candidate contributions for *our* project)

From the 2025–26 literature, the live limitations — any one is a legitimate mini-contribution:

- **Cost scales with grid size.** Training memory/time grow ~quadratically with resolution;
  information propagates only one cell per step, so large images need many steps
  ([From Cells to Pixels](https://arxiv.org/abs/2506.22899)). *→ our 6 GB budget forces small
  grids anyway, which is fine.*
- **Training is tuning-sensitive & sometimes mysteriously unstable** — the attractor landscape
  is only starting to be mapped ([Attractor Landscape, 2026](https://arxiv.org/html/2604.10639)).
- **Weak compositional generalization** — 2–3× performance drops across benchmark variants
  ([NCA for ARC-AGI](https://arxiv.org/html/2506.15746v1)).
- **Conditioning is underexplored** — one trained NCA usually grows *one* target; making a
  *single* NCA grow a *chosen* target on demand (class-conditional) is not yet routine.
- **Architectural upgrades** (attention, gating, adaptive #steps) are flagged as promising but
  under-tried while keeping updates local.
- **Robustness studies** — generalization of Med-NCA across scanners/domains is itself a 2024–25
  research thread ([Generalization of NCA for Med Seg](https://arxiv.org/html/2408.15557v1)).

## 5. Training recipe (what Phase 3 will implement, from scratch)

> **In simple terms.** Train it like teaching a lump of clay to become a shape *and stay* that
> shape even if you poke it. Show it its own past attempts, damage some, and reward it for
> converging back to the target.

- **Loss:** pixel-wise MSE between rendered RGBA and target, after `T` steps.
- **Sample pool (persistence):** maintain N past final states; each batch seed from a mix of
  clean seed + pooled states; replace the worst with a fresh seed to anchor "grow from scratch."
- **Damage augmentation (regeneration):** zero out random circular patches of some pooled
  samples so the model must regrow them.
- **Stability tricks that matter:** residual (delta) updates; stochastic cell updates;
  **normalize/clip gradients** (per-cell grads vary wildly — this is the usual "why won't it
  train" fix); randomize `T` per batch; keep hidden channels small.
- **Optimizer:** Adam, lr ~2e-3 with step decay; a few thousand steps suffices for emoji-scale
  targets — **minutes on the RTX 3050**, so ablations are cheap.

## 6. Datasets & tasks (in feasibility order)

| Task | Data | License | Why start here |
|---|---|---|---|
| Grow a fixed emoji/glyph | Single 40×40 RGBA image | trivial/self-made | Fastest path to a working, demoable model; reproduces Distill 2020 |
| Class-conditional growth | A handful of emoji + a class embedding | self-made | Our first *novel-ish* twist (§4 conditioning gap) |
| Self-classifying MNIST | [MNIST](http://yann.lecun.com/exdb/mnist/) | CC-BY-SA 3.0 | Discriminative NCA; standard, tiny |
| Texture synthesis | DTD / single textures | research-permissive | Best-looking demo |
| Lightweight 2D med-seg | [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (Hippocampus) | CC-BY-SA 4.0 | The "useful" endgame; direct Med-NCA baseline. **License permits redistribution — satisfies PRD NFR3** |

## 7. Compute-fit check (RTX 3050, 6 GB) — the deciding factor

An NCA forward pass is `T` iterations of two small convs on a `(C,H,W)` grid; backprop
unrolls all `T` steps (memory ~ `T × H × W × C`). Concretely: `C=16`, grid `64×64`, `T≈96`,
batch 8 → activations on the order of a few hundred MB in fp32, comfortably inside 6 GB, and
**smaller in bf16** (supported — verified Day 0). Emoji-scale training is **minutes**; even
med-seg experiments are **hours, not days**. This is the only shortlisted concept where the
GPU is not a bottleneck — which is exactly why NCA won.

## 8. Demo plan (Phase 5) — NCA's superpower

> **In simple terms.** The model is so small it runs *in the visitor's browser*. They watch a
> shape grow live on a canvas, click to erase part of it, and watch it heal — no server, free
> to host forever.

Export the update rule to ONNX; run the iteration loop in-browser with
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) (WebGPU backend, with WASM
fallback). Precedent exists: the [Neural Particle Automata demo](https://selforg-npa.github.io/)
runs trained self-organizing models live via WebGL. Frontend: React + Vite + TypeScript on a
`<canvas>`; interactions = seed placement, erase/damage brush, play/pause, speed. Because
inference is client-side, we deploy as a **static site (GitHub Pages)** — zero running cost.

## 9. Proposed scope (three tiers — pick at Phase 2)

- **Tier A — "Reproduce & understand" (safe floor):** growing + regenerating emoji NCA from
  scratch, faithfully reproducing Distill 2020, with our own clean code, tests, ablations
  (perception filters, hidden-channel count, pool on/off), and browser demo. *Guarantees a
  shippable, correct, well-documented result.*
- **Tier B — "A real twist" (target):** Tier A **+ class-conditional NCA** — one model grows a
  *chosen* target from a class code — directly probing the §4 conditioning gap, with an
  ablation on how conditioning is injected. *This is the novel-enough-to-be-interesting core.*
- **Tier C — "Useful reach" (stretch):** apply the conditional NCA to lightweight 2D medical
  segmentation (Decathlon Hippocampus), benchmarked against a small U-Net on Dice/params. *Only
  if A+B land with time to spare.*

## 10. Open questions for the design phase

- [ ] Lock the scope tier (recommend committing to **A**, aiming for **B**).
- [ ] Exact channel count / grid size / step schedule (small ablation to set defaults).
- [ ] Conditioning mechanism for Tier B: input concat vs. FiLM/gating on the update MLP.
- [ ] Confirm ONNX export handles the iterated loop cleanly, or export one step and loop in JS
      (**verify early — de-risks the whole demo**).

## Sources

- [Growing NCA — Distill 2020](https://distill.pub/2020/growing-ca/) ·
  [Self-classifying MNIST NCA](https://distill.pub/2020/selforg-mnist/)
- [Med-NCA (2302.03473)](https://arxiv.org/pdf/2302.03473) ·
  [M3D-NCA (2309.02954)](https://arxiv.org/abs/2309.02954) ·
  [MECLabTUDA/M3D-NCA code](https://github.com/MECLabTUDA/M3D-NCA)
- [NCA survey: biology & beyond (2509.11131)](https://arxiv.org/abs/2509.11131) ·
  [From Cells to Pixels (2506.22899)](https://arxiv.org/abs/2506.22899)
- [NCA for ARC-AGI (2506.15746)](https://arxiv.org/html/2506.15746v1) ·
  [Generalization of NCA for Med Seg (2408.15557)](https://arxiv.org/html/2408.15557v1)
- [Attractor Landscape (2604.10639)](https://arxiv.org/html/2604.10639) ·
  [Differentiable Logic CA (2506.04912)](https://arxiv.org/pdf/2506.04912)
- [Neural Particle Automata + live demo (2601.16096)](https://arxiv.org/pdf/2601.16096) /
  [selforg-npa.github.io](https://selforg-npa.github.io/) ·
  [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)
