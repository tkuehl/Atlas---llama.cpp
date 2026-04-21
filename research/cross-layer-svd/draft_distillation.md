# Draft Distillation — Research Plan

Track C of the speculative decoding research. Produces tiny draft
models on demand for targets that don't have a published matched draft.

Upstream of [speculative_decoding.md](speculative_decoding.md) Track B,
which validated that a good same-family draft is worth ~1.56× decode
speedup on mixed workloads at zero quality cost. This track asks: **what
do we do when the target family doesn't publish a tiny draft?**

Status: **planning**. No training runs yet.

## Problem statement

Given a target model `T` with:
- no published same-family tiny draft, or
- a published draft that's too big to run alongside `T` on the user's
  hardware (e.g. published 2B draft when only 1 GB VRAM headroom
  available), or
- a published draft that's undertrained / unaligned relative to `T`

produce a draft model `D` that:

1. Has a **compatible tokenizer** — every token `D` emits, `T` can
   read.
2. Has a **compatible chat template** — prompts that work on `T`
   work the same way on `D`.
3. Gets **high acceptance rate** (target ≥ 70% on Track B's 25-prompt
   fixture, matching the off-the-shelf Qwen3-0.6B baseline we already
   established).
4. Is **small enough** to run alongside `T` on consumer hardware
   (target: ≤ 1 GB VRAM for `D` when `T` is 8B-class, ≤ 2 GB when `T`
   is 70B-class).
5. Is producible in **hours of wall time**, not days, on a single
   consumer GPU. "On demand" implies the turnaround is short enough
   that a user waiting for their custom draft can do other things and
   come back to it the same day.

And is loadable by upstream llama.cpp via `-md` — no fork-only
runtime extensions (per the project invariant).

## Why this is worth doing

Track B established the upper bound for "ideal case with a published
draft." But many widely-used targets lack a good match:

| Target family | Size | Published tiny match? |
|---|---|---|
| Qwen3 | 8B | yes (0.6B) |
| Qwen3 | 14B, 32B | yes (via 0.6B / 4B) |
| Llama 3.1 | 8B | Llama-3.2-1B (not same-series) |
| Llama 3.1 | 70B | no small official match |
| Gemma 2 | 9B, 27B | 2B exists but is positioned as a standalone model, not a draft |
| Gemma 3/4 | all sizes | no explicit tiny draft at same revision |
| Mistral 7B / Mixtral | various | no same-rev tiny |
| Newer releases | — | always lag the ecosystem |

So "just use the published draft" solves ~half the real-world cases
and none of the newly-released ones. A pipeline that produces a
usable draft *on demand* for any target closes the rest of the gap.

## Invariants

- **Upstream-only runtime.** Output is a GGUF loaded via `-md`. No
  Medusa/EAGLE-style prediction heads, no forward-path modifications.
- **Consumer hardware for training.** A single RTX 4090 / 5080 / 5090
  class GPU. No cluster assumption, no multi-day runs.
- **Reproducible inputs.** Training corpus drawn from public datasets
  (FineWeb-Edu, The Stack, OpenOrca, UltraChat, etc.) or from the
  target model's own rollouts. No private data.
- **Honest baselines.** Every custom draft is compared against (a) the
  baseline-no-draft throughput from Track B's fixture, and (b) the
  best published draft for that target if one exists. A custom draft
  that loses to the off-the-shelf option is a failed recipe, not a
  valid "ours works."

## What's known from prior art (to verify before we reinvent)

Before writing code, confirm the state of the literature. These are
the papers that land closest to our problem; each should be read and
summarized under a "prior art" section of this doc before the first
training run:

- **DistillSpec** (Zhou et al., 2023 / arxiv 2310.08461) — distills a
  draft from the target using target's logits, specifically optimizing
  for speculation acceptance rather than perplexity. Reports meaningful
  acceptance-rate gains vs off-the-shelf drafts. Likely our closest
  reference.
- **Medusa** (Cai et al., 2024 / arxiv 2401.10774) — prediction heads,
  not standalone drafts. Out of scope for our upstream-only constraint
  but methodology on training for acceptance is relevant.
- **EAGLE / EAGLE-2** (Li et al., 2024 / arxiv 2401.15077 / 2406.16858)
  — trains a lightweight autoregressive head that shares target's
  hidden states. Upstream-incompatible but informative on
  hidden-state-conditioned draft quality.
- **SpecDec++ / DraftRetriever / TriForce** — various other
  speculation-draft training schemes worth cataloging.
- **Distillation without teacher logits** — training-data-only recipes
  (SLM student fine-tuned on target rollouts) are the simplest form
  of this. Worth knowing the published quality ceiling.

None of these ship a "give me a command, get a GGUF" pipeline for
consumer hardware. That packaging is where our marginal value sits,
even if the underlying algorithms are all published.

## Approach space (ranked by tractability on consumer hardware)

### Option A — Fine-tune an existing tiny model on target rollouts

Most practical. Take a small pretrained model with compatible
tokenizer (easier to find than you'd think — Llama-family tinies are
plentiful, same for Qwen) and fine-tune it on text *generated by the
target*. Standard supervised fine-tuning loop, no logits needed, a
few hours on consumer hardware for a 0.5-1B model.

- **Cost:** hours to a day of wall time on 16 GB VRAM.
- **Ceiling:** decent acceptance but likely below DistillSpec-style
  logit-KD. Reasonable baseline though.
- **Tokenizer risk:** if no same-tokenizer tiny exists, this is
  blocked. `--spec-replace` is an escape hatch but lossy per our
  bench methodology.

### Option B — Logit distillation from target (DistillSpec-style)

Train the same starting point as Option A, but the loss is KL
divergence against the target's per-token logits rather than
cross-entropy on a target rollout. Usually gets higher acceptance.

- **Cost:** ~2-4× Option A because every training step needs a
  target forward pass. Still feasible on consumer hardware for a
  small student.
- **Ceiling:** higher acceptance than Option A per DistillSpec.
- **Infrastructure:** needs a pipeline that can run target + student
  forward passes in the same process without OOM — solvable with
  frozen target in 8-bit or via CPU/GPU offload.

### Option C — Acceptance-rate-optimized training

Distill, but loss function explicitly optimizes what the target
would accept. E.g., train to maximize P(target argmax at position t+1
| draft's top token at t). Less studied publicly; DistillSpec touches
this.

- **Cost:** same as Option B plus some complexity.
- **Ceiling:** potentially better than plain logit KD on the specific
  "acceptance rate" metric.
- **Risk:** may overfit to bench-style greedy acceptance and lose
  generalization.

### Option D — Prune the target

Take the target itself and aggressively prune / quantize it down to
~0.5-1B effective parameters. The result is "the same model, just
smaller." Our Track A (cross-layer SVD) showed this is hard to do
without quality collapse at aggressive ratios; doing it specifically
to serve as a draft (where quality matters less than alignment with
the target) might lower the bar.

- **Cost:** days of research per target.
- **Novelty:** unclear vs off-the-shelf distillation. Most prior art
  concludes distillation beats pruning at the small-student scale.
- **Probably deprioritize** — Options A-C are better-trod and easier.

### Option E — Retrieve the best available

Not a training approach — a "matching" approach. Given target T,
automatically pick the best existing published small model with a
compatible tokenizer and a plausible output distribution. Cheap,
instant, but capped at what's already out there.

- **Cost:** zero.
- **Ceiling:** whatever the best off-the-shelf is.
- **Value:** acts as a "free baseline" the custom-training options
  must beat. Worth building even just as a reference.

## Tracks within Track C

### C.1 — Retrieval pipeline (cheapest first)

Before any training runs, build the matching pipeline:

1. Given a target GGUF, extract its tokenizer metadata.
2. Search a small curated registry of published tiny models for
   tokenizer-compatible candidates.
3. Spin up the candidate alongside the target, run the 25-prompt
   fixture, report acceptance + speedup.
4. Return the winning candidate as the "best available today" draft.

**Output:** a `bench/draft_registry.json` listing published tinies
with their tokenizer fingerprint + target-compatibility notes, plus
a `pick_draft.py` script that, given a target path, identifies the
best registered candidate.

**Value:** solves the "no published match" problem for targets where
there *is* a cross-family compatible tiny that no one has thought to
try. Costs one download + one 30-min bench per candidate. Informs
which targets actually need custom training (C.2) vs which already
have a usable draft if you look.

### C.2 — Option A prototype on a known target

Before committing to the hardest path (logit-KD, Option B), validate
the infrastructure works end to end on a target we already have a
good answer for:

- **Target:** Qwen3-8B-Q4_K_M (same target as Track B, baseline of
  71.5% acceptance with off-the-shelf Qwen3-0.6B-Q8_0).
- **Seed student:** Qwen3-0.6B at *random init* OR Qwen2.5-0.5B as a
  near-family starting point.
- **Training data:** 10k-100k samples from the target on a public
  prompt corpus (instruction-following + code + reasoning).
- **Training recipe:** plain supervised fine-tuning, cross-entropy
  loss, a few epochs, AdamW.
- **Goal:** acceptance rate within 10 percentage points of the
  off-the-shelf 71.5%. Not exceeding it — just demonstrating the
  pipeline produces something reasonable.

If this works, Option B (logit KD) becomes the natural next step to
push acceptance above the off-the-shelf baseline. If the retrieval
pipeline (C.1) already gets acceptable results for the target at
hand, logit KD is the only reason to bother training anything.

### C.3 — Option B on a target with no published match

Once C.2 infrastructure is solid, run Option B on a target without a
published tiny:

- **Candidate:** Gemma-2-9B-it or Llama-3.1-8B-Instruct (both widely
  used, neither has a same-series tiny with aligned training).
- **Seed student:** ~0.5B random-init with matching tokenizer, or a
  pretrained same-tokenizer small model.
- **Recipe:** logit distillation from target.
- **Goal:** acceptance ≥ 70% on Track B's fixture, speedup ≥ 1.4×
  overall.

Success here is the headline result for this track: **"Take any
8B-class model, turn it into an on-demand draft in a day."**

## First experiment (for when we actually start)

**Retrieval-first (C.1), no training yet.** Before spending a single
GPU-hour on fine-tuning, map the landscape of what's already
possible via cross-family draft pairing.

Steps:

1. Pick a target that lacks a same-series tiny, e.g.
   `Gemma-2-9B-it-Q4_K_M`.
2. Enumerate same-tokenizer or near-tokenizer published smalls.
   Candidates:
   - `Gemma-2-2B-it` (same tokenizer, but 2B is big for a draft)
   - `Qwen3-0.6B` via `--spec-replace` (mismatched tokenizer)
   - Any newly-released Gemma-family tiny
3. Run the Track B fixture against each candidate paired with the
   Gemma-2-9B target using the existing `run_condition.py`
   infrastructure.
4. Report acceptance + speedup per candidate. If any candidate hits
   ≥ 70% acceptance and ≥ 1.3× speedup, we've answered this target's
   question via retrieval.
5. If retrieval fails for this target, that's the motivation for
   C.2/C.3 with a clear case study.

**Expected duration:** 3-4 hours including downloads and benches,
no training.

**Decision points:**
- Retrieval gets ≥70% acceptance → publish the registry as the first
  public artifact of this track, move to C.2 on a different target.
- Retrieval gets all < 30% → this target is a strong C.3 candidate;
  go straight to logit-distillation prototype.

## Open design questions

- **Where does the trained GGUF live?** The `model_prep_pipeline.md`
  doc has `emit_gguf` as a stage. A distilled draft should emit via
  the same path so the output is interchangeable with downloaded
  ones.
- **What's the regression-test harness?** Track B's 25-prompt fixture
  is good for acceptance characterization but not a training loss.
  A proper training pipeline needs a quick-to-compute validation
  metric that correlates with acceptance — likely target KL or
  top-5 agreement on a held-out set.
- **Tokenizer fingerprinting.** C.1 needs a reliable way to decide
  "these two models share a tokenizer." Tokenizer SHA of the vocab
  file? Byte-level round-trip test on a corpus? Open question.
- **How aggressive on draft size?** The published Qwen3-0.6B works
  at 0.6B. Would a 0.3B distilled draft still hit 70% acceptance?
  A smaller draft reduces per-step overhead and could raise speedup
  on moderate-acceptance slices (e.g. conversational). Worth
  characterizing alongside the acceptance bar.
- **Prior-art consultation.** Read DistillSpec, EAGLE-2, and recent
  2024-2025 draft-training papers before the first training run;
  update this doc with a "prior art" section before C.2 starts.

## Not in scope (this doc)

- **Training infrastructure hardening.** Beyond a consumer-GPU single-
  process loop. Multi-GPU, distributed training, anything cluster-
  shaped is out of scope until proven unnecessary on one GPU.
- **Speculation runtime changes.** Covered by the upstream-only
  invariant. This track produces GGUFs; everything else is out of
  scope.
- **Per-target custom architectures.** We train within the
  off-the-shelf transformer shape. Medusa-style or EAGLE-style
  architectural surgery is research-interesting but fails our
  deployment invariant.

---

Ongoing progress lives below this line as dated subsections.
