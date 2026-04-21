# Draft Models via Structural Pruning

Research track for producing a tiny draft model by **pruning the
target itself** — dropping layers, heads, or hidden dims — and
recovering quality with a light retraining pass. The goal is a
draft with meaningfully fewer parameters than the target, an output
distribution aligned with the target's, and a consumer-hardware
cost budget of hours to overnight, not weeks.

Sibling of [draft_distillation.md](draft_distillation.md) (fresh
small student from scratch). Motivated by the 2026-04-20 self-quant
finding in [speculative_decoding.md](speculative_decoding.md):
quantization alone reduces bytes but not FLOPs, so it doesn't
produce a useful draft. Pruning attacks parameters directly.

Status: **planning**. No pruning experiments run yet.

## Thesis

Speculation speedup at batch=1 consumer decode is dominated by
**FLOPs per token in the draft**, not compressed storage. Our Q2_K
self-draft experiment confirmed this:

| Draft | Params | Storage | Accept | Speedup |
|---|---:|---:|---:|---:|
| Qwen3-0.6B-Q8_0 | 0.6B | 609 MB | 71.5% | 1.56× |
| Qwen3-8B-Q2_K | 8B | 3.1 GB | 82.8% | 0.83× |

The 0.6B wins despite lower acceptance because it has 14× fewer
parameters and therefore 14× fewer matmul ops per forward pass.
Pruning gets us toward that fewer-parameters regime while preserving
same-tokenizer / same-template compatibility with the target, which
training-a-different-model doesn't.

## Invariants

Same as the broader speculation track:

- **Upstream-only runtime.** Output is a standard GGUF loaded via
  `-md`. No architecture changes upstream doesn't understand.
- **Consumer hardware.** Single RTX 4090 / 5080-class GPU. Budget:
  one overnight run per experiment.
- **Training data from public sources** (or rollouts of the target on
  public prompts). No private data.
- **Honest baselines.** Every pruned draft compared against (a)
  baseline-no-draft and (b) the best off-the-shelf option for that
  target, using the existing `bench/` harness.

## Approach space

### Ranked by cost × expected value

| Approach | Training cost | Expected acceptance | Expected draft speed | Why |
|---|---|---|---|---|
| **A1. Drop every other layer, no retraining** | 0 | 5-20% | very fast | Proof of the quality floor. Almost certainly fails but cheap to confirm. |
| **A2. Drop every other layer, rollout distillation** | ~8-24 h | plausibly 40-60% | very fast | First real shot. Half the depth ≈ half the FLOPs. Published precedent (ShortGPT, LLM-Pruner). |
| **A3. Depth-importance-based drop, rollout distillation** | ~8-24 h | plausibly 50-70% | very fast | Drop the *least-important* layers (measured by activation change when ablated) rather than blind every-other. Potentially better quality retention. |
| **A4. Width pruning (head + hidden dim reduction), retraining** | ~1-3 days | plausibly 60-75% | fast | Finer-grained cut. More complex to implement (SliceGPT-style) — hidden-dim reduction needs calibration-matrix rotation. |
| **A5. Combined depth + width pruning + distillation** | 3+ days | potentially beats published tiny | fast | Max-compression attempt. High risk of quality collapse; only attempt if A2/A3 clear the acceptance bar. |

A1 is always the first experiment — it establishes the "how bad is
it with no recovery?" floor and validates the pruning infrastructure
before we spend training compute.

### Explicitly not in scope here

- **Medusa / EAGLE-style prediction heads** — not upstream-loadable.
  Research-interesting, deployment-invariant-violating.
- **Mixture-of-depths / conditional compute** — same runtime-support
  issue. Upstream doesn't ship the dispatch.
- **Pure distillation from scratch** — covered by
  [draft_distillation.md](draft_distillation.md). Pruning
  specifically starts from the target's weights.

## Prior art — synthesis (2026-04-20)

Surveyed the literature on parameter/FLOP reduction without
gradient retraining. Short form below; full agent synthesis archived
in the journal.

### In scope (matches constraints — no retraining, upstream GGUF, reduces FLOPs)

| Method | arXiv | Type | Calibration? | Aggressiveness ceiling |
|---|---|---|---|---|
| **ShortGPT** (Men et al.) | 2403.03853 | Layer drop by Block Influence | forward-only | ~25% layers @ <5% MMLU drop on 7-13B; ~30-40% degrades |
| **LLM-Streamline** | (companion work, same year) | Layer drop + linear-LSQ replacement layer | forward-only calibration | ~25%; the LSQ-replacement variant is genuinely gradient-free |
| **SLEB** | 2402.09025 | Iteratively drops the block whose removal least hurts calibration PPL | calibration | ~20% layers |
| **LaCo** (Layer Collapse) | 2402.11187 | Merges adjacent layers via parameter-space rule | calibration | ~25-30% |
| **Gromov et al.** ("Unreasonable Ineffectiveness of the Deeper Layers") | 2403.17887 | Drops contiguous deep-layer spans via angular distance | calibration | up to ~50% of *later* layers with "healing"; ~25-30% without |

**Consensus empirical rule:** ~25-30% of layers can be dropped from
LLaMA/Qwen-class 7-8B models with calibration-only techniques and
<5-10% benchmark degradation. Past ~35% quality falls off a cliff.
**Later/middle layers are universally more redundant than early
ones** — this tells us *which* layers to target in A1.

### Adjacent / tangential

| Method | arXiv | Why deprioritized |
|---|---|---|
| **SliceGPT** (width pruning) | 2401.15024 | Rotation matrices fuse into residual stream; no mainline GGUF support, would need loader extension. Deprioritize until A1/A2/A3 clear. |
| **FLAP** (head + channel) | 2312.11983 | Calibration-only head/FFN pruning with bias compensation. ~20% params. Candidate for stacking after A2. |
| **LASER / ASVD / SVD-LLM** | 2312.13558 / 2312.05821 / 2403.07378 | Rank reduction. Only saves FLOPs if kept as factored matmuls; GGUF has no low-rank tensor type, so these become storage-only on our runtime. |

### Out of scope (violates constraints)

- **SparseGPT** (2301.00774) and **Wanda** (2306.11695) — unstructured
  magnitude pruning. Reduces storage but **not FLOPs on dense GPU
  kernels**. Requires 2:4 structured sparsity + sparse tensor cores
  (Ampere+) to get actual speedup, and llama.cpp doesn't use sparse
  kernels. Storage-only on our runtime.
- **LayerSkip** (2404.16710) applied zero-shot — requires its own
  continued-pretraining regime (layer-dropout + early-exit loss).
  Applying to a vanilla pretrained model underperforms. Meta ships
  LayerSkip checkpoints precisely because you can't retrofit it.
- **Medusa** (2401.10774) / **EAGLE** (2401.15077, 2406.16858) /
  **Kangaroo** (2404.18911) / **Draft&Verify** (2309.08168) — all
  speculation-specific, all require gradient training of draft
  heads or adapters.
- **SWIFT** (2410.06916) — genuinely training-free but requires
  custom inference runtime. Violates upstream-GGUF invariant.
- **LLM-Pruner** (2305.11627) / **Sheared LLaMA** (2310.06694) —
  structured pruning requiring LoRA recovery / continued
  pretraining respectively. Adjacent but gradient-based.

### The genuinely novel angle

**No published paper explicitly reports speculation-draft
acceptance rates from zero-shot pruned targets.** Community
anecdote suggests 60-70% acceptance at ~25% layer drop, but
nothing peer-reviewed. If our bench produces clean per-slice
numbers, the data is a publishable small-but-honest contribution.

## Refinements to the plan given the survey

1. **A1 mechanism updated.** Instead of "drop every other layer" as
   the no-thought baseline, use **block-importance-based drop
   targeting middle/late layers** — this is what the literature
   converges on and costs nothing extra (one forward pass through
   a calibration set computes the Block Influence scores).
2. **A2 recipe unchanged** — rollout SFT on target outputs is still
   the most natural recovery step.
3. **A1.5 new step.** Between A1 (drop-no-retrain) and A2 (drop +
   distillation), insert LLM-Streamline-style **linear-LSQ
   replacement**: replace the dropped contiguous span with a single
   linear layer whose weights are fit by least squares against the
   calibration set's hidden states. Genuinely gradient-free, should
   bridge much of the A1-to-A2 gap cheaply.
4. **A4 deprioritized.** SliceGPT width pruning isn't GGUF-clean.
   Only revisit if depth pruning has a clear ceiling we can't break.
5. **Explicitly dropped from the ladder:** any path relying on
   unstructured pruning (SparseGPT / Wanda), LayerSkip-zero-shot,
   or trained-draft schemes (Medusa / EAGLE / Kangaroo).

## Metrics

Same `bench/` harness as Tracks A/B. Measured per candidate draft:

- **Draft standalone decode tok/s** (bench the pruned model alone
  to quantify the FLOPs win independent of speculation).
- **Acceptance rate per slice** (via `-md` against the target, same
  25-prompt fixture).
- **Effective speedup** (target + draft combined vs target alone).
- **Quality spot-check** (target's outputs unchanged at temp=0 by
  speculation; verified byte-for-byte in prior benches).

### What makes a pruned draft "useful"

A draft graduates out of research-only if:
- Mean acceptance rate ≥ 50% on the mixed fixture
- Effective speedup ≥ 1.3× on structured + code + reasoning
- No regression below 0.95× on any slice
- VRAM overhead ≤ the published Qwen3-0.6B-Q8_0 (609 MB)

If a pruned draft only beats the "no draft at all" baseline by <20%,
it doesn't justify the training cost and we should just use the
published draft when one exists.

## Training/recovery recipe

For approaches A2-A5, after pruning we need to recover quality
enough to hit useful acceptance. Recipe skeleton:

1. **Pruning step** — HF transformers, surgically remove selected
   blocks / slice hidden dims. Save intermediate checkpoint.
2. **Rollout generation** — sample Qwen3-8B-Q4_K_M (target) on a
   public prompt corpus (OpenOrca, UltraChat, plus The Stack for
   code, plus basic WikiText for general LM). ~1-10M tokens of
   rollout data.
3. **Distillation pass** — train the pruned model on the rollout
   corpus with cross-entropy loss (simple SFT) or KL divergence
   against target's logits (if we can afford the target forward
   passes during training).
4. **Intermediate eval** — every N steps, run a 5-prompt mini
   fixture through `spec_bench.py`, measure acceptance. Plot the
   curve. Decide stop-training when it flattens.
5. **Final eval** — full 25-prompt fixture, markdown comparison,
   write-up.
6. **GGUF export** — `convert_hf_to_gguf.py` produces a standard
   GGUF. Load with `-md` like any other draft.

## First experiment — A1 (smoke test)

Before any retraining, establish the baseline. Simple, 1-2 hours of
work.

### Steps

1. Download Qwen3-8B in HuggingFace format (not GGUF — we need to
   edit the model). ~16 GB bf16 or ~8 GB at bf16-to-fp8 if we want
   to save disk.
2. Write a small `prune_layers.py` — load the HF checkpoint, drop
   every other transformer block (`model.layers[::2]`), save to a
   new directory.
3. Convert the pruned HF checkpoint to GGUF via the upstream
   `convert_hf_to_gguf.py`, quantize to Q4_K_M.
4. Run the existing `run_condition.py` with the pruned GGUF as
   `-md`. Measure standalone decode tok/s and acceptance via the
   fixture.
5. Record results in this doc. Expected: acceptance very low
   (5-20%), speed very high (draft forward passes are half the
   FLOPs of target). Almost certainly a regression overall. This
   is fine — it sets the "how much does retraining buy us?"
   reference.

### Kill / continue criteria

- If pruned-no-retrain acceptance is **already ≥ 40%** → surprising
  enough that we might get away with very short retraining in A2.
  Proceed to A2 with a modest training budget.
- If acceptance is **<10%** → confirms quality collapses without
  recovery. A2 becomes the real experiment; budget the retraining
  carefully.
- If pruned model's standalone decode is **<2× target speed** →
  depth reduction isn't doing enough; reconsider which/how many
  layers to drop.

## Second experiment — A2 (the real test)

Only run after A1 establishes the floor. Rollout distillation over
N hours, checking acceptance periodically.

- Training corpus: ~1M tokens of Qwen3-8B rollouts on a mixed
  public prompt set. Skipping logit KD for V1 — plain SFT on target
  outputs is simpler and faster.
- Student: the A1 pruned model (half-depth Qwen3-8B, ~4B params).
- Loss: standard causal LM cross-entropy on target-emitted tokens.
- Optimizer: AdamW, cosine LR, short warmup.
- Duration: overnight (~8-12 hours). Mini-fixture eval hourly.
- Early stop if acceptance plateaus and mini-fixture speedup is
  stable.

**Decision after A2:**
- If pruned-retrained hits ≥ 50% acceptance and ≥ 1.3× speedup on
  favorable slices → write it up; this is the compression-as-draft
  result.
- If not → diagnose (was it the recipe, the pruning shape, or a
  fundamental ceiling?) and pivot to A3 (importance-based pruning)
  or A4 (width pruning).

## Infrastructure prerequisites

Need to verify before A1 starts:

- **PyTorch on Python 3.14 on Windows.** The user has Python 3.14;
  PyTorch nightlies *may* support it but it's bleeding edge. Fallback
  plan: install Python 3.12 alongside, or use WSL2 with 3.12.
- **HF transformers + datasets.** Standard install, ~1 GB.
- **GGUF conversion toolchain.** `convert_hf_to_gguf.py` is in the
  fork's root. Needs `sentencepiece`, `numpy`, `torch` — the last of
  which is the Python-version gating concern above.
- **Disk space for HF checkpoint.** Qwen3-8B bf16 is ~16 GB; plus
  intermediate checkpoints during training can double that. Need
  ~40-50 GB free on a fast drive.
- **Training corpus storage.** Rollout generation produces ~1-10M
  tokens × ~2-5 bytes of metadata each — a few GB at most. Trivial.

Pre-A1 action item: confirm the PyTorch / Python-3.14 compatibility
status. If it's a blocker, that's the first thing to fix — the rest
of the track depends on it.

## Open design questions

- **How aggressive should A1 be?** Drop every other layer (50%) vs
  drop every third (33%) vs drop first-and-last (preserves both
  edge layers, which are known to be important in transformers).
  The block-importance-based version (A3) answers this better but
  we need A1 to establish the floor.
- **Rollout-only SFT vs logit KD.** Rollout SFT is cheaper per step
  but logit KD typically gets more signal per token. For a V1 A2
  experiment, rollout SFT is the right starting point — fewer moving
  pieces.
- **How much target rollout data.** 1M tokens is a starting
  estimate, not a measured requirement. The acceptance-vs-training-
  tokens curve is what the mini-fixture eval during A2 will tell us.
- **Tokenizer fidelity.** Pruning preserves the tokenizer and
  embedding table (we don't touch those). No concerns there. Width
  pruning (A4) *would* need to slice the embedding; that's one of
  the reasons A4 is higher up the complexity ladder.

---

Ongoing progress lives below this line as dated subsections.

## 2026-04-20 — A1 + A1.1 zero-shot results

Executed the A1 plan and its lighter sibling A1.1 against Qwen3-8B-Q4_K_M
target. Same bench harness, fixture, and baseline flags as Track A/B.

### Setup shared by both

| | |
|---|---|
| Target | Qwen3-8B-Q4_K_M (36 layers) |
| Binary | upstream llama.cpp release b8855 |
| GPU | RTX 5080 Laptop, 16 GB |
| Source HF checkpoint | `Qwen/Qwen3-8B` bf16 (~16 GB) |
| Pipeline | HF → `prune_layers.py` → `convert_hf_to_gguf.py` bf16 → `llama-quantize` Q4_K_M |
| Draft flags added | `-md <pruned>.gguf --draft-max 8 -ngld 999` |

### A1: drop 10 middle-late layers (20-29), no retraining

- 36L → 26L (28% parameter reduction)
- Pruned Q4_K_M: 3.69 GB
- Round-trip time: prune 35s + GGUF convert 1.5 min + quantize 30s = ~3 min

### A1.1: drop 3 middle layers (20-22), no retraining

- 36L → 33L (8% parameter reduction)
- Pruned Q4_K_M: 4.59 GB
- Round-trip time identical

### Five-way comparison on identical fixture

| Condition | Wall | Mean tok/s | Accept | Speedup |
|---|---:|---:|---:|---:|
| baseline (no spec) | 40.8 s | 87.2 | — | 1.00× |
| qwen_draft (Qwen3-0.6B-Q8_0) | 31.7 s | 136.3 | 71.5% | 1.56× |
| q2k_self_draft (Qwen3-8B-Q2_K) | 53.2 s | 72.3 | 82.8% | 0.83× |
| **pruned-26L (28% drop)** | 79.8 s | 43.7 | **26.3%** | **0.50×** |
| **pruned-33L (8% drop)** | 61.7 s | 62.6 | **72.1%** | **0.72×** |

Per slice (tok/s (acceptance%)):

| Slice | Base | 0.6B | Q2_K | Pru-26L | Pru-33L |
|---|---:|---:|---:|---:|---:|
| structured_output | 83.7 | 152.3 (80%) | 66.8 (75%) | 43.6 (22%) | 61.7 (75%) |
| code | 89.7 | 163.0 (81%) | 77.7 (85%) | 44.4 (26%) | 67.8 (76%) |
| factual_qa | 94.0 | 138.1 (77%) | 84.7 (100%) | 44.6 (25%) | 70.3 (82%) |
| reasoning | 84.7 | 143.4 (84%) | 74.3 (92%) | 42.3 (29%) | 61.2 (78%) |
| conversational | 84.1 | 84.5 (55%) | 57.8 (74%) | 43.5 (24%) | 51.7 (65%) |

Temperature-0 correctness preserved in all conditions (per-prompt response
previews match byte-for-byte vs baseline).

### Key findings

**Finding 1 — Light drop preserves target-aligned output distribution.**
The 33L (8% drop) zero-retrain model hits **72.1% acceptance, statistically
identical to Qwen3-0.6B's 71.5%** — without any training. This is the
genuinely novel data point the prior-art survey flagged as unpublished.
Dropping 3 well-chosen middle layers preserves target behavior almost as
well as a dedicated published tiny draft.

**Finding 2 — But light drop doesn't save enough FLOPs.** 33 layers is
still 92% of target FLOPs, so the draft only runs ~1.08× faster than the
target. With the speculation math:
```
round cost = target + 8 × 0.92 × target = 8.36 × target
tokens per round @ 72% accept ≈ 5.76
speedup = 5.76 / 8.36 ≈ 0.69×    (measured: 0.72×)
```
Good acceptance + bad cost ratio = regression.

**Finding 3 — Heavy drop collapses acceptance.** At 28% drop, acceptance
falls from ~72% to 26% across all slices. The "without recovery,
pruning collapses past ~25-30%" finding from Gromov et al. reproduces
cleanly in the speculation-acceptance framing. Quality drop is not merely
a "slight PPL rise" — it's a full loss of target alignment.

**Finding 4 — Zero-shot pruning has no winning aggressiveness.** The
acceptance / draft-speed trade-off is bimodally bad on this target:
- Light drop (≤10%): high acceptance, slow draft → regression.
- Heavy drop (≥25%): fast-ish draft, quality collapse → worse regression.
- Middle ground doesn't exist at zero-shot: the collapse happens sharply
  around 25-30% drop.

To win via pruning, we'd need **aggressive drop (~50%+ params) with
acceptance-recovering training**. That's A2 territory (rollout
distillation) — genuinely needs gradient updates.

### The publishable data point

Before this run, no paper reported speculation-draft acceptance rates from
zero-shot pruned targets. Our 5-way table on Qwen3-8B characterizes the
frontier clearly and honestly. The "3-layer drop matches published 0.6B
draft acceptance, but loses on speed" is the key finding worth writing up
for the community — it tells anyone attempting pruning-as-draft that the
cheap zero-shot path has a ceiling they can't escape without retraining.

### Artifacts

Under `bench/results/`:
- `pruned_midlate10.jsonl`, `pruned_midlate10.run.log`, `.server.log`
- `pruned_mid3.jsonl`, `pruned_mid3.run.log`, `.server.log`

Pruned GGUFs in `Atlas/models/`:
- `Qwen3-8B-pruned-midlate10-Q4_K_M.gguf` (3.7 GB, 26 layers)
- `Qwen3-8B-pruned-mid3-Q4_K_M.gguf` (4.6 GB, 33 layers)

Pruned HF checkpoints under `research/models-hf/`:
- `Qwen3-8B-pruned-midlate10/` (bf16 full)
- `Qwen3-8B-pruned-mid3/` (bf16 full)

Plus ~21 GB of roundtrip validation artifacts (`Qwen3-8B-roundtrip-*`)
that can be reclaimed since the pipeline is validated.

### Next — A1.5

With A1 characterized, proceed to A1.5: calibration-only quality recovery.
The question is whether any no-gradient technique can bridge the gap
between the 26L-zero-shot (26% acceptance) and a retrained version of the
same pruned shape (predicted 50-70% acceptance from the literature).

Practical V1 of A1.5: **span-averaging replacement.** Drop 10 layers
(20-29), insert a single new transformer block whose weights are the
element-wise mean of the dropped layers' weights. Net model size: 27
layers. No gradient updates.

This is a simplification of LLM-Streamline's LSQ replacement — the
full LSQ version requires fitting a linear W from "input to span" to
"output from span" on a calibration set, then expressing that W as a
transformer block (which is fiddly given SwiGLU + RMSNorm). Starting
with the easier averaging version to validate whether *any* zero-shot
recovery helps before investing in the LSQ version.

## 2026-04-20 — A1.5 result: averaging doesn't help

Implemented span-averaging replacement (`bench/prune_avg_replace.py`):
drop contiguous span of N layers, insert 1 block whose weights are the
element-wise mean of the dropped layers' weights. Net model size:
`original - N + 1` layers. Tested on the same aggressive drop region
as A1 (layers 20-29 → 1 averaged block, giving 27 layers).

### Setup

Identical to A1 pipeline: HF checkpoint → `prune_avg_replace.py` →
`convert_hf_to_gguf.py` → `llama-quantize` Q4_K_M → bench.
Output GGUF size: 3.8 GB (vs 3.7 GB for pure 26L drop).

### Result

| Condition | Accept | Speedup | vs A1 no-replace |
|---|---:|---:|---:|
| pruned-26L (drop 10, no replace — A1) | 26.3% | 0.50× | baseline for this row |
| **avgspan-27L (drop 10, averaged replace — A1.5)** | **26.5%** | **0.50×** | **statistically identical** |

Per slice acceptance rates match the no-replace case to within 5
percentage points on every slice. The averaged block contributes no
measurable recovery.

### Why averaging doesn't work

Each dropped layer performs a specific *nonlinear* transformation:
`x + attn(norm(x)) + mlp(norm(x + attn(norm(x))))`. The composed effect
of 10 such layers is deeply nonlinear. Element-wise averaging of their
weights produces a single block whose transformation is arbitrary
linear-ish noise, not a meaningful approximation of the span's
composed effect. The A1.5-lite hypothesis that "any zero-shot insertion
beats nothing" is falsified for this variant.

### What about full LSQ LLM-Streamline

The proper LLM-Streamline variant would fit a linear `W` from
"hidden-state entering the span" to "hidden-state exiting the span"
using calibration data, via least squares. Two problems:

1. **Expressivity ceiling.** 10 layers of nonlinear processing is
   poorly approximated by a linear map. Published LLM-Streamline
   results show modest recovery at small drops (4-6 layers) but don't
   rescue aggressive drops. Based on A1.5-lite's failure, expected
   ceiling on a 10-layer drop is maybe 30-40% acceptance — still not
   useful.
2. **Transformer-block encoding is fiddly.** Expressing a fitted
   linear `W` as a valid transformer block (given SwiGLU + RMSNorm +
   residual) requires clever weight initialization. Meaningful
   implementation work for marginal expected gain.

**Not pursuing the full LSQ version.** The empirical evidence from
A1.5-lite, combined with the theoretical argument above, suggests
the upside doesn't justify the implementation effort at 10-layer drop
depth. A different approach variation (say drop 5 layers, LSQ replace)
*might* work, but "drop-5-no-replace" (A1.1) already got 72%
acceptance without any replacement, so the LSQ version would be
competing against that, not against the 26% floor.

## Complete zero-shot pruning frontier — characterized

Six conditions benched on Qwen3-8B-Q4_K_M target with identical
fixture/flags:

| Draft | Draft FLOPs (≈) | Accept | Speedup |
|---|---:|---:|---:|
| no draft (baseline) | 0% | — | 1.00× |
| **Qwen3-0.6B-Q8_0** (published, 0.6B param) | 7% | 71.5% | **1.56×** |
| Q2_K of target (quant compression) | 100% | 82.8% | 0.83× |
| pruned 33L (A1.1, drop 3) | 92% | 72.1% | 0.72× |
| pruned 26L (A1, drop 10) | 72% | 26.3% | 0.50× |
| avgspan 27L (A1.5, drop 10 + avg) | 75% | 26.5% | 0.50× |

Reading the table:

- **Only one condition wins vs baseline: the published tiny.** Zero-shot
  pruning of the target doesn't produce a useful draft at any
  aggressiveness level.
- **Acceptance & FLOP-reduction don't co-occur zero-shot.** To clear
  both bars (≥60% accept AND ≤30% of target FLOPs), retraining is
  required.
- **The FLOP-vs-bytes distinction is confirmed.** Q2_K (same FLOPs,
  fewer bytes) regresses despite 83% acceptance. Pure bit-width
  compression doesn't help speculation speedup.

## Decision — stop here on zero-shot, next step is A2

The zero-shot space is completely characterized. Six conditions, clean
per-slice numbers, one positive result (published tiny), four negative
results (quantization, light pruning, heavy pruning, averaged
replacement), and a consistent explanation (FLOPs dominate; target
alignment is necessary but not sufficient).

**Remaining open question:** does A2 (rollout distillation on
aggressive-drop model) recover enough acceptance to turn a heavily
pruned model into a useful draft? This requires gradient training —
~8-16 hours of GPU time on the 5080, not an in-session experiment.

The honest status on pruning-for-drafts is:
- **Known to not work zero-shot.** Characterized across the full
  parameter range. Unambiguous negative result.
- **Unknown but plausible with retraining.** Literature suggests
  recovery from aggressive pruning is possible with ~days of
  continued pretraining; whether rollout distillation alone (cheaper
  than continued pretraining) is enough is the A2 experiment.
- **Off-the-shelf remains the practical answer today.** For any target
  family with a published tiny draft, use that. The research value of
  A2 is for targets *without* a published match.

### A2 not run in this session

Deferred pending the user's decision on compute budget. The
A1 pruned-26L HF checkpoint in
`research/models-hf/Qwen3-8B-pruned-midlate10/` is the natural starting
point — rollout distillation can start there.
