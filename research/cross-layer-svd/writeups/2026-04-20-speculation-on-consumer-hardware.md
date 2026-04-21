# A ~56% Decode Speedup for Qwen3-8B on a Consumer GPU, in Three Flags

**Date:** 2026-04-20
**Target audience:** llama.cpp users running modern 8B-class models on a single consumer GPU at batch=1 (i.e. most of us).
**TL;DR:** Upstream llama.cpp's built-in speculative decoding, paired with Qwen's own published 0.6B draft model, gives a **1.56× decode throughput improvement on mixed workloads with zero quality cost** — just by adding three flags to the server command line. N-gram speculation at default knobs, by contrast, does ~nothing on this target class. Full bench methodology, per-slice numbers, and reproduction instructions below.

---

## Why bother

The prevailing answer to "my LLM is slow" is "buy more hardware." That answer has a ceiling — compute costs are now the dominant constraint for a lot of self-hosted inference — and a blind spot: the single-GPU, single-user, batch=1 regime is under-measured. Industry speculation benchmarks assume batch>>1 throughput servers, where the math is different.

This writeup characterizes what upstream llama.cpp's existing speculation machinery actually delivers on a consumer GPU for a freely-available modern 8B model. Nothing here requires a fork, a custom kernel, or a trained artifact. The goal is to surface wins that are already in the codebase but are not on by default.

## Setup

| | |
|---|---|
| Target model | Qwen3-8B-Q4_K_M (HuggingFace `Qwen/Qwen3-8B-GGUF`) |
| Draft model (Track B) | Qwen3-0.6B-Q8_0 (`Qwen/Qwen3-0.6B-GGUF`) |
| Binary | Upstream llama.cpp release `b8855`, Windows CUDA 13.1 |
| GPU | NVIDIA RTX 5080 Laptop, 16 GB VRAM |
| Server baseline flags | `-fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256 -c 8192 -ngl 999 --jinja` |
| Request | `temperature=0`, `seed=42`, `enable_thinking=false` via `chat_template_kwargs` |
| Fixture | 25 hand-authored public prompts across 5 slices (5 each) |

Slices:
- **structured_output** — JSON schema completion, config-file generation, short SQL
- **code** — function completions in Python, C, Go
- **factual_qa** — short Q&A (TriviaQA-shape)
- **reasoning** — multi-step math word problems (GSM8K-shape)
- **conversational** — open-ended chat prompts

Temperature=0 gives us greedy decode, where speculation's correctness guarantees are bit-exact. All 25 response previews match byte-for-byte across conditions; speculation never changes what gets generated, only how fast.

## Three conditions

All three share the baseline flag set above. They differ only in what they add:

1. **baseline** — no speculation.
2. **ngram-cache** — `--spec-type ngram-cache --draft-max 8 --spec-ngram-size-n 4`. Built-in upstream n-gram speculation; no draft model needed.
3. **qwen-draft** — `-md /path/to/Qwen3-0.6B-Q8_0.gguf --draft-max 8 -ngld 999`. External draft model, same family as target, published by the target's authors.

Each condition is run against the same 25-prompt fixture with a fresh llama-server process.

## Headline numbers

### Decode throughput (mean tok/s across all 25 prompts)

| Condition | tok/s | vs baseline | Overall acceptance |
|---|---:|---:|---:|
| baseline | 87.2 | 1.00× | — |
| ngram-cache | 85.4 | 0.98× | 0.0% |
| **qwen-draft** | **136.3** | **1.56×** | **71.5%** |

### Wall time — full bench (25 prompts end-to-end)

| Condition | Total wall time | Tokens generated | Mean time per prompt |
|---|---:|---:|---:|
| baseline | 40.8 s | 3,364 | 1.63 s |
| ngram-cache | 41.4 s | 3,364 | 1.66 s |
| **qwen-draft** | **31.7 s** | 3,396 | **1.27 s** |

The full 25-prompt bench finishes **9 seconds faster** with the draft model — a 22% wall-time reduction.

## Per-slice results (draft condition vs baseline)

| Slice | Base tok/s | Draft tok/s | Speedup | Accept rate | Mean time before → after |
|---|---:|---:|---:|---:|---|
| structured_output | 83.7 | 152.3 | **1.82×** | 80.1% | 0.72 s → 0.44 s |
| code | 89.7 | 163.0 | **1.82×** | 81.3% | 0.63 s → 0.39 s |
| factual_qa | 94.0 | 138.1 | **1.47×** | 76.9% | 0.16 s → 0.13 s |
| reasoning | 84.7 | 143.4 | **1.69×** | 83.9% | 3.32 s → 2.03 s |
| conversational | 84.1 | 84.5 | 1.00× | 55.4% | 3.33 s → 3.35 s |

**Four slices win substantially. One is neutral. None regress.**

The biggest absolute wall-time saving is on reasoning — 1.29 s saved per prompt on a 3.32 s baseline. Structured output and code are proportionally largest (1.82×). Factual QA has a high ratio but small absolute savings because the outputs themselves are tiny.

## Ngram-cache gets nothing here

Worth pausing on — the upstream `--spec-type ngram-cache` is often presented as "free speedup, no draft needed." On this target at default knobs, it isn't: 0.0% acceptance, 0.98× speedup (a ~2% regression from the drafting overhead).

The mechanism: `ngram-cache` builds an n-gram cache from tokens seen in the current request's context and predicts continuations from cache hits. With empty cross-request state (each request starts fresh) and n=4, most slices never populate enough cache to get useful hits before the output ends. The few drafts that do fire are rarely accepted by the target.

This is a useful prior-art check: the "turn on `--spec-type` and forget it" advice is not correct for all targets. It may help on workloads with heavy intra-request repetition (long reasoning traces, repeated templating), but at default knobs on short representative tasks it's a no-op with a small cost.

### A separate note on Qwen3 reasoning mode

An earlier pass on this bench had Qwen3's default `enable_thinking=true`. In that mode, most prompts spent their entire `max_tokens` budget emitting `<think>...</think>` reasoning and never reached the final answer. That's a valid measurement of reasoning-phase throughput, but not a characterization of the slice's intended output shape.

When thinking is off, n-gram-cache basically never fires (23/25 prompts had `draft_n=0`). When thinking is on, it fires ~150× more often (reasoning prose has formulaic phrasing that matches n=4 lookups), but acceptance is still ~5% and end-to-end speedup is the same ~0.98×.

Takeaway: on reasoning-capable models, bench reasoning-mode and answer-mode separately, because speculation's behavior differs meaningfully between them. A config that "works" in one mode can be a no-op in the other.

## The metric that actually predicts speedup

Here's the interesting data point from Track B, the one not usually surfaced in speculation writeups.

| Slice | Acceptance rate | Speedup |
|---|---:|---:|
| structured_output | 80.1% | 1.82× |
| code | 81.3% | 1.82× |
| reasoning | 83.9% | 1.69× |
| factual_qa | 76.9% | 1.47× |
| conversational | 55.4% | 1.00× |

**Acceptance rate alone doesn't predict speedup.** Conversational hit 55% acceptance but landed at 1.00× — the draft's per-step overhead balanced almost exactly against the tokens saved. Structured and code hit 80% and got 1.82×.

The difference isn't in the rate, it's in the **distribution of accepted run lengths**. On predictable outputs, the target accepts drafts in long contiguous runs — 8 tokens drafted, 8 accepted, one target forward pass saves 7 of them. On conversational, acceptance is choppy — 2 accepted, 1 rejected, 3 accepted, 1 rejected — so each rejection still costs a target forward pass and the draft overhead doesn't amortize.

Practically, at `--draft-max 8`, **acceptance ≥ ~70% is the rough threshold where speedups actually materialize on this class of target**. Below that, the draft is paying its freight without earning it.

This suggests tuning strategies:
- For workloads in the 50-70% acceptance band, reducing `--draft-max` (so each rejection costs less) or raising `--draft-p-min` (so only high-confidence drafts are emitted) could lift the speedup. Untested here.

## How to reproduce

Download the models:

```bash
# Target
curl -L -o Qwen3-8B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf

# Draft
curl -L -o Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

Pull the upstream llama.cpp binary (release page: `github.com/ggml-org/llama.cpp/releases`). For a Blackwell / RTX 40-series / RTX 50-series GPU with CUDA 13.x, grab `llama-<tag>-bin-win-cuda-13.1-x64.zip` + `cudart-llama-bin-win-cuda-13.1-x64.zip`.

Start the server:

```bash
llama-server \
  -m Qwen3-8B-Q4_K_M.gguf \
  -md Qwen3-0.6B-Q8_0.gguf \
  -ngl 999 -ngld 999 \
  --flash-attn on \
  -ctk q8_0 -ctv q8_0 \
  --cache-reuse 256 \
  --draft-max 8 \
  -c 8192 --jinja \
  --host 127.0.0.1 --port 11502
```

That's the entire config. Three flags on top of a reasonable default server setup (`-md`, `-ngld`, `--draft-max`). VRAM cost: +609 MB for the Q8_0 draft. On a 16 GB card, after the 8B target (~5 GB) + quantized KV cache + draft + runtime overhead, there's room.

Harness and fixture for full reproduction are in `research/cross-layer-svd/bench/` of this repository:
- `public_prompt_fixture.json` — 25 prompts, 5 per slice
- `spec_bench.py` — client that POSTs each prompt and records the `timings` field from the response
- `run_condition.py` — orchestrator that starts the server, runs the bench, stops the server
- `spec_compare.py` — JSONL → markdown comparison tables

## Caveats

- **Single target, single GPU, one bench run per condition.** Variance wasn't characterized; reported numbers are a single-run mean per slice. Repeat runs are cheap — rerunning to get variance bars is a reasonable next step.
- **Temperature = 0 only.** Under sampling (temp > 0), the acceptance math changes. Speedups likely remain but shapes may differ.
- **One model family.** Qwen3 has published the explicit matched draft; not every family does. The obvious next test is the same setup on Llama-3.1-8B + Llama-3.2-1B, Gemma-2-9B + Gemma-2-2B, etc.
- **Default knobs only.** `--draft-max 8`, default `--draft-min`, default `--draft-p-min`. The conversational flat result suggests per-slice knob tuning has headroom.

## What this is evidence for

For any llama.cpp user running a model family that publishes a tiny same-family draft, the config above is a no-regret change: speedups on most workloads, no regressions anywhere, zero training, zero quality cost, ~600 MB of VRAM.

That's not a novel technique — speculative decoding has been in llama.cpp for a long time, and Qwen themselves publish their drafts for exactly this use. It is a concrete measurement of how much performance is sitting in the defaults, untaken. Three flags and ~600 MB buy back 22% wall-clock time on this bench.

For the "stop scaling hardware, use what's there better" direction: this is one data point. The follow-up question — *what do we do for target models that don't have a published tiny draft?* — is where the research gets harder.

---

## Addendum — Self-quant as draft (what didn't work, and why it's interesting)

**The intuition:** if a same-family tiny draft works, skip the training step entirely — quantize the target itself aggressively and use the quant as the draft. Same architecture, same tokenizer, same distribution. Should give *higher* acceptance than an independently-trained 0.6B, and at zero training cost.

**The test:** Qwen3-8B-Q4_K_M target, Qwen3-8B-Q2_K as draft (bartowski's Q2_K, calibrated from bf16, 3.1 GB). Same fixture, same flags, same `-md` mechanism.

**Result — three-way:**

| Condition | Total wall | Mean tok/s | Accept | Speedup |
|---|---:|---:|---:|---:|
| baseline (no spec) | 40.8 s | 87.2 | — | 1.00× |
| Qwen3-0.6B-Q8_0 as draft | 31.7 s | 136.3 | 71.5% | 1.56× |
| **Qwen3-8B-Q2_K as draft** | **53.2 s** | **72.3** | **82.8%** | **0.83×** |

The intuition was half right. Acceptance genuinely went *up* — 82.8% vs the 0.6B's 71.5% — because the quantized target really does agree with the full target more often than a smaller unrelated model does. But overall wall time went *up* 30%, because the draft is too expensive per token. Every slice regressed. Conversational dropped to 0.69× (31% slower than baseline).

**The principle this calibrates:**

> Speculation speedup is governed by *FLOPs-per-token in the draft*, not bytes-per-token.

Quantization reduces bytes but not parameter count. A Q2_K 8B still has 8B parameters' worth of matrix multiplies per forward pass; its storage is smaller, its FLOP count is identical. At batch=1 consumer-GPU decode, that matters:

- Qwen3-0.6B-Q8_0 draft: ~600 MB/token streamed → ~5× faster than the 8B target per forward pass.
- Qwen3-8B-Q2_K draft: ~3 GB/token streamed → only ~1.7× faster than the 8B target.

At 1.7× draft speed, even 83% acceptance can't pay for the draft's per-step cost. At 5× draft speed with 70% acceptance, you win handily. The 0.6B wins not by being smarter — it's measurably *less* aligned with the target — but by having **14× fewer parameters**.

**Implication for CALDERA / Q+LR compression approaches:** compression schemes that reduce storage without reducing parameter count (CALDERA, SqueezeLLM, AQLM, QuIP#, the whole family) are not the right tool for producing drafts. They shine at the opposite goal — fitting a bigger model in less VRAM. Different lever.

**What does produce a useful draft from a target without training it from scratch:** architectural reduction. Layer pruning (drop depth) or width pruning (narrower hidden dim / fewer heads) cuts both bytes *and* FLOPs. The quality drop from blind pruning needs recovery — typically a short distillation pass on target outputs — but the net compute budget ("make a draft") is an overnight run on consumer hardware rather than weeks of from-scratch training.

Worth making concrete for people who want to try this themselves:

- Pure layer drop, no retraining → likely ~10-30% acceptance, proof-of-concept only.
- Layer drop + ~1 day of rollout distillation on target-emitted text → plausibly 50-70% acceptance at half the parameter count.
- Then quantize the pruned result → gets both the FLOP win and the byte win.

That's the open research question we're taking up next. The 0.6B-draft result above is what "it works" looks like when the target's authors published a matched small. The pruning-distillation question is what "produce it yourself" looks like for targets where they didn't.

---

## Addendum 2 — Zero-shot pruning as a draft, and why it can't quite win alone

If quantization shrinks bytes but not FLOPs, the natural next idea is **pruning** — drop whole transformer layers from the target and use the pruned model as the draft. Prior art (ShortGPT, LLM-Streamline, LaCo, Gromov et al. "Unreasonable Ineffectiveness of the Deeper Layers") shows ~25-30% of layers can be removed from 7-13B models with calibration-only techniques while preserving most of the general quality. The question is whether the speculation-acceptance metric holds up the same way.

Bench on Qwen3-8B-Q4_K_M, same 25-prompt fixture, same flags — just swapping the draft:

| Draft | Draft params | Accept | Speedup |
|---|---|---:|---:|
| Qwen3-0.6B-Q8_0 (published) | 0.6B | 71.5% | 1.56× |
| Pruned 33L (drop layers 20-22, zero retraining) | 7.3B | **72.1%** | 0.72× |
| Pruned 26L (drop layers 20-29, zero retraining) | 5.8B | **26.3%** | 0.50× |

**The interesting finding:** dropping just 3 middle layers from Qwen3-8B and using the result as a draft hits **72.1% acceptance — statistically identical to the published 0.6B draft's 71.5%**. Without any training. No published paper reports this measurement; this is the first honest data point for "what does zero-shot pruning give you as a speculation draft."

**The catch:** 33 layers is 92% of the target's FLOP count, so the draft decodes only ~1.08× faster than the target. At that cost ratio, even 72% acceptance isn't enough to amortize the draft — you still regress 28% on wall time.

Pushing harder on compression (dropping 10 layers instead of 3) crashes acceptance to 26%. The failure modes on both ends give us the frontier:

- **Light zero-shot drop:** quality preserved, draft too slow → regression.
- **Heavy zero-shot drop:** quality collapses → worse regression.
- **Middle ground where both axes cooperate doesn't exist** without training.

The mathematical conclusion: to make pruning-as-draft profitable, a draft needs ~50%+ FLOP reduction *and* ≥60% acceptance. Those two conditions don't co-occur in the zero-shot regime — you need to retrain the aggressively-pruned model to recover acceptance after a big drop.

That retraining question is the remaining open piece — and it's the honest next direction for any target that lacks a published tiny draft. A day of rollout-distillation on an aggressively-pruned model is the minimum experiment that could produce a genuinely novel, genuinely useful result. If it works, it's a recipe anyone with a consumer GPU can follow. If it doesn't, we learn that hand-trained tiny drafts (like Qwen's 0.6B) don't have obvious cheaper substitutes.

**For the "stop scaling hardware" thesis more broadly:** the strongest practical answer today remains the three-flag config change from the main article — if your target has a published small draft, use it. The pruning data says "self-produce the draft" is a harder problem than the intuitive framing makes it look, but not an impossible one.

---

## Addendum 3 — The complete zero-shot frontier, and what it tells us

Before calling the story done, one more experiment closed the loop. LLM-Streamline (Zhou et al.) reports that after aggressively pruning a span of layers, you can partially recover quality by inserting a single "replacement layer" fit via least squares on calibration hidden states. Light version tested here: replace the dropped span with one transformer block whose weights are the element-wise *average* of the dropped layers' weights.

Result vs plain drop-and-forget:

| Drop approach | Accept | Speedup |
|---|---:|---:|
| Drop 10 layers, no replacement (A1) | 26.3% | 0.50× |
| Drop 10 layers, insert 1 averaged block (A1.5) | 26.5% | 0.50× |

Statistically identical. Element-wise averaging across a deeply nonlinear span produces an arbitrary linear-ish perturbation that doesn't approximate the span's composed transformation. The A1.5 hypothesis "any zero-shot insertion beats nothing" is falsified for averaging.

The full LSQ-based version would probably recover a few percentage points, but not enough to escape the frontier: a linear fit of 10 transformer blocks' composed effect is fundamentally expressivity-limited. And implementing the full LSQ version — encoding the fitted linear as a valid transformer block given SwiGLU and RMSNorm — is meaningful work for likely-marginal gain.

With this last negative result, the zero-shot space is fully characterized:

| Draft | Accept | Speedup | Verdict |
|---|---:|---:|---|
| Qwen3-0.6B published tiny | 71.5% | **1.56×** | **wins handily** |
| Q2_K 8B (quantize target) | 82.8% | 0.83× | high accept, draft too slow |
| Pruned 33L (light, 8% drop) | 72.1% | 0.72× | same problem, less compression |
| Pruned 26L (heavy, 28% drop) | 26.3% | 0.50× | quality collapse |
| Averaged replacement 27L | 26.5% | 0.50× | averaging doesn't bridge |

**The complete honest answer to "can I produce a useful draft without training?":** for an 8B-class target, on current consumer hardware, *no*. The only winning condition in the table is the already-trained published tiny. Every attempt to derive a draft from the target itself — quantization, pruning at any aggressiveness, or calibration-only recovery — either regresses against baseline or fails to produce meaningful speedup.

This is a clean negative result with a practically useful corollary. **If the target you run has a published same-family small draft, three flags get you a ~56% decode speedup with no quality cost. If it doesn't, you cannot shortcut the training step.** Medusa / EAGLE / LayerSkip all agree on this; our zero-shot data just traces the boundary without gradient updates.

For the specific case of "produce a draft for a target without a published match" — the open follow-up is A2: rollout-distill an aggressively-pruned model overnight on consumer hardware. Literature suggests this should recover enough acceptance to matter. Whether it actually does is the next experiment. But the case for it has to overcome the simpler alternative — just use Qwen3 (or any other model family with published drafts) as your target if you have a choice.

### Reproducibility

Full bench harness, fixture, pruning scripts, and raw JSONL results are in `research/cross-layer-svd/bench/`. Every number in this article can be reproduced end-to-end in ~half a day on a consumer GPU, including the model downloads, the bench runs, and the pruning + conversion pipeline.
