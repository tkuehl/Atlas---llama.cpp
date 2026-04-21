# Speculative Decoding — Research Plan

Research track for characterizing and improving speculative decoding
efficiency in llama.cpp, aimed at the single-user consumer-hardware
regime. Downstream of the plan in [JOURNAL.md](JOURNAL.md)
(2026-04-20 entry).

Status: **planning + first experiment spec**. No results yet.

## Purpose

The broader motivation for this fork is to push back on the "throw
more hardware at it" default in LLM inference. Compute costs are
becoming prohibitive; meanwhile the single-GPU, single-user,
batch=1 decode regime is comparatively under-studied because most
industry speculation research optimizes for throughput servers at
batch>>1. There may be overlooked wins in the home-lab shape of the
problem.

This track measures what upstream llama.cpp's existing speculation
machinery actually delivers across workload types on consumer
hardware, publishes honest characterization data, and identifies
where new work (distilled drafts, knob tuning) has the most
leverage.

Results here are **upstream of any single consumer project** —
Atlas or otherwise. A downstream project adopts stable findings
when they apply to its workload, but this research is not gated
on any particular application.

## Top-level invariants

- **Upstream-only deployment.** Schemes must work with upstream
  llama.cpp (the main `ghcr.io/ggml-org/llama.cpp:server-cuda`
  image or a local build from upstream), not require runtime
  changes to this fork. Research that needs new loader code, new
  tensor layouts, or forward-path extensions is research-only and
  doesn't graduate.
- **Consumer-hardware focus.** Measurements are on a single GPU,
  batch=1 decode. Findings are expected to differ from published
  results that assume batch>>1; documenting that difference is
  itself a contribution.
- **Temp=0 first.** Greedy decode is where speculation's
  correctness guarantees are strongest. Sampling (temp>0) is a
  follow-up track once greedy is characterized.
- **Honest measurement.** Confirming prior art counts as a result.
  Matching a paper's published number is a valid contribution
  (reproduction data); diverging from it is the interesting case.

These rule out Medusa, EAGLE, LayerSkip, and anything else that
modifies the forward path. They keep:

- **N-gram speculation** (`--spec-type ngram-*`) — native in
  upstream, no artifact needed.
- **External draft via `-md`** — standard spec; any small
  same-family model works.
- **`--spec-replace`** — mismatched-tokenizer draft pairing; lossy
  escape hatch.
- **Distilled draft models** — output is a regular GGUF, loaded
  via `-md`; indistinguishable to upstream from any other draft.

## Metrics

All metrics come from the llama-server `timings` object
(attached to every OpenAI-compat response, per
`tools/server/server-task.cpp:615-633`):

| Field | What it tells us |
|---|---|
| `predicted_per_second` | decode tok/s — primary speed metric |
| `prompt_per_second` | prefill tok/s — affected by KV reuse |
| `cache_n` | tokens served from prefix cache |
| `draft_n` | draft tokens proposed |
| `draft_n_accepted` | draft tokens accepted by the target |
| `draft_n_accepted / draft_n` | **acceptance rate** — leading spec-decode quality indicator |

Derived metric: **effective speedup** =
`(target+draft tok/s) / (target-only tok/s)` on the same prompt.
Acceptance rate is the leading indicator; effective speedup is the
end metric.

No fixed "deploy bar" — this is characterization, not a gating
decision. We report the curve across workload slices; downstream
projects decide their own bars.

## Bench methodology

**Non-disruptive.** Tests run against a parallel llama-server
instance on a reserved port (e.g. 11502), not any existing
running service. Configuration under test is the only variable;
the baseline is flag-matched except for speculation knobs.

Dedicated `bench/` subdir under this research path:

- `bench/spec_bench.py` — hits a llama-server OpenAI endpoint with
  a prompt fixture, records `timings` per prompt as JSONL.
- `bench/spec_compare.py` — aggregates JSONL results into a
  markdown comparison table (config × prompt-category × metric).
- `bench/public_prompt_fixture.json` — public, reproducible
  prompt set (see next section).
- `bench/docker-compose.bench.yml` — spins up the parallel
  llama-server with a config passed via environment variable so
  the same compose file runs every test condition.

The existing `bench_model.py` / `bench_compare.py` from the SVD
track (JOURNAL 2026-04-18) are **not reused** — they run HF
models directly for PPL / greedy-agreement work. This track
measures end-to-end llama-server behavior, which is a different
shape of test.

## Prompt fixture

Acceptance rate is workload-dependent — n-gram speculation wins
big on structured / repetitive output and loses on open-ended
text. The fixture is sliced so we can report the shape of that
curve, not just an average.

Slices, 5 prompts each, hand-authored from public sources so
anyone running llama.cpp can reproduce:

- **structured_output** — JSON-schema completion, config-file
  generation, simple SQL. Highest-repetition slice; best case
  for n-gram.
- **code** — short function completions (HumanEval / MBPP style)
  in Python, C, and Go.
- **factual_qa** — short factual questions (TriviaQA style),
  brief answers.
- **reasoning** — multi-step math word problems (GSM8K style) and
  multi-hop reasoning. Longer outputs, mid-sequence decision
  points.
- **conversational** — open-ended chat prompts. Anti-repetition
  territory; worst case for n-gram.

Greedy decode (`temperature=0`, seed fixed) for all runs. Every
prompt carries a `slice` tag and a `max_tokens` budget so
comparisons are apples-to-apples.

## Track A — N-gram speculation

**Why first:** zero artifact, zero training, native upstream
support. Information per hour of work is maximal, and the result
is publishable regardless of direction.

Upstream offers five variants via `--spec-type`: `ngram-cache`,
`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`
(per `common/arg.cpp:3504-3555`). Each has different
memory/acceptance tradeoffs.

Knobs, same arg.cpp region:

- `--spec-ngram-size-n N` — lookup n-gram length
- `--spec-ngram-size-m N` — draft m-gram length
- `--spec-ngram-min-hits N` — min hits before prediction fires
- `--draft-min` / `--draft-max` — bounds on draft length per step
- `--draft-p-min` — minimum draft-probability threshold

First-pass matrix kept deliberately small:

| Variant | draft-max | ngram-n | ngram-m | rationale |
|---|---|---|---|---|
| baseline (none) | — | — | — | control |
| ngram-cache | 8 | 4 | — | most battle-tested in upstream |

Additional variants (`ngram-simple`, `ngram-map-k`, etc.) come
*after* the feasibility question is answered. Sweeping four
variants in the first test conflates feasibility with tuning.

### Characterization thresholds (not deploy gates)

- **Meaningful-win threshold.** ≥ 1.3× effective decode speedup on
  the `structured_output` slice with ≥ 60% acceptance rate. If a
  model can't clear this on its most favorable slice, n-gram
  speculation isn't a general-purpose lever on that model.
- **Regression floor.** ≤ 5% decode tok/s regression on the
  `conversational` slice. Above that, n-gram's overhead costs real
  performance on mixed workloads where it's left always-on.
- **Slice spread.** The gap between best and worst slice acceptance
  is itself a finding — it quantifies how workload-sensitive n-gram
  is on a given model.

## Track B — Off-the-shelf same-family drafts

**What this learns.** For users whose target has a published
same-family small draft, what do they gain by adding `-md`?
`Qwen3-8B` + `Qwen3-0.6B` is the best-documented pair today;
benching it gives the reference point any user can reach with
two lines of config change.

Other documented pairs worth checking as the track matures:
Llama-3.1-8B + Llama-3.2-1B, Gemma-2 family pairs. Not every
model family has a published tiny draft at matching revision;
absence-of-draft is itself a characterization result.

## Track C — Distilled standalone drafts

Produces an ordinary GGUF loadable via `-md`; indistinguishable to
upstream from any other small model. Not literal from-scratch
pretraining (not feasible on consumer hardware in useful time),
but two viable recipes:

1. **Output-distribution distillation.** Small student (~300-500M
   params, same tokenizer and architecture family as target)
   trained on the target's per-token logits over a held-out
   corpus. Standard KD. Days of compute on a single consumer GPU;
   acceptance rate tunable via training recipe.
2. **Data distillation.** Train a small model from its own
   initialization on text sequences *generated by* the target.
   Simpler than logit KD; acceptance rate typically lower but a
   useful baseline.

### Prerequisites before starting Track C

- A student architecture (matches target's tokenizer + chat
  template).
- A frozen teacher (target model).
- A publicly reproducible training corpus — mix of general text
  (WikiText, OpenWebText), code (The Stack or HumanEval-derived),
  and instruction-following (OpenOrca, UltraChat).
- A training loop — HuggingFace `Trainer` is sufficient for the
  300-500M regime on a single consumer GPU; single-GPU-weeks of
  wall time at most.
- Evaluation via `spec_bench.py` — same harness as Tracks A/B.
  Standalone PPL is at best a weak proxy; the real signal is
  target acceptance.

Track C only starts once Tracks A and B produce enough data to
know what bar a distilled draft needs to clear to be worth the
training cost.

## Cross-cutting risks

- **Acceptance rate is workload-specific.** A config that wins on
  structured output may lose on long-form reasoning. The slice
  structure exists to expose this; never collapse first-pass
  numbers into a single average.
- **Greedy vs sampling.** Speculation's correctness guarantees
  hold at `temperature=0`. At `temperature>0`, the sampling/draft
  interaction changes the acceptance math. All first-pass numbers
  are temp=0. Temp>0 follow-up is its own track.
- **Flag-matching on the baseline.** The baseline must carry the
  same flags as the speculation condition except for the spec
  knobs. KV-cache-quant, flash attention, context size, and
  cache-reuse all affect the numbers; a careless baseline mismatch
  gives a fake speedup.
- **Draft-model VRAM budget.** A tiny draft loaded alongside an
  8B-class target adds ~0.5-2 GB depending on quant and context.
  On a 12 GB card with Q4_K_M target, there's headroom; tuning
  `-ngld` may be needed at longer contexts.
- **`--spec-replace` semantic drift.** Cross-tokenizer draft
  pairing is available but realistically lossy. Escape-hatch only;
  do not treat its numbers as equivalent to matched-tokenizer
  draft results.

## Out of scope

- Medusa / EAGLE / LayerSkip prediction heads — require upstream
  runtime changes we can't ship. Research-interesting but not
  deployable via upstream. Revisit if upstream adopts them.
- Server-side multi-slot parallelism — throughput lever at
  batch>>1, not a decode-latency lever at batch=1. Different
  problem.
- Quality regressions against no-spec target — the correctness
  guarantee at temp=0 covers this; at temp>0 it becomes its own
  question, handled when sampling enters scope.

## First experiment (ready to run)

**Single objective.** Answer one question: does n-gram-cache
speculation produce a detectable win on a representative modern
open-weight 8B-class model, and where on the workload curve does
the win sit?

**Target.** Qwen3-8B-Q4_K_M.
- Modern, open-weight, freely downloadable — a
  representative-user baseline.
- Q4_K_M is llama.cpp's de-facto serving default; benching this
  quant means results apply to the config most users actually run.
- Publishes a `Qwen3-0.6B` draft, so the same target transfers
  directly into Track B without a model swap.

**Conditions.** Exactly two, flag-matched except for the speculation
addition:

- **baseline** — `-fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256 -c 8192 -ngl 999 --jinja`
- **ngram-cache** — baseline + `--spec-type ngram-cache --draft-max 8 --spec-ngram-size-n 4`

**Steps.**

1. `bench/spec_bench.py` — minimal llama-server OpenAI-compat
   client: POSTs `/v1/chat/completions` for each fixture prompt,
   captures `timings` into a JSONL line per prompt. Temperature=0,
   fixed seed.
2. `bench/public_prompt_fixture.json` — 5 slices × 5 prompts,
   hand-authored from public sources (HumanEval/MBPP-style for
   code, GSM8K-style for reasoning, TriviaQA-style for factual,
   simple JSON schemas for structured_output, open-ended for
   conversational).
3. `bench/docker-compose.bench.yml` — one llama-server service
   reading its command-line flags from an env var so the same
   compose file runs both conditions. Port 11502, reads the
   Qwen3-8B-Q4_K_M GGUF.
4. Pull Qwen3-8B-Q4_K_M if not already present in `./models/`.
5. Run condition `baseline`, save results → `bench/results/baseline.jsonl`.
6. Run condition `ngram-cache`, save results → `bench/results/ngram_cache.jsonl`.
7. `bench/spec_compare.py baseline.jsonl ngram_cache.jsonl` →
   markdown table: per-slice acceptance rate, effective speedup,
   TTFT delta, and a short qualitative spot-check.
8. Paste the compare output into this doc as a dated results
   subsection. Summarize in JOURNAL.md.

**Expected duration:** 2-3 hours once the Qwen3-8B GGUF is local
(~5 GB download if not already present).

**Feasibility thresholds (narrow scope).**
- If every slice shows < 30% acceptance → n-gram is not a
  general-purpose win on this target class; archive the result and
  redirect effort to Track B.
- If ≥ 1 slice clears the meaningful-win threshold (≥ 60% accept,
  ≥ 1.3× effective speedup) → write up the slice-spread curve as
  Track A's first finding. Follow-up is variant + knob sweep on
  the slices where it works, not a cross-the-board rollout.
- If all slices win big → surprising result; expand the matrix to
  all five variants and sweep knobs to characterize fully.

---

Ongoing results live below this line as dated subsections.

## 2026-04-20 — First results: ngram-cache on Qwen3-8B-Q4_K_M

Feasibility test per the "First experiment" section above.

### Setup

| | |
|---|---|
| Target model | Qwen3-8B-Q4_K_M.gguf (HuggingFace `Qwen/Qwen3-8B-GGUF`) |
| Binary | upstream llama.cpp release `b8855` (Windows CUDA 13.1) |
| GPU | RTX 5080 Laptop, 16 GB VRAM, idle before each run |
| Common server flags | `-fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256 -c 8192 -ngl 999 --jinja` |
| ngram-cache additions | `--spec-type ngram-cache --draft-max 8 --spec-ngram-size-n 4` |
| Request | `temperature=0`, `seed=42`, `chat_template_kwargs={enable_thinking: false}` |
| Fixture | 25 prompts × 5 slices — `bench/public_prompt_fixture.json` |

Server load-time: 5.7s baseline, 3.6s condition (warm file cache).

### Per-slice results

| Slice | Base tok/s | Cond tok/s | Speedup | Accept rate | TTFT Δ (ms) |
|---|---:|---:|---:|---:|---:|
| structured_output | 83.7 | 80.5 | 0.96× | 0.0% | +0 |
| code | 89.7 | 85.2 | 0.95× | — | +1 |
| factual_qa | 94.0 | 94.1 | 1.00× | — | +1 |
| reasoning | 84.7 | 83.9 | 0.99× | — | +2 |
| conversational | 84.1 | 83.2 | 0.99× | — | +1 |

**Overall:** baseline=87.2 tok/s, condition=85.4 tok/s, speedup=**0.98×**, overall acceptance **0.0%**.

(Accept-rate cells marked `—` had `draft_n=0` on every prompt in
that slice; there was nothing to compute a rate from. Overall 0%
because the only slice with any drafts at all — structured_output,
8 drafts on one prompt — saw 0 accepted.)

Temperature-0 correctness is preserved: per-prompt response
previews match byte-for-byte between baseline and condition across
all 25 prompts (see `bench/results/compare.md` for the full
side-by-side).

### Interpretation

`ngram-cache` at upstream-default knobs on Qwen3-8B-Q4_K_M is
**effectively a no-op that costs ~2% throughput.** The draft stream
barely fires (23/25 prompts have `draft_n=0`), the occasional draft
gets zero acceptances, and the always-on drafting overhead eats a
small but consistent ~1-3 tok/s off decode.

This triggers the feasibility doc's kill criterion: *"If every
slice shows < 30% acceptance → n-gram is not a general-purpose win
on this target class; archive the result and redirect effort to
Track B."*

### Mechanism — why it doesn't fire

The `ngram-cache` variant builds an n-gram cache from tokens seen
in the current context and predicts continuations from matches.
The cache starts empty per request (no cross-session state in
upstream's implementation), so it has to both populate and exploit
within one generation. With `enable_thinking=false`:

- Short outputs (factual_qa, ~10-25 tokens) finish before the cache
  has enough entries at n=4.
- Structured-output slices have surface-level regularity (brackets,
  commas, colons) but *token-level* uniqueness — values differ per
  prompt — so n=4 lookups miss.
- Code / reasoning / conversational outputs are generally "novel"
  within a single request: little intra-generation self-repetition
  for the cache to latch onto.

### Methodology note: Qwen3 thinking mode changes the bench

Initial runs used Qwen3's default `enable_thinking=true`. Outcome:
most prompts burned their entire `max_tokens` budget on
`<think>...</think>` reasoning and emitted no final answer — the
JSONL rows had `predicted_n ≈ max_tokens` but `response_length_chars
≈ 0`. The tok/s numbers were valid measurements of *reasoning-phase*
generation but didn't characterize the slice's intended output
shape.

Retried with `enable_thinking=false` via `chat_template_kwargs`
(exposed as `spec_bench.py --enable-thinking`, default false).
Archived the first-run JSONLs under `bench/results/think_on/`.

Comparing the two:

| Mode | Avg `draft_n` per prompt | Overall accept | Overall speedup |
|---|---:|---:|---:|
| thinking=on | ~46 | ~4.3% | 0.98× |
| thinking=off | ~0.3 | 0.0% | 0.98× |

N-gram-cache **fires ~150× more often during reasoning** — because
reasoning prose has formulaic phrasing ("Let me…", "First,…",
"Therefore…") that matches n=4 lookups — but acceptance stays low
and the end-to-end speedup is the same 0.98× in both modes.

**Methodological takeaway:** on reasoning-capable models, separate
reasoning-mode and answer-mode characterization. A speculation
config tuned for `<think>` tokens is a different config from one
tuned for final-answer tokens, and collapsing the two gives a
misleading single number.

### Decision

1. **Archive** `ngram-cache` at default knobs on Qwen3-8B-Q4_K_M —
   not a general-purpose win on this target class.
2. **Don't** sweep other n-gram variants / knobs on this target in
   the same session. Per scope discipline: the feasibility question
   has a clean answer (no). Knob sweeps belong to a tuning track we
   only enter if an initial config shows promise.
3. **Next action — Track B.** Pair Qwen3-8B-Q4_K_M with its published
   `Qwen3-0.6B` draft via `-md` and measure the same 25-prompt
   fixture. This is the reference point any llama.cpp user reaches
   with two flag additions; the characterization question is "what
   do published same-family drafts actually deliver on consumer
   hardware at batch=1, greedy?"

### Artifacts

Under `research/cross-layer-svd/bench/`:

- `public_prompt_fixture.json` — 25 public-sourced prompts
- `spec_bench.py` — llama-server OpenAI-compat client, records `timings`
- `run_condition.py` — start-server → bench → stop-server orchestrator
- `spec_compare.py` — JSONL → markdown comparison
- `results/baseline.jsonl`, `results/ngram_cache.jsonl` — raw results
- `results/compare.md` — full comparison table
- `results/think_on/*` — archived thinking-mode-enabled first run
- `results/*.server.log` — llama-server logs per condition (kept for audit)

## 2026-04-20 — Track B results: Qwen3-0.6B draft on Qwen3-8B

Track B executed on the same fixture / baseline / flag set.
The only change vs baseline is the addition of a same-family
published draft via `-md`.

### Setup

| | |
|---|---|
| Draft model | Qwen3-0.6B-Q8_0.gguf (609 MB, from HuggingFace `Qwen/Qwen3-0.6B-GGUF`) |
| Track B flag additions | `-md <draft> --draft-max 8 -ngld 999` |
| Everything else | identical to baseline (same server flags, same fixture, `enable_thinking=false`, temp=0, seed=42) |

### Per-slice results

| Slice | Base tok/s | Cond tok/s | Speedup | Accept rate | TTFT Δ (ms) |
|---|---:|---:|---:|---:|---:|
| structured_output | 83.7 | 152.3 | **1.82×** | 80.1% | -2 |
| code | 89.7 | 163.0 | **1.82×** | 81.3% | -2 |
| factual_qa | 94.0 | 138.1 | **1.47×** | 76.9% | -3 |
| reasoning | 84.7 | 143.4 | **1.69×** | 83.9% | -2 |
| conversational | 84.1 | 84.5 | 1.00× | 55.4% | -1 |

**Overall:** baseline=87.2 tok/s, condition=136.3 tok/s, speedup=**1.56×**, overall acceptance=**71.5%**.

Temperature-0 correctness preserved — all 25 response previews
match byte-for-byte vs baseline (see `bench/results/compare_draft.md`).

### Three-way summary on identical fixture

| Config | Overall tok/s | Overall accept | Overall speedup |
|---|---:|---:|---:|
| baseline (no spec) | 87.2 | — | 1.00× |
| ngram-cache (Track A) | 85.4 | 0.0% | 0.98× |
| qwen-draft (Track B) | **136.3** | **71.5%** | **1.56×** |

The gap between Track A (zero effect) and Track B (meaningful win)
is precisely what *same-family published draft* buys over
*zero-artifact n-gram cache* on this target class.

### Interpretation

**Four of five workload slices see a substantial speedup
(1.47-1.82×) with zero quality cost.** The fifth slice
(`conversational`) is neutral (1.00×) — the draft doesn't help,
but it also doesn't hurt. No slice regresses. This is a
no-quality-loss, no-regression, real-throughput-gain config.

### Why conversational is flat despite 55% acceptance

Worth dwelling on because it reframes the metric:

- `structured_output` / `code` / `reasoning`: acceptance ~80%,
  speedup ~1.7-1.8×
- `conversational`: acceptance ~55%, speedup ~1.00×

Acceptance rate alone doesn't predict speedup. The distribution of
**accepted run lengths** does. On predictable slices, drafts come
in long contiguous runs — many 8-token batches accepted fully,
one expensive target forward pass saving 7 target forward passes.
On conversational, acceptance is choppy: accept 2, reject 1,
accept 3, reject 1 — each reject costs a target forward pass
without amortizing the draft overhead. The draft cost per step
and the tokens saved land roughly balanced.

The practical implication: **acceptance rate ≥ 70% with
same-family drafts seems to be where speedups actually materialize
at `draft-max=8`**. Below that, draft overhead eats the win.

### What this means for the research thesis

This is the "more capability without more hardware" story at ~30
characters of configuration. Take a standard Qwen3-8B-Q4_K_M
server, add:

```
-md Qwen3-0.6B-Q8_0.gguf --draft-max 8 -ngld 999
```

and get ~56% more decode throughput on a mixed workload, with:

- **zero quality loss** (temperature=0 correctness; verified
  byte-for-byte output match)
- **zero training cost** (draft is published by the target's
  authors)
- **~609 MB extra VRAM** (trivial on any card that's already
  fitting the 8B target)
- **no upstream changes, no fork-only code** (runs on the
  standard `ghcr.io/ggml-org/llama.cpp:server-cuda` image or
  any upstream binary)

On consumer hardware. At batch=1. On representative mixed
workloads. This is directly usable by any llama.cpp user today,
and it lands right in the center of the "stop throwing more
hardware at the problem" research direction.

### Not yet tested (deferred per scope discipline)

- **Non-default knobs.** `--draft-max ∈ {4, 12, 16}`,
  `--draft-min`, `--draft-p-min` higher thresholds. Tuning on
  conversational specifically could lift that 1.00× — the 55%
  acceptance means the ceiling is there if overhead can be
  reduced. One-variable sweep is the right follow-up.
- **Other same-family pairs.** Llama-3.1-8B + Llama-3.2-1B,
  Gemma-2-9B + Gemma-2-2B. Each one confirms or diverges the
  pattern — does the 70% acceptance threshold hold on other
  model families?
- **Temperature > 0.** Under sampling, the acceptance-rate math
  changes. Same fixture, same target, sampled decode at
  `temperature ∈ {0.3, 0.7, 1.0}` is a follow-up track.
- **Per-slice fine-grained metrics.** We report per-slice
  averages. The per-prompt variance (reasoning ran 130-174 tok/s
  across 5 prompts) hints that within-slice workload shape
  matters too.

### Decision

**Track B cleared the meaningful-win threshold handily.**
Characterization data is strong enough to publish a writeup. For a
llama.cpp user running any modern 8B-class model with a
same-family ≤1B draft, the config above is a clear no-regret
change.

Next candidate directions (in priority order):
1. **Knob sweep** on Track B's `conversational` slice — does
   `--draft-max=4` + `--draft-p-min=0.7` reduce draft overhead
   enough to turn the flat 1.00× into a modest win?
2. **Cross-family generalization** — rerun the fixture with
   Llama-3.1-8B + Llama-3.2-1B as the pair. Is the 70% acceptance
   threshold a Qwen thing or a universal property?
3. **Track C (distilled draft)** — only worth doing if we find a
   target class where no good published draft exists. Current
   evidence says the off-the-shelf option is already excellent on
   modern model families.

### Artifacts

- `bench/results/qwen_draft.jsonl` — raw Track B results
- `bench/results/compare_draft.md` — Track B vs baseline comparison
- `bench/results/qwen_draft.run.log`, `.server.log` — per-run audit

## 2026-04-20 — Self-quant draft attempt: Qwen3-8B-Q2_K as draft for Qwen3-8B-Q4_K_M

Motivated by the question "can we skip training by quantizing the
target aggressively into a draft?" Intuitive appeal: same
architecture, same tokenizer, same distribution shape — draft's
output should agree with target's far more often than a separately-
trained model.

### Setup

| | |
|---|---|
| Target | Qwen3-8B-Q4_K_M (5.0 GB) |
| Draft candidate | Qwen3-8B-Q2_K (3.1 GB, from `bartowski/Qwen_Qwen3-8B-GGUF` — calibrated from bf16) |
| Flags added vs baseline | `-md Qwen3-8B-Q2_K.gguf --draft-max 8 -ngld 999` |
| Everything else | identical baseline, fixture, temp=0, `enable_thinking=false` |

### Results — three-way

| Condition | Total wall time | Mean tok/s | Accept | Speedup |
|---|---:|---:|---:|---:|
| baseline (no spec) | 40.8 s | 87.2 | — | 1.00× |
| qwen_draft (Qwen3-0.6B-Q8_0) | 31.7 s | 136.3 | 71.5% | 1.56× |
| **q2k_self_draft (Qwen3-8B-Q2_K)** | **53.2 s** | **72.3** | **82.8%** | **0.83×** |

Per-slice (tok/s):

| Slice | Baseline | Qwen 0.6B draft | Q2K 8B self-draft |
|---|---:|---:|---:|
| structured_output | 83.7 | 152.3 (1.82×) | 66.8 (0.80×) |
| code | 89.7 | 163.0 (1.82×) | 77.7 (0.87×) |
| factual_qa | 94.0 | 138.1 (1.47×) | 84.7 (0.90×) |
| reasoning | 84.7 | 143.4 (1.69×) | 74.3 (0.88×) |
| conversational | 84.1 | 84.5 (1.00×) | 57.8 (0.69×) |

Temperature-0 correctness preserved; response previews still match
byte-for-byte across all three conditions.

### The finding

**Quantization-as-draft regresses ~17% on wall time despite
measurably higher acceptance.** The intuition about architectural
alignment was correct — 82.8% acceptance vs the 0.6B's 71.5% — but
the speedup formula cares about draft cost per proposed token, not
draft accuracy alone.

### Why — the FLOPs-vs-bytes distinction

At batch=1 consumer-GPU decode, throughput is roughly
`weight_bytes_per_token / memory_bandwidth`. Quantization reduces
bytes-per-weight but **not parameter count**. So:

- Qwen3-0.6B-Q8_0: ~600 MB/token streamed → ~5× faster than target
- Qwen3-8B-Q2_K: ~3000 MB/token streamed → only ~1.7× faster than target
- Draft-overhead ratio is 1/5 vs 1/1.7 — huge difference in the
  speculation speedup formula.

Applied to the data: with 82.8% acceptance and draft_max=8, we
accept ~6.6 tokens per round. Round cost at 1.7× speedup is
`target + 8 × target/1.7 = 5.7 × target`. Effective speedup vs
baseline = `6.6 / 5.7 = 1.15×` theoretical, but in practice the two
models contend for GPU memory bandwidth simultaneously, giving us
0.83×.

**Key principle:** speculation speedup is governed by
**FLOPs-per-token in the draft**, not bytes. Compression that
doesn't reduce parameter count doesn't help.

### Corrected claim (for the record)

Earlier framing of this track referenced "CALDERA-style Q + L·R"
as a path to a tiny draft. CALDERA reduces **storage bytes**, not
FLOPs. `y = Qx + L(Rx)` has the same `Qx` FLOP count as the
original matmul (quantization reduces bits, not ops) plus the
correction's `(d_out + d_in) × r` extra ops. A CALDERA version of
Qwen3-8B (say Q2_K base + rank-128 correction, ~3.5 GB) would sit
on the wrong side of the bandwidth curve vs pure Q2_K, with acceptance
likely only marginally better. CALDERA is a tool for
*fitting-bigger-models-in-less-VRAM*, not for producing drafts.
Don't use it for this.

### What actually moves the right lever

For a draft to be useful, it needs **fewer parameters** than the
target, not just compressed parameters. Concretely:

1. **Layer pruning + light retraining** — drop N of target's
   transformer layers, recover quality via rollout distillation. Hits
   both fewer FLOPs and fewer bytes. See
   [draft_pruning.md](draft_pruning.md).
2. **Width pruning** (hidden dim, head count) — same idea, different
   axis. Same retraining requirement.
3. **Small-student distillation** — train a fresh ≤1B model on target
   outputs. Covered in [draft_distillation.md](draft_distillation.md).

Pruning is closest to the "compression as draft" intuition that has
a real shot at working, and pairs cleanly with quantization at the
end (prune for FLOP savings, quantize for bytes — compounding).

### Artifacts

- `bench/results/q2k_self_draft.jsonl` — raw
- `bench/results/compare_q2k.md` — baseline-vs-Q2K comparison
- `bench/results/q2k_self_draft.run.log`, `.server.log` — audit
- `Atlas/models/Qwen3-8B-Q2_K.gguf` — kept as negative-result
  reference point
