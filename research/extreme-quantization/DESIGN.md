# Extreme Quantization — Phase 0 Research Vision

**Status:** Phase 0 (vision / scope). No code yet.
**Date:** 2026-04-20
**Relates to:** [CALDERA](../cross-layer-svd/) — complementary, not competing. Rotation and QAT stages compose *before* and *after* CALDERA decomposition respectively; see §5.

---

## 1. Thesis

Modern information-dense models (Llama-3, Qwen3, DeepSeek-V3 class) are
dramatically harder to quantize than the Llama-2-era checkpoints that
seeded the "2-bit 70B on one consumer GPU" narrative. The gap is real
and well-documented: at 2-bit, Llama-3-70B loses multiple perplexity
points where Llama-2-70B loses a fraction of one. More training tokens
per parameter → less weight redundancy → less tolerance for
quantization noise.

The gap does not yield to any single quantization method. It yields to
a **stack**: rotation preprocessing → sensitivity-aware calibration →
non-uniform bit allocation → low-bit quantization → short QAT recovery.

This research vein investigates building that stack as new stages in
the existing `HF → optimize-stages → GGUF` pipeline, composable with
CALDERA rather than replacing it.

---

## 2. Why this direction, why now

Three converging 2024 data points:

- **BitNet b1.58** (Microsoft) — ternary-native training matches FP16
  at ≥3B scale. The "weights must be real-valued matrices" assumption
  is empirically broken at pretraining scale. (Not directly actionable
  without full retraining, but it raises the ceiling on what
  post-training methods can aspire to.)
- **QuaRot / SpinQuant** — Hadamard and learned orthogonal rotations
  eliminate the activation outlier channels that cause most low-bit
  quantization error on modern dense models. Applied pre-quantization,
  folded into adjacent weights, zero inference-time overhead.
- **EfficientQAT / PV-Tuning** — a few epochs of post-quantization
  fine-tuning recover 50-80% of the remaining quality gap at sub-3-bit.
  Costs ~1% of pretraining FLOPs.

None of these are composed in any open-source pipeline today. That's
the opening.

---

## 3. Target outcomes

Concrete quality/hardware targets. "Ppl gap" measured on Wikitext-2 +
a held-out mixed domain set (code, math, multilingual), not Wikitext
alone — this repo's experience with Llama-3 shows Wikitext-only
calibration overfits to a narrow distribution.

| ID | Model              | Bit budget | Quality target              | Hardware                                   |
|----|--------------------|------------|-----------------------------|--------------------------------------------|
| T1 | Qwen3-8B (current daily driver) | 2.0 bpw avg | ≤0.5 ppl gap vs FP16 | 6 GB consumer GPU, 8K KV          |
| T2 | Llama-3-8B         | 2.0 bpw avg | ≤0.5 ppl gap vs FP16        | 6 GB consumer GPU, 8K KV                   |
| T3 | Llama-3-70B        | 2.5 bpw avg | ≤1.5 ppl gap vs FP16        | Single 24 GB card (4090/3090), 8K KV       |
| T4 | Llama-3-70B        | 2.0 bpw avg | ≤3.0 ppl gap vs FP16 (stretch) | Single 24 GB card, 4K KV                |

T1/T2 are the proving ground. T3 is the headline. T4 is a reach target
that tells us where the pipeline's floor actually lies.

Non-goals: outperforming BitNet's from-scratch ternary numbers (we're
not retraining), outperforming dense FP16 (ceiling is FP16 minus
quantization loss), upstream merge (this is a private fork stage set).

---

## 4. The stack

Each stage is orthogonal and independently testable. The stack is the
contribution; none of the individual components are novel.

### 4.1 Rotation preprocessing (new stage)

Apply orthogonal rotations `R` to weight matrices such that `R W` has
flatter per-channel magnitude distributions. Fold the inverse into the
adjacent downstream weight matrix so the rotation vanishes at
inference.

Two flavors to evaluate:

- **Hadamard rotations (QuaRot-style)** — free, deterministic,
  closed-form. One-shot preprocessing. Empirically strong on Llama-3.
- **Learned rotations (SpinQuant-style)** — optimized via small
  calibration loop. Marginally better on paper, more complex to
  implement.

**Decision point:** start with Hadamard. Only escalate to learned
rotations if Hadamard leaves a ≥0.3 ppl gap we can't close downstream.

Folding points in Llama-3 architecture: attention QKV input, attention
output, MLP gate/up input, MLP down output. RMSNorm weights absorb the
pre-rotation scale.

### 4.2 Sensitivity-aware calibration

Replace Wikitext-only imatrix with a diverse corpus that matches the
target deployment:

- ~40% general web (C4 / RedPajama subset)
- ~25% code (StarCoder subset)
- ~15% math (OpenWebMath subset)
- ~10% multilingual
- ~10% instruction-following / chat format

Evaluate whether imatrix quality alone is worth ~0.3-0.5 bits of
effective precision on Llama-3. Prior CALDERA work in this repo
already shows calibration sensitivity matters; quantification is what
we need.

### 4.3 Non-uniform bit allocation

Score each weight tensor's sensitivity via a single-pass Hessian-diag
proxy or via activation-weighted reconstruction error. Allocate bits:

- Embeddings, output head, attention projections: protected (≥4-bit).
- MLP `down_proj`, `gate_proj`, `up_proj`: aggressive (can drop to
  2-bit with rotations).
- Attention `o_proj`: middle ground (3-bit).

This mirrors the allocations EXL2 and SqueezeLLM converge on
empirically. The question for us is whether the *pipeline* can make
the decision automatically from calibration data rather than relying
on hand-tuned heuristics.

### 4.4 Quantization (existing stage, augmented)

Feed rotated + calibrated + bit-budgeted weights into the existing
quantization stage. Current options that compose cleanly:

- **llama.cpp i-quants (IQ2_XS, IQ2_S, IQ3_XXS)** — production-ready,
  ship in our GGUF path today.
- **AQLM / QuIP#** — research-grade; would require new GGUF format
  extension. Defer unless i-quants prove insufficient after stages
  4.1-4.3.
- **CALDERA** — already our active vein. Rotation composes with
  CALDERA decomposition; QAT composes on top of CALDERA output.

### 4.5 QAT recovery (new stage)

Short fine-tuning pass at the quantized bit-width to recover quality.

- **EfficientQAT-style** — optimizes quantized weights directly via
  LoRA-shaped updates plus straight-through estimation.
- **Target cost:** <24 GPU-hours for 8B, <1 week for 70B on a single
  consumer card (the stage itself runs at target bit-width, so memory
  pressure is similar to inference).
- **Dataset:** small instruction-following + general text mix, ~100M
  tokens. Longer runs show diminishing returns per the literature.

**Decision point:** QAT is the heaviest stage and the one with the
longest implementation tail. Gate its inclusion on whether stages
4.1-4.4 leave a meaningful quality gap. For T1/T2 they likely do not.
For T3/T4 they likely do.

---

## 5. Composition with CALDERA

CALDERA decomposes `W ≈ Q + L·R` where `Q` is aggressively quantized
(2-bit LDLQ) and `L·R` is a low-rank 4-bit correction. Stack ordering:

```
HF checkpoint
  → [rotate]         (new — QuaRot-style Hadamard)
  → [calibrate]      (existing imatrix, diversified corpus)
  → [score+allocate] (new — per-tensor bit budget)
  → [decompose]      (existing — CALDERA Q + L·R)
  → [quantize]       (existing — i-quants or CALDERA-native)
  → [QAT recover]    (new — optional, gated on quality delta)
  → GGUF
```

Rotation commutes with CALDERA decomposition: rotating before
decomposing gives `R W = R Q + (R L) R` — the rank-k correction
structure is preserved, the quantized residual is easier to represent.

QAT composes on the CALDERA output by fine-tuning `L` and `R` while
holding `Q` frozen. This is cheap (low-rank factors are small) and
sidesteps the harder problem of straight-through gradients into `Q`.

---

## 6. Open questions to resolve in Phase 1

1. **Does Hadamard rotation alone close the Llama-3 gap at 3-bit?**
   If yes, T1/T2 become trivial and the research interest shifts to
   T3/T4 only.
2. **What's the QAT floor?** At what bit-width does QAT stop
   recovering quality? BitNet suggests 1.58 from-scratch; post-training
   QAT floor is almost certainly higher.
3. **Does rotation interact constructively or destructively with
   CALDERA?** The math says constructively. Empirics need to confirm.
4. **Can the bit-allocation decision be made from a single calibration
   pass?** If we need iterative search, pipeline cost multiplies.
5. **How much does calibration corpus diversity actually buy on
   Llama-3?** Need an ablation: Wikitext vs mixed, at matched bit-width.
6. **Does our T3 target (2.5 bpw Llama-3-70B on 24 GB) survive 8K KV
   cache pressure?** KV quantization may become mandatory.

---

## 7. Success criteria for Phase 1

Phase 1 is scoped to T1 (Qwen3-8B, 2.0 bpw, ≤0.5 ppl gap). It exits
when:

- A reproducible pipeline produces the target GGUF from an HF
  checkpoint in one command.
- Benchmark harness in this repo (`bench_model.py` pattern from
  cross-layer-svd) reports ppl on Wiki + mixed held-out set.
- An ablation table quantifies contribution of each stage
  (rotation-only, +calibration, +allocation, +QAT) to final ppl.

T3 (Llama-3-70B) becomes Phase 2 only if Phase 1's ablation shows the
stack is additive as hypothesized. If Phase 1 shows one stage
dominates and the others are noise, we refocus.

---

## 8. References (seed list)

- QuaRot — arxiv:2404.00456
- SpinQuant — arxiv:2405.16406
- AQLM — arxiv:2401.06118
- QuIP# — arxiv:2402.04396
- BitNet b1.58 — arxiv:2402.17764
- EfficientQAT — arxiv:2407.11062
- PV-Tuning — arxiv:2405.14852
- Llama-3 quantization sensitivity — discussion in Llama-3 paper §5
  and follow-up empirical work from /r/LocalLLaMA community throughout
  2024.

Related prior work in this repo:

- [../cross-layer-svd/DESIGN.md](../cross-layer-svd/DESIGN.md) —
  CALDERA design, composes with this stack.
- [../cross-layer-svd/model_prep_pipeline.md](../cross-layer-svd/model_prep_pipeline.md)
  — current pipeline architecture this vein extends.

---

## 9. What this document is not

This is a vision + scoping document. It does not commit to any
implementation schedule, does not choose final algorithms where
alternatives remain plausible, and does not specify the GGUF-level
binary format impact. Those decisions belong in a Phase 1 DESIGN.md
once Phase 0 investigation resolves the open questions in §6.
