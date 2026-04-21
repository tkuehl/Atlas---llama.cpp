# Unexplored efficiency gains — consolidation review

Systematic review of all accumulated LittleBit research for gaps,
missed combinations, and under-optimized pipeline steps. The theme
is **combining techniques that individually worked** — either in our
own prior research or in literature.

Reviewed:
- [littlebit_math.md](littlebit_math.md) — method + sanity + QAT
  results
- [littlebit_enhancements.md](littlebit_enhancements.md) — known
  optimization catalog
- [memory_efficient_training_research.md](memory_efficient_training_research.md) —
  external survey
- [savings_exploration_plan.md](savings_exploration_plan.md) —
  current ablation plan
- [scale_to_30b_architecture.md](scale_to_30b_architecture.md) —
  NVMe + cached-teacher architecture
- [JOURNAL.md](JOURNAL.md) 2026-04 LittleBit entries
- Archived CALDERA / SVD work

## 1. High-leverage combinations we haven't explored

These pair techniques that each already produced positive results.

### 1.1 CALDERA initialization + LittleBit QAT

**Status in our research**: CALDERA Stage 1 (2026-04-19 JOURNAL)
achieved ~20-55% Σ-rel-err reduction in the rank-r subspace vs pure
quant. Dual-SVID leaves ~61% of rank-r subspace discarded (§12
structural constants).

**The combination**: instead of Dual-SVID initializing `W_pri`,
start from a **CALDERA decomposition** `W ≈ Q + L·R` — quantize
`Q`, binarize `L` and `R` for LittleBit scales, use as initial
state.

**Why it might work**: CALDERA already solves "find the best
rank-r + residual" for a given W. Its rank-r subspace is weighted
by activation importance (via gramian), not just Frobenius.
Starting LittleBit's QAT from CALDERA's subspace gives the student
a better subspace to sign-quantize.

**Expected gain**: higher initial captured energy → less work for
QAT → faster convergence or higher ceiling.

**Cost**: 1 day to wire CALDERA output into LittleBit init path.
CALDERA script already exists in the fork.

**Validation**: single matrix first. Compare
`||W - W_pri(CALDERA_init)||_F^2` vs `||W - W_pri(Dual-SVID)||_F^2`
on our standard test matrix. If CALDERA init gives >20% lower init
Frobenius error, proceed to full-model QAT.

### 1.2 Speculative decoding with LittleBit student as draft

**Status**: our fork has a complete speculative decoding research
track ([speculative_decoding.md](speculative_decoding.md)). Track B
showed 1.56× decode speedup on Qwen3-8B when paired with a
published tiny draft.

**The combination**: once we have a LittleBit-compressed 7B student,
use it as the **draft model** for an unmodified full-precision 7B
teacher. Draft proposes, teacher verifies.

**Why**:
- Student's entire training objective was matching teacher's
  outputs — so draft acceptance rate should be **very high**
- Student is much smaller in memory — teacher can be bf16/fp16,
  student in sub-1-BPW
- Combined deployment is "fast + accurate" — student for speed,
  teacher for correctness

**Expected acceptance rate**: based on our results so far, student
matches teacher probabilities well at PPL-level. If PPL ratio is
~2× (paper-grade result), acceptance rate probably 60-80% — enough
for meaningful speedup.

**Cost**: zero training cost. Just needs:
1. LittleBit checkpoint (already producing)
2. GGUF-loadable binary format for upstream `llama.cpp` (this is
   the gap)

**Validation**: benchmark via the existing
`research/cross-layer-svd/bench/spec_bench.py` infrastructure.

**Why we haven't discussed this**: these two research tracks have
been treated as parallel. They're not — the compression enables the
speculation, and the speculation justifies the compression cost.

### 1.3 Progressive two-stage training (KL first, then KL+MSE)

**Status**: Run 2 (KL only) achieved PPL 54.8 but broken generation.
Run 3 (KL+MSE from step 1) achieved better hidden-state alignment
but slower convergence and worse PPL.

**The combination**: stage the objectives:
- **Stage 1** (first N steps): KL only. Student's logits align
  quickly. Fast convergence — Run 2 showed clean descent.
- **Stage 2** (rest of training): Add MSE term. Student's hidden
  states now align on top of already-good logits.

**Why**: our Run 3 trajectory shows MSE and KL **fight each other
early** — student's sign patterns optimized for MSE-matching hidden
states don't map to teacher-matching logits until both settle.
Stage 1 lets KL establish the logit-matching baseline before MSE
perturbs it.

**Literature precedent**: several distillation papers
(TinyBERT, DistilGPT2) use stage-wise objectives. None in the
LittleBit line that we've seen.

**Cost**: 1 flag in the training loop (schedule for
`inter_mse_weight`). Free ablation.

**Validation**: compare Phase B (concurrent MSE from step 1) vs
staged (KL-only for first 1000, then KL+λMSE).

### 1.4 Incremental rank training

**Status**: not tested in our pipeline. Related concept in
literature (IncreLoRA: "Incremental LoRA ranks").

**The combination**: train at low rank first, expand:
- Step 1-2000: train at r=128 (4× cheaper per step)
- Step 2000-5000: expand to r=256 (keeping r=128 factors, adding
  fresh rank-128 residual)
- Step 5000-20000: expand to r=512

**Why**: early training at r=128 captures dominant singular
directions fast. Later ranks only need to refine residual.
Potentially faster convergence AND better final quality (avoids
spreading optimizer effort across full r=512 subspace when early
signal is rank-dominant).

**Cost**: parameter-expansion code path — nontrivial but ~200
lines. Keep optimizer state across rank expansions.

**Validation**: r=128 fully-trained vs r=128→256 progressive vs
r=512 direct. Compare at matched compute budget.

### 1.5 Per-layer non-uniform rank

**Status**: we currently use uniform r everywhere. Paper does
uniform r too.

**Observation from §12**: structural constants (rank-1 separability
~0.63, discard fraction ~0.61) are shape-invariant. But **actual
singular value distributions differ** between attention projections
and MLP projections.

**The combination**: allocate rank budget based on per-layer
singular value decay:
- Attention projections (small matrices, fast-decaying singular
  values): `r=64-128` sufficient
- MLP gate_proj / up_proj / down_proj (large matrices, slower
  decay): `r=512`

**Why**: same total BPW, but spent where it matters.

**Cost**: per-layer rank config. ~20 lines in wrap function.

**Validation**: single matrix per layer-type, measure FP-SVD err at
each rank, pick operating point that matches overall PPL target.

### 1.6 Cached-teacher + speculative decoding + LittleBit student

**The combination**: three-way synthesis:
1. Teacher cache (no teacher during training)
2. LittleBit QAT on student
3. Trained student becomes spec-decoding draft at inference

**Why**: each research track amortizes across all three:
- Teacher cache: one-time cost, used for QAT + spec research
- LittleBit compression: trained once, used for compression
  benchmark + speculation draft
- Speculation infrastructure: benchmarks all student checkpoints
  as they're produced

**Effect on Atlas deployment**: fast-fp16-teacher + tiny-
littlebit-student is a stronger deployment than either alone.

### 1.7 Relation-based distillation on hidden states

**Status**: Run 3 uses per-position MSE on hidden states. Works
(80% energy captured) but per-position matching is brittle to
coordinate-system drift.

**The alternative**: match **relations** between pairs of positions,
not absolute hidden values:

```python
def relation_mse(s_hidden, t_hidden):
    # Cosine-similarity matrix of positions
    s_rel = F.cosine_similarity(s_hidden.unsqueeze(1),
                                 s_hidden.unsqueeze(2), dim=-1)
    t_rel = F.cosine_similarity(t_hidden.unsqueeze(1),
                                 t_hidden.unsqueeze(2), dim=-1)
    return F.mse_loss(s_rel, t_rel)
```

**Why**: if student's hidden space is rotated vs teacher's but
internal structure is preserved, relation-based MSE forgives the
rotation. Our §15.2 finding (per-layer rel-err 1.05 but generation
partially coherent under KL) hints that student's hidden space is
indeed rotated — relation-MSE might be a better target than
absolute MSE.

**Literature**: PKD (Patient Knowledge Distillation),
Relation-KD. Well-established.

**Cost**: one function + loss-term coefficient. Very cheap.

## 2. Under-optimized current pipeline steps

Reviewing the actual code path, things we could do better without
any new research:

### 2.1 Post-training scale LSQ fit

**Observation**: after QAT converges, signs are fixed. We COULD
re-solve h, g, ell analytically given the fixed signs to get
Frobenius-optimal scales. Our Run 2 scales-only ablation showed
Dual-SVID init was near-optimal for random signs, but post-QAT
signs may admit a better scale fit.

**Cost**: 1 second per matrix at inference, free at deploy time.

**Expected gain**: ~1-3% PPL improvement from "free" post-hoc
refinement.

### 2.2 Gradient accumulation at higher effective batch

Currently Phase B uses `grad_accum=4`. Paper uses `batch=4`
directly. Could try:
- `grad_accum=8` → effective batch=8 (beyond paper)
- Is there evidence bigger batch helps QAT specifically?

**Literature**: Unclear. Large-batch training usually needs scaled
LR. Worth one ablation.

### 2.3 Loss weight scheduling for MSE

Paper uses `λ=10` constant. Could decay:
- Steps 1-5000: `λ=10` (force hidden alignment early)
- Steps 5000-20000: `λ=5` (relax as model matures)
- Final steps: `λ=2` (let logit-KL dominate last-mile refinement)

**Why**: our Phase B trajectory shows MSE-loss component stays
high throughout. Decay might let the final polish focus on logits.

**Cost**: 10 lines. Adds one more hyperparameter to tune.

### 2.4 Multi-resolution MSE

Raw-hidden MSE is pixel-level matching. Could add:
- MSE on average-pooled hidden states (spatial smoothing over seq)
- MSE on `hidden @ random_projection` (random-projection robustness)

**Effect**: more tolerant of small positional perturbations.
Regularization effect.

### 2.5 bitsandbytes 4-bit Adam (not just 8-bit)

We use `AdamW8bit`. bitsandbytes also has `AdamW4bit` and
`AdamW_paged_4bit`. Cuts optimizer state by another 2×.

**Risk**: 4-bit Adam has more compression-noise than 8-bit;
stability unclear for QAT.

**Cost**: one-line swap. Free to test.

### 2.6 Batch-level token prefetching

Current loop: `batch = next(it).to(device)` is synchronous.
Each batch waits for tokenizer + `.to(device)`.

**Optimization**: run a background thread that pre-tokenizes next
N batches and pre-moves to device while current batch computes.
PyTorch's `DataLoader(num_workers=2, pin_memory=True)` handles
this.

**Cost**: minor refactor of training loop. ~15 lines.

**Gain**: ~5-15% per-step wall clock reduction (depending on I/O
overlap).

### 2.7 Fused RMSNorm / RoPE / SwiGLU from Liger Kernel

Liger Kernel offers these beyond FusedLinearCrossEntropy. None of
them change the math. Just faster Triton implementations.

**Cost**: `pip install liger-kernel`, then
`apply_liger_kernel_to_qwen2(student)` after wrapping.

**Risk**: our custom LittleBitLinearHF replaces nn.Linear. Liger's
patches target specific HF modules. If Liger's patches touch
`self.mlp.gate_proj` (now a LittleBitLinearHF), they might break.
Need to apply Liger BEFORE our wrap, or ensure Liger only targets
norm/RoPE/SwiGLU (not our linears).

**Gain**: ~10-20% per-step speed, ~1-2 GB memory.

## 3. Missed optimizations in evaluation pipeline

Not just training — our eval is also suboptimal:

### 3.1 Full-test PPL computed every eval checkpoint

During training we run 25k-token PPL every 500 steps. 17 evals ×
5-7 seconds = ~1.5 min of the run's 8h wall clock. Negligible.

But for ablation runs we re-evaluate full teacher from scratch
every time. Teacher PPL only needs to be computed once per
(model, dataset) pair — it doesn't change.

**Fix**: cache teacher PPL. Skip teacher eval on any rerun with
same model+tokenizer.

**Gain**: ~20-30 seconds per ablation run. Adds up.

### 3.2 Per-layer activation drift on subset of layers

Current: hook every layer, dump hidden states, compute rel-err.
At seq=512 × 4 batches × 24 layers × 896 dim × 4 bytes = 175 MB of
activation storage just for the drift measurement.

**Fix**: sample every 4th layer for drift. 90% of info, 25% of
memory and compute.

### 3.3 Generation samples: increase diversity

Currently 5 fixed prompts. For ablation, 5 isn't enough to
distinguish "slightly better" from "slightly worse" generation.

**Fix**: 25-prompt standard suite (could reuse our public prompt
fixture from the speculative-decoding research). More stable
signal.

## 4. Missing entirely from our plans

Items I haven't documented anywhere else:

### 4.1 Curriculum learning for QAT

Start training on easy samples (shorter context, common tokens),
gradually increase difficulty. Well-established for standard LM
pretraining. Haven't seen it explicitly for QAT.

Our random-window sampling gives uniform difficulty. A
length-sorted curriculum (seq=128 first, then 256, then 512) might
speed early convergence.

**Cost**: data loader modification. Moderate.

### 4.2 Lion optimizer for sign-heavy parameters

Lion uses only `sign(momentum)` for updates. For our `U_fp, V_fp`
which themselves live under a sign() operation, Lion might be a
natural fit — both the parameter and optimizer are sign-oriented.

**Literature**: Lion paper claims ~1-2x speedup on LM training vs
AdamW. Hasn't been tested specifically for SmoothSign-parametrized
models (because those are rare).

**Cost**: one-line optimizer swap.

**Risk**: Lion's sign-update might mess up the scale vectors (h, g,
ell) which need precise updates. Mixed optimizer (Lion for U_fp,
V_fp; AdamW for scales) is the sensible config.

### 4.3 Sharpening the teacher for low-confidence positions

Teacher's softmax at some positions is flat (legitimately
uncertain). KL drives student to match this uncertainty.

**Alternative**: sharpen teacher's softmax at positions where
teacher is already decisive (high max-prob), leave alone where
teacher is uncertain. `T_effective = 1 + (1 - max_prob)`.

**Why**: student benefits from confident signal where available;
doesn't need to match teacher's doubt where teacher is genuinely
unsure.

**Cost**: small modification to KL computation.

### 4.4 Gradient centralization

Subtract mean from gradient before applying optimizer. Well-known
to help some QAT regimes (quantization-specific training dynamics).

**Cost**: 5 lines.

**Expected gain**: 0-3% PPL. Small but free.

### 4.5 Soft sign with learned tau

`tau` in SmoothSign is fixed at 100. What if it were learnable
per-layer or per-tensor? Let the model choose its own surrogate
sharpness.

**Risk**: adds more parameters, might not help and could destabilize.

**Literature**: some papers (Learnable Sign, etc.) do this for
binary networks.

## 5. Combination-of-combinations — the "kitchen sink" config

If we stacked ALL the high-value combinations that survive their
individual validations, the final recipe looks like:

1. **Teacher**: nf4-quantized, cached offline (top-256 logits + 3
   key hidden-state layers) — frees GPU VRAM entirely
2. **Student init**: CALDERA-based instead of Dual-SVID — better
   starting point
3. **Training stage 1**: KL only, r=128, 3000 steps — fast early
   convergence
4. **Rank expansion**: grow to r=256 at step 3000, r=512 at step
   8000
5. **Training stage 2**: KL + MSE (λ scheduled 10 → 2), 17000 steps
6. **MSE type**: relation-based on hidden states, not absolute
7. **Optimizer**: Lion on U_fp/V_fp + AdamW8bit on scales
8. **Data**: wikitext + C4 curriculum (short seq → long seq)
9. **Memory**: chunked KL + fused RMSNorm/RoPE/SwiGLU via Liger
10. **Per-layer rank**: r=128 for attention, r=512 for MLP
11. **Post-training**: LSQ refine scales

Speculatively: PPL ratio 2.0-2.5× over FP16 teacher, vs Phase B's
likely 5-6× and Run 3's 7.7×. Generation coherent.

**This is too many simultaneous changes to validate empirically
without isolating each.** But it's the direction.

## 6. What's on the cutting room floor (skipped intentionally)

- **BitNet-style from-scratch training**: different regime, not
  LittleBit
- **NAS over LittleBit hyperparameters**: too expensive
- **Multi-GPU**: we only have one
- **Reinforcement learning-based distillation**: exotic, unproven
  for QAT

## 7. Revised ablation priority (now post-review)

Updating [savings_exploration_plan.md](savings_exploration_plan.md)'s
ranking with the new ideas:

| New Rank | Technique | Source | Priority reason |
|---:|---|---|---|
| 1 | Chunked KL (DIY) | original plan | Zero risk, unlocks seq |
| 2 | nf4 teacher | original plan | 10 GB savings at 7B |
| 3 | **Staged KL-then-MSE training** | new §1.3 | Free ablation of a paper deviation |
| 4 | **Relation-based MSE** | new §1.7 | Addresses §15.2 rotation issue |
| 5 | **CALDERA init** | new §1.1 | Better starting point |
| 6 | Rank r=256 | original plan | Closer to paper BPW |
| 7 | Forward hooks for hidden states | original plan | 2 GB savings |
| 8 | **Per-layer non-uniform rank** | new §1.5 | Budget efficiency |
| 9 | KD temperature T=4 | original plan | Small quality win |
| 10 | tau warmup 10→100 | original plan | Gradient coverage |
| 11 | **Incremental rank training** | new §1.4 | Convergence efficiency |
| 12 | Liger Kernel partial | original plan | Speed + memory |
| 13 | **Pre-loaded DataLoader** | new §2.6 | Easy speed win |
| 14 | torch.compile | original plan | Speed only |
| 15 | Top-k logit caching | original plan | 30B+ unlock |
| 16 | **Lion optimizer on U_fp/V_fp** | new §4.2 | Might match sign-oriented params |
| 17 | **MSE λ decay schedule** | new §2.3 | Refine late training |
| 18 | COAT FP8 training | original plan | Research-grade |
| 19 | **Spec-decoding with LittleBit student** | new §1.2 | Deployment combination |

Items 3, 4, 5, 8, 11, 13, 17 are net-new additions from this
review.

## 8. Recommended new first-wave ablations

If Phase B succeeds and we want to run **three** high-leverage new
ablations next (not in the original plan):

1. **Staged KL-then-MSE** (§1.3) — cheapest, most likely to improve
   our result at zero cost
2. **Relation-based MSE** (§1.7) — directly addresses the "student's
   hidden space is rotated" diagnosis from §15.2
3. **CALDERA init + LittleBit QAT** (§1.1) — reuses archived work,
   possibly big win on initial-point quality

All three are single-variable changes, Short (4000 steps) tier
budget. Each produces a clean datapoint for inclusion in the final
writeup.

## 9. Meta-observation

Reviewing everything together, our research has been **sequential**
(one technique at a time, compare to baseline). Literature we've
surveyed tends to stack techniques. The three recommended ablations
above test whether simple stackings beat paper's unit approach.

If ANY of them meaningfully improves over Run 3 / Phase B, we've
shown paper's recipe isn't fully optimal and have a genuine
contribution beyond reproduction.
