# LittleBit QAT — savings exploration plan

> **Part of the LittleBit plan set.** See [README.md](README.md)
> and [consolidated_implementation_roadmap.md](consolidated_implementation_roadmap.md).
> Related: [wall-time](wall_time_reduction_plan.md) ·
> [unexplored gains](unexplored_efficiency_gains.md) ·
> [scale-to-30B](scale_to_30b_architecture.md) ·
> [memory research](memory_efficient_training_research.md).

Systematic ablation plan for exploring **every tractable source of
memory / compute savings** in our QAT pipeline, with explicit
quality-degradation gates so no technique is adopted if it damages
model quality catastrophically.

Downstream of:
- [littlebit_math.md §14-15](littlebit_math.md) — Phase B baseline
  establishment
- [littlebit_enhancements.md](littlebit_enhancements.md) — catalog of
  candidate optimizations
- [memory_efficient_training_research.md](memory_efficient_training_research.md) —
  state-of-the-art techniques survey
- [scale_to_30b_architecture.md](scale_to_30b_architecture.md) —
  local-only architecture for 30B+; why scale-up needs memory
  optimization

Status: **planning**. Written while Phase B training runs. Actual
ablations begin after Phase B completes.

## 1. Purpose

Two pressures:
1. **Scale**: Phase B will finish at some PPL / generation quality
   on 0.5B. Moving to 1.5B / 3B / 7B requires memory we don't yet
   have. We need savings.
2. **Paper alignment**: our recipe deviates from paper on multiple
   dimensions (see [JOURNAL.md](JOURNAL.md) run-3 entry). Closing
   those gaps costs memory too.

**Goal of this plan**: identify every savings technique, test each
in isolation against the Phase B baseline, keep what works, drop
what breaks quality.

**Catastrophic degradation bar**: A technique is rejected if it
worsens the PPL-vs-Phase-B ratio by **more than 20%** or collapses
generation to Run-2-quality word-salad.

## 2. Phase B baseline to preserve

After Phase B (~7 more hours), we will have:

- **Trained checkpoint**: `littlebit_qat_checkpoint_r512_phaseB.pt`
- **Training-eval PPL** at 25k-token wikitext cap (Run 3 was 92.9,
  Phase B likely 60-75)
- **Full-test PPL** at 250k tokens (Run 3 was 133.7)
- **Per-layer mean rel-err** (Run 3 was 0.42, Phase B likely ~0.30)
- **Generation samples** on 5 reference prompts
- **Wall clock** per opt-step (~1.78s after step 1000)

Every ablation run is compared against these four numbers under the
same evaluation protocol.

## 3. Ablation methodology

### 3.1 Single-variable runs

Each ablation changes **one thing** vs Phase B:
- Same seed (`torch.Generator().manual_seed(0)`)
- Same hyperparameters
- Same teacher model
- Same data mix
- Same evaluation harness

### 3.2 Start from init cache, not Phase B checkpoint

Why fresh: avoids distillation inertia. If we resume from Phase B's
trained state and swap (say) the teacher to nf4, gradients
immediately drift toward nf4-teacher's distribution — we can't
distinguish "recovers cleanly" from "inherited Phase B's quality."
Fresh starts from Dual-SVID init give clean A/B.

### 3.3 Short vs full runs

Not every ablation needs full 20k steps. Three tiers:

| Tier | Steps | Wall | When to use |
|---:|---:|---:|---|
| Smoke | 500 | ~15 min | Does it not crash? |
| Short | 4000 | ~2h | Does it hit Run-3-quality? |
| Full | 20000 | ~10h | Does it match Phase B? |

Most ablations get the **Short** tier unless they're passing
cleanly and we want to confirm against Phase B's full trajectory.

### 3.4 Decision criteria

After each ablation, classify as one of:

| Verdict | PPL delta | Gen quality | Action |
|---|---|---|---|
| **Keep** | within ±3% | comparable or better | Adopt, stack |
| **Conditional keep** | +3% to +15% | comparable | Adopt if savings justify it |
| **Drop** | +15% to +20% | noticeably worse | Not worth it |
| **Catastrophic** | >20% or gen broken | Reject, document |

## 4. Techniques, ranked by priority × risk-reward

### 4.1 Run A — `nf4` teacher via bitsandbytes

**Memory savings**: ~0.75 GB at 0.5B, **10.5 GB at 7B** (critical
for scale-up).

**Mechanism**:
```python
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
teacher = AutoModelForCausalLM.from_pretrained(
    model, quantization_config=quant_config,
)
```

**Risk**: Teacher's hidden states (used in MSE loss) are drawn from
nf4-quantized weights. Student learns to match a slightly distorted
target. Distillation literature suggests minimal PPL impact (<0.5
PPL on non-QAT distillation) but no ablation under LittleBit's
exact MSE recipe.

**Test**: Short tier (4000 steps). Compare to Phase B trajectory at
step 4000. Decision: keep if PPL within +15%.

**Implementation effort**: ~5 lines. Existing code infrastructure
handles it (`load_in_4bit` doesn't change forward API).

**Downstream unlock if successful**: 7B local feasibility.

---

### 4.2 Run B — Liger Kernel FusedLinearCrossEntropy

**Memory savings**: ~1.2 GB at 0.5B seq=512 (peak logits region);
~20 GB at 7B.

**Mechanism**: `pip install liger-kernel`, then
`from liger_kernel.transformers import apply_liger_kernel_to_qwen2`
(or `apply_liger_kernel_to_llama` for Llama teacher).  Replaces
Qwen's `lm_head` → cross-entropy pipeline with a chunked Triton
kernel that never materializes the full logits tensor.

**Risk**: Liger's fused cross-entropy expects standard LM loss
(one-hot targets). We use KL against soft targets. Direct swap
won't work. Two options:

- **Option 1**: Adapt the Liger kernel to accept soft targets.
  Non-trivial Triton work.
- **Option 2**: DIY chunked KL (see Run C). Parallel effort, lower
  risk.
- **Option 3**: Apply Liger to *just* the RMSNorm / RoPE / SwiGLU
  replacements, skipping the fused-CE part. Partial gains, no risk.

**Test**: Smoke (500 steps) to confirm it initializes and trains.
If stable, Short (4000 steps) for PPL comparison.

**Implementation effort**: ~1 hour if adopting RMSNorm / RoPE only,
~1 day if trying to adapt fused CE to KL.

**Downstream unlock**: complements nf4 teacher to make 7B local
possibly viable.

---

### 4.3 Run C — Chunked KL loss (DIY)

**Memory savings**: ~2-3 GB at seq=512 fp32 logits. Scales with
seq_len and vocab_size (both of which we want to increase).

**Mechanism**: Two-pass log-sum-exp chunked KL. Never materializes
full softmax over 152k-vocab:

```python
def chunked_kl_div(s_logits, t_logits, vocab_chunk=8192):
    # Pass 1: compute log-sum-exp normalizers over vocab chunks
    # Pass 2: compute per-chunk KL using pre-computed normalizers
    # Peak memory: chunk-sized, not vocab-sized
```

**Risk**: **None, mathematically**. Log-sum-exp is exact, just
different memory profile. Potential numerical stability if we get
the math wrong; cover with unit test against naive full-vocab KL
before running training.

**Test**: Smoke + Short. Should match Phase B exactly (or match
first 500 steps of Phase B bit-for-bit).

**Implementation effort**: ~50 lines + unit test. 1-2 hours.

**Downstream unlock**: seq=1024 batch=1 without OOM. Or batch=2 at
seq=512. Important for batch exploration.

---

### 4.4 Run D — Hidden-state capture via forward hooks

**Memory savings**: 1-2 GB (defeats `output_hidden_states=True`
list retention, lets gradient checkpointing actually fully drop
activations).

**Mechanism**: Register forward hooks on decoder layers, capture
hidden state per-layer into a list we manage, then free
immediately after MSE consumption. Avoids HF's internal
`output_hidden_states` path which retains lists during backward.

**Risk**: Low. Gradient flow through hooks is well-understood; just
need to ensure hooks capture the pre-checkpointing tensor
appropriately.

**Test**: Smoke + Short. Compare per-layer rel-err at step 4000.

**Implementation effort**: ~30 lines.

**Downstream unlock**: makes gradient checkpointing more effective,
frees ~1.5-2 GB activation budget.

---

### 4.5 Run E — `tau` warmup schedule (10 → 100)

**Memory savings**: **none.** (Listed because it's in enhancement
doc.)

**Quality upside**: potentially significant. SmoothSign at tau=100
has a gradient window of only `|x| < 0.0002` — 1-5% of U_fp, V_fp
entries receive non-negligible gradient per step. Warmup from
tau=10 would widen the window to `|x| < 0.002` early, letting more
signs flip cheaply, then narrowing for final refinement.

**Risk**: tau=10 might flip too many signs early, creating
instability. Watch first 200 steps for loss spike.

**Test**: Short (4000 steps), compare convergence rate vs Phase B.

**Implementation effort**: ~15 lines.

---

### 4.6 Run F — Temperature in KD (T=4)

**Memory savings**: none.

**Quality upside**: softer teacher targets. Standard KD trick,
expected 2-5% PPL improvement.

**Test**: Short. If beneficial, adopt as new baseline before stacking
more memory techniques.

**Implementation effort**: ~5 lines.

---

### 4.7 Run G — Reduced rank (r=256)

**Memory savings**: 0.75 GB at 0.5B (U_fp, V_fp halved). Scales to
~5 GB at 7B.

**Quality impact**: this is a **compression-level change**, not an
optimization. Reduces trainable params by ~40%, and brings us to
~0.35 BPW (vs r=512's ~0.7). Closer to paper's 0.1 BPW headline.

**Risk**: The format might not have enough capacity at r=256 under
KL + MSE. Run 2/3 only tested r=512.

**Test**: Short. If PPL trajectory stays within +15% of Phase B, r=256
is the clearly better operating point (cheaper, closer to paper BPW).

**Implementation effort**: trivial — just change `--rank` flag.

---

### 4.8 Run H — `torch.compile` on student

**Speed upside**: 30-50% per-step improvement (may or may not apply
given our custom autograd function).

**Memory upside**: modest; compile can sometimes reduce memory via
op fusion, sometimes increase via recomputation.

**Risk**: `SmoothSignEfficient` custom autograd might trigger graph
breaks. Test carefully.

**Test**: Smoke — does it compile? Does loss at step 500 match
Phase B's loss at step 500?

**Implementation effort**: ~5 lines + fallback path.

---

### 4.9 Run I — Top-k logit caching for teacher

**Memory savings**: **100% of teacher VRAM during training** (teacher
only runs once, offline).

**Mechanism**: Pre-compute teacher's top-32 logit values + indices
for every training token. Store to disk (~500 MB for our corpus).
At training time, load from disk instead of running teacher forward.

**Risk**: Top-k truncation loses tail mass. Distillation literature
suggests k=32 preserves >99% of KL signal; k=8 preserves >95%.
Haven't validated for our specific recipe.

**Test**: Offline — generate cache, load, compare to live teacher
on a handful of batches. If KL values match within 5%, proceed.

**Implementation effort**: ~3 hours. New preprocessing step, loader
changes. One-time cost per dataset/teacher pair.

**Downstream unlock**: teacher-free training. Enables 30B+ local
without teacher VRAM overhead. Pairs beautifully with nf4 or
even no teacher at all.

**Catch**: we still need teacher hidden states for MSE. Caching
24 layers × 4.4M tokens × 896 dim × 2B = 190 GB. Not feasible on
disk. Could cache hidden states for a **subset** of layers (e.g.
every 4th), reducing to ~48 GB. Still a lot.

Alternative: cache ONLY logits for KL, drop MSE entirely. Paper
uses MSE so dropping is a recipe change. But if logit-caching
enables 30B training we might take the tradeoff.

---

### 4.10 Run J — COAT FP8 training (speculative)

**Memory savings**: ~50% vs bf16 (activations + optimizer state).

**Mechanism**: Native FP8 training via
[COAT's Dynamic Range Expansion](https://arxiv.org/html/2410.19313v1).
Requires Hopper (H100+) or Blackwell (RTX 5080 Laptop / 5090)
hardware.

**Risk**: High. SmoothSign's tanh(100·x) in FP8 has limited dynamic
range; unclear if surrogate gradient remains meaningful. FP8
multiplies in backward could have precision issues near x=0 where
the surrogate is most significant.

**Test**: Very careful. Smoke first (500 steps). Watch for NaN,
grad explosion, loss not descending. If clean, Short.

**Implementation effort**: Days. Requires either Transformer Engine
or manual FP8 autocast. Not trivial.

**Verdict**: research-grade. Only pursue if Runs A-G establish
enough savings without it.

---

### 4.11 Run K — Gradient accumulation scaling

**Already in Phase B**: `grad_accum_steps=4`. Could scale further:

| accum | effective batch | wall per opt-step | total tokens |
|---:|---:|---:|---:|
| 1 | 1 | 0.45s | Run 3's 4M |
| 4 | 4 | 1.78s | Phase B's 16M-41M |
| 8 | 8 | 3.6s | 82M (too slow for 20k) |

Not worth scaling further; we're near paper's 4× effective batch
already.

---

### 4.12 Run L — Larger batch via sequence shortening

**Idea**: If seq=512 batch=1 fits, and seq=256 batch=2 uses the same
memory, we get 2× effective batch without grad_accum latency.

**Memory math**: activations scale with seq², logits scale with
seq. At seq=256 batch=2 vs seq=512 batch=1:
- Activations: `2 × 256²` vs `1 × 512²` → 0.5 × 
- Logits: `2 × 256 × 152k` vs `1 × 512 × 152k` → 1.0 ×
- Attention: `2 × 256²` vs `1 × 512²` → 0.5 ×

**Quality impact**: shorter sequences mean weaker long-range
gradient signal. But paper's seq=2048 is already long; we're all
below that.

**Test**: Smoke + Short. Compare to Phase B.

**Implementation effort**: trivial. Just change flags.

---

## 5. Recommended ablation order

Based on **safety × savings × unlocks**:

1. **Run C (Chunked KL, DIY)** — zero quality risk, unlocks
   seq=1024. Highest confidence adoption.
2. **Run A (nf4 teacher)** — high savings at 7B, moderate risk at
   0.5B. Single biggest enabler for local scale-up.
3. **Run D (forward hooks for hidden states)** — low risk, moderate
   savings.
4. **Run G (rank=256)** — closer to paper's operating point, cheaper.
   If it survives quality check, new default.
5. **Run F (KD temperature T=4)** — free quality win, small code.
6. **Run E (tau warmup)** — potentially significant quality win.
7. **Run B (Liger Kernel RMSNorm/RoPE/SwiGLU)** — modest speed +
   memory. Only after everything else works.
8. **Run H (torch.compile)** — speed only, late-stage optimization.
9. **Run I (Top-k logit caching)** — big engineering investment,
   only justified if we commit to 30B+ local.
10. **Run J (COAT FP8)** — research-grade, last resort.

## 6. Decision tree after Phase B

```
Phase B result:
├── Generation coherent + PPL < 90 (clear win)
│   ├── Run A (nf4 teacher) next
│   ├── If A survives: Run C + D + G in parallel (all safe)
│   └── Aim for 7B local run once Sprint 3 (teacher cache) lands;
│       0.5B stack transfers directly
├── Generation word-level-broken + PPL 60-90 (partial win, like Run 3)
│   ├── Priority on quality ablations: E (tau warmup), F (KD temp)
│   ├── Then memory: A, C, D
│   └── Reproduction story less clear; 7B attempt deferred until
│       quality ablations lift generation off the partial-win plateau
└── Generation still-digit-spam OR PPL > 150 (fail)
    ├── Debug: did MSE weight fire correctly? Hidden state capture?
    ├── Try increasing MSE weight (λ=20) or adding temperature
    └── If no recovery, paper's method may not extend to 0.5B at
        our compute budget; document as honest negative result
```

## 7. Stacked savings projection

If Runs A, C, D, G all succeed, stacked memory profile on 7B:

| Component | Current Phase B (0.5B) | Stacked 7B projection |
|---|---:|---:|
| Student (fp32 params + grads) | 3.2 GB | 9.6 GB (bf16) or 19 GB (fp32) |
| 8-bit Adam | 0.4 GB | 2 GB |
| Teacher | 1 GB bf16 | **3.5 GB nf4** |
| Logits / KL region | 2.5 GB peak | **0.5 GB** (chunked) |
| Hidden states (MSE) | ~0.2 GB | **0.3 GB** (hook-freed) |
| Activations (grad ckpt) | ~1 GB | ~3 GB |
| **Total** | **~8 GB** (well under 13.7 GB cap) | **~19-24 GB** |

At 24 GB total on 7B, we're close to consumer 24 GB GPU envelope
(RTX 4090 / 5080 Desktop). Laptop 16 GB still needs **either** the
extra push from FP8 (-5 GB) **or** CPU offload for one component.

## 8. Open questions this plan doesn't resolve

- **Generation coherence metric**: currently visual inspection.
  Should we add BLEU/ROUGE/custom against a short prompt corpus
  for quantitative comparison? Time-boxed at maybe 2 hours to add.
- **Zero-shot benchmark**: paper reports PIQA / HellaSwag / ARC.
  None of our runs test these. One ablation slot should include
  zero-shot measurement on the winning checkpoint to confirm
  downstream usability.
- **Longer training**: even at full paper recipe on 0.5B, we've
  only done 1-2 epochs. 5 epochs would take ~2 days locally.
  Worth it only if earlier ablations hit a clear plateau.
- **Data quality**: C4 subset is sample-N-from-start. Paper likely
  uses more curated selection. Could we improve by filtering?

## 9. Budget for all ablations

Wall clock if running serially:
- Run C smoke+short: 2.5h
- Run A smoke+short: 2.5h
- Run D short: 2h
- Run G short: 2h
- Run F short: 2h
- Run E short: 2h
- Run B smoke: 15min (if we adopt limited scope)
- Run H smoke: 15min
- Run I offline setup: 3h
- Run J: not planned in this budget

**Total**: ~17 hours of ablation time, spread over 2-3 days.

If several run in parallel via different checkpoints on same GPU:
probably compressible to 10 hours.

## 10. Exit criteria

This plan's done when:
- **All Runs A-G have Verdicts assigned** (Keep / Conditional /
  Drop / Catastrophic)
- **Stacked savings estimate** for 7B has been validated by
  running the adopted stack on 0.5B
- **Honest recommendation** written in JOURNAL: either
  (a) "7B local training is feasible with our stacked ablations —
  go for it once Sprint 3 teacher cache lands" or
  (b) "7B local is not yet feasible on current hardware even with
  stacked savings — defer 7B until NVMe tier
  ([scale_to_30b_architecture.md §11](scale_to_30b_architecture.md))
  lands, or until hardware (RAM / GPU) is upgraded"

Both are acceptable outcomes. The point is to reach a defensible
decision backed by data. Cloud compute is not an option — no
scenario in this plan routes through cloud GPU rental.
