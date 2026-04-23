# Stage 4 execution plan — BRECQ-style Fisher-weighted block-local MSE

> **Parent plan:** [one_shot_ptq_plan.md](one_shot_ptq_plan.md) §3 Stage 4.
> **Foundations:** [littlebit_math.md §13.2, §15.2](littlebit_math.md)
> (activation-weighted capacity + composition collapse).
> **External precedent:** BRECQ ([arXiv 2102.05426](https://arxiv.org/abs/2102.05426)),
> AQLM Phase 3 ([arXiv 2401.06118](https://arxiv.org/abs/2401.06118)).
>
> **Status:** Planning → S4.0 implementation. 2026-04-22.

This is the concrete execution plan for the single highest-leverage
stage of the one-shot sprint. Stage 4 alone is expected to reduce
the 7B QAT budget by 3–5× by addressing the §15.2 composition failure
directly.

## 1. Why Stage 4 first (theory claim)

The squared-loss gap between quantized and FP models, Taylor-expanded
around the FP solution, is approximately

`L(ŵ) − L(w) ≈ ½ Δzᵀ H Δz`

where `Δz = ẑ − z` is an intermediate activation perturbation and `H`
is the Hessian of the final loss w.r.t. that intermediate. BRECQ §3.2's
second-order analysis establishes that `H` is approximately
**block-diagonal in the transformer-block sense** — off-diagonal
coupling is concentrated *inside* blocks (q→k→v→softmax, gate×up
interaction), not across block boundaries.

Consequences:

- **Per-matrix calibration cannot capture intra-block coupling.** This
  is why [§13.2](littlebit_math.md)'s 90% per-matrix capture composed
  down to 3.6% globally — the per-matrix objective didn't see the
  q/k/v coupling through softmax.
- **End-to-end KL training sees all of `H` but only indirectly.** Many
  hidden-state configurations produce similar output logits; the
  student settled into an orthogonal basin that matched logits at
  teacher-conditioned prefixes but failed under exposure to its own
  predictions.
- **Block-level Fisher-weighted MSE is the smallest objective that
  captures `H` correctly.** Each block sees its own intra-block
  coupling; Fisher weighting ensures capacity is spent on gradient-
  important directions.

## 2. Locked design decisions (2026-04-22)

### 2.1 Sign handling: **RELAXED**

- `U_fp, V_fp` remain learnable via SmoothSign at reduced LR during
  block calibration. Both signs and scales `(h, g, ℓ)` receive gradients.
- **Rationale:** [§13.1](littlebit_math.md) showed scales-only QAT
  recovers essentially nothing (+0.001 rel-err) — signs are load-bearing
  in LittleBit. AQLM Phase 3 uses the same "discrete parameters
  learnable after init" pattern successfully.
- **Ablation follow-up:** once S4.0 validates, run BRECQ-strict
  (signs frozen, scales only) for the same budget to quantify how
  much signs contributed. Informs whether Stage 3 matching pursuit
  obviates the need for gradient sign search.

### 2.2 Block input: **PURE TEACHER**

- Each block is calibrated with `X_b^teacher` loaded from cache;
  never with the student's own predecessor output.
- **Rationale:** BRECQ's Hessian block-diagonal argument says this
  is sufficient if intra-block coupling dominates. All published
  sub-1-bit PTQ methods (BRECQ, AQLM, BTC-LLM) use pure teacher.
  Enables per-block parallelism and eliminates the error-accumulation
  chain that partially caused §15.2.
- **Ablation follow-up:** if final-model PPL shows per-block rel-err
  is low but global PPL gaps, propagated inputs (Stage 4.5) become
  the fix. Don't pre-commit.

### 2.3 Parameter scope within a block: **JOINT (A)**

- All 7 linears of a Qwen block (`q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj`) trained simultaneously against
  a single block-output loss.
- **Rationale:** Intra-block coupling (`softmax(QKᵀ)V`, gate×up) is
  exactly what we're trying to capture. Breaking the block into
  sub-pieces defeats the theoretical basis. Matches BRECQ, AQLM.
- **Scope exclusions:** RMSNorms, embeddings, LM head frozen at
  FP16 (standard LittleBit convention; paper confirms).

## 3. Formal objective

For block `b` with parameters `θ_b = (U_fp, V_fp, h, g, ℓ)_{l ∈ b}`:

```
L_b(θ_b) = (1/N) Σ_n || f_b^{1/2} ⊙ (Z_b^student(X_b^teacher_n; θ_b) − Z_b^teacher_n) ||²
```

where:
- `X_b^teacher_n`, `Z_b^teacher_n`: teacher's FP32 hidden states entering/leaving block `b` on calibration sample `n`.
- `Z_b^student(x; θ_b)`: LittleBit-quantized block `b` applied to input `x`.
- `f_b ∈ ℝ^d`: Fisher diagonal at block `b`'s output, `f_{b,i} = 𝔼[(∂L_ce / ∂z_{b,i})²]`, collected from FP teacher.
- `⊙`: elementwise product; `L_ce`: cross-entropy to ground-truth next-token.

Training: for each `b = 1..B`, initialize `θ_b` from Dual-SVID
(Stage 2 version if available), run AdamW on `L_b(θ_b)` for N steps.

## 4. Implementation architecture

### 4.1 File plan

| File | Status | Purpose |
|---|---|---|
| `littlebit_fisher.py` | **New** | Collect Fisher diagonals per block from teacher |
| Teacher cache extension | **New (small)** | Store `{X_b^teacher, Z_b^teacher}` for all b |
| `littlebit_qat_brecq.py` | **New (fork of qat_model.py)** | Per-block training loop with Fisher-weighted MSE |
| `littlebit_qat_model.py` | Reused | `LittleBitLinear`, `SmoothSign` classes |
| Sprint-3 teacher cache | Reused | `mmap` reader + storage format |
| `littlebit_eval.py` | Reused | End-to-end eval after per-block training |

### 4.2 Data flow

```
[FP16 teacher model]
    ↓ (one-time, ~30 min on 7B)
[calibration forward + backward]
    ├──→ {X_b^teacher, Z_b^teacher} for b = 1..B    (cache extension)
    └──→ f_b ∈ ℝ^d  for b = 1..B                     (fisher cache)

[Dual-SVID init from teacher weights]
    ↓
[per-block training loop]
    for b = 1..B:
        load X_b^teacher, Z_b^teacher, f_b from cache
        init block_b from Dual-SVID
        for step = 1..N:
            sample batch from cache
            Z_s = block_b(X_b^teacher[batch])
            loss = mean((f_b^0.5 * (Z_s - Z_b^teacher[batch]))^2)
            loss.backward(); optimizer.step()
        save block_b state

[assemble full student model]
    ↓
[evaluate: PPL, hidden-state drift, generation coherence]
```

### 4.3 Reuse vs new

**Reused from Sprint 0:** TF32, bf16 model saves, plateau early-stop,
`torch.compile` where it doesn't break on SmoothSign.

**Reused from Sprint 3:** Teacher cache infrastructure (mmap reader,
storage format, extraction loop). Extended to store intermediate
hidden states, not just final logits.

**New infrastructure:**
1. Fisher collection (one-time per model).
2. Teacher-cache extension to intermediate hidden states (one-time per model).
3. Per-block training loop with scoped optimizer.
4. Block-wise model assembler.

## 5. Experiment S4.0 — single-block proof of concept

**Goal:** validate that Fisher-weighted block-output MSE does meaningful
work per-block at modest step count, before committing to the full
24-block pipeline.

**Target:** Qwen 2.5 0.5B, block 12 (contains our canonical
`gate_proj L12` layer from [§13](littlebit_math.md)).

**Setup:**
- Fisher collected from 128 seqs × 2048 tok on WikiText-2 train.
- `X_12^teacher`, `Z_12^teacher` extracted on-the-fly (no cache extension
  needed yet for single-block validation).
- Dual-SVID init at r=512 for all 7 linears in block 12.
- BRECQ-relaxed: SmoothSign + scales, AdamW, lr=1e-3, tau=100.
- Train 500 steps, log block-output rel-err every 25 steps.

**Metrics:**
- **Primary:** Block-output rel-err `||Z_s − Z_t||_F / ||Z_t||_F` at init
  and at each checkpoint. Goal: drops from ~0.88 (Dual-SVID init) to ≤0.3.
- **Secondary:** activation-rel-err at each of the 7 internal matrices
  (q, k, v, o, gate, up, down) — shows intra-block coupling capture.
- **Tertiary:** loss curve, sign-flip count per step (how many signs
  actually flip during block calibration).

**Go signal:** block-output rel-err ≤ 0.3 at step 500.

**Partial-credit signals:**
- Rel-err 0.3–0.5: Fisher-weighted MSE works but needs more steps or
  better init (Stage 2 rotation / activation-weighted).
- Rel-err 0.5–0.7: block-local objective helps but format or parameter
  scope may need adjustment. Investigate Choice 1 ablation.

**Kill signal:** rel-err > 0.7 after 500 steps, or training unstable
(loss diverges, NaN). Indicates either implementation bug or that
Fisher-weighted block MSE is insufficient for LittleBit's factored
form. Debug before extending.

**Cost:** ~1 day engineering + <1 hr compute.

## 6. Follow-up experiments (conditional on S4.0)

### S4.1 — Full 24-block pipeline at 0.5B

Run S4.0's protocol on all 24 blocks sequentially. Assemble, evaluate
end-to-end. Measure PPL, hidden-state drift, generation coherence.

**Success criterion:** WikiText full-test PPL ≤ 20 (target: near FP16
which is 16.4 on Qwen 2.5 0.5B). Coherent generation past 20 tokens on
10-prompt test set. Per-layer hidden-state rel-err < 0.5 end-to-end.

**Cost:** ~2 days engineering (cache extension, pipeline assembly)
+ ~1 hr compute (24 blocks × ~2 min).

### S4.2 — Budget sweep at 0.5B

Vary N ∈ {100, 250, 500, 1000, 2000} steps per block. Populate the
QAT-budget curve. Find the knee.

**Cost:** ~5 runs × ~1 hr = 5 hr compute. No engineering.

### S4.3 — BRECQ-strict ablation

Repeat S4.1 with signs frozen after Dual-SVID init, scales only.
Quantifies sign-flipping contribution. Informs whether Stage 3
matching pursuit could replace gradient sign search.

**Cost:** ~1 hr compute.

### S4.4 — Scale to 1.5B then 7B

Full teacher-cache extension. Sprint-3 infrastructure integration.
Production-grade per-block parallelism if memory permits (Sprint 4
concern).

**Cost:** ~1 week total including infra.

## 7. Open questions (resolve during S4.0)

1. **Fisher target distribution.** Ground-truth next-token (standard,
   cheaper) or teacher's own distribution (self-Fisher, more principled)?
   **S4.0 default: ground-truth.** Revisit if S4.0 underperforms.

2. **Calibration set size.** 128 seqs × 2048 tok for Fisher collection;
   32 seqs for training batches (Sprint-0 default).
   **S4.0 default: as stated.** Sensitivity analysis deferred to S4.2.

3. **Sign LR vs scale LR.** SmoothSign at reduced LR — how much reduced?
   Current full-QAT uses lr=1e-3 jointly. For BRECQ-relaxed, try
   `lr_signs = 0.3 × lr_scales`. First pass.
   **S4.0 default: single lr=1e-3 for both, tune if S4.0 shows sign
   oscillation.**

4. **SmoothSign tau.** Current full-QAT uses tau=100. For shorter
   block-local training, may need tau warmup (10 → 100) or fixed
   lower tau.
   **S4.0 default: tau=100 constant, match existing QAT.**

5. **Batch size within a block.** Block-local memory is much smaller
   than full-model — can use batch sizes the full QAT can't afford.
   **S4.0 default: batch=4 (matches Sprint-0), opportunistically scale.**

## 8. First commits

Implementation order:

1. **`littlebit_fisher.py`** — standalone script that runs the FP
   teacher forward + backward on calibration data, collects Fisher
   diagonals, saves per-block to disk. No LittleBit infrastructure
   dependency. Can validate in isolation before wiring to training.
2. **`littlebit_qat_brecq.py`** — per-block training loop. Single-block
   mode first (S4.0), extended to full pipeline in S4.1.
3. **Teacher-cache extension** — deferred to S4.1; S4.0 collects
   `X_12, Z_12` on-the-fly.

Deliverable for first PR: S4.0 result — a plot of block-output rel-err
vs step, plus the per-matrix breakdown.
