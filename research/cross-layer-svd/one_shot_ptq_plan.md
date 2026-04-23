# One-shot PTQ sprint plan — block-local calibration instead of full-model QAT

> **Part of the LittleBit plan set.** See [README.md](README.md) and
> [consolidated_implementation_roadmap.md](consolidated_implementation_roadmap.md)
> for context. Related plans:
> [wall-time](wall_time_reduction_plan.md) ·
> [savings](savings_exploration_plan.md) ·
> [unexplored gains](unexplored_efficiency_gains.md) ·
> [scale-to-30B](scale_to_30b_architecture.md).
>
> **Foundations:** [littlebit_math.md §13](littlebit_math.md) (per-matrix
> activation-weighted result) and §15 (full-model composition collapse).
> **External survey:** [docs/research/quantization-papers-2025.md](../../docs/research/quantization-papers-2025.md).

This sprint plan scopes a research program to **incrementally reduce
the QAT budget** needed to reach LittleBit's quality target, by
stacking block-local calibration techniques from the BRECQ / AQLM /
BTC-LLM lineage on top of the current pipeline. Each stage is an
independently-evaluable contribution to a shared "QAT-budget curve."
The end state may be **zero QAT** (fully one-shot) or a **short
training cycle** (e.g., 30–60 min at 7B instead of 4–9 hours) —
both are valid outcomes; the sprint picks whichever the evidence
supports.

**Framing (load-bearing, per user guidance 2026-04-22):**
- 0.5B is a **rapid-prototyping harness**, not a target. Real
  acceptance is measured at 7B. Adverse effects from individual
  methods are expected and acceptable; the question is the net
  QAT-budget curve at 7B.
- "One-shot" is not the binary goal. The goal is **QAT-budget
  reduction**, on a curve. Each stage is scored by how much it
  pulls the curve down at matched quality; the sprint stops
  when the curve bottoms out or further stages stop helping.
- Stages ship incrementally. Each is a valid contribution on its
  own even if subsequent stages fail or regress.

**Status:** Proposal, not accepted. Execution gated on §4.1
establishing a QAT-budget baseline on the current pipeline and
§4.2 confirming that block-local optimization reduces it
measurably at 0.5B.

## 0. Motivation

[littlebit_math.md §13.2](littlebit_math.md) established that the
LittleBit factored form at r=512 **can represent ~90% of the
activation-relevant subspace per matrix** under an activation-weighted
objective. §15.2 then showed that **KL-only full-model QAT composes
this 90% local capacity down to 3.6% global** — per-layer hidden-state
rel-err reached 1.05 (orthogonal to teacher), generation collapsed
to digit loops within 5 tokens.

The existing plan response ([§15.6](littlebit_math.md),
[littlebit_enhancements.md](littlebit_enhancements.md) #1) is to add
intermediate-MSE (paper's λ=10) to the QAT loss, then scale-validate
at 1.5B / 7B. This is the "grind through composition with more
gradient" path.

**This plan proposes a parallel path:** replace end-to-end QAT
with block-local calibration in the BRECQ lineage. The hypothesis
is that the composition failure is not a capacity problem but an
objective-and-scope problem — per-matrix calibration can't see
intra-block dependencies (e.g., `softmax(QKᵀ)V`), and end-to-end
KL loss is too indirect to constrain hidden-state geometry. Block-
level Fisher-weighted reconstruction constrains both directly at
closed-form cost, and recent literature (BTC-LLM, AQLM) shows
sub-1-bit models reaching within ~1 PPL of FP16 with this approach
on 13B-scale models.

If this works, the wall-clock cost drops an order of magnitude,
fits the [no-cloud-compute constraint](../../../../.claude/projects/c--Users-tk199-source-repos-Atlas/memory/feedback_no_cloud_compute.md)
cleanly, and opens the door to a LittleBit variant that downstream
llama.cpp users could apply to their own models without
multi-hour training windows.

## 1. Literature landing — what the four papers teach us

Fetched 2026-04-22 from full-text arXiv HTML. Detailed notes inline.

### 1.1 Summary table (Llama-2, WikiText-2)

| Method | "One-shot"? | Opt. scope | Cost (7B) | 7B PPL | 13B PPL |
|---|---|---|---:|---:|---:|
| FP16 baseline | — | — | — | 5.12 | 4.88 |
| **BiLLM** (1.08 bit) | Yes — no gradient | GPTQ-style Hessian, column-seq OBS | 0.5 hr on A100 | **32.48** | **16.77** |
| **AQLM** (2.02 bit) | No — block FT | Beam + Adam codebook + block FT | Not stated | **6.64** | **5.65** |
| **BTC-LLM** (0.8 bit) | No — block-local | Cayley-SGD on (Λ, R) per block | 56–66 min | **6.60** | **5.83** |
| **BRECQ** (2 bit CNN) | No — block-local | AdaRound + LSQ | 20 min (ResNet-18 on 1080Ti) | N/A | N/A |
| **LittleBit** (0.55 bit) | No — full QAT | End-to-end KL + MSE | Multi-hour | Paper-claim FP16-comp. | Paper-claim FP16-comp. |

### 1.2 Key technical findings

**BRECQ** ([arXiv 2102.05426](https://arxiv.org/abs/2102.05426))
- Block = a transformer block / residual block, justified by Fisher block-diagonality:
  off-diagonal loss is concentrated *inside* blocks, not across them (§3.2).
- Objective is Fisher-diagonal-weighted reconstruction:
  `min 𝔼[Δz^(ℓ)ᵀ · diag((∂L/∂zᵢ^(ℓ))²) · Δz^(ℓ)]` (Eq. 10).
- **Block inputs are FP32 teacher activations**, not quantized-predecessor
  outputs. Errors don't accumulate across blocks during calibration because
  each block is calibrated independently against teacher.
- Only rounding parameters + activation step sizes get gradients (AdaRound + LSQ scope);
  weights themselves frozen.
- Mixed-precision via genetic search on Fisher sensitivity, 2-bit only.
- CNN-only in the paper; no transformer evals.

**AQLM** ([arXiv 2401.06118](https://arxiv.org/abs/2401.06118))
- Weight groups of size `g` represented as sum of `M` codewords from
  learned codebooks: `W_group ≈ Σₘ Cₘ · bᵢⱼₘ` (Eq. 2).
- Three-phase alternating optimization:
  1. Beam search over discrete codes (MRF inference).
  2. Adam update on codebooks (100 steps, lr 1e-4).
  3. Block fine-tuning — backprop through transformer blocks,
     frozen codes. Target = pre-quantization teacher block output.
- Block FT is 10–30% of total calibration time.
- Uniform bit allocation per model, no per-layer adaptation.
- Llama2-7B at 2.02 bit: 6.64 WikiText PPL vs FP16 5.12.

**BiLLM** ([arXiv 2402.04291](https://arxiv.org/abs/2402.04291))
- **Pure PTQ — no gradient updates anywhere.** Hessian-based column-
  sequential quantization (OBS/GPTQ lineage).
- Salient weight selection: `sᵢ = wᵢ² / [H⁻¹]ᵢᵢ²` (Eq. 3), column-
  structured, ~5–10% of weights.
- Salient weights: binary residual `W ≈ α_o* B_o* + α_r* B_r*` (Eq. 6–7).
- Non-salient: bell-shape split at percentile-searched break-point `p*`,
  each half binarized independently.
- **Scale behavior is the critical finding:** 7B at 1.08 bit reaches
  32.48 PPL (6.3× FP16 — not usable). 13B 1.08 bit: 16.77 PPL.
  70B 1.08 bit: 8.41 PPL (~2.7× FP16, marginal but coherent).
- **Pure calibration-only PTQ is insufficient at 7B scale sub-1-bit.**
  This is load-bearing for our target regime.

**BTC-LLM** ([arXiv 2506.12040](https://arxiv.org/abs/2506.12040))
- Learnable transformation `(Λ, R)`: diagonal scaling × orthogonal
  rotation. Learned via Cayley SGD (preserves orthogonality),
  network weights frozen.
- Forward: `y = (xRΛ) · B(Λ⁻¹Rᵀ Wᵀ)` (Eq. 3), binarizer `B(·)`.
- Absorption: `(Λ, R)` fuses into adjacent weight matrices at
  zero runtime cost except for attention projections / FFN down
  where Hadamard is used (still fusable via residual-stream math).
- Binary codebook: Hamming-distance E-step, sign-based majority
  M-step. Codebook size varies with vector length (ablated [4..20]).
- **Cost: 56–66 min on 7B (single GPU).**
- Llama2-13B 0.8 bit: 5.83 PPL (+0.95 vs FP16), 61.91% mean
  zero-shot across 7 tasks (+8.06 pp vs STBLLM).
- **Best published sub-1-bit PTQ result.** No LittleBit comparison.

### 1.3 Implications for LittleBit

Three structural takeaways:

1. **"One-shot" in the published literature universally means
   "no full-model QAT," not "no gradient anywhere."** Pure
   calibration-only (BiLLM) fails at 7B scale sub-1-bit. The viable
   regime is block-local gradient optimization on a small number of
   parameters (rotations, scales, codebooks) with frozen weights
   — 30–90 minutes on consumer hardware.

2. **Block granularity is load-bearing.** BRECQ's Fisher analysis
   explains why: intra-block dependencies (`QKᵀV`, gate × up projection
   interaction) are captured in block objectives but not per-matrix
   objectives. This matches our §15.2 failure mode.

3. **No paper does per-layer rank / bit allocation for factored
   schemes.** Unoccupied territory for a LittleBit-native
   heterogeneous-rank formulation.

## 2. The QAT-budget curve (what this sprint measures)

The core metric is the **minimum QAT budget needed to reach a fixed
quality bar**, as a function of which stages are enabled. Adding each
stage ideally pulls the curve down; at some point the curve bottoms
out (floor). Success of the sprint is the floor being materially below
the current baseline; "one-shot" is the special case of the floor being
zero.

**Quality bar (anchor point).** For the 7B target:
- WikiText-2 full-test PPL ≤ FP16 + 2.0 PPL
- Mean per-layer hidden-state rel-err ≤ 0.5 (cf. §15.2's 1.05 current)
- Coherent generation past 20 tokens on a 10-prompt standard set
- Zero-shot PIQA accuracy materially above random (≥55%)

This is tighter than the paper's reported numbers in some places
(hidden-state rel-err isn't reported in the paper but is load-bearing
for our generation-coherence concern) and looser in others (PPL
anchored relative to FP16 rather than absolute).

**The curve.** Conceptual shape we'll populate with measurements:

| Stages enabled | Expected QAT budget (7B, hours) | Measured? |
|---|---:|---:|
| Baseline (current: naive Dual-SVID init + KL + 10·MSE) | 4–9 | from existing runs |
| + Stage 2 (activation-weighted Dual-SVID init) | ? | 4.2 |
| + Stage 1 (rotation preprocessing) | ? | 4.4 |
| + Stage 4 objective (Fisher-weighted block-local MSE) | ? | 4.5 |
| + Stage 3 (closed-form matching pursuit for signs) | ? | 4.3 + 4.6 |
| + Stage 5 (heterogeneous rank) | ? | 4.7 |
| Full stack (one-shot candidate) | 0 or "short cycle" | 4.8 |

The experiments in §4 populate this table. **The result we care about
is the column "Measured budget at 7B at matched quality"** — not
pass/fail on any single stage.

**What "success" means in this framing.**
- **Floor = 0:** true one-shot — block-local calibration alone reaches
  the quality bar. Best case.
- **Floor > 0 but ≤ 1 hour at 7B:** short training cycle. Usable for
  downstream llama.cpp users on consumer hardware without a multi-
  hour window. Good case.
- **Floor > 1 hour but ≤ 3 hours at 7B:** material reduction from the
  current 4–9 hour baseline. Worth shipping as a pipeline improvement.
  OK case.
- **Floor ≥ current baseline:** the stack doesn't help. Characterization
  data, fall back to current Sprint 4/5 plan.

**What "adverse effects" means in this framing** (per user guidance).
Individual stages may hurt quality when added in isolation; that's
acceptable as long as the combined curve trends down. Track adverse
effects per stage and report — some may matter only in combination.

## 3. Proposed pipeline

Five orthogonal stages, each independently ablatable. Each stage's
contract follows the [model_prep_pipeline.md](model_prep_pipeline.md)
"requires / produces" convention.

### Stage 1 — Rotation preprocessing (BTC-LLM-derived)

- **Requires:** FP16 teacher model, calibration data (~1024 seqs × 2048 tok).
- **Produces:** Per-layer `(Λ_l, R_l)` with `R_l` orthogonal, storable
  inline with the factored weights.
- **Method:** Learn `(Λ_l, R_l)` to minimize L2 between rotated-and-
  binarized block output and FP32 block output. Cayley SGD
  ([Lezcano-Casado 2019](https://arxiv.org/abs/1901.08428)) for the
  orthogonal manifold, diagonal free. Block-local gradient — no
  backprop across blocks.
- **Cost estimate:** ~10–20 min on 0.5B, ~60 min on 7B.
- **Absorption:** Λ fuses into `h`, `g` scale vectors of LittleBit.
  R fuses into adjacent linear (`out_proj` / residual path) where
  possible; Hadamard fallback where not.

### Stage 2 — Activation-weighted Dual-SVID on rotated weights

- **Requires:** Rotated weights `W_l' = Λ_l⁻¹ R_lᵀ W_l` from Stage 1,
  calibration activation Gramian `H_l = 𝔼[X_lᵀ X_l]` from a single
  forward pass.
- **Produces:** Initial `sign(U_l), sign(V_l), h_l, g_l, ℓ_l` per
  layer at rank `r_l` (rank decided in Stage 5).
- **Method:** Closed-form Dual-SVID as in
  [littlebit_math.md §4](littlebit_math.md), but on rotated weights.
  No gradient. Single forward pass.
- **Cost estimate:** <5 min for any model size (dominated by SVD, not training).

### Stage 3 — Matching-pursuit sign refinement (novel, LittleBit-specific)

- **Requires:** Stage 2 init, `H_l` from Stage 2.
- **Produces:** Refined `sign(U_l), sign(V_l), ℓ_l` per layer.
- **Method:** Greedy rank-by-rank activation-weighted matching pursuit.
  For each rank `k = 1..r_l`:
  1. Compute residual `R_k = W_l' - Σⱼ<k sign(uⱼ) ℓⱼ sign(vⱼ)ᵀ`.
  2. Find best rank-1 binary-factored approximation of `R_k` minimizing
     `tr((R_k − u ℓ vᵀ) H_l (R_k − u ℓ vᵀ)ᵀ)`:
     - Alternate `v = sign(R_kᵀ H_l u)`, `u = sign(R_k H_l v)`
       (sign-constrained power iteration) to convergence (typically <10 iters).
     - `ℓ` closed-form: `ℓ* = (uᵀ R_k H_l v) / (uᵀ H_l u · vᵀv)`
       *(exact form to derive in implementation).*
  3. Commit, proceed to rank `k+1`.
- **Prior art:** Binary Matrix Factorization (Zhang et al. 2007,
  Miettinen 2011) for the rank-1 sign-constrained subproblem.
  Not previously applied to LLM weight compression in factored form,
  per a quick lit pass.
- **Cost estimate:** Closed-form; ~5–15 min per layer at r=512.
  O(L · r) work total, embarassingly parallel per rank within a layer.
- **Fallback if matching pursuit underperforms:** retain Stage 2 init,
  skip refinement. Signs remain from SVID.

### Stage 4 — BRECQ-style block reconstruction

- **Requires:** Stage 2 + optional Stage 3 init.
- **Produces:** Optimized `(U_fp, V_fp, h_l, g_l, ℓ_l)` per layer
  under relaxed-signs configuration.
- **Design choices (locked 2026-04-22):**
  - **Signs:** relaxed. `U_fp, V_fp` remain learnable via SmoothSign
    at reduced LR; scales `(h, g, ℓ)` also learnable.
  - **Block input:** pure teacher. `X_b^teacher` loaded from cache,
    never propagated from student predecessor output.
  - **Parameter scope within block:** joint. All 7 linears
    (q, k, v, o, gate, up, down) of a Qwen block trained
    simultaneously against the single block-output loss.
- **Method:** For each transformer block `b = 1..B`:
  - Input: FP32 teacher activation `X_b^teacher` from calibration data.
  - Target: FP32 teacher block output `Z_b^teacher`.
  - Student: LittleBit-factored block with Stage 2+3 init.
  - Objective (Fisher-weighted, BRECQ Eq. 10 adapted):
    `min 𝔼[(Z_b^student − Z_b^teacher)ᵀ diag(f_b) (Z_b^student − Z_b^teacher)]`
    where `f_b = (∂L/∂z_b)²` is the empirical Fisher diagonal.
  - Gradient scope: all 7 block linears' `(U_fp, V_fp, h, g, ℓ)`.
    Norms, embeddings, LM head frozen at FP16.
  - Optimizer: AdamW, ~500–1000 steps per block.
- **Cost estimate:** ~1–3 min per block. 24 blocks × 2 min = ~50 min on 0.5B.
  ~3 hours on 7B (extrapolated linearly by block count).
- **Detailed plan:** See `stage_4_brecq_plan.md` (to be written).

### Stage 5 — Heterogeneous rank allocation (Fisher-weighted)

- **Requires:** Per-layer Fisher sensitivity `s_l` from one backward
  pass on calibration data; total parameter budget `B`.
- **Produces:** Per-layer rank `{r_l}` subject to `Σ_l 2 r_l = B / d`.
- **Method:** Lagrangian water-filling under locally-additive loss
  model `L ≈ Σ s_l / r_l` gives **`r_l ∝ √s_l`** (closed form).
  Round to integer budget, enforce `r_l ≥ r_min` (say 32) and
  `r_l ≤ min(d_in, d_out)`.
- **Sensitivity measurement options** (all forward-only except Fisher):
  1. **Empirical Fisher diagonal** — one backward pass through teacher
     on calibration data, sum squared grads per layer output. Standard.
  2. **Block-output rel-err at reference rank** — cheapest, run
     Stage 2 at fixed `r_ref = 128` per layer, measure block-output
     rel-err. No backward pass.
  3. **Angular sensitivity** — perturb each layer output by small
     Gaussian noise, measure KL increase at final logits.
- **Integrates with:** Stages 2–4. Runs first to decide `r_l`, then
  stages consume.
- **Cost estimate:** <10 min regardless of model size (one pass).

## 4. Experiments — populate the QAT-budget curve

Each experiment contributes a **data point** to the curve in §2.
The "QAT budget" for a given stage combination is the minimum step
count at which the §2 quality bar is reached (or the plateau quality
if the bar isn't reached). Experiments are ordered by information-
per-engineering-day; later experiments build on infrastructure from
earlier ones but don't gate on pass/fail of specific quality targets.

Every experiment's deliverable is the same shape: **a QAT-curve
measurement + per-layer-rel-err + generation-coherence sample**,
regardless of whether quality improves. Adverse effects are recorded,
not filtered out.

All Qwen 2.5 0.5B experiments are a **rapid-prototyping harness** —
they validate that infrastructure runs and the curve has the right
shape. The actual budget floor only matters at 7B (§4.8); the 0.5B
numbers are for debugging and stage selection.

### 4.1 Baseline calibration — anchor the curve

**Question:** What's the current QAT budget at 7B to reach the
§2 quality bar, before any sprint changes?

**Method:** Measure from existing Sprint-0 + Sprint-3 infrastructure
runs. Extract: (i) budget at 0.5B to hit the bar, (ii) extrapolated
budget at 7B using current scaling trend. No new experiment — just
auditing existing JOURNAL data.

**Deliverable:** The "Baseline" row of the §2 table filled in with
measured 0.5B and extrapolated 7B numbers. If the bar isn't reachable
with the current pipeline at any budget, that's the real finding and
this sprint becomes essential rather than incremental.

**Cost estimate:** ~0.5 day (analysis of existing runs).

### 4.2 Activation-weighted Dual-SVID init (Stage 2 contribution)

**Question:** How much does swapping naive Dual-SVID for activation-
weighted init move the QAT-budget curve?

**Method:** On Qwen 2.5 0.5B, initialize via Stage 2 (rotated-free,
activation-weighted Dual-SVID using calibration Gramian). Run the
current QAT loop (unchanged KL + MSE) for varying step budgets.
Plot quality vs budget. Compare to baseline curve.

**Deliverable:** The "+Stage 2" row of the §2 table. Adverse effects
noted (e.g., if the better init biases the optimizer into a worse
basin at long training times — which would surprise, but worth checking).

**Cost estimate:** ~2 days engineering (Gramian collection + init
code) + ~3 hours compute per budget point × 4–5 budget points.

### 4.3 Closed-form matching-pursuit single-matrix test (Stage 3 probe)

**Question:** Can closed-form sign refinement match gradient-based
SmoothSign on per-matrix activation-rel-err? If yes, Stage 3 becomes
a zero-QAT contribution; if not, Stage 3 still helps as better init
but needs gradient afterward.

**Method:** Take Qwen 0.5B gate_proj L12 (the §13 canonical layer).
Compute `H` from same calibration set. Run matching-pursuit to
convergence (<30 iterations expected). Measure activation-rel-err
vs §13.2's 0.307 (from 5000 SmoothSign+Adam steps). Also measure
intermediate rel-err at each rank-1 commit to see if partial MP is
a useful "warm init for reduced QAT" signal.

**Deliverable:** Single-layer curve: activation-rel-err vs MP
iteration. Comparison to §13.2's gradient result. Characterization
of where MP plateaus vs gradient refinement.

**Cost estimate:** ~1 day, existing `littlebit_qat_activation.py`
infrastructure reusable.

### 4.4 Rotation preprocessing alone (Stage 1 contribution)

**Question:** How much does Stage 1's `(Λ, R)` rotation, applied
*before* any other change, move the QAT-budget curve? This is the
BTC-LLM technique in isolation, using our existing QAT loop.

**Method:** Port a minimal Cayley-SGD + (Λ, R) learnable transform,
applied to Qwen 2.5 0.5B weights before the usual Dual-SVID init.
Run existing QAT loop afterward. Plot quality vs budget, compare to
§4.2 curve.

**Deliverable:** "+Stage 1" and "+Stage 1+2" rows of the table.
Adverse effects: rotation may interact poorly with our factored
form's sign structure — document if so.

**Cost estimate:** ~3 days engineering (Cayley SGD isn't in our
stack yet) + ~3 hours compute × 4 budget points.

### 4.5 Fisher-weighted block-local MSE objective (Stage 4 contribution)

**Question:** Does swapping the current KL + 10·MSE end-to-end
objective for Fisher-weighted per-block MSE against teacher block
outputs (BRECQ Eq. 10) reduce the QAT budget? This is the single
change most likely to directly address §15.2's composition failure.

**Method:** Fork `littlebit_qat_model.py`. Replace loss with per-block
Fisher-weighted MSE; keep teacher activations as each block's input
during training (BRECQ-style). Run Qwen 2.5 0.5B at varying budgets,
compare to baseline curve.

**Deliverable:** "+Stage 4" row. This is the most load-bearing
single-stage experiment — if the block-local objective alone
collapses the QAT budget by 3× or more, most of the "one-shot" story
is already written and Stages 1+3 become polish.

**Cost estimate:** ~2 days engineering (loss swap + per-block
gradient scoping) + ~3 hours compute × 5 budget points (incl. zero-
step "post-init-only" point).

### 4.6 Per-layer Fisher sensitivity profile (Stage 5 prerequisite)

**Question:** Is the sensitivity distribution across layers peaked
enough that heterogeneous rank allocation materially helps?

**Method:** One backward pass through Qwen 2.5 0.5B on calibration
data, collect per-layer Fisher diagonal sums. Compute water-filling
rank allocation `r_l ∝ √s_l` at matched total parameter budget.
Repeat on Qwen 2.5 1.5B if 0.5B looks promising — sensitivity
profile may be architecture- and scale-dependent.

**Deliverable:** Sensitivity distribution plot, water-filling rank
table, expected quality gain estimate under locally-additive error
model. Input to §4.7.

**Cost estimate:** ~1 day, runs on existing infra.

### 4.7 Combined stack incremental add-in (0.5B)

**Question:** Do the stages compose? Each stage individually moves
the curve; do they stack, interfere, or some combination?

**Method:** On Qwen 2.5 0.5B, add stages cumulatively in the order
best-performing-first (from §4.2–§4.5 results). Measure the QAT-
budget curve at each step. Also try the worst-first order to test
ordering sensitivity.

**Deliverable:** Full table from §2, 0.5B column. Adverse-effect map:
which stages regress when combined with which others.

**Cost estimate:** ~1 week engineering (integration) + ~8 hours compute.

### 4.8 Scale to 7B (the actual measurement)

**Question:** Where does the QAT-budget curve bottom out at 7B?

**Method:** Apply the best-performing stage combination(s) from §4.7
to Qwen 2.5 7B, using the Sprint-3 teacher cache. Sweep QAT budget
from 0 to the current full-run budget. Find the smallest budget that
reaches the §2 quality bar.

**Deliverable:** Final QAT-budget floor at 7B. This is the number
the sprint produces.

**Cost estimate:** ~1–2 weeks depending on how many stage combinations
need validation at scale. Each 7B run: teacher-cache build (~30 min
one-time) + QAT run (variable by budget, capped at baseline ~9 hr).

### 4.9 Optional: sub-0.5 BPW differentiating regime (scale down BPW, not model)

**Question:** Below ~0.5 BPW where BTC-LLM degrades sharply and
STBLLM collapses, does LittleBit's factored form hold up with our
stack?

**Method:** Apply the §4.8 pipeline to Qwen 2.5 7B at rank
corresponding to 0.3, 0.2, 0.1 BPW. Measure quality and QAT budget.
Characterize the sub-0.5-BPW regime specifically, where LittleBit's
format has a structural advantage over codebook-based schemes.

**Deliverable:** Quality / budget / BPW table for the LittleBit-
unique regime. Publishable even if 4.8 shows no one-shot win at
0.8 BPW — the sub-0.5 BPW regime is unoccupied in published PTQ
literature.

**Cost estimate:** ~1 week, reuses §4.8 infrastructure.

## 5. Deliverable — the filled QAT-budget curve

No experiment has pass/fail criteria. The sprint produces a single
populated table:

| Stage combination | 0.5B budget (steps) | 0.5B quality @ budget | 7B budget (hr) | 7B quality @ budget | Adverse effects |
|---|---:|---:|---:|---:|---|
| Baseline | from §4.1 | | | | |
| +Stage 2 | from §4.2 | | | | |
| +Stage 4 | from §4.5 | | | | |
| +Stage 2+4 | from §4.7 | | | | |
| +Stage 1+2+4 | from §4.7 | | | | |
| +Stage 1+2+3+4 | from §4.7 | | | | |
| +Stage 1+2+3+4+5 | from §4.7 | | from §4.8 | | |

Plus the sub-0.5-BPW characterization from §4.9 (if pursued).

**How to read the deliverable.** The 7B budget column is the sprint's
primary output. Interpretation:
- Budget = 0 → one-shot path validated. Full stack replaces QAT.
- Budget ≤ 1 hr → short-cycle path. Stack plus short polish QAT.
- Budget 1–3 hr → material reduction. Stack adopted as default init
  for existing QAT pipeline.
- Budget ≥ current → stack doesn't help at 7B; fall back to Sprint 4/5.

**The "correct" outcome depends on the user's tolerance.** This
sprint produces the curve; the decision on how far to push (and
whether "short cycle" is acceptable vs requiring zero QAT) is the
user's, post-measurement.

## 6. Cost estimates — total sprint budget

| Phase | Engineering | Compute | Wall-clock |
|---|---:|---:|---:|
| 4.1 Baseline audit | ~0.5 day | 0 (existing runs) | ~0.5 day |
| 4.2 Stage 2 init | ~2 days | ~12 hr (budget sweep) | ~3 days |
| 4.3 MP single-matrix | ~1 day | <1 hr | ~1 day |
| 4.4 Stage 1 rotation | ~3 days | ~12 hr | ~4 days |
| 4.5 Stage 4 BRECQ loss | ~2 days | ~15 hr | ~3 days |
| 4.6 Fisher sensitivity | ~1 day | <1 hr | ~1 day |
| 4.7 Combined stack @ 0.5B | ~1 week | ~8 hr | ~1.5 weeks |
| 4.8 Scale to 7B | ~2 weeks | ~30 hr | ~3 weeks |
| 4.9 (optional) sub-0.5 BPW | ~1 week | ~15 hr | ~1.5 weeks |
| **Total (without §4.9)** | **~4 weeks** | **~80 hr** | **~5 weeks** |
| **Total (with §4.9)** | **~5 weeks** | **~95 hr** | **~6.5 weeks** |

Stages can parallelize where they don't share code: §4.3 (single-
matrix MP), §4.4 (rotation infra), and §4.6 (sensitivity profile)
can run concurrently with no interference. §4.5 (objective swap)
and §4.7 (integration) are sequential.

**Early-exit rule.** If §4.5 alone reduces the 0.5B QAT budget by
≥3× at matched quality, treat it as a standalone ship and pause
§4.1+§4.4 work until after §4.8 at 7B measures whether it scales.
This is the cheapest plausible path to a useful result.

## 7. Risks and honest caveats

**7.1 Scale discontinuity.** BiLLM works at 70B but not 7B. Our target
regime is 7B. Nothing published guarantees that block-local methods
survive at 7B at sub-0.5 BPW — only BTC-LLM's ~0.8 BPW result sits in
that zone, and it's on 7B Llama2, not Qwen 2.5. The further we push
below 0.8 BPW, the more we're extrapolating published evidence.

**7.2 Factored form penalty.** BTC-LLM uses a dense binary codebook;
LittleBit uses a factored binary form with rank truncation. The
rank bottleneck is an additional source of capacity loss that the
block-local methods weren't designed around. 4.5's full-pipeline
test is the first joint evaluation — at-risk of revealing that
the factored form is fundamentally harder to one-shot than the
codebook form.

**7.3 Matching-pursuit convergence isn't guaranteed.** Sign-
constrained power iteration on non-convex binary spaces can cycle
or plateau far from optimum. The closed-form formulation in Stage 3
requires derivation review and empirical validation (4.3) before
commitment.

**7.4 Calibration data sensitivity.** BTC-LLM uses RedPajama;
LittleBit paper uses wikitext. Our current evals use wikitext.
If calibration distribution shift matters (likely for sub-1-bit),
cross-domain eval becomes a hidden cost.

**7.5 This is fork-only research.** The LittleBit factored form has
no kernel in mainline llama.cpp. Per [project_llama_cpp_fork_goal.md](../../../../.claude/projects/c--Users-tk199-source-repos-Atlas/memory/project_llama_cpp_fork_goal.md)
the deployment invariant is upstream-loadable GGUFs, which LittleBit
currently doesn't satisfy. A successful one-shot pipeline is a
characterization result ("sub-1-bit PTQ in factored form is viable at 7B
in 90 min on consumer hardware") and a motivator for a future upstream
kernel conversation, not itself a shipping artifact.

**7.6 BTC-LLM may have better absolute numbers at our target BPW.**
If BTC-LLM at 0.8 BPW is reachable with block-local optimization and
LittleBit at 0.55 BPW via one-shot is materially worse, then the
"one-shot LittleBit" program is dominated by "adopt BTC-LLM" for the
0.8-BPW tier. The compelling LittleBit niche is below 0.5 BPW where
BTC-LLM and peers degrade sharply (STBLLM falls off a cliff at 0.7 BPW
per the BTC-LLM Table 3). Sprint should explicitly target 0.1–0.3 BPW
as the differentiating regime in 4.5–4.6.

## 8. How this fits into the existing sprint sequence

Existing sprint plan status (per [JOURNAL.md](JOURNAL.md) 2026-04-22):

| Sprint | Scope | Status |
|---|---|---|
| 0 | Wall-time stack (TF32 + Liger + compile + bf16 saves + early-stop) | Shipped |
| 1–2 | Memory + wall-time polish | Open |
| 3 | Teacher cache + chunked KL + mmap reader | Phase I shipped |
| 4 | Quality enhancements (19 items from [littlebit_enhancements.md](littlebit_enhancements.md)) | Open |
| 5 | Scale validation at 7B | Open, gated on Sprint 3 |
| 6 | 30B teacher cache + NVMe streaming | Open, gated on Sprint 5 |
| **7 (this plan)** | **One-shot PTQ via block-local calibration** | **Planning** |

**Positioning.** This sprint is **orthogonal to Sprints 1–6**, not
sequential. Sprints 1–6 optimize and scale the current QAT pipeline;
this sprint explores whether the QAT pipeline can be replaced outright.
Both paths can proceed in parallel — if Sprint 7 succeeds, Sprint 6
may not be needed (30B one-shot is cheap); if Sprint 7 fails, Sprint
6 remains the scaling strategy.

**Interaction with Sprint 3.** Sprint 3's teacher cache is reusable
here — block-local calibration also needs teacher activations, and
the existing cache format works. 4.6 at 7B scale benefits directly.

**Interaction with Sprint 4.** The 19-item enhancement catalog
([littlebit_enhancements.md](littlebit_enhancements.md)) lists several
items that overlap with this plan's stages (notably #1 intermediate-MSE,
which is superseded by Stage 4's BRECQ objective; #5 tau warmup, which
applies only if gradient sign search remains post-Stage 3). If Sprint 7
ships, a subset of Sprint 4's catalog becomes obsolete.

## 9. Execution cadence — iterate and measure

There are no gating pass/fail decisions. The sprint runs as a
measurement campaign with standing rules:

**Order of attack:**
1. §4.1 (baseline audit — 0.5 day) anchors the curve.
2. §4.5 (Stage 4 BRECQ objective) runs first because it's highest-
   likelihood-single-cause for §15.2's failure. If it alone
   reduces the 0.5B QAT budget materially, §4.4 and §4.2 run as
   extensions on top of the new baseline rather than independent tests.
3. §4.3 (single-matrix MP) and §4.6 (sensitivity profile) run
   concurrently with §4.5 — cheap, independent, and inform §4.7.
4. §4.2 and §4.4 run after §4.5 on the new baseline.
5. §4.7 integrates at 0.5B once the single-stage contributions
   are characterized.
6. §4.8 scales to 7B with whichever combination §4.7 showed lowest floor.
7. §4.9 optional characterization of sub-0.5 BPW regime.

**Standing rule after each experiment:**
- Record measured budget + quality + adverse effects in the §5 table.
- Update §2 curve estimate with new data.
- Let the measurement decide whether to proceed or pivot — no
  pre-committed kill criteria.

**Incremental ship points.** Each stage that measurably helps at 7B
ships as a pipeline enhancement independent of the others, updating
[model_prep_pipeline.md](model_prep_pipeline.md) with a new default
stage. The "full stack" result in §4.8 is the aggregate target but
not the only useful output — a 2× QAT budget reduction from §4.5
alone is a shipping improvement.

**When to stop.** Sprint ends when any of:
- The §5 table is fully populated at 7B and the floor is acceptable.
- Two consecutive added stages fail to reduce the floor by ≥10%.
- Resources run out and the partially-filled table is the result.

Partial results are valid deliverables; the characterization is the value.

---

**Open questions for the user before sprint starts:**

1. **Factored form commitment.** Keep LittleBit's factored form fixed
   across all experiments, or scope to include a head-to-head at the
   same BPW against a dense binary codebook form (BTC-LLM-style)?
   Adding the latter doubles §4.8 compute but produces a stronger
   publishable characterization. Default in plan: fixed factored form.

2. **Acceptable QAT floor.** The sprint produces a curve; the user
   decides when the floor is "short cycle" enough. Rough anchor
   points for 7B:
   - ≤ 10 min → "zero-effort" regime; equivalent to one-shot for UX.
   - ≤ 1 hour → clearly "short cycle"; runs while making coffee.
   - ≤ 3 hours → overnight on laptop; still useful vs current 4–9 hrs.

   Pre-commit a tolerance or leave as "see what the curve says"? The
   plan assumes the latter.

3. **Target BPW range.** 0.55 BPW (paper headline), 0.8 BPW (BTC-LLM-
   comparable), or sub-0.3 BPW (LittleBit-unique)? Each has different
   curve shape expectations. Default in plan: 0.55 BPW for §4.1–§4.8,
   sub-0.3 BPW as optional §4.9 characterization.
