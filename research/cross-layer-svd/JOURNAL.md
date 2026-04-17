# Factored Inference Research Journal

Running log of the research project: run 20GB+ (ideally 70B-class) models on a single 12 GB GPU by factoring weights and streaming per-layer coefficients from system RAM over PCIe during inference.

## Hard constraints (locked)

- **Model-agnostic.** Scheme must work on any transformer family, not just Qwen. No assumptions about hidden-state distributions, layer counts, or architecture-specific structure.
- **Post-training only.** No retraining, no fine-tuning, no distillation. Decomposition runs on existing pretrained weights, with at most a calibration pass (forward-only inference on a small held-out corpus).
- **Runtime application.** Decomposition happens offline or at model load, but the streaming runtime must serve any model the decomposer was pointed at.

## Target architecture (unchanged)

```
┌─────────────── GPU (12 GB) ───────────────┐      ┌──── CPU RAM ────┐
│                                            │       │                 │
│   Shared basis B  (resident)               │       │  Per-layer      │
│                                            │  ←→   │  coefficients   │
│   KV cache, activations, current layer     │ PCIe  │  A_1, A_2, ... │
│                                            │       │                 │
└────────────────────────────────────────────┘       └─────────────────┘
```

Per decode step: compute `y_i = B · (A_i · x)` for current layer `i`. Transfer `A_{i+1}` concurrently via `cudaMemcpyAsync` on a second stream, overlapping with `y_i` compute.

## Timeline

### 2026-04-16 — Direction locked
- Approach: cross-layer SVD with shared basis + streamed per-layer coefficients.
- Optimization stack agreed: activation-aware SVD, non-uniform rank per layer, `y = B(Ax)` execution, double-buffered PCIe, int8 streamed coefficients.
- Critical empirical gate identified: need rank ≤ ~500 for the project to be viable.

### 2026-04-17 — First experiments

#### Experiment 1: cross-layer horizontal SVD (shared output basis)
**Setup:** Qwen 2.5 0.5B, 32 calibration samples from WikiText-2 train, evaluate PPL on WikiText-2 test. For each weight role (Q/K/V/O/gate/up/down), stack all 24 layer copies horizontally, apply activation-aware SVD, truncate to rank r, reconstruct via projection.

**Bug fix:** First attempt at reconstruction used `W_r = U_r Σ_r V_r^T @ S^{-1}` which is numerically unstable (ill-conditioned `S^{-1}` spikes fp16-cast values). Rewrote to use the mathematically equivalent but stable projection form `W_r = U_r (U_r^T W)`, sidestepping `S^{-1}` entirely.

**Result:** Broken at every rank. PPL 27,269 → 14,236 → 86,137 as rank grows from 64 to 512 (baseline 9.4). PPL is non-monotonic in rank.

**Diagnosis from rel_err probes:** The SVD math is correct (relative error decreases monotonically with rank, zero error at full rank). The problem is that for MLP projections, shared-output-basis across layers throws away 84-99% of the matrix's energy at any rank we tested — each layer's MLP wants its own output subspace, and forcing 24 layers through one shared r-dimensional bottleneck is catastrophic.

#### Experiment 2: three-mode comparison
**Setup:** Same corpus/eval as Exp 1, compare three decomposition geometries at ranks {64, 128, 256, 512}:
- `cross-layer-h` — shared output basis (original hypothesis)
- `cross-layer-v` — shared input basis (vertical stacking)
- `per-matrix` — no cross-layer sharing, independent SVD per weight

**Results (PPL, baseline 9.4):**

| rank | per-matrix | cross-layer-h | cross-layer-v |
|------|-----------|---------------|---------------|
|   64 |     5,463 |        27,269 |         6,302 |
|  128 |     7,265 |        40,761 |        13,280 |
|  256 |    13,852 |        14,236 |        14,719 |
|  512 |    **86** |        86,137 |        12,266 |

**Findings:**
1. **Per-matrix is the only mode that works.** At rank 512 it drops to PPL 86 — still 9× baseline, but two orders of magnitude better than either cross-layer variant.
2. The rank-512 per-matrix jump is partly because K/V projections have `max_rank=128` (GQA head-dim), so at r=512 they reconstruct exactly. Attention K/V are disproportionately important.
3. Cross-layer sharing in both orientations is broken at every rank we tested. The inductive bias "layers share a basis" is wrong for this model.
4. PPL is non-monotonic in rank even for per-matrix. This isn't a bug — we minimize weighted reconstruction error, but PPL depends on how errors interact with LayerNorm/RoPE/downstream layers. Different rank cuts produce different residual-error patterns, some of which survive downstream while others are masked.

**Compression math reality check:** For a square d×d attention matrix, rank-r factoring uses `2dr` params vs `d²` original — only compresses when `r < d/2`. At r=512 for Qwen 2.5 0.5B (d=896), we're *inflating* attention storage, not compressing. Real compression wins must come from MLP (where d_out=4864 ≫ d_in=896).

## Research findings (cross-field survey, 2026-04-17)

Dispatched a research agent to survey matrix compression in non-ML fields. Key leads:

1. **Hierarchical matrices (HSS / HODLR)** — computational fluid dynamics / integral equations. Partition matrix into quadtree; each block gets its own rank based on local importance. Directly addresses our "uniform rank assumption is wrong" problem.
2. **Tensor trains (MPS)** — quantum many-body physics. Sequential bond structure across layers instead of one shared basis. Better inductive bias for transformer stacks than plain cross-layer SVD.
3. **Balanced truncation** — control theory. Weights by both input reachability AND output observability (gradient covariance). Strictly more principled than activation-aware SVD; likely makes PPL monotonic in rank.

Full survey transcript lives in the session log. Canonical references:
- Hackbusch, *Hierarchical Matrices: Algorithms and Analysis* (Springer 2015)
- Oseledets, "Tensor-Train Decomposition" (SISC 2011)
- Gugercin & Antoulas, "Balanced Truncation Model Reduction" survey

## Active hypothesis

The winning scheme is likely **hierarchical per-matrix decomposition with balanced-truncation weighting, with cross-layer sharing only at the hierarchy's coarse level**. Specifically:
- Each weight → HODLR/HSS with non-uniform rank per block.
- The coarse/root basis is shared across layers for the same role (the resident part on GPU).
- Finer blocks are per-layer (the streamed part).
- All decompositions use balanced-truncation weighting instead of plain ASVD.

This hypothesis is untested. The immediate next experiment is whether HODLR on a *single* MLP weight beats flat SVD at matched storage — that's the cheapest way to validate the "non-uniform rank" part of the hypothesis.

## Open questions

- Does HODLR beat flat SVD at matched storage on one MLP matrix? (Exp 3, in progress.)
- If HODLR works per-matrix, does adding cross-layer sharing at the coarse level preserve quality?
- Does balanced-truncation weighting (input + output gradient covariance) fix the non-monotonic PPL behavior?
- What's the per-matrix PPL ceiling as rank → max? (Need sweep at r ∈ {768, 1024, 2048} on MLP roles specifically.)
- How does this generalize to Llama-family, Mistral, etc.?

## Files

- `prototype.py` — rank sweep runner. Modes: `cross-layer-h`, `cross-layer-v`, `per-matrix`.
- `compare_modes.py` — orchestrator that runs all three modes and prints a diff table.
- `benchmarks/` — backend-agnostic evaluation suite (consistency, throughput, quality via lm-eval-harness wrapper). Not fully exercised yet; consistency suite ready.
- `.env` — HF_TOKEN for HuggingFace downloads (gitignored).
- `results_<mode>.json` — last rank-sweep output per mode.

## Next experiment queue

1. **HODLR single-matrix test** — decomposing one `gate_proj` from Qwen 2.5 0.5B with varying depth and off-diagonal rank. Measure rel_err and storage vs flat SVD at matched byte budget.
2. **Per-matrix high-rank sweep** — r ∈ {512, 768, 1024, 2048} on MLP only. Establishes the quality ceiling we're measuring everything else against.
3. **Balanced-truncation weighting** — add gradient covariance from a calibration backward pass; compare to ASVD at matched rank.
4. **Tensor train across layers** (if HODLR+balanced-trunc validates).
