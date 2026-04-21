# LittleBit — math walkthrough and findings

Math reconstruction of LittleBit (Lee et al., Samsung Research, NeurIPS
2025, arXiv [2506.13771](https://arxiv.org/abs/2506.13771)) with
commentary grounded in this fork's archived per-matrix SVD results.

Downstream of the 2026-04-21 JOURNAL entry, which flagged LittleBit as
"the most direct continuation of the per-matrix SVD work" because it
uses the same `W ≈ U·Vᵀ` skeleton we rank-swept in April but with `U`
and `V` binarized and QAT used to recover quality.

Status: **math reconstruction and analysis, no reproduction yet.**
A one-file numerical sanity check on a toy matrix is the natural next
step before committing to a full reproduction.

## 1. Scope of this doc

Paper contains four components, all of which are reconstructed here:

1. Factored-binary forward pass (Eq. 4).
2. Efficient computation form (Proposition 1 / Eq. 5).
3. Dual Sign–Value-Independent Decomposition initialization (Eq. 6–8).
4. Residual compensation (Eq. 9–10) and QAT loss (Eq. 11).

Not in scope here:

- Kernel implementation on CUDA / Metal.
- Ablations on individual components (paper has these; we haven't
  reproduced).
- Empirical PPL / zero-shot numbers beyond the headline claims.

## 2. Notation

For a single weight matrix `W ∈ ℝ^{d_out × d_in}` and chosen rank `r`:

| Symbol | Shape | Role | Storage |
|---|---|---|---|
| `U_sign = sign(U)` | `d_out × r` | left binary factor | 1 bit / entry |
| `V_sign = sign(V)` | `d_in × r` | right binary factor | 1 bit / entry |
| `h ∈ ℝ^{d_out}` | row | per-output scale | FP16 |
| `g ∈ ℝ^{d_in}` | col | per-input scale | FP16 |
| `ℓ ∈ ℝ^r` | rank | per-latent scale | FP16 |

The three FP16 vectors are the "multi-scale compensation" of the paper.
Their dimensions are the distinguishing choice: one scale per row, per
column, and per latent rank — not per block, not per tensor.

## 3. The primary reconstruction (Eq. 4)

```
Ŵ_pri = diag(h) · sign(U) · diag(ℓ) · sign(Vᵀ) · diag(g)          (4)
```

Elementwise this is

```
(Ŵ_pri)_{ij} = h_i · g_j · Σ_k ℓ_k · sign(U)_{ik} · sign(V)_{jk}
```

The inner sum is integer-valued (sum of `r` terms, each ±1, weighted by
`ℓ_k`). The FP16 cost lives entirely in the scalings, not in the matmul.

### 3.1 Bit budget — reconstructed

Total storage per weight matrix:

```
bits = (d_out + d_in) · r     // sign matrices
     + 16 · (d_out + d_in + r) // FP16 scale vectors
```

Bits per weight, dividing by `d_out · d_in`:

```
BPW = r · (1/d_in + 1/d_out) + 16 · (1/d_in + 1/d_out + r/(d_out · d_in))
```

For a square `d × d` projection (`d_out = d_in = d`):

```
BPW ≈ 2r/d + 32/d + 16r/d²
    ≈ 2r/d   (the other two terms are O(1/d) at r ≪ d)
```

So **the compression knob is `r / d`**. At Llama2-13B `d = 5120`:

| `r` | `r/d` | BPW |
|---:|---:|---:|
| 16 | 0.0031 | 0.013 |
| 64 | 0.0125 | 0.031 |
| 256 | 0.050 | 0.106 |
| 1024 | 0.200 | 0.404 |

The paper's headline "0.1 BPW" lands at `r ≈ 256` on `d = 5120`. This
matches its claim of "~31× smaller than FP16" (16/0.5 ≈ 32 at the
0.5 BPW operating point they benchmark).

### 3.2 The first note worth logging

**Sub-1-BPW is only possible because `r` decouples parameter count from
weight count.** Any per-weight scheme has a floor at 1 bit even before
format overhead. Factored schemes sit below that floor trivially at
`r < d/2`.

This is the same reason standard low-rank SVD compresses (fewer
parameters in `U·V`) — the novel piece is doing it with 1-bit factors
instead of FP16. The compression ratio vs FP16 SVD at the same rank is
exactly **16×** from the factor precision change alone, before any
scale vectors or training.

## 4. Proposition 1 — efficient forward (Eq. 5)

**Claim** (Prop. 1, paper Section 3.1):

```
Y = X · Ŵ_priᵀ = ((((X ⊙ g) · V_sign) ⊙ ℓ) · U_signᵀ) ⊙ h          (5)
```

### 4.1 Derivation

Start with `Y = X · Ŵ_priᵀ` and substitute Eq. 4:

```
Y = X · ( diag(h) · sign(U) · diag(ℓ) · sign(Vᵀ) · diag(g) )ᵀ
  = X · diag(g) · sign(V) · diag(ℓ) · sign(Uᵀ) · diag(h)
```

Reading left-to-right:

1. `X · diag(g)` = elementwise-multiply each column of `X` by `g`:
   broadcasting, cost `O(B · d_in)`.
2. Result `· sign(V)`: a `B × r` output from a **binary** matmul over
   the `d_in` axis.
3. `· diag(ℓ)`: broadcast on the rank axis, cost `O(B · r)`.
4. `· sign(Uᵀ)`: a `B × d_out` output from another binary matmul.
5. `· diag(h)`: broadcast on the output axis, cost `O(B · d_out)`.

So `Y` is computed without ever materializing `Ŵ_pri`. The claim is
algebraically exact — diag-matrix multiplication is broadcasting
commuted with the adjacent matmul, and transpose reverses factor order.

### 4.2 FLOPs vs bytes

Two binary matmuls replace one FP16 matmul:

- **FLOPs**: unchanged to first order. Binary matmul still counts a
  multiply-accumulate per entry — just with 1-bit multiplicands. The
  total op count is `2 · B · r · (d_in + d_out) ≈ 4 · B · r · d` for
  square `d`, vs baseline `2 · B · d²`. Speculation: this is smaller
  by `d / (2r)` = `10×` at the 0.1 BPW config, assuming the integer
  MACs can be packed into available SIMD/tensor-core paths.
- **Bytes moved**: sign matrices are 1 bit per entry. At `r = 256`,
  `d = 5120`, the two sign matrices total `(2 · 5120 · 256) / 8 = 327 KB`
  vs `5120² · 2 = 52 MB` for FP16. `160×` reduction.

The paper reports "up to 11.6×" speedup at 0.1 BPW on Llama2-70B MLP
layers, with the caveat (Section 5) that "inference at small batch
sizes is often dominated by memory access rather than raw computation."
**Per-batch-1 numbers are not given.**

This is the biggest open question for our hardware profile — the
5070-class consumer decode case, where we already know (2026-04-19
Cloudflare investigation) that memory bandwidth dominates MMVQ at
batch=1. An 11.6× kernel speedup that only materializes at batch≥N is
not useful for the workloads this fork cares about. Reproduction plan
for this is in §10.

## 5. Dual-SVID initialization (Eq. 6–8)

### 5.1 The three-step procedure

Paper Section 3.2 prescribes:

1. Truncated-rank-`r` SVD: `W ≈ U' · Σ · V'ᵀ`. Fold `Σ` into `U'`
   and/or `V'` to get two rank-`r` factors.
2. **Sign init**:

   ```
   U_sign,0 = sign(U'),   V_sign,0 = sign(V')                     (6)
   ```

3. **Magnitude decomposition** — this is the "SVID" step:

   ```
   |U'| ≈ h_0 · ℓ_{u,0}ᵀ    (rank-1 approximation of |U'|)        (7a)
   |V'| ≈ g_0 · ℓ_{v,0}ᵀ    (rank-1 approximation of |V'|)        (7b)
   ℓ_0 = ℓ_{u,0} ⊙ ℓ_{v,0}                                          (8)
   ```

   where `|·|` is element-wise absolute value and the rank-1 fits are
   themselves computed by SVD of the magnitude matrices.

### 5.2 Why this is heuristic, not derived

The paper is explicit (Appendix A.2 per the public HTML) that this is
presented as a procedure, not derived from an optimization criterion.
The key things that are **asserted, not proven**:

- **Why rank-1 of the magnitudes.** The claim is that row-wise variation
  (captured by `h`) and rank-wise variation (captured by `ℓ_u`) are
  approximately separable, i.e. that `|U'|_{ik} ≈ h_i · ℓ_{u,k}`. This
  is a multiplicative-separability assumption. If the truncated SVD
  factors don't actually exhibit this structure, the rank-1 fit will
  have large residual and the initialization will be poor.
- **Why `ℓ_0 = ℓ_{u,0} ⊙ ℓ_{v,0}`.** The primary-path has a single rank
  axis `ℓ`, but the two magnitude decompositions each produce their
  own `ℓ_u`, `ℓ_v` vectors. The paper takes their Hadamard product.
  No proof is given that this is optimal vs (say) geometric mean,
  element-wise max, or a learned combination. It is chosen to match
  dimensions and because `ℓ_u · ℓ_v` appears as a scalar contribution
  in the product `|U'_{ik}| · |V'_{jk}|` under the separability
  assumption.

**Finding worth logging:** Dual-SVID is a warm-start heuristic, not an
optimal initializer. Its value is almost certainly in getting QAT out
of a flat region of the loss — not in producing a good
zero-shot reconstruction. Expect the initial `Ŵ_{pri,0}` to be far from
`W` in Frobenius; expect QAT to do most of the work. This reframes
Dual-SVID as "a sign/magnitude split that QAT can then refine," not as
the compression itself.

### 5.3 Relation to our archived SVD floor

Our 2026-04 per-matrix SVD sweep on Qwen 2.5 0.5B was:

| Rank | PPL |
|---:|---:|
| baseline | 9.4 |
| r = 512 | 86 |

That was FP16 factors, no binarization, no QAT. LittleBit's starting
point — before QAT — would be **worse** than this because of the sign
truncation. The interesting question is whether QAT can recover below
our baseline-PPL floor with only `r = 256` binary factors.

We don't need to speculate about this. It's directly measurable.

## 6. Residual compensation (Eq. 9–10)

After initializing the primary path, the paper computes the residual:

```
W_{res,0} = W − Ŵ_{pri,0}                                           (implicit before 9)
```

Then runs Dual-SVID on `W_{res,0}` to init a second factorized-binary
path with its own `h_res, g_res, ℓ_res, U_{res,sign}, V_{res,sign}`:

```
Ŵ_res = diag(h_res) · U_{res,sign} · diag(ℓ_res) · V_{res,sign}ᵀ · diag(g_res)    (9)
Ŵ     = Ŵ_pri + Ŵ_res                                               (10)
```

### 6.1 Notes

- The residual doubles the storage: `2 · r / d` BPW instead of `r / d`.
  At 0.1 BPW (`r = 256, d = 5120`) this becomes 0.2 BPW with the
  residual path active.
- The paper uses **exactly one residual stage**. No iterative "residual
  of residual" is explored. This is worth flagging as an obvious ablation
  — the natural question is whether the sequence of residuals decays
  fast enough to justify more stages at a given total BPW budget.
- Both paths are trained **jointly** under QAT (paper Section 3.3), not
  staged.

## 7. QAT loss (Eq. 11)

```
L_QAT = L_out + λ · L_inter                                          (11)
```

where (per public HTML commentary):

- `L_out`: KL divergence between student (LittleBit model) and teacher
  (full-precision original) output logits.
- `L_inter`: MSE between matched intermediate hidden activations across
  student and teacher.
- `λ = 10` — empirically fixed.

### 7.1 Backward pass for sign

The sign function is non-differentiable. Paper uses **SmoothSign**
(Section A.2.2) rather than the standard straight-through estimator:

- Forward: `sign(x)`
- Backward: gradient = `d/dx tanh(100·x) = 100 · (1 − tanh²(100·x))`

This is a sharp-but-smooth surrogate. The surrogate is nonzero only
near `x = 0`, which is where sign flips could reduce loss.

**Comment:** The `100` factor in `tanh(100x)` is a temperature — higher
makes the surrogate closer to a delta at `x = 0` (less gradient flow),
lower blurs it (more gradient but less fidelity to `sign`). The paper
doesn't ablate this choice. If we reproduce, this is one of the first
hyperparameters to sanity-check.

## 8. What the paper proves vs what it claims

| Component | Proven / Derived | Heuristic / Asserted |
|---|---|---|
| Eq. 4 reconstruction form | — (definitional) | — |
| Proposition 1 efficient forward | Algebraic identity. Proven. | — |
| Bit budget | Counting argument. Reconstructed in §3.1. | — |
| Dual-SVID (Eq. 6–8) | — | Separability of `\|U'\|` as `h·ℓ_uᵀ`; Hadamard product for `ℓ_0`. |
| Residual one-stage | — | Choice; not ablated vs iterative. |
| SmoothSign temperature 100 | — | Choice; not ablated. |
| QAT loss weights λ=10 | — | Empirical. |
| 11.6× speedup at 0.1 BPW | Kernel benchmark, MLP only | Batch size unspecified for the headline figure. |

**The core math — Eq. 4 and Prop. 1 — is exact.** Everything that sets
the initial point and the training dynamics is empirical. The
compression ratio is real and falls out of the storage layout; the
quality recovery is the part that needs reproduction.

## 9. Direct relevance to this fork's archived SVD track

| Archived result | LittleBit offset |
|---|---|
| Per-matrix FP16 SVD at `r = 512` on Qwen 0.5B: PPL 86 vs baseline 9.4. Deemed not useful as post-training. | LittleBit adds QAT + binarization + multi-scale compensation on top of the same skeleton. Claims usable at rank far below 512 on much larger models. |
| CALDERA Stage 2 (2026-04-19): K-quants beat CALDERA at matched BPW. | LittleBit doesn't compete with K-quants at 4+ BPW — it only targets sub-1-BPW where no K-quant exists. Different operating point. |
| Cross-matrix shared-dictionary: DEAD (matrices anti-aligned). | LittleBit is per-matrix; the archived finding doesn't apply. |
| Depth-smoothness: DEAD. | Same — per-matrix, orthogonal. |
| 2026-04-19 synthesis: post-training compression near ceiling; need retraining. | LittleBit is a QAT method. Exactly the class the synthesis said we'd need to relax the constraint toward. |

LittleBit is the cleanest single continuation of the archived SVD track
given the 2026-04-19 synthesis. Whether it beats the archived floor at
any operating point is an open empirical question.

## 10. Open questions and next actions

Ranked by cost × information value.

### 10.1 Cheap math sanity checks (hours, single matrix)

1. **Numerically verify Prop. 1 on a toy matrix.** Pick a 256 × 256
   matrix, generate random `U_sign`, `V_sign`, random `h, g, ℓ` at
   `r = 16`. Compute both forms of `Y`. Assert bit-for-bit equality
   modulo FP rounding. Catches any symbol-flip in our reconstruction.
2. **Reconstruct Dual-SVID on a single matrix from our 7B
   calibration dump.** Measure `||W − Ŵ_{pri,0}||_F / ||W||_F`
   before any QAT. This establishes the initial point quality, which
   is the floor QAT has to start from. Directly comparable to CALDERA's
   initial-point numbers from Stage 1.
3. **Sweep rank-1 separability of `|U'|` from the same matrix's SVD.**
   Compute `|U'|` from truncated SVD at various `r`, run rank-1 SVD,
   report residual energy. If separability is good (say >95% of
   Frobenius captured), Dual-SVID is well-founded empirically; if it's
   bad (say <70%), the initializer is weak and QAT is doing
   essentially all the work.

Expected wall time: an afternoon. No GPU needed for 256×256 toy; the
7B matrix fits on CPU in float64.

### 10.2 Reproduction prereqs (days, full model)

1. **QAT trainer.** A minimal implementation of the sign-forward
   SmoothSign-backward pass plus the KL + MSE loss. Re-uses our
   `venv-research/` stack.
2. **Teacher-student pipeline.** For Qwen 0.5B → LittleBit-Qwen 0.5B,
   the teacher is the FP16 model, student is the LittleBit model. Our
   existing calibration corpus works as the training data.
3. **Acceptance bar.** PPL at `r = 256` beats our archived `r = 512`
   FP16 floor of PPL 86? If yes, LittleBit has unlocked a regime that
   our SVD sweep couldn't reach. If no, the retraining constraint was
   not what was blocking us.

### 10.3 Kernel-level questions (harder to answer without implementation)

1. **Batch-1 decode speedup.** Paper claims 11.6× at 0.1 BPW but
   doesn't break out batch=1. Our Cloudflare investigation (2026-04-19)
   established batch-1 is memory-bandwidth-bound on consumer cards. A
   factored-binary GEMV kernel has to demonstrate it stays
   bandwidth-efficient per-token at batch=1, not just at batch=N.
2. **GGUF support.** LittleBit's factored-binary layout is not any
   existing `llama.cpp` tensor type. Upstream-deployment (our fork's
   invariant, from `speculative_decoding.md`) requires either an
   upstream loader PR or GGUF-side adaptation. Until a kernel path
   exists upstream, LittleBit is research-only by our deployability
   rules — same status as EAGLE / Medusa.

## 11. Summary finding

- **The compression is real and storage math is exact.** Sub-1-BPW
  comes from decoupling parameter count (via rank `r`) from weight
  count, combined with 1-bit precision on the factors.
- **The quality recovery is empirical.** Dual-SVID gets QAT to a
  warm start; QAT + multi-scale compensation does the actual
  quality work. Neither the initializer nor the compensation form
  is derived from optimality; both are well-motivated design choices.
- **Prop. 1 is exact.** The efficient forward form is algebraic, so
  kernel speedups are about packing binary ops efficiently — not
  about any approximation.
- **Batch-1 efficacy is unanswered.** Paper doesn't give batch-1
  numbers; our hardware profile makes this the decisive question.
- **Deployment under our upstream-only invariant is not possible
  today.** LittleBit's tensor layout has no upstream loader. Any
  reproduction stays in this fork (or in training-only code) until
  upstream ships a factored-binary tensor type or a loader
  extension.

Next action: §10.1 (math sanity checks). These are cheap enough that
verifying the paper's math on a single matrix is lower friction than
continuing to reason about its implications in the abstract.

---

## 12. Sanity-check results (2026-04-21)

Ran the §10.1 checks via
[`littlebit_sanity.py`](littlebit_sanity.py). Raw output in
[`littlebit_sanity.json`](littlebit_sanity.json). Target matrix:
**Qwen 2.5 0.5B, `model.layers.12.mlp.gate_proj`**,
shape `[4864, 896]`, `||W||_F = 42.53`.

### 12.1 Proposition 1 is numerically exact — confirmed

Toy verification on random `U_sign, V_sign, h, g, ℓ, X` with `d_out = 128,
d_in = 96` at multiple ranks:

| `r` | max abs diff | rel diff |
|---:|---:|---:|
| 4 | 7.8e-14 | 8.5e-16 |
| 16 | 9.9e-14 | 6.5e-16 |
| 64 | 2.8e-13 | 7.9e-16 |
| 256 | 5.1e-13 | 6.5e-16 |

All under 1e-10 absolute and at floating-point epsilon relative.
Confirms Eq. 5 is algebraic: the efficient form is bit-for-bit equal
to the naive materialize-then-matmul modulo FP rounding.

### 12.2 Dual-SVID initial point is strictly worse than my framing

Rank sweep on the real matrix (4864 × 896):

| rank | BPW | fp-SVD err | DualSVID err | sep \|U\| | sep \|V\| |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.026 | 0.984 | 0.991 | 0.752 | 0.693 |
| 8 | 0.032 | 0.973 | 0.988 | 0.696 | 0.661 |
| 16 | 0.042 | 0.955 | 0.981 | 0.663 | 0.650 |
| 32 | 0.064 | 0.925 | 0.970 | 0.646 | 0.635 |
| 64 | 0.106 | 0.874 | 0.952 | 0.636 | 0.630 |
| 128 | 0.191 | 0.791 | 0.924 | 0.630 | 0.627 |
| 256 | 0.360 | 0.652 | 0.881 | 0.626 | 0.628 |
| 512 | 0.700 | 0.416 | 0.824 | 0.624 | 0.630 |

Columns: `fp-SVD err = ||W - U_r·V_rᵀ||_F / ||W||_F` (the archived
SVD baseline), `DualSVID err = ||W - Ŵ_{pri,0}||_F / ||W||_F` (the
post-init, pre-QAT reconstruction), `sep = σ_1² / Σσ_i²` of the
respective magnitude matrix.

### 12.3 What this means

**At `r = 512` (the archived SVD baseline operating point):**
- Optimal rank-512 FP-SVD captures 58% of matrix energy
  (err = 0.42).
- Dual-SVID at the same rank captures only 18%
  (err = 0.82).
- **Binarization + three-vector scale compensation throws away
  ~75% of the information rank-512 FP-SVD had access to.**
  `(0.82² - 0.42²) / 0.82² ≈ 0.74`.

**The rank-1 separability assumption holds partially.** `|U'|` is
62–75% rank-1 separable depending on `r`, plateauing at ~62% for
`r ≥ 128`. The paper's implicit assumption that `|U'|_{ik} ≈ h_i · ℓ_{u,k}`
captures only about five-eighths of the actual magnitude variation.
The remaining ~38% is discarded in Dual-SVID.

**This revises the framing in §5.2 more strongly than I expected.**
The doc already said "expect QAT to do most of the work." The
numbers say **QAT must recover essentially everything** — the
initializer is not a useful warm start in any quantitative sense.
At `r = 512, d = 896`, we're starting from a matrix whose
reconstruction has 82% relative Frobenius error. That's roughly
the error you'd get from returning a rescaled random matrix.

### 12.4 Consequences for reproduction

1. **The paper's reported end-to-end quality is entirely a QAT
   result.** There is no "initialization magic"; Dual-SVID is a
   symmetry-breaker, not a compressor. This matters because
   reproduction effort is dominated by the QAT training budget,
   not by the init math.
2. **The rank-1 separability claim (Eq. 7) is empirically weak
   but not invalid.** 62–75% separability is "real structure"
   even if it's not close to 100%. An ablation that swaps
   Dual-SVID for a straight `h, g, ℓ = 1` init would isolate how
   much the warm-start actually helps QAT convergence vs just
   providing sign patterns.
3. **Caveat on target-model scale.** This test was on Qwen 2.5
   0.5B — our archived SVD baseline model, chosen for direct
   comparability. The paper headlines numbers on Llama2-13B and
   above. Trained weights at larger scale may have flatter
   singular spectra (more rank needed to explain a given fraction
   of energy) or stronger multiplicative structure in their
   magnitudes. Would not be surprised if `sep |U|` climbs to
   ~0.8 at 13B. Worth re-running on a 7B matrix (we have the
   extraction script) before concluding anything global.
4. **The BPW axis matches the paper's operating points.** At
   `r = 256`, our BPW comes out to 0.36 on `d = 896`; the paper
   hits 0.1 BPW because they run on `d = 5120`. Same rank, same
   format, but the per-weight bit cost scales with `2r/d`. This
   confirms the §3.1 reconstruction.

### 12.5 Open question newly raised by this data

The ratio `(DualSVID_err² − fp_SVD_err²) / (1 − fp_SVD_err²)` measures
what fraction of the *available* rank-`r` information Dual-SVID
discards. From the table:

| `r` | discarded fraction |
|---:|---:|
| 64 | 0.78 |
| 128 | 0.78 |
| 256 | 0.77 |
| 512 | 0.77 |

**Remarkably rank-independent.** Dual-SVID consistently throws away
~77% of the rank-`r` subspace no matter how much is available.
Implication: the compression-via-binarization loss is a fixed
fraction of the pre-binarization information, not an absolute loss.
QAT's job is to recover that fraction; whether it succeeds at `r = 64`
vs `r = 512` is a question of how much absolute information is
present in the rank-`r` SVD to begin with (columns 3 and 4 of the
main table).

This reframes the reproduction question cleanly:

> **Does QAT recover the ~77% of rank-`r` information that Dual-SVID
> discards via sign truncation?**

If yes at `r = 256` on Qwen 0.5B, then LittleBit beats our archived
post-training SVD floor (PPL 86 at `r = 512`, FP16 factors) at
*lower* BPW (0.36 BPW at `r = 256` vs our archived `r = 512` setup
which was not BPW-measured since we didn't binarize). That would be
the clean win condition.

### 12.6 Next concrete action

Given the above, the priority ordering in §10 needs slight
revision:

- **§10.1.1 (Prop. 1 toy)** — DONE, confirmed.
- **§10.1.2 (Dual-SVID init quality)** — DONE, finding is "very
  bad init, QAT carries it all."
- **§10.1.3 (rank-1 separability sweep)** — DONE, separability ≈ 0.63
  at plateau.
- **NEW: §10.1.4 — run the same sanity on a 7B matrix** to test
  the "larger models have more multiplicative magnitude structure"
  hypothesis from §12.4 point 3. Cheap: one download + rerun.
- **§10.2 (QAT reproduction)** — unchanged in scope, but the
  expected training budget is now the whole story (not just the
  recovery margin).

If the 7B separability is also ~0.63, the reproduction bar gets
harder — QAT has to do even more work than the paper's headline
number suggests (because the paper's 13B/70B targets may not be
representative of sub-10B consumer-card workloads). If it climbs
to ~0.8, the paper's approach is better founded at larger scale
and the 0.5B regime is an unfortunate test bed.
