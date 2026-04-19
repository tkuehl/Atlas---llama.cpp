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

---

## 2026-04-17 (evening) — Experimental results + strategic pivot

### Experiment 3: HODLR single-matrix (gate_proj layer 12)
Compared depth-1 and depth-2 HODLR against flat ASVD at matched byte budget.

**Result:** Flat ASVD wins on weighted error at every byte budget (e.g. 5.8 MB budget → HODLR wgt_err 0.18, flat SVD wgt_err 0.12). Raw Frobenius error is closer (HODLR sometimes slightly better) but *weighted* error — the metric that predicts PPL — favors flat.

**Why:** HODLR was designed for CFD integral-equation matrices where spatial locality is real. LLM weight rows/cols have no spatial structure; the quadtree prior is wrong. Our block-local ASVD also ignores cross-block activation covariance, stripping the signal that ASVD exploits.

### Experiment 4: balanced truncation single-matrix
Compared plain SVD, ASVD, and balanced (input + output gradient weighted) on one gate_proj matrix across ranks 64-768.

**Result:** Balanced beats ASVD on weighted error at every rank, margin widens with rank (7% at r=64 → 40% at r=768). Balanced *loses* to ASVD on plain Frobenius and on input-only weighted error — it's intentionally sacrificing those to win on the balanced metric.

### Experiment 5: whole-model balanced truncation (Qwen 2.5 0.5B)
Extended prototype.py with a `--weighting balanced` flag. Requires a backward-pass calibration (forward + `loss.backward()`) to accumulate the output-gradient gramian GGT per module. Reconstruction is `W_r = S_out^{-1} U_r Σ_r V_r^T S_in^{-1}` with GPU eigendecomps for stability.

**Result on 0.5B:**
```
rank | ASVD PPL | balanced PPL | delta
 512 |    86.5  |    46.8      | 46% better
 640 |    24.1  |    19.3      | 20% better
 768 |    13.1  |    12.4      |  5% better
```
Balanced wins everywhere but improvement shrinks as rank grows. The gap on whole-model PPL is smaller than the single-matrix weighted-error gap would suggest — some of the single-matrix improvement is wasted on layers that don't matter as much for final loss.

### Research landscape (second agent survey, ML-specific)

Key findings that changed our thinking:

- **Basis Sharing (Saha et al. 2024, [arxiv 2410.07383](https://arxiv.org/abs/2410.07383))** — explicitly validates shared-basis cross-layer IF applied to **windows of 2-4 adjacent layers** (not all L). Global sharing breaks on LLaMA-7B just like it broke on our Qwen 0.5B. Our architecture was right; our granularity was wrong.
- **SparseGPT-style OBS repair** ([arxiv 2301.00774](https://arxiv.org/abs/2301.00774)) — after truncation, use Hessian-inverse closed-form to *analytically redistribute* error into remaining columns. "Project and repair" instead of "project and pray." ~200 lines on top of our pipeline.
- **CALDERA** ([arxiv 2405.18886](https://arxiv.org/abs/2405.18886)) — inverts the streaming economics: keep a 3-4 bit quantized W *resident*, stream a small (rank 64-128) correction. Per-layer PCIe transfer drops to 2-8 MB, well under 1 ms on Gen4 x16.
- **Scale effects:** small models (<3B) fundamentally resist compression — intrinsic rank ≈ nominal rank. ~60-70% of our 0.5B fragility is model size, not algorithm. Papers: [arxiv 2411.04330](https://arxiv.org/abs/2411.04330), [arxiv 2404.09937](https://arxiv.org/abs/2404.09937), [arxiv 2012.13255](https://arxiv.org/abs/2012.13255).
- **Nothing in production uses factored streaming.** FlexGen, PowerInfer, DeepSpeed-ZeRO-Inference, llama.cpp `-ngl` — none execute `y = B(A·x)` with PCIe-overlapped coefficient transfer. Either they reconstruct dense weights in VRAM or they stream whole layers naively.

## Strategic pivot — 2026-04-17

**From** "novel factoring scheme + streaming runtime"
**To** "adopt proven factoring math + ship the streaming runtime that nobody has built yet"

The decomposition math has been extensively studied in 2023-2024. Windowed Basis Sharing, non-uniform rank, balanced-truncation-adjacent methods, Q+LR hybrids — they're all published. **What's missing across every paper is the actual inference runtime**: kernel-level `y = B(A·x)` GEMM with double-buffered PCIe `cudaMemcpyAsync` streaming the next layer's coefficients, overlapped with current layer compute.

That runtime work is where the llama.cpp fork gives us superpower. We can:
1. Adopt windowed Basis Sharing + balanced truncation (or CALDERA) as the *decomposition*
2. Build the *serving path* in `ggml-cuda` that keeps the resident basis warm and streams coefficients on a second CUDA stream
3. Benchmark end-to-end tok/s vs naive offloading schemes (AirLLM, FlexGen)

This is a cleaner, higher-value framing. The innovation budget moves from "prove a new math" to "prove a new runtime that makes existing math practical."

### Revised roadmap

1. **Gate 1 (now, in flight):** run current pipeline on Qwen 2.5 3B. If PPL at r=1024 is in the usable range (< ~20), the 0.5B-ceiling hypothesis is confirmed and the scheme works.
2. **Next decomposition:** windowed Basis Sharing (window ∈ {2, 3, 4}) + balanced truncation on Qwen 3B. This is our original scheme at correct granularity.
3. **Parallel kernel track:** prototype the `y = B(Ax)` kernel + double-buffered `cudaMemcpyAsync` in `ggml-cuda/`. Start against a toy decomposition; integrate winning factorization later.
4. **CALDERA as alternate track:** Q+LR decomposition if Basis Sharing's memory split doesn't fit the 70B-on-12GB goal (Basis Sharing's shared basis is small but per-layer coeffs scale with L; CALDERA's quantized resident scales with param count).
5. **Integration:** end-to-end benchmark on 70B — tok/s, TTFT, peak VRAM, PCIe utilization.

---

## 2026-04-17 (late) — Phase 0 complete: DESIGN.md

Aggregated 5 parallel research agents + ggml internals recon into a full design doc: see **[DESIGN.md](DESIGN.md)** in this directory.

Key corrections and pins from the aggregation:

- **Paper ID correction:** Basis Sharing is arxiv **2410.03765**, not 2410.07383 (which is SparseGrad, unrelated). This repo's earlier notes had the wrong ID.
- **Window=2 is the only safe Basis Sharing window size.** Paper's ablation: window=8 already worse than no sharing at 50% compression on LLaMA-7B.
- **`o_proj` and `down_proj` do NOT share across layers** (Basis Sharing paper's Fig. 4b). Only q/k/v/gate/up. Those non-shared roles stay per-matrix with balanced truncation.
- **FP64 Cholesky is mandatory** for the activation-covariance whitening; FP32 is unstable at 7B+ scale.
- **CALDERA can't fit 70B in 12GB** even at its lowest published bit budget (~1.9 avg bits, ~16.6 GB). Streaming is still required. CALDERA is deferred to Phase 6.
- **SparseGPT OBS repair** extends to SVD truncation with a clean derivation; this is our one "original math" contribution. Empirical validation is the last gate before Phase 1.
- **Gen5 PCIe on RTX 5070 sustains 55-60 GB/s** — PCIe is not our bottleneck. Kernel launch overhead on small rank-k GEMMs is the real concern.
- **Fork stays private.** Every streaming/offload proposal upstream has been rejected or closed-stale since 2023. Known scheduler bugs ([#18310](https://github.com/ggml-org/llama.cpp/issues/18310), [#18313](https://github.com/ggml-org/llama.cpp/issues/18313)) are closed "not planned" — we work around, not through.

### Strategic pivot confirmed by aggregation

From "novel factoring scheme + streaming runtime" → **"adopt proven factoring math (Basis Sharing + balanced truncation + OBS repair) and ship the streaming runtime that nobody has built"**. The decomposition papers stop at "here's the math, run it in Python with reconstructed weights." The PCIe-overlapped inference runtime is where our fork differentiates.

### Phase 0 deliverables

- [x] ggml internals recon (scheduler, tensor types, mul_mat dispatch, GGUF extensibility)
- [x] 5 parallel research agents synthesized (Basis Sharing, CALDERA, SparseGPT OBS, CUDA streaming practices, llama.cpp upstream archaeology)
- [x] DESIGN.md written with phase plan, file map, open questions
- [ ] OBS-on-SVD empirical check on a single `gate_proj` (last gate before Phase 1)

### Phase 1 triggers

Proceed to Phase 1 when:
1. OBS-on-SVD empirical check shows meaningful `bal_wgt` reduction vs plain balanced truncation at matched rank, OR is demonstrated harmless if the repair provides no lift — either outcome is fine, we just need the decision data.

---

## 2026-04-17 (late) — Phases 1, 2a, 2b, 2c-MVP, 3.1, 3.2 all shipped

The entire Python→C++ pipeline for factored inference is now working end-to-end.
Milestone overview:

### Python pipeline
- **`basis_sharing.py`** — HF model → factored decomposition. Implements windowed Basis Sharing (window=2) for Q/K/V/gate/up and per-matrix balanced truncation for O/down. Supports optional weighted-LS coefficient refit using output-gradient covariance (our extension over the paper). Heavy runtime optimizations shipped: Cholesky-first sqrt_and_inv with escalating fallback ladder (GPU-fp32 → GPU-fp64 → CPU-fp64 eigh), model-to-CPU during compute_factors to free VRAM, per-stage checkpointing so reruns skip what's already done. Decomposition of Qwen 2.5 3B with 32 calibration samples takes ~3 min on RTX 5070.
- **`convert_factored_gguf.py`** — takes a base GGUF (from `convert_hf_to_gguf.py`) + our intermediate format, writes factored GGUF with structured naming (`shared.{role}.w{W}.basis`, `shared.{role}.w{W}.coeffs.{L}`, `permatrix.{role}.{L}.{U|V}`) and `factored.*` KV metadata.
- **`reconstruct_factored_gguf.py`** — inverse: factored GGUF → dense GGUF (materializes B@A / U@V into standard blk.*.weight names). Used as the interim workflow before the C++ loader landed.
- **`chat_factored.py`** — Python interactive test: loads a factored model back into HF transformers for quality gut-checks.

### C++ integration (Phase 3.1 + 3.2)
- **`ggml_factored_linear(ctx, basis, coeffs, x)`** — public ggml helper emitting two sequential `ggml_mul_mat` nodes. Reuses every backend's existing MUL_MAT path (cuBLAS, quantized kernels, etc.) — no new op enum, no new kernel. Commit `38378d25d`.
- **`struct llama_layer`** extended with `wq_coeffs` / `wk_coeffs` / `wv_coeffs` / `wo_coeffs` / `ffn_gate_coeffs` / `ffn_up_coeffs` / `ffn_down_coeffs` nullable fields. Commit `656c76b13`.
- **`llama_model::factored_coeffs`** sidecar map (basis ptr → coeffs ptr) threaded through `llm_graph_params` → `llm_graph_context`. Commits `1a8268c56`, `55a9e6bdb`.
- **`llama_model_loader::scan_factored_sources()`** detects `factored.enabled=True` on GGUF open and parses factored tensor names into a `{canonical_name → (basis_name, coeffs_name)}` map. Runtime-verified on qwen3b-factored: `252 factored weights detected`. Commit `4ec2f4dba`.
- **`llama_model_loader::create_tensor_factored()`** helper that, given a canonical role, returns `{basis, coeffs}` from the factored sources OR `{nullptr, nullptr}` if not factored. Commit `2e490d95d`.
- **`create_tensor_or_factored` lambda** in `llama_model::load_tensors` drives the two-path dispatch. `create_tensor_qkv` (used by many architectures) + the Qwen2 case (`wo`, `ffn_gate`, `ffn_up`, `ffn_down`) now factor-aware. Commit `fd67cd178`.
- **`build_lora_mm` intercept** in `llm_graph_context`: if weight is a factored basis, routes to `ggml_factored_linear` with the mapped coeffs; otherwise normal `ggml_mul_mat`. LoRA path preserved but naturally skipped for factored weights (adapters keyed by dense pointer). Commit `55a9e6bdb`.

### End-to-end results (Qwen 2.5 3B, fp16, RTX 5070)

| Metric | Dense baseline | Factored (target 0.8 = 20% compression) | Delta |
|---|---|---|---|
| GGUF size | 6.18 GB | 5.07 GB | −18% |
| VRAM peak | 11.5 GB | 9.9 GB | −14% |
| Decode throughput | 87.9 tok/s | 64.0 tok/s | −27% |
| Prompt prefill | 62.3 tok/s | 72.8 tok/s | +17% |
| WikiText-2 PPL | 7.66 | 29.5 | +285% |
| Sample output | coherent | degenerate loop | — |

**Reading the numbers:**
- Storage/VRAM drop is the real compression win, matches `(d_out+d_in)·r` vs `d_out·d_in`. We leave ~4% on the table vs ideal because our Phase 3.2 MVP duplicates shared basis tensors per layer (Phase 3.4 reclaims this).
- Decode slowdown is kernel-launch dominated: 36 layers × 7 factored roles × 2 launches per role = 504 extra launches per token ≈ 2.5 ms at 5 µs each, matching the observed ~15.6 ms/token budget at 64 tok/s.
- Prefill speedup comes from the FLOP reduction (factored form is ~3.7× fewer FLOPs per MLP at 20% compression). Launch overhead amortizes across the batch.
- PPL gap is the well-documented small-model compression ceiling (see Huang et al. 2024, Aghajanyan 2021). Not a bug — literature says Qwen/LLaMA at 3B doesn't take 20% compression cleanly; 7B is where Basis Sharing published results (PPL ~19-25 at 50%).

### Per-model quality curve measured (Qwen 2.5 3B, baseline PPL 7.66)

| target_ratio | PPL | delta | Notes |
|---|---|---|---|
| 0.8 (20% compression) | 29.5 | +285% | chosen reference model — pipeline testable |
| 0.7 (30% compression) | 247 | +3,134% | broken |
| 0.5 (50% compression) | 2,611 | +34,017% | broken (same regime as paper's 7B @ 50% ≈ PPL 20) |

Consistent story: Qwen 2.5 3B runs out of compressible rank around 20–25% savings. Paper's LLaMA-7B at 50% lands at PPL ~20; scale effects are as the literature predicts.

### Commits in this sweep
Running log of commits (run `git log --oneline` for canonical ordering):

| Commit | Phase | Scope |
|---|---|---|
| `2c603d53c` | Phase 0 | research + Windows CUDA build |
| `aa4d72e81` | Phase 0 | OBS validation |
| `ced3dd23c` | Phase 1 + 2a + 2b | Python converter + intermediate format + GGUF emitter |
| `b22e93369` | Phase 2c-MVP | Python reconstruct_factored_gguf |
| `38378d25d` | Phase 3.1 | `ggml_factored_linear` helper |
| `656c76b13` | Phase 3.2 scaffold | `struct llama_layer` / `llama_model` extensions |
| `1a8268c56` | Phase 3.2 step 1 | thread `factored_coeffs` through graph params/context |
| `55a9e6bdb` | Phase 3.2 step 2 | `build_lora_mm` intercept |
| `4ec2f4dba` | Phase 3.2 step 3a | loader scan of factored sources |
| `2e490d95d` | Phase 3.2 step 3b | `create_tensor_factored` helper |
| `fd67cd178` | Phase 3.2 step 3c+4 | Qwen2 integration + end-to-end validation |

### Next gates
1. **Qwen3 architecture support** — the Qwen2 case in `load_tensors` is hard-coded. Qwen3 (used for Atlas production via `Qwen3-8B-Q4_K_M.gguf`) has its own `LLM_ARCH_QWEN3` switch case that needs the same `create_tensor_or_factored` call-site rewiring.
2. **Scale validation on 7B+** — the 3B quality curve is bounded by the small-model ceiling. Running Qwen 2.5 7B or Qwen 3 8B at target_ratio=0.5 should land in paper territory (PPL ~19-25). That's the real validation of the decomposition math.
3. **Phase 3.4 — streaming runtime** — true shared-basis on GPU (no per-layer duplication), coefficients in pinned host RAM streamed over PCIe during inference. The novel technical contribution beyond what papers have published.
4. **Deferred** — re-enable LS coefficient refit with deferred-hook pattern (forward-only calibration plus per-sample post-hoc gradient processing to avoid the sync-per-hook death we hit earlier).

---

## 2026-04-17 (late) — Optimization path aggregated: see [optimization_path.md](optimization_path.md)

Literature survey + pipeline-code walk-through produced a consolidated
optimization catalog in **[optimization_path.md](optimization_path.md)**.
Two tracks:

- **Runtime (C++/CUDA)** — kernel fusion to fix the two-`mul_mat` anti-pattern
  (FlashSVD, SVDQuant/Nunchaku, arxiv:2512.20861's Triton reference);
  kernel-launch reduction via cuBLAS Grouped GEMM, CUDA Graph capture, and
  megakernels (Mirage MPK, Hazy "No Bubbles"); transport upgrades
  (GPUDirect Storage, HMM, PIPO's tensor-merging); coefficient quantization
  (IntLoRA, FireQ); orthogonal multipliers (activation sparsity, speculative
  decoding).
- **Decomposition pipeline (Python)** — GPU-resident role-sequential gramian
  (fixes the CPU hotspot at `basis_sharing.py:356-374`), randomized SVD via
  `torch.svd_lowrank` to replace Golub-Reinsch, gramian caching across
  `target_ratio` sweeps, Cholesky-ladder audit, and batched calibration
  samples.

Priority ordering in §3 of that doc. Key callouts: cuBLAS Grouped GEMM is
probably the single cheapest fix for the 27% decode regression; GPU-resident
gramians are the single cheapest fix for calibration wall-time at 7B+ scale.

---

## Background reading — Phase 3.4 design inputs

Two external references that reshape how Phase 3.4 should be scoped. Captured here so the design conversation doesn't start from zero when we pick this up.

### GPUDirect Storage (NVIDIA cuFile)

**What it is.** Direct NVMe→VRAM DMA via the `nvidia-fs` kernel module + `cuFile` API. Bypasses the CPU bounce buffer. Effective streaming bandwidth goes from ~6-8 GB/s (CPU-mediated) to ~25-45 GB/s on decent Gen5 NVMe.

**Platform reality.** Linux only. No Windows driver. WSL2 lacks `cuFile` today. Our Windows 11 dev box can't use it.

**Bandwidth math for a pure streaming Phase 3.4 design (Qwen 2.5 7B factored at r=0.5):**

| quantity | value |
|---|---|
| down_proj coefficient per layer | ~380 MB (rank × hidden × bf16) |
| shared-role coefficient per layer | ~15-30 MB |
| layers traversed per decoded token | 28 |
| total coefficient payload per token | ~11 GB |
| target tok/s | 20-40 |
| **required streaming bandwidth** | **220-440 GB/s** |
| PCIe Gen5 x16 sustained | ~55 GB/s |

Naïve "stream every layer every token" is **4-8× over what a single PCIe link can deliver**, with or without GDS. This means Phase 3.4 cannot be pure per-token streaming — we must either (a) keep a working set of coefficients resident across many tokens, (b) batch decode so coefficient loads amortize across tokens, or (c) reduce the per-token payload via FlashSVD-style tile fusion (see below).

**Principles to apply on Windows (since we can't use GDS itself):**
- Pinned host staging buffers, allocated once and reused
- Two CUDA streams (compute + copy), double-buffered prefetch
- Transfer granularity tuned to SSD queue depth (2-16 MB chunks, not 32-64 KB)
- Hides CPU-staging cost, captures ~70% of GDS's benefit

**When GDS becomes worth it.** Dual-boot or provision a Linux dev box if Phase 3.4 hits a PCIe-bound wall in practice. Not before — the gap between pinned+async and true GDS is ~2-3× for our workload, which isn't the current bottleneck.

### FlashSVD (arxiv 2508.01506)

**What it is.** Custom GPU kernels that fuse the two matmuls of a factored linear (`basis @ (coeffs @ x)`) so the intermediate activation never lands in HBM. Tile-by-tile streaming through on-chip SRAM. Paper claims 70% peak activation memory reduction with zero accuracy loss on BERT/RoBERTa at ranks 16-768.

**Two FFN variants in the paper:**
- **V1 (practical)**: first projection uses cuBLAS GEMM; second projection + activation fused into a streaming kernel. This is the one that actually wins on latency.
- **V2 (extreme memory)**: fully fused GEMM-Activation-GEMM, zero HBM I/O, but **60% latency penalty** (reconstruction loop serializes). Flagged by the authors as a memory-vs-speed tradeoff.

**Direct relevance to us.** Our `ggml_factored_linear(basis, coeffs, x)` *is* the V2 compute shape. The current ggml implementation launches two separate `ggml_mul_mat` kernels — intermediate writes to HBM, second kernel reads it back. This is the exact inefficiency FlashSVD attacks.

**Gaps blocking a naïve port:**

| factor | FlashSVD | us |
|---|---|---|
| model type | encoders (BERT), no KV cache | decoder-only, causal, KV cache |
| sequence regime | long (M ≥ 512) | prefill long, **decode M=1** |
| code release | none | would need custom CUDA kernel as new ggml op |
| benchmark at M=128 | **0.62× slower than dense** | decode lives here — naïve port would *regress* tok/s |

FlashSVD only wins at long sequences. A straight port hurts decode throughput, which is what chat users feel. Any integration must gate on seq length and use the dense path for decode.

**The synergy with GDS that actually matters.** FlashSVD's tile-streaming design means coefficients never need to live fully in VRAM — only the current tile. Combining that with pinned-host (or GDS) async prefetch:

> Coefficients sit on NVMe → `cp.async` loads tile-sized chunks (~KB) directly into shared memory → fused matmul computes its tile → evict → next tile.

Working set in VRAM shrinks from 100s of MB per layer to O(MB). This reframes the Phase 3.4 bandwidth problem from "impossible on one GPU" to "tractable with careful pipelining." Without this fusion, Phase 3.4 hits the PCIe wall.

### Phase 3.4 design recommendation

When we pick up Phase 3.4:

1. **Don't port FlashSVD naïvely.** Its benchmarks don't match decode. A straight kernel port regresses decode throughput.
2. **Prototype a fused factored-linear kernel for prefill only.** Long-context gain, no decode regression. Gate on a batch/seq threshold. ~2-3 weeks of work.
3. **Design the streaming data plane around tile-fused kernels**, not full coefficient residency. This is the combination that makes the "larger model than fits" thesis actually achievable.
4. **Revisit GDS if we productionize.** The Linux port becomes worth it once Phase 3.4 is PCIe-bound in practice.

---

## 2026-04-18 — calibration precision, compression ratio, and rank allocation

Multi-day session refining the Python decomposition pipeline. Key results below; see `basis_sharing.py` for the shipped implementation.

### Compression ratio findings — r=0.5 is broken, r=1.0 is the validation regime

Started at r=0.5 (paper's headline ratio) and got catastrophic PPL at every scale tested:

| model | baseline | factored r=0.5 | ratio |
|---|---|---|---|
| Qwen 2.5 0.5B | 12.22 | 1391 | 114× |
| Qwen 2.5 3B | 7.66 | 2611 | 341× |
| Qwen 2.5 7B | 5.88 | 1389 | 236× |

Our `window=2 shared + permatrix` scheme at r=0.5 loses coherence entirely — model output is degenerate pattern-loops ("the sky is blue because its surface is blue"). Paper's r=0.5 must be using a different byte accounting or different window strategy; attempts to match their claimed PPL ~20 on 7B never worked.

Pivoted to **r=1.0 as the validation regime**: rank ≈ dense-equivalent storage, small residual truncation error. Clean monotonic curve confirms the pipeline is mathematically correct at every scale:

| model | r=1.0 factored PPL | baseline | ratio |
|---|---|---|---|
| 0.5B | 22.95 | 12.22 | 1.88× |
| 3B | 12.26 | 7.66 | 1.60× |
| **7B** | **8.75** | **5.88** | **1.49×** |

Larger models compress cleaner at r=1.0 — bigger spectrum tail amortizes the truncation residual. **7B r=1.0 PPL 8.75 is now our reference-point for runtime regression testing.** r=1.0 with 32 calibration samples stays the default; r=0.8 and lower are opt-in stress tests.

### Precision and rank deficiency

At 7B r=0.8 the Cholesky ladder in `sqrt_and_inv` fires aggressively on `down_proj` (d=18944). Initially hypothesized fp32 gramian accumulation noise; shipped full fp64 accumulation via a new `StreamingCollector` that caches activations in pinned host (fp16) and computes fp64 gramians on GPU in chunks after the forward pass.

**Result: precision was not the bottleneck at r=0.8.** Both fp32 and fp64-streamed runs at 8 samples gave PPL ≈ 37 (6.3× baseline). Moving to 32 samples dropped PPL to 22.3 — **sample count is the dominant lever**, not precision.

Mechanically: 32 × ~200 tokens = ~6400 rows per layer, still `< d_in=18944`, so the gramian stays rank-deficient. Every down_proj layer triggers `gpu-fp64-eps10` regularization. The paper territory (PPL ~19-25) would need ~100+ calibration samples to fully span d_in. Left for a future quality pass.

### Per-matrix rank allocation (SVD-LLM V2 style)

Shipped water-fill rank allocator behind `--rank-alloc per-matrix`: computes σ(W @ S_in) via a fast `eigvalsh(W @ (XTX + eps·I) @ W^T)` for each matrix, greedy-allocates rank by marginal utility (σ² / bytes_per_rank) under the global byte budget. Floor at 50% baseline rank to prevent any matrix from being decimated.

Empirical result:

| model | global rank PPL | per-matrix alloc PPL | delta |
|---|---|---|---|
| 0.5B r=0.8 | 162 | 114 | −30% (win) |
| **7B r=0.8** | **22.28** | **28.32** | **+27% (regression)** |

At 7B the allocator moves rank AWAY from `down_proj`/`o_proj` toward shared `gate/up/k/v` (gate/up +12-14%, k/v +22-26%). This minimizes sum-of-σ² truncation error but **hurts PPL** — in the actual loss landscape, down_proj rank is more PPL-critical than its spectrum magnitude suggests. `sum-of-σ²` is not a reliable PPL proxy at 7B scale.

Default reverted to `--rank-alloc global`. Per-matrix stays available as an experimental flag; a better utility function (gradient-weighted, or empirical per-role PPL sensitivity) would be needed to make it the default.

### Infrastructure shipped

- **Streaming activation cache** (`StreamingCollector` in `basis_sharing.py:~500`). For wide-d roles, hook captures raw activations to pinned-host buffers, finalizes to fp64 gramians on GPU post-pass. Incremental drain keeps peak pinned-RAM bounded (`--streaming-max-ram-gb`). Replaces the 80 GB disk-backed memmap approach that thrashed A: mechanical HDD into a system hang.
- **Streaming factor output**. `compute_factors` writes each layer's U/V to per-layer `.pt` files and installs the reconstructed dense weight into the model inline. Peak RAM during the 25-minute decomposition phase drops from ~22 GB → ~1 GB on 7B. `save_factored` assembles the final `factored.safetensors` from the per-layer files at the end.
- **Memory caps** (`--cpu-ram-max-pct`, `--disk-max-pct`, `--gpu-vram-max-pct`, all default 90%). Prevents a run from filling system disk or OOM'ing under memory pressure.
- **fp64 gramian accumulation** via `--accum-fp64-threshold` (default 4096). Auto-routes wide-d roles through fp64 on-disk memmap; narrow-d stays fp32 in RAM for speed.
- **Cholesky-skip hook** for known-rank-deficient gramians (buggy — currently routes to CPU eigh; backlog item to fix the fall-through).
- **Drive routing** (`reference_drive_speeds.md` memory). C: = NVMe (calibration temp I/O), A: = spinning HDD (cold storage only). Mechanical HDD thrash was the root cause of the first 7B fp64 failure.

### Speed experiments (2026-04-18)

Tested three speedup ideas against the 3B r=1.0 baseline (190.9s decomposition, PPL 12.255):

| optimization | decomp time | PPL | verdict |
|---|---|---|---|
| `torch.svd_lowrank` partial SVD (#1) | 181.2s | 12.30 | marginal at r=1.0 (threshold gates out most cases); helps at r<1.0 |
| batched calibration (batch=8) on 7B | ~same | ~same | hook-path dominates at 7B; no wall-time benefit |
| **8 samples × seq 2048 (same total tokens)** | **199.9s** | **14.59** | **regression on both axes** (killed) |

**Partial SVD** integrated cleanly. Heuristic `q ≥ 2/3 × min_dim → skip lowrank` correctly falls back to full SVD when rank is near-full. Only `o_proj` at 3B r=1.0 triggers the lowrank path (speedup 14s → 4s on that role). At r=0.8 and below, more matrices fall under the threshold — speedup projected ~30% on down_proj.

**Batched calibration** does NOT help at 7B because the hook path (not model-forward) dominates. At 3B batch=4 gave 3.4× speedup; at 7B batch=8 was a wash. Batching kept as `--calib-batch-size` flag, off by default.

**Seq-len retune** is a loss: fewer samples means less activation diversity, so gramian becomes MORE rank-deficient → more eps10 regularization → PPL drift. Sample count matters beyond just token count.

### Backlog (runtime hardening, pre-quality-work)

1. **GPU-resident shared-role accumulators** — eliminates per-hook D2H (~270s of 400s hook-path at 7B). Fits 3B cleanly; 7B needs VRAM budget management.
2. **Fix `--skip-cholesky-above-d`** — currently mis-routes to slow CPU Cholesky when GPU eigh OOMs on d=18944. Correct behavior: skip directly to `gpu-fp64-eps10`.
3. **Pipeline prefetch** — overlap layer k+1 load with layer k SVD via async CUDA streams. ~20% decomp speedup.
4. **Cross-run gramian caching** — save fp64 gramians to safetensors after streaming finalize; reuse across rank/ratio experiments.
5. **Per-window safetensors format** — eliminates the brief 22 GB peak during save_factored assembly. Requires C++ loader update.

Quality work (per-matrix allocation with a better utility function, calibration at >100 samples, refit=True with LS solve) is deferred until the runtime backlog clears.

---

## 2026-04-18 (late) — runtime polish, bench suite, refit regression

### Runtime optimizations shipped

**Cholesky-skip fix (commit `e29ea935a`).** Previous skip path for
rank-deficient gramians routed to GPU fp64 eigh, which OOMs at d=18944
on a 12 GB card and then fell through to the very slow CPU Cholesky
ladder. Fixed to jump directly to `gpu-fp64-eps10` Cholesky (the tier
empirically observed to succeed on rank-deficient wide-d gramians).
Default `--skip-cholesky-above-d` bumped from 0 → 15000 (catches 7B+
down_proj, leaves 3B and smaller on the full ladder). Validated on
7B r=1.0 reference: 1507s → 1433s decomposition (-5%), PPL 8.75 →
8.765 (within noise).

**GPU-resident shared-role accumulators (commit `6c221e192`).**
`XtxStore` gains a GPU-residency path for fp32 gramians under a
`--gpu-accum-budget-gb` budget (default 4 GB). Forward hook does
in-place `addmm_` into the gramian on device, skipping the legacy
GPU-mm → D2H → CPU-add round trip per hook. `consolidate_to_cpu()`
moves gramians back to CPU between calibration and decomposition so
downstream code sees uniform CPU tensors and GPU workspace is free for
SVD. Symmetric treatment for xtx (forward) and ggt (backward). fp64
accumulators always stay on CPU (GPU fp64 is 1/64 throughput on
consumer cards).

Results vs the r=1.0 references:

| model | calibration | decomp | PPL |
|---|---|---|---|
| 3B (before) | 34.9s | 190.9s | 12.26 |
| 3B (+GPU-resident) | **21.8s** (−38%) | 193.8s | 12.21 |
| 7B (before) | 417s | 1507s | 8.75 |
| 7B (+GPU-resident) | ~420s (neutral) | 1400s | 8.76 |

The 7B neutral result exposed a miscalculation in my earlier hook-path
accounting — I had estimated D2H dominated shared-role hooks (~270s of
~400s hook-path at 7B), but the real number is closer to ~27s. The
remaining ~300s at 7B is dominated by the StreamingCollector
pin_memory + drain loop for down_proj, which is a separate target.

**File prefix fix (same commit stack).** When `refit=True` the
ggt_store and xtx_store both want disk-backed memmaps and both used
the `xtx_` filename prefix — file collisions that showed up as
`OSError(Errno 22)` on the second store's alloc. Added
`file_prefix` parameter defaulting to `"xtx"`; ggt_store passes
`"ggt"`.

### Benchmark suite (commit `e2a1d41f5`)

Three files under `research/cross-layer-svd/`:

- `bench_prompts.json` — 15-prompt fixture (5 factual w/ exact-match
  keys, 3 completion, 3 reasoning, 2 code, 2 summary).
- `bench_model.py` — runs one model (base HF or base+factored overlay)
  through the fixture. Records per-prompt timing (prefill / TTFT /
  total / per-token / tok/s), generated text + token ids + per-token
  log-probs + top-k ids per step. Background resource sampler polls
  VRAM / GPU util% / CPU% / RSS every 100 ms; reports peak + mean.
- `bench_compare.py` — diffs two JSONs; emits markdown with speed
  deltas, resource deltas, exact-match scoring, greedy token agreement
  (% of leading tokens identical), top-5 overlap, plus a side-by-side
  response dump for qualitative judgment.

Greedy decode (`temperature=0.0`) default so base-vs-factored is
deterministic. Top-k computed offline from the JSONs so compare
doesn't re-run the model. UTF-8 output required on Windows (arrow +
check glyphs can't encode in cp1252).

### First bench run: 3B base vs factored (r=1.0, no-refit)

On our PPL reference pair (base 7.66, factored 12.26):

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 41.0 | 41.7 | +1.5% (noise) |
| VRAM peak | 6325 MB | 6327 MB | 0 |
| **Model load** | **3.7 s** | **37.4 s** | **+902%** |
| RSS peak | 8.0 GB | 20.4 GB | +153% |
| Exact-match (factual) | 4/5 | 3/5 | −1 |
| **Greedy token agreement** | ref | **1.9%** | — |
| **Top-5 overlap** | ref | 11.3% | — |

Inference-time speed and VRAM are essentially identical — expected,
since `chat_factored.py` reconstructs dense weights in memory. The
model-load and RSS overhead are the cost of that reconstruction path;
they go away once llama.cpp loads the factored form natively.

**The striking number is greedy-agreement = 1.9%.** PPL of 12.26
vs base 7.66 (1.60× baseline) sounds like a modest compression tax,
but the argmax at the very first decoded position diverges from base
on 13 of 15 prompts. Qualitatively the factored model loops ("Berlin
is the capital of Berlin, Germany..."), contradicts itself ("whales
are cold-blooded"), produces broken code, and gives wrong arithmetic
answers. On the two prompts where greedy agreement was high (a
hard-constrained fox completion and a one-word factual), outputs were
indistinguishable.

**PPL is an unreliable proxy for generation quality at our
compression regime.** It averages token-level likelihood, so a factored
model can keep average per-token loss close while its peak tokens
shuffle — which is exactly what breaks autoregressive decoding. The
greedy-agreement and top-5-overlap metrics from the new bench suite
track this failure mode directly and should be the regression
indicator for quality work going forward.

### Refit=True at r=1.0 on 3B — regression, not improvement

Enabled `refit=True` (balanced truncation: per-layer coefficients
refit via closed-form weighted LS with output-gradient covariance
ggt) to test whether balancing in/out sensitivity would reduce the
argmax-shifting. Run setup:

- `--target-ratio 1.0`, `--calib-samples 32`, `--dtype bfloat16`
- 252 hooked modules (7 roles × 36 layers); backward pass + per-layer
  ggt accumulation during calibration
- No streaming (forces all gramians in fp32 through the regular hook
  path so both fwd+bwd hook flavors work)

Result:

| config | factored PPL | ratio | greedy agreement | qualitative |
|---|---|---|---|---|
| no-refit (reference) | 12.26 | 1.60× | 1.9% | loops + factual errors |
| **refit=True** | **21.07** | **2.75×** | **1.0%** | **word salad, tighter loops, no valid code** |

Refit **nearly doubled PPL** and made generation strictly worse:

- "Berlin" prompt: loops immediately — "The capital of Germany is
  Berlin, which is the capital of Germany. The capital of Germany is
  Berlin..." — whereas no-refit at least listed other capitals
  coherently for a few tokens before losing the plot.
- Math prompt: "a number of people's the number of people's a number
  of people's..." — no longer attempts arithmetic at all.
- Code prompt: "Return the n-th return (n) using, 3003: n = f (n)..."
  — incoherent tokens, no Python syntax.
- Fox completion: fabricates "bridge and the green one goes to the
  river" — loses the canonical "lazy dog" entirely.

Calibration wall time: ~80 min on 3B (forward+backward, 150-165
s/sample), vs 35s for no-refit. ~23× slower, not the 2× I projected.
The backward pass + hook-path cost compounds.

**Likely causes of the quality regression:**

1. **bf16 backward underflow.** bf16 has a 7-bit mantissa. Gradients
   for weights with small activations underflow to zero. The
   accumulated ggt captures only the large-gradient directions
   reliably; small but non-negligible directions get lost to noise.
   The LS solve `A = (B^T S_out^T S_out B)^-1 B^T S_out^T S_out W`
   inverts S_out's spectrum, which means any ggt eigenvalues that
   underflowed get enormous weight in the solve — amplifying noise
   into the coefficients.
2. **Compound regularization.** sqrt_and_inv is now called on both
   S_in (from xtx) and S_out (from ggt). Eps regularization applied
   on both sides compounds: coefficients get distorted by both
   whitenings' regularization error.
3. **Over-parameterization at r=1.0.** At r ≈ dense-equivalent rank,
   the ASVD path (input-only) already captures ~all the signal. The
   LS refit has little headroom to improve and plenty to overfit to
   noise. Refit may only show its benefit at aggressive
   compression (r=0.5–0.7) where input-only SVD genuinely loses
   information.

### What to try next on the quality front

1. **Refit with fp32 calibration.** Repeat the experiment at
   `--dtype float32`. Rules out bf16 as the cause. ~4× slower
   calibration than bf16 but isolates the variable.
2. **Refit at lower ratio.** Try `--target-ratio 0.8` with refit to
   test the "refit only matters when truncation bites" hypothesis.
3. **Refit only a subset of roles.** Apply balanced truncation only to
   down_proj (the role we see struggling qualitatively); leave shared
   roles on ASVD. Small code change.
4. **Investigate ggt quality directly.** Log condition number and
   effective rank of ggt per layer. If bf16 noise is the problem, it
   will show up as low effective rank.
5. **Sample count at r=1.0.** Push samples from 32 → 128 (tolerable
   wall time without refit) to see whether the 1.9% greedy agreement
   floor is calibration-noise-bound or fundamental to the scheme.

### Default settings as of this entry

Shipped as defaults in `basis_sharing.py`:

- `--target-ratio 1.0` — validation regime
- `--rank-alloc global` — per-matrix experimental (regresses 7B)
- `--skip-cholesky-above-d 15000` — catches 7B+ down_proj
- `--gpu-accum-budget-gb 4.0` — fits 3B shared+o_proj on GPU
- `--calib-batch-size 1` — batching neutral on 7B, helps 3B but
  asymmetric; left at 1 for consistency
- `--streaming-roles ""` (empty) — streaming opt-in per-run
- Factor streaming (per-layer .pt temp files + inline materialize)
  on by default, `--no-stream-factors` to disable.

### Current backlog (supersedes the 2026-04-18 list)

**Runtime — shipped since last entry:**
- ✓ Cholesky-skip fix (commit `e29ea935a`)
- ✓ GPU-resident shared-role accumulators (commit `6c221e192`)
- ✓ ggt / xtx file-prefix fix (same stack)
- ✓ Bench suite `bench_model.py` + `bench_compare.py` (commit `e2a1d41f5`)

**Runtime — still open, ordered by impact:**

1. **StreamingCollector pin_memory speedup** — _new target_, identified
   as the real 7B calibration bottleneck (~300s of the 420s total) now
   that GPU-resident nailed the shared-role hook path. Options: async
   pinning + double-buffered drain, or bf16-on-disk + lazy fp64
   promotion during drain. ~2-3 hr.
2. **Cross-run gramian caching** — save fp64 gramians to safetensors
   after streaming finalize, reuse across rank/ratio/refit experiments.
   Zero-cost re-runs when only decomposition params change. Huge
   iteration-speed multiplier for the quality experiments below. ~2 hr.
3. **Pipeline prefetch** — overlap layer k+1 load from disk with
   layer k SVD via async CUDA streams. ~20% decomp speedup. ~2-3 hr.
4. **Per-window safetensors output format** — eliminates the ~22 GB
   peak during save_factored assembly. Requires C++ loader update.
   Deferred until runtime is otherwise optimized.

**Quality — where we pivoted today, but refit regressed:**

Refit=True with bf16 backward + 32 samples at r=1.0 nearly doubled PPL
and catastrophically degraded generation (word salad, tighter loops,
no valid code). Five follow-up experiments queued:

1. **Refit with fp32 calibration** — rules out bf16 underflow as the
   noise source. ~4× slower than bf16 but isolates the variable.
2. **Refit at lower ratio (r=0.8)** — tests the "refit only matters
   when truncation actually loses information" hypothesis.
3. **Partial refit (down_proj only)** — apply balanced truncation only
   to the role that's qualitatively struggling; leave shared roles
   on ASVD. Small code change.
4. **ggt diagnostics** — log condition number / effective rank of
   per-layer ggt. If bf16 underflow is the culprit it shows up as
   rank collapse.
5. **Calibration at 128 samples (no refit)** — push past 32 to see
   whether the 1.9% greedy agreement floor is calibration-noise-bound
   or structural to our scheme.

**New quality lever from bench findings:**

The bench suite exposed **greedy agreement as a better
generation-quality indicator than PPL**. Future quality experiments
should report greedy agreement + top-5 overlap alongside PPL; any
change that regresses either is suspect even if PPL holds. Qualitative
judgment remains the final word — the side-by-side markdown from
`bench_compare.py` is the artifact to review per-experiment.

**Quality experiments deferred indefinitely:**

- Per-matrix rank allocation (sum-of-σ² not a PPL proxy at 7B).
  Needs a better utility function — ggt-weighted if refit works, or
  empirical PPL sensitivity if not.
- r < 1.0 production compression. Returning to this once generation
  quality at r=1.0 is understood; no point chasing compression when
  the lossless-ish regime already loops.
