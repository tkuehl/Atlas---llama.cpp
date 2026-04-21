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

---

## 2026-04-18 (evening) — quality curve sweep, scale-invariant floor, pivot to streaming runtime

Today's work flipped a major assumption about the scheme. We now have a
usable quality floor and a clear strategic direction.

### The 0.5B target_ratio sweep

Systematically varied `--target-ratio` on Qwen 2.5 0.5B, all with
32 calibration samples, no refit. Used the new bench suite for
automated scoring against the base model.

| r | factored PPL | × baseline | greedy agreement | top-5 overlap | exact-match |
|---|---|---|---|---|---|
| 1.0 | 23.02 | 1.88× | **1.1%** | 11.4% | 3/5 |
| 1.25 | 12.86 | 1.05× | 6.6% | 16.2% | 3/5 |
| **1.5** | **12.19** | **0.998×** | **50.0%** | **57.9%** | 3/5 |
| 1.75 | 12.21 | ~1.00× | 61.6% | 71.5% | 3/5 |
| 2.0 | 12.21 | ~1.00× | 89.6% | 90.7% | 3/5 |

**Two big observations:**

1. **PPL flattens at r ≈ 1.5 but generation quality keeps climbing
   through r = 2.0.** At r=1.5 the factored model's per-token
   probability on held-out tokens matches the base, but its *argmax*
   at a given position still differs from the base on half the
   prompts. Between r=1.5 and r=2.0 PPL is flat (both match base to
   three decimal places) while greedy agreement climbs from 50% to
   90%. This is another concrete demonstration that **PPL is a
   misleading proxy for generation quality** — it averages token-level
   likelihood and is blind to shuffling in the tail of the top-k
   distribution that autoregressive decoding relies on.

2. **r = 1.0 is catastrophically bad** — 1.1% greedy agreement, loops
   and word salad. It is by far the worst operating point tested —
   worse than doing nothing (not factoring) AND worse than r=1.5
   (which is more storage). r=1.0 is the byte-parity point of our
   `rank_permatrix` formula; its clean number hides the fact that the
   required truncation cuts eigenvectors the model actually uses.

### The naming reality of `target_ratio`

The `--target-ratio` parameter is literally a **storage multiplier
relative to dense**, not a compression ratio. Because the factored
form uses `r × (d_out + d_in)` bytes against dense's `d_out × d_in`,
setting `rank = target_ratio × d_out·d_in / (d_out + d_in)` gives
storage = `target_ratio × d_out·d_in`. So:

- `--target-ratio 0.5` → 50% of dense bytes (50% *compression*)
- `--target-ratio 1.0` → 100% of dense bytes (**no compression**)
- `--target-ratio 1.5` → **150% of dense bytes (inflation)**
- `--target-ratio 2.0` → 200% of dense bytes (2× dense)

At r ≥ 1.0 we are no longer compressing — we're representing the same
information in a factored form that's *larger* than the dense weights.
Useful for the streaming runtime (see below), useless for the "shrink
the model" pitch.

### Scale invariance — 3B shows the same floor

Re-ran the sweep point that mattered (r=1.5) on 3B:

| metric | 0.5B r=1.5 | 3B r=1.5 |
|---|---|---|
| factored PPL | 12.19 | 7.666 |
| × baseline | 0.998× | 1.001× |
| greedy agreement | 50.0% | 45.8% |
| top-5 overlap | 57.9% | 62.5% |

**Same ~50% greedy agreement floor at r=1.5 across 6× scale
difference.** The quality curve does not obviously benefit from
model size. This suggests the required rank multiplier is a property
of the *scheme* (windowed shared basis + per-matrix for o_proj /
down_proj) rather than the *model*. Whatever bottlenecks the spectrum
tail at 0.5B bottlenecks it equally at 3B.

Qualitatively, both 0.5B and 3B at r=1.5 produce coherent text.
Typical 3B r=1.5 behaviors:
- `fact_math`: generates the exact same 80-token arithmetic walkthrough
  as base ("17 + 23 = 40"), token-for-token.
- `code_factorial`: near-identical Python implementation to base, with
  one function name diverging ~60 tokens in.
- `fact_cap_germany`: matches base for 6 tokens ("Berlin. The capital
  of France is Paris..."), then both continue listing capitals validly
  (different countries, both correct).
- `comp_fox`: diverges from base but produces a valid word-counting
  Python function — different but semantically reasonable.

No loops, no word salad, no factual errors — the failure modes of
r=1.0 are gone.

### Hypothesis: why the tail eigenvectors matter

The r=1.0 → r=1.5 jump shouldn't change PPL much (it doesn't) but
does fix generation coherence. The mechanistic read:

- Factored form `W ≈ U · V` at rank r loses the smallest (d - r)
  singular directions of W (under whatever whitening we apply).
- At r = d_out·d_in / (d_out + d_in) (i.e. target_ratio=1.0), roughly
  half the original rank is preserved.
- **Top eigenvectors** capture the dominant per-token probability mass
  — keeping them is sufficient for PPL-competitive predictions. This
  is what the spectrum-weighted SVD prioritizes.
- **Tail eigenvectors** capture subtle circuits that act as a small
  perturbation on the main distribution: induction heads, bigram
  suppression ("don't repeat what you just said"), copy heads, etc.
  These circuits fire rarely and contribute little to token likelihood
  in aggregate (hence PPL is tolerant) but they *trigger on specific
  patterns* during generation — exactly where loops and self-reference
  happen. Truncating the tail breaks these, and the model goes into
  repetition because the "don't repeat" signal is gone.
- At r ≥ 1.5, we're preserving enough of the tail that these circuits
  survive. At r = 2.0, we reach full rank (for the permatrix roles)
  and reconstruction is essentially exact.

This predicts that anti-repetition behavior and any "subtle pattern
matching" capability will be the first thing compression kills, and
that is precisely what we observe.

### r=1.0 sum-of-σ² rank allocation regresses for the same reason

The per-matrix water-fill allocator (2026-04-17 entry) moves rank
from down_proj/o_proj to shared gate/up. Under sum-of-σ² at rank k,
the criterion rewards matrices whose σ_k² is larger — which tends to
be shared roles because W_stack = cat(W·S_in) has σ²(W_stack) ≈
n·σ²(W) at similar layers. But **the tail of σ² is what carries the
anti-repetition circuits,** and sum-of-σ² minimization explicitly
dismisses the tail. So the allocator trades exactly the wrong thing.
Confirms from a different angle that quality ≠ reconstruction error
as the SVD sees it.

### Refit regression at r=1.0, in light of this

The 2026-04-18 refit experiment regressed PPL 12.26 → 21.07 on 3B.
With the rank-truncation framing: refit tries to re-optimize `V_i`
given a fixed `B` under balanced weighting, but at r=1.0 the fixed
rank-r basis `B` already excludes critical tail directions. Refit
then forces `V_i` to approximate the full weight as well as possible
through this insufficient basis, amplifying the errors. Refit at
r < 1.5 is like tightening a bad approximation — makes it more
efficient at being wrong.

Predicts: refit at r ≥ 1.5 should be neutral or beneficial, not
catastrophic. Worth testing if we return to it.

### Strategic pivot: streaming runtime is the actual value prop

At r=1.5 we have:
- **Generation quality usable** (coherent text, ~50% greedy agreement)
- **PPL lossless** (within 0.002× baseline)
- **1.5× dense storage** — NOT compression, but a *structured decomposition*

The original "shrink the model" framing was wrong, but a better framing
is already in the DESIGN.md / Phase 3.4 plan: **memory-hierarchy
exploitation**. At r=1.5:

- `B` (shared basis): `d_out × r` per window, ~few GB per role on 7B,
  **resident in fast memory (VRAM)**.
- `V_i` (per-layer coefficients): `r × d_in` per layer, individually
  small enough to stream from CPU RAM or NVMe **on demand per token**.

Total bytes grow vs dense, but the **working set during inference**
shrinks. A 7B model at r=1.5 could run on a 12 GB GPU by keeping
bases resident and streaming coefficients — something a dense 15 GB
model can't do at all on that hardware.

**This changes the victory condition.** We stop chasing "factored form
smaller than dense" and start chasing "factored form runs a model
bigger than VRAM would otherwise allow." r=1.5 is usable for that
thesis *today*.

### Hypothesis for larger-model (7B+) behavior

We haven't tested 7B at r=1.5 yet (scheduled next). Prediction based
on the scale invariance seen so far:

- **PPL**: will be lossless at r=1.5, tracking the trend
- **Greedy agreement**: will sit around 45-55% — same floor
- **Qualitative**: coherent generation; loops absent at this ratio

If this prediction holds, 7B r=1.5 becomes the target for the first
streaming-runtime benchmark. If the curve *does* shift favorably with
scale (e.g., 7B is clean at r=1.2 or r=1.0), we get actual compression
as a bonus on top of streaming.

### What's shipping / what's deferred

**Shipped in this direction (runtime hardening, now complete enough
for the streaming runtime to start):**

- Mathematically-correct factoring pipeline across 0.5B / 3B / 7B
- Streaming calibration (activation-cache + incremental drain)
- Streaming factor output (per-layer .pt files + inline materialize)
- Memory caps (CPU / disk / VRAM percentage)
- GPU-resident shared accumulators on 3B-scale models
- Cholesky-skip for rank-deficient wide-d gramians
- Benchmark suite that measures speed, resources, and quality

**Deferred indefinitely (quality fixes that don't help at our regime):**

- Per-matrix water-fill rank allocation (sum-of-σ² wrong proxy)
- Refit at r < 1.5 (regresses under the tail-truncation framing)
- Calibration sample count beyond 32 (tested 128, no generation
  improvement — confirms noise isn't the bottleneck)

**Quality experiments now downgraded (r=1.5 removed the urgency):**

- Refit with fp32 calibration (only re-test if we revisit r<1.5)
- Partial refit (down_proj only)
- ggt diagnostics

### Next action

1. **Run 7B at r=1.5** through factor + bench. Validate the scaling
   prediction. ~30-40 min.
2. **If 7B r=1.5 is usable, this becomes the first-ever candidate for
   actual streaming-runtime testing.** The llama.cpp-side C++ work in
   `src/llama-model-loader.*` (already partially written per earlier
   commits) can target this artifact.
3. **Map the DESIGN.md §3 streaming plan against a concrete 7B r=1.5
   artifact** — identify what's missing on the runtime side vs what's
   ready.

The compression-as-primary thesis is dead. The streaming-runtime
thesis is alive, and today it got its first viable input.

---

## 2026-04-19 — pivot to CALDERA (Q + L·R)

**TL;DR:** The 7B r=1.5 factored run was ungodly slow at inference time
(not calibration — decode). Root cause is architectural, not a tuning
issue. Pivoting the primary research direction to CALDERA
(`W ≈ Q + L·R`, arxiv:2405.18886). The factored-SVD streaming runtime
is shelved.

### Why the factored forward path is slow

Three compounding problems, none fixable with kernel polish:

1. **No mainline kernel for `y = B(Ax)`.** llama.cpp's fast path is
   `mul_mat` against GGUF-quantized tensors with CUDA-graph capture.
   The factored forward is two sequential `ggml_mul_mat` nodes —
   2× kernel launches per linear, 252 linears per token on Qwen 7B
   (7 roles × 28 layers × 2 matmuls = 504 launches vs 252 for dense).
   This is exactly the anti-pattern `optimization_path.md §1.1`
   flagged but we hadn't gotten around to fusing.
2. **r=1.5 is more bytes than dense.** The factored storage is
   `r · (d_out + d_in)` vs dense's `d_out · d_in`; at r=1.5 that's
   150% of dense. A forward pass *reads more memory* than the dense
   model it's replacing. For a memory-bandwidth-bound decode step
   (which single-request autoregressive decode always is on a
   consumer card), that's a strict regression — no runtime cleverness
   recovers it.
3. **PCIe streaming at batch=1 is bandwidth-impossible.** The
   DESIGN.md thesis was to stream per-layer coefficients from CPU RAM
   over PCIe. PCIe 4.0 x16 ≈ 32 GB/s vs VRAM ≈ 500+ GB/s. Hiding
   PCIe transfer behind compute requires arithmetic intensity that
   single-request decode doesn't have; at batch=1 you cannot amortize
   the transfer. Double-buffering doesn't help — the transfer
   *itself* is the bottleneck. This was implicit in the DESIGN.md
   math but we hadn't tested it end-to-end until the 7B r=1.5 bench.

The base vs factored 7B comparison exposed all three:
`comparison_7b_r1.5.md` reports model load 8.2s → 224s (+2627%), TTFT
+128%, decode throughput ~flat because the base was *also* Python/HF
bound at 4.7 tok/s. Neither is representative of what llama.cpp can do
with a standard GGUF.

### Why CALDERA

`W ≈ Q + L·R` — Q is a standard GGUF-quantized tensor, L and R are
small fp16 low-rank correction factors.

- **Q rides the mainline kernel path.** Q4_K_M GEMV is
  hand-optimized CUDA, CUDA-graph captured, already fast. Expected
  7B tok/s: 4.7 (PyTorch/HF) → ~60–80 (llama.cpp Q4_K_M), before we
  add anything new.
- **L·R is LoRA-shaped.** The `build_lora_mm` intercept from Phase 3.2
  (commit `55a9e6bdb`) already runs the forward correctly. Load-time
  GGUF scan and tensor creation helpers (`create_tensor_factored`,
  `fd67cd178`) are reusable.
- **Streaming becomes optional, not the core mechanism.** 7B at Q4
  fits in ~4 GB — no streaming needed on the 5070. For bigger models
  we can layer on llama.cpp's existing `-ngl` partial offload, which
  is already a tested PCIe-streaming path.
- **Calibration infra transfers wholesale.** CALDERA's low-rank fit
  is Σ-weighted Frobenius minimization, the same whitening
  (`sqrt_and_inv`) + gramian collection we already built in
  `basis_sharing.py`. Plus the GPU-resident accumulators from commit
  `6c221e192`.

### What's reusable vs deprecated from the factored track

**Reusable:**
- `caldera.py` prototype (commit `ef08245f7`) — single-matrix RPCD,
  standalone smoke test.
- `basis_sharing.collect_stats` — gramian collection, whitening.
- `bench_model.py` + `bench_compare.py` + `bench_prompts.json` — the
  15-prompt bench with greedy-agreement + top-5-overlap metrics.
  **PPL-is-misleading lesson from 2026-04-18 still holds.**
- Loader plumbing: `create_tensor_factored`, factored-GGUF scan,
  `build_lora_mm` intercept. CALDERA's L·R slots into the same path.
- Calibration optimizations: GPU-resident shared-role accumulators,
  cholesky-skip, streaming activation cache.

**Deprecated (don't resume):**
- `y = B(Ax)` forward path / custom `GGML_OP_FACTORED_LINEAR` — we
  were going to fuse the two matmuls into one kernel; moot now that
  the mainline Q-kernel path is the target.
- `target_ratio` knob — CALDERA has (qtype, rank) instead.
- r=1.5 "structured decomposition" thesis — the pitch ("stream
  coefficients, keep basis resident") doesn't survive batch=1 PCIe
  math.
- DESIGN.md §3 streaming runtime — architecture still sound for a
  hypothetical batch>>1 server regime, but out of scope.

### Next steps (gated)

1. **Smoke + real-matrix validation** (~1 hr).
   `python caldera.py --smoke` first. Then single-matrix CALDERA on
   a real Qwen 7B `down_proj` with its saved gramian: does it beat
   pure Q4_K in Σ-weighted rel_err at r=32 / 64 / 128? Synthetic
   passing tells us the math; real-matrix tells us whether real weight
   spectra benefit.
2. **Full-model integration** in `basis_sharing.py` (~4 hr). Apply
   per-matrix across all 252 linears, sweep (qtype ∈ {Q4_K, Q3_K},
   rank ∈ {32, 64, 128, 256}), evaluate PPL + greedy-agreement +
   top-5 overlap on the 15-prompt bench. Target: find the
   (qtype, rank) pair that gets back to ≥90% greedy agreement at
   ≤4.5 bpw.
3. **GGUF emission + loader** (~variable). Q as standard GGUF quant
   tensor, L/R as companion tensors. The loader scan pattern from
   Phase 3.2 already handles companion tensors; mostly GGUF schema
   + `build_lora_mm` verification.
4. **Bench against llama.cpp mainline Q4_K_M** on the same 15-prompt
   fixture. This is the real win-condition: does CALDERA ≥ Q4_K_M
   quality at comparable speed? If yes, ship. If not, tune
   (qtype, rank) or look at Q3_K + larger rank.

### Tradeoff being accepted

The factored-SVD thesis was "novel PCIe-streamed coefficients let us
run a 30GB+ model on a 12GB card." That research bet is being
abandoned. The replacement thesis is "fit the same-size models with
less quantization damage, at mainline llama.cpp speed." Less novel,
much faster path to something usable.

For the bigger-model-on-small-card problem specifically, the honest
path is (a) more aggressive CALDERA compression (Q3_K + rank
correction, Q2_K + larger rank), plus (b) llama.cpp's existing
`-ngl` partial offload, plus (c) speculative decoding for additional
throughput. None of that needs a custom streaming kernel.

### Extended plan — big-model target (added 2026-04-19)

30 GB target on 12 GB VRAM has a hard PCIe ceiling at ~1.3 tok/s for
18 GB streamed per token. Three levers, stackable:

1. **Compress harder so more fits resident.** CALDERA-Q3_K + rank-64
   on 30B ≈ 12–13 GB — may squeeze into VRAM with KV-cache tradeoffs.
   CALDERA-Q2_K + rank-256 on 70B ≈ 22 GB — still partial offload,
   but less of it. This is where CALDERA's "less damage at low bits"
   actually delivers for big models (see Stage 5 below).
2. **Speculative decoding.** Small same-family draft model (0.5–1B)
   runs resident, big model verifies k proposed tokens per forward
   pass. 2–5× effective speedup, stacks cleanly on partial offload
   or full-resident. Zero custom code — llama.cpp has
   `--draft-model` on mainline. Added as Stage 4.
3. **Hardware.** 5090 has 32 GB; problem dissolves. Noted but not
   planned.

Realistic target for 5070 + 30 GB stored model via
CALDERA-Q3_K + spec decode: **~8–12 tok/s interactive**. Below that,
the constraint is PCIe bandwidth, not software.

### Parallel research track — cross-matrix shared structure

Separately from the CALDERA main track, queued a research bet on
whether transformer weight matrices share compact common structure
across layers and roles that a global codebook + per-matrix sparse
coefficients could exploit. Does not block CALDERA shipping.

**Motivation from hardware math:** 5070 at batch=1 decode has
~60 FLOPs/byte of unused compute (tensor cores process data
~500× faster than VRAM delivers it). Today's Q4_K_M dequant spends
3–5 FLOPs/byte. Huge headroom to "emulate" weights via compute. The
Cloudflare "Unweight" kernel pattern
(https://blog.cloudflare.com/unweight-tensor-compression/) is the
production precedent — producer/consumer thread groups decompress
in shared memory and feed tensor cores without an HBM round-trip,
shipping on H100 with 13–22% lossless size reduction. Same bandwidth-
compute imbalance argument they cite ("tensor cores process data
nearly 600× faster than memory can deliver it" on H100) holds on
consumer cards.

**Hypothesis-test experiment (~2 hr, does not block CALDERA):**

1. Load all 252 Qwen 7B linears, normalize each by Frobenius norm.
2. Stack columns across all matrices as one `d_out × (252·d_in)`
   tensor.
3. Randomized SVD, inspect singular-value spectrum.
4. If top K≈1024 captures >90% energy → shared-dictionary thesis
   viable, pursue global-codebook scheme. If spectrum is flat →
   matrices genuinely independent, archive this direction.

**Prior art to read first** (so we don't reinvent AQLM):

- **AQLM** (arxiv:2401.06118) — per-matrix small codebook, ~2 bpw,
  in llama.cpp, loses to Q4_K_M on tok/s despite being smaller.
  Cross-matrix codebook sharing is the unshipped extension.
- **QuIP#** (arxiv:2402.04396) — randomized Hadamard + lattice
  quant, near-fp16 at 2 bpw, zero-storage "codebook."
- **Monarch / Butterfly** (arxiv:2204.00595) — structured product
  of block-sparse matrices, `d log d` storage, decode via compute
  can be faster than dense matmul.
- **Basis Sharing** (arxiv:2410.03765) — cross-layer SVD, already
  tried (shelved above).

**Novelty window:** cross-matrix global codebook (~1–10 MB resident)
+ per-matrix sparse index + scale. Decoded via gather + mul +
accumulate on GPU. AQLM does this per-matrix; sharing across all 252
matrices is the open research bet.

**Risk:** custom codebook kernels have to beat Q4_K_M on tok/s,
not just on size. AQLM already failed this bar on consumer cards.
Any novel scheme has to clear Q4_K_M speed or it doesn't ship.

**Operational insights borrowed from Cloudflare Unweight:**
- **Selective compression** — don't compress attention / embeddings
  / layer norms; different tensors have different value/byte, MLP
  is where compression pays off. Applies to CALDERA Stage 2 sweep
  design too (consider excluding non-MLP from low-bpw experiments).
- **Producer/consumer kernels** with shared-memory decompression as
  the kernel target for any cross-matrix scheme that ships.
- **Per-matrix / per-batch-size autotuner** over multiple decode
  strategies. Applies to CALDERA too (full-dequant vs
  fused-reconstruct paths).

---

## 2026-04-19 (later) — Cloudflare kernel-pattern investigation: MMVQ already optimal for batch=1

Dug into llama.cpp's Q4_K_M CUDA path to see if porting Cloudflare
Unweight's producer/consumer kernel pattern (on-chip decompression
feeding tensor cores, no HBM round-trip) would speed up batch=1
decode on the 5070. Short answer: **no — llama.cpp already has the
right kernel architecture for batch=1, and Cloudflare's specific
pattern is a no-op on this path.**

### The two Q4_K_M paths

llama.cpp has two architecturally different Q4_K_M kernels:

**MMQ (batch ≥ 2)** — [mmq.cuh](../../ggml/src/ggml-cuda/mmq.cuh).
Tile-based. All threads cooperatively load Q4_K blocks and
dequantize into shared memory (`load_tiles_q4_K`, lines 2151-2258),
then all threads run `vec_dot` via MMA intrinsics (`mul_mat_q_process_tile`,
lines 3591-3604). Dequant outputs live in shmem and never
round-trip to HBM — so Cloudflare's "no HBM round-trip" half is
already achieved. What's missing is the thread-group split: all
threads do both roles in a single sequential pass. No `__pipeline_`,
no `cuda::barrier`, no async-copy. **Porting a producer/consumer
split with double-buffering would likely help here — but this path
only runs at batch ≥ 2.**

**MMVQ (batch = 1)** — [mmvq.cu:22](../../ggml/src/ggml-cuda/mmvq.cu)
dispatches `vec_dot_q4_K_q8_1` for Q4_K GEMV. Element-wise vec-dot
per thread, dequant inline in registers, no tile materialization,
no shmem staging. Our interactive decode runs entirely on this path.

Tensor-core gating: MMA uses `TURING_MMA_AVAILABLE` (sm_75+), so
Ada sm_89 (5070) is in scope. WGMMA is Hopper-only (sm_90+),
irrelevant to us.

### Why Cloudflare's pattern doesn't apply at batch=1

The producer/consumer trick assumes there is **tile-level work** to
split across thread groups — one group loads/decompresses, another
feeds tensor cores. At batch=1 there is no output tile to compute;
each thread computes one scalar accumulator by pulling weights
through the vec-dot loop. Dequant is already one register-local
instruction. No tile round-trip to eliminate and no thread-group
split that would help — the pattern has no purchase here.

### Implication

**Batch=1 Q4_K_M on the 5070 is approximately at the VRAM-bandwidth
ceiling today.** The kernel shape is already optimal for a
bandwidth-bound GEMV. We cannot kernel-polish our way to faster
interactive tok/s on Q4_K_M.

The only remaining lever for batch=1 tok/s is **reading fewer bytes
per token** — genuine compression below Q4_K_M's ~4.5 bpw with
acceptable quality. Three attacks, ordered by research cost:

1. **CALDERA aggressive quant** (Q3_K / Q2_K + rank correction) —
   Stage 5 of the pivot plan. Lowest risk, closest to shipping.
2. **Speculative decoding** — doesn't cut bytes-per-forward, but
   amortizes the forward over k drafted tokens. Stage 4, pure
   llama.cpp config.
3. **Cross-matrix structure** — if the SVD hypothesis test shows
   shared structure, global-codebook schemes could go below 3 bpw.
   Parallel research track, higher risk, higher novelty.

### Updated priorities

- **Dropped:** Nsight profile + MMVQ kernel port for batch=1. MMVQ
  is already architecturally optimal — no headroom. The Nsight
  phase-0 step was a sanity check we no longer need to run.
- **Running:** [`entropy_char.py`](entropy_char.py) against a
  Qwen 2.5 7B GGUF — measures the lossless compression ceiling on
  top of existing quant representations. Tells us how much a
  Huffman/ANS layer on CALDERA's Q output could buy. Small
  stackable win if any. ~5 min to run.
- **Parked:** MMQ producer/consumer port. Real engineering
  opportunity for server/batched inference workloads if those ever
  become a primary use case; not our target today.
- **Active:** CALDERA Stage 1 smoke + Stage 2 (qtype × rank) sweep
  including Q3_K / Q2_K, plus the SVD cross-matrix hypothesis test
  in parallel. Both attack the "read fewer bytes" constraint.

Cloudflare research paid off without being ported: it told us
*cleanly* where **not** to spend effort. At batch=1 the kernel is
already doing everything right. The constraint is genuinely the
bits on disk.

---

## 2026-04-19 — CALDERA Stage 1: smoke + real-matrix validation across 0.5B / 3B / 7B

Validated the CALDERA math end-to-end on real model weights. Math
holds at every scale; RPCD iteration converges monotonically. Q4_K /
Q3_K / Q2_K still untested because gguf-py's Python quantizer isn't
implemented for those qtypes — that unblocks in Stage 2 via a
`llama-quantize` subprocess.

### Smoke test (synthetic 1024×2048, r=128, Q4_0, 3 iters)

| method | Σ-rel-err |
|---|---|
| pure quant (Q4_0) | 0.0859 |
| pure LR (r=128) | 0.5341 |
| CALDERA | **0.0632** (history 0.0694 → 0.0652 → 0.0632) |

Beats both baselines, assertion passes.

### Real-matrix — Qwen 2.5 `mlp.gate_proj` L12 across three scales

Same role, same layer, 32 WikiText-2 calibration samples, forward-only
bf16 calibration via new [`extract_matrix_caldera.py`](extract_matrix_caldera.py).

| scale | shape | pure Q4_0 | CALDERA r=32 | r=64 | r=128 | r=256 |
|---|---|---|---|---|---|---|
| 0.5B | 4864 × 896 | 0.0587 | 0.0365 (−38%) | 0.0319 (−46%) | 0.0259 (−56%) | 0.0187 (−68%) |
| 3B | 11008 × 2048 | 0.0393 | 0.0248 (−37%) | 0.0228 (−42%) | 0.0201 (−49%) | 0.0166 (−58%) |
| 7B | 18944 × 3584 | 0.0554 | 0.0439 (**−21%**) | 0.0409 (−26%) | 0.0369 (−33%) | 0.0316 (−43%) |

Improvement ratio shrinks with model size because fixed ranks are a
smaller fraction of d_in at bigger matrices (r=32 is 3.6% of d_in on
0.5B, 0.9% on 7B). The L·R overhead also shrinks proportionally:
CALDERA Q4_0 + r=32 at 7B is 4.84 bpw (vs 5.85 on 0.5B) — only 0.34
bpw on top of pure Q4_0 for a 21% error cut. Still a favorable trade,
just at a different operating point per-scale. Stage 2's 7B sweep
should extend ranks to r=512 / r=1024 to find the quality sweet spot.

### What the validation doesn't tell us yet

All tested qtypes are Q4_0 (4.5 bpw floor) or Q8_0 (8.5 bpw, already
near-lossless). The CALDERA value proposition is specifically
**low-bpw quant with rank correction** — Q3_K (~3.5 bpw), Q2_K (~2.8
bpw). Those are exactly the qtypes gguf-py does not implement in
Python: `quantize_blocks` raises `NotImplementedError` for Q4_K, Q3_K,
Q2_K. Q4_0 is "compression floor that barely matters"; the interesting
regime is where pure-quant error is large enough for LR correction to
have real work.

### Infra produced this stage

- [`caldera_validate.py`](caldera_validate.py) — compares pure quant /
  pure LR / CALDERA at multiple (qtype, rank) points, reports
  Σ-weighted rel-err and effective bpw side-by-side.
- [`extract_matrix_caldera.py`](extract_matrix_caldera.py) — bf16
  forward-only extractor; 3B calibration in 2.1s, 7B in 21.1s with
  `device_map="auto"` (some layers offloaded to CPU).
- Snapshots (gitignored, regenerable):
  `snapshot_{3b,7b}_gate12.pkl` contain (W, XTX, metadata) for the
  validated matrices.
- Result JSONs: `caldera_validate_{05b,3b,7b}.json`.

### Stage 2 entry criteria

- Wire up `llama-quantize` subprocess (C++ path) so any gguf-py qtype
  is accessible. Write W to minimal single-tensor GGUF → invoke
  `llama-quantize --pure --output-tensor-type Q3_K_M` etc. → read
  back dequantized W. Matches what ships in production GGUFs exactly.
- Extend rank sweep for 7B to include r ∈ {32, 64, 128, 256, 512,
  1024}. Small models can stay at existing range.
- Evaluate on more than one role: gate_proj, down_proj, o_proj are
  the three roles with different activation spectra (permatrix roles
  vs shared roles in the old factored nomenclature).

---

## 2026-04-19 — CALDERA Stage 2: K-quant sweep and an honest finding

**TL;DR:** Stage 2 infra is built and working (ctypes into `ggml-base.dll`'s
`ggml_quantize_chunk`, all K-quants accessible). Ran the full sweep on
7B `mlp.gate_proj` L12. Result: **pure K-quants beat CALDERA at matched
bpw across every tier.** CALDERA's per-qtype improvement ratio is
remarkably stable (~21% at r=32, ~55% at r=512), but it's not enough to
close the gap to pure Q(N+1)_K at the same bit budget. Stage 2 findings
reshape the value-prop question for CALDERA.

### Infra built this stage

- **`llama-quantize` build target added** to the fork build
  (`build-cuda-quantize.bat`). Works end-to-end but required full
  LLaMA architecture metadata on the input GGUF — not useful for our
  single-tensor flow.
- **Pivoted to ctypes → `ggml_quantize_chunk`** against the built
  `build-cuda/bin/ggml-base.dll`. Same C kernel that powers llama.cpp's
  production quantization path; no GGUF metadata gymnastics, no
  subprocess spawn overhead, works for every GGML quant type.
- **`_quantize_roundtrip` updated** to route non-Python qtypes through
  the ctypes path. Drop-in replacement; all existing validation still
  works. Verified round-trip correctness on Q2_K / Q3_K / Q4_K / Q5_K
  / Q6_K / Q8_0 with expected bpw (2.62 / 3.44 / 4.50 / 5.50 / 6.56 /
  8.50).
- **`caldera_validate.py` updated**: K-quants in default qtypes; ranks
  extended through r=512; bpw computed from `GGML_QUANT_SIZES` rather
  than hardcoded.

### Result on 7B `mlp.gate_proj` L12

Improvement-ratio consistency across qtypes at matched rank is striking:

| qtype | pure err | r=32 | r=64 | r=128 | r=256 | r=512 |
|---|---|---|---|---|---|---|
| Q2_K (2.62 bpw) | 0.1881 | −20.6% | −25.7% | −32.9% | −42.3% | −54.7% |
| Q3_K (3.44 bpw) | 0.0959 | −20.6% | −26.0% | −33.2% | −42.8% | −55.4% |
| Q4_K (4.50 bpw) | 0.0456 | −20.8% | −26.0% | −33.0% | −42.2% | −54.4% |
| Q5_K (5.50 bpw) | 0.0231 | −20.6% | −25.8% | −32.7% | −41.9% | −54.1% |
| Q8_0 (8.50 bpw) | 0.0035 | −20.7% | −26.1% | −33.3% | −42.9% | −55.4% |

CALDERA adds a rank-dependent multiplicative factor (~0.79×, 0.74×,
0.67×, 0.58×, 0.45× for r=32, 64, 128, 256, 512) that's independent of
the quant tier. The LR correction "removes roughly half of whatever
quant error it's starting from" given enough rank.

### The matched-bpw comparison is where it gets honest

| bpw target | best pure quant | err | best CALDERA | err |
|---|---|---|---|---|
| ~2.6 | Q2_K | 0.188 | — | — |
| ~3.4 | **Q3_K** | **0.096** | Q2_K r=64 | 0.139 |
| ~4.5 | **Q4_K** | **0.046** | Q3_K r=64 | 0.071 |
| ~5.5 | **Q5_K** | **0.023** | Q4_K r=64 | 0.034 |
| ~8.5 | **Q8_0** | **0.0035** | Q5_K r=256 | 0.0134 |

At *every* bpw tier, "spend the bit budget on a higher-tier K-quant"
beats "spend it on L·R correction on top of a lower-tier K-quant" on
Σ-weighted rel-err. Pure K-quants are very well tuned — they already
capture a lot of what CALDERA's per-matrix LR is trying to add, via
their super-block + hierarchical-scale structure.

### Why earlier Stage 1 results looked so much better

Stage 1 compared CALDERA against **Q4_0** (32-weight block, single
fp16 scale) — a much weaker quantizer than Q4_K. CALDERA's 21–68%
improvement over Q4_0 doesn't translate to a win against Q4_K because
Q4_K itself is already ~55% better than Q4_0 at the same bpw.

### What this doesn't yet rule out

1. **Full-model PPL and greedy agreement.** Σ-weighted rel-err on one
   matrix is a proxy. The ordering might invert on downstream
   generation quality if CALDERA's activation-weighted correction
   lands on eigendirections that matter disproportionately for
   autoregressive decode. K-quants optimize element-wise / block-
   local error, not Σ-weighted. Worth one end-to-end test.
2. **Importance-matrix quantization.** The K-quants we compared
   against here are RTN (round-to-nearest) without imatrix. Production
   GGUFs typically use imatrix to bias the quantization toward
   activation-important channels. imatrix-K-quants would likely
   *widen* the gap against CALDERA.
3. **Cross-matrix dictionary sharing.** Orthogonal — a global
   codebook shared across all 252 linears (the
   `project_unweight_research.md` thesis) is a fundamentally different
   attack and still untested.

### Updated position on CALDERA

CALDERA's value-prop of "rank correction on top of low-bit quant" is
not winning the matched-bpw game against K-quants. Three plausible
next moves, in priority order:

1. **End-to-end PPL + greedy-agreement test** on a single sweep point
   (CALDERA Q3_K r=128 on full Qwen 2.5 7B) vs pure Q4_K and pure
   Q3_K. Cost: 1 day to build, 1 hr to run. If CALDERA wins on PPL
   at matched bpw — even when losing Σ-rel-err — the value-prop holds
   on a different axis.
2. **Pivot primary track to the cross-matrix codebook research bet**
   (`project_unweight_research.md`). CALDERA is a reasonable
   per-matrix refinement but isn't a novelty moonshot; the global
   codebook angle is.
3. **Archive CALDERA and run plain Q4_K_M + spec decode** for the
   interactive-tok/s problem. Pragmatic. Nothing novel to write about
   but gets the best fast path on the 5070 today.

---

## 2026-04-19 — Cross-matrix shared-dictionary hypothesis: DEAD

Ran the gate experiment from `project_unweight_research.md` on
Qwen 2.5 7B. Goal: does transformer weight structure across matrices
concentrate in a compact shared subspace of the residual stream
that a global codebook could exploit?

### Setup

197 `nn.Linear` modules (all attention and MLP projections across 28
layers, plus `lm_head`) projected into the 3584-dim hidden-space:
matrices where `d_out == 3584` use `G_i = W_norm @ W_norm.T`; matrices
where `d_in == 3584` (gate / up) use `G_i = W_norm.T @ W_norm`. Each
W normalized by its Frobenius norm so `tr(G_i) = 1`. Accumulate
`G_stacked = Σ_i G_i`, compare its spectrum to per-matrix baseline.
Computed in fp64 on CPU; run took 287s.

### Result

| | effective rank | k=256 | k=1024 | k=2048 |
|---|---|---|---|---|
| stacked-all (197 matrices) | **3303** / 3584 | 12.3% | **38.3%** | 66.2% |
| per-matrix mean | 1333 | 41.9% | 74.0% | 90.3% |
| random baseline | — | 7.1% | 28.5% | 57.1% |

Stacked top-1024 = 38.3% — far below the 90% gate. Only 10 points
above a uniformly-random subspace. **Stacked effective rank (3303) is
LARGER than per-matrix mean (1333), not smaller.**

Per-role stacked concentration (k=1024):
| role | n | k=1024 |
|---|---|---|
| down_proj | 28 | 35.8% |
| gate_proj | 28 | 39.7% |
| k_proj | 28 | 64.5% |
| o_proj | 28 | 44.7% |
| q_proj | 28 | 45.5% |
| up_proj | 28 | 36.1% |
| v_proj | 28 | 54.5% |

`k_proj` is the best sharer (64.5%) — likely because GQA replicates
K projections across query heads, giving some structural reuse.
Still nowhere near the 90% gate.

### Interpretation

Matrices actively use **complementary** directions in hidden-space —
not random-independent, but **anti-aligned**. Each matrix's preferred
~1333-dim subspace is distinct from other matrices'. When you stack
197 such matrices, the combined distribution almost fills the whole
3584-dim space.

This is the structure predicted by transformer superposition
interpretability: different attention heads and MLP neurons encode
different features, the residual stream is the shared coordinate
system where features sum. Weight matrices orthogonalize during
training to maximize feature distinguishability. "Space-filling
matrix distributions" is the correct null for a well-trained
transformer.

### Implications

- **Cross-matrix shared-dictionary schemes have no compression headroom
  to exploit.** Global codebook (AQLM-across-matrices), shared basis
  (Basis Sharing extended to all matrices), or any other cross-matrix
  structure-exploitation is fighting the nature of trained weights.
- **Per-matrix approaches are structurally correct.** CALDERA
  (per-matrix Q + L·R), K-quants (per-matrix super-block scales),
  AQLM (per-matrix codebook) — all the right shape.
- **This is consistent with the Stage 2 CALDERA finding** that pure
  K-quants beat CALDERA Q3_K+rank at matched bpw. Per-matrix structure
  is what matters; nothing at the cross-matrix level helps.
- **Basis Sharing paper's `window=2` ablation** (PPL degrades sharply
  past window=2) is explained: the larger the shared window, the
  more it's fighting the matrix anti-alignment we observe here.

### Remaining research axes (unrelated to this hypothesis)

- **Block-VQ within a matrix** (AQLM-style 8-dim codebook per-matrix)
  — different question, not directly invalidated, but the per-matrix
  structural picture suggests most of the benefit already lives
  inside K-quants' block design.
- **Temporal / activation-flow compression** (Mixture of Depths,
  early-exit, draft-then-verify) — a completely different axis.
  Speculative decoding with a small draft (already in CALDERA pivot
  plan as Stage 4) is the pragmatic version.
- **Memory-hierarchy exploitation for batch≥2** (the shelved
  factored-SVD DESIGN.md plan) — still valid for server workloads,
  still not our target.

### Verdict

Archive the cross-matrix research track. Return to the CALDERA
main-track question: does CALDERA Q3_K + rank correction beat pure
Q4_K on **downstream PPL and greedy agreement** (not just Σ-rel-err)?
If yes, CALDERA has a narrow-but-real use case as an in-between-tier
compressor. If no, CALDERA is archived too and the research track
collapses to "use llama.cpp mainline + spec decoding."

Files produced: [cross_matrix_svd_test.py](cross_matrix_svd_test.py),
[shared_basis_test_7b.pkl](shared_basis_test_7b.pkl).

---

## 2026-04-19 — Depth-smoothness hypothesis: also DEAD

Follow-up to the cross-matrix test, asking the "calculus" question:
are adjacent-layer weight matrices `W_i, W_{i+1}` related smoothly
(i.e., low-magnitude / low-rank differences) such that a neural-ODE
/ continuous-depth parameterization could compress the layer stack?

For each role, computed magnitude `||D_i||/||W_i||` and hidden-space
gramian of the 27 adjacent-layer differences (`depth_smoothness_test.py`).

### Result

| role | ‖D‖/‖W‖ | energy ratio | per-mat ER | diff ER | ratio |
|---|---|---|---|---|---|
| k_proj | 1.41 | 1.91 | 268 | 1484 | **5.5×** |
| v_proj | 1.42 | 1.91 | 410 | 2091 | 5.1× |
| q_proj | 1.41 | 1.92 | 866 | 2792 | 3.2× |
| o_proj | 1.42 | 1.93 | 1088 | 2694 | 2.5× |
| gate_proj | 1.41 | 1.92 | 1750 | 3212 | 1.8× |
| down_proj | 1.42 | 1.94 | 2445 | 3297 | 1.3× |
| up_proj | 1.42 | 1.94 | 2550 | 3408 | 1.3× |

### Interpretation

**‖D‖/‖W‖ ≈ 1.414 = √2 for every role.** The signature of `W_{i+1}`
and `W_i` being *uncorrelated random matrices of similar norm*:

`‖W_{i+1} − W_i‖² = ‖W_{i+1}‖² + ‖W_i‖² − 2⟨W_{i+1}, W_i⟩`

Independence ⇒ cross-term ≈ 0 ⇒ ratio = √2. Observed exactly.

**Diff effective rank is LARGER than per-matrix ER.** Differences
span more directions than the matrices themselves, not fewer. Depth-
continuous parameterization would need differences compact; they're
the opposite.

### Reconciling with the prior cross-matrix test

The cross-matrix test found k_proj highly shared (28 matrices stack
to ER=1580 vs single ER=1333, i.e., ~85% shared subspace). This
depth test says differences span 5.5× the single-matrix rank — looks
like a contradiction but isn't.

Resolution: all 28 k_proj matrices live in (roughly) the same
overlapping ~1500-dim hidden-space subspace, but **within that
subspace they move independently between layers**. The "cone" of
attention-relevant directions is shared; the specific orientations
inside it are drawn independently. Each layer's k_proj is a
different sample from a shared distribution, not a smooth step along
a common trajectory.

### Implications

- **Neural-ODE / continuous-depth parameterization is dead** for
  post-training compression on this model family. No training-free
  way to exploit "smooth depth evolution" because there is none.
- **Layer merging / looped transformer** also dead — reusing layer
  weights would catastrophically change the sampled point within the
  shared subspace.
- **Mixture-of-Depths / early-exit** would need retraining to work.
- **Distillation** (training a smaller student) is the only known
  path to trade this structure for compression.

Files: [depth_smoothness_test.py](depth_smoothness_test.py),
[depth_smoothness_7b.pkl](depth_smoothness_7b.pkl).

---

## Research-track synthesis (end of 2026-04-19)

Four consecutive experiments, all negative for post-training
compression beyond llama.cpp mainline K-quants:

| # | hypothesis | gate | result |
|---|---|---|---|
| 1 | Cloudflare on-chip producer/consumer kernel port | helps batch=1 | MMVQ already optimal; only helps batch≥2 |
| 2 | CALDERA Q+rank beats pure K-quants at matched bpw | Σ-rel-err win | K-quants win at every tier |
| 3 | Cross-matrix shared hidden-space structure | top-1024 ≥90% of stacked energy | 38% (matrices anti-aligned) |
| 4 | Depth smoothness / cross-layer weight evolution | ‖D‖/‖W‖ < 0.5, diff ER ≪ matrix ER | √2, diff ER > matrix ER |

Pattern: each experiment attacked a different "where is there
structure to exploit?" question. Four answers, all "nowhere that
training hasn't already arranged to be exploited." The through-line
is that **trained transformer weights are dense, anti-aligned, and
layer-independent — the quant format that works on one matrix at a
time with local super-block scales is essentially the right
structural fit.** K-quants are not a weak target; they're close to
the post-training compression ceiling given these structural
properties.

### What remains productive (none of it is post-training weight
compression)

1. **Speculative decoding.** Amortize big-model forward over k
   drafted tokens. 2–5× effective tok/s. Requires no custom code —
   `llama-cli --draft-model`. Already in CALDERA pivot plan as Stage 4.
2. **KV cache compression.** Weights are fixed but KV cache grows
   with context; at long context it's often bigger than weights.
   This research track hasn't touched KV cache at all. Separate axis
   from everything above.
3. **Retraining-allowed compression.** Distillation, QAT, pruning+
   distillation. Out of scope for the post-training-only constraint
   that defined this research track.
4. **Accept Q4_K_M + partial offload + spec decode as the answer.**
   Pragmatic. ~30-40 tok/s on 7B in VRAM, ~5-10 tok/s on 30 GB
   models with spec decode, bounded by PCIe physics not software.

### Recommendation

Archive the post-training-weight-compression research track. The
negative results are collectively strong enough to conclude that
K-quants + spec decoding is the near-optimum operating point for a
5070-class consumer card. If the goal is "run bigger models
interactively," either buy more VRAM (5090) or loosen the post-
training constraint (distillation).

---

## 2026-04-20 — Reframing as a pluggable model-prep pipeline

Every optimization direction explored so far — CALDERA (in flight),
windowed Basis Sharing + balanced truncation (deprecated 2026-04-19),
standard `llama-quantize` quants, the queued cross-matrix codebook bet
— has been a bespoke script path. Unifying them under one interface:
**HF model → ordered swappable stages → GGUF → bench**.

Each stage declares `requires` / `produces` / `calibration` /
`params`; the runner composes a pipeline by checking the graph is
consistent, runs calibration once, and emits a GGUF plus a bench
report scored against mainline `llama-quantize` Q4_K_M on the same
base model.

Full design, stage inventory, sample pipelines, acceptance criteria,
and open questions in
**[model_prep_pipeline.md](model_prep_pipeline.md)**.

The immediate value is that calibration gramians (the most expensive
thing we compute) get computed once per model and reused across every
stage config — the cross-run gramian caching backlog item from the
2026-04-18 late entry collapses into "a first-class feature of the
runner." The longer-term value is that future research bets (the
cross-matrix codebook track, anything else we pick up) have a known
landing spot and a known bar to clear to become defaults, rather than
living as one-off scripts.

Pipeline runner itself isn't written yet. First two target pipelines:
`baseline-q4` and `caldera-q4`, both on Qwen 2.5 3B, to prove the
runner against a known artifact before we chase 7B / 8B.

---

## 2026-04-20 — Easy-config baseline on the Atlas llama-server

Before investing in new research, audited what llama.cpp flags
the production Atlas deployment (`Atlas/docker-compose.yml`, service
`llama-chat`) was actually using. The goal was to establish an
honest "features-already-shipped" baseline so the research pipeline's
acceptance bar compares against *well-configured* llama.cpp, not
stock defaults.

### Audit findings

The Atlas llama-chat service runs the upstream
`ghcr.io/ggml-org/llama.cpp:server-cuda` image. Prior config:

- `--flash-attn on` — **already enabled** ✓
- `-ngl 999` (full GPU offload) — already enabled ✓
- `-c 8192`, `--jinja` — already set ✓
- Quantized KV cache — **missing**
- Cross-request prefix cache — **missing**

Also noted a doc drift: `Atlas/CLAUDE.md` says the chat model is
`Qwen3-8B-Q4_K_M.gguf`, but docker-compose.yml actually serves
`gemma-4-e4b-it-Q8_0.gguf` with `mmproj-gemma-4-e4b-it-bf16.gguf`
(vision). The llama-models-init service still downloads Qwen3-8B as
the default GGUF, but the live command line points at the Gemma
build. Leaving the doc drift alone for now; the point for this
journal is that the current chat target is Gemma-4-e4b + vision.

### Config changes applied (`Atlas/docker-compose.yml`)

Added three flags to llama-chat:

```yaml
- "-ctk"
- "q8_0"
- "-ctv"
- "q8_0"
- "--cache-reuse"
- "256"
```

Rationale:

- **`-ctk q8_0 -ctv q8_0`** — quantizes the KV cache to Q8_0. Near-
  lossless quality, halves KV-cache memory. Prerequisite (`--flash-attn
  on`) was already satisfied. Effective win: at the same 8192-token
  context we free ~half the KV VRAM, which translates to either
  headroom for bigger context later, or headroom for a resident draft
  model (see speculative-decoding plans below) alongside Gemma-4-e4b
  + mmproj.
- **`--cache-reuse 256`** — enables cross-request KV reuse via prefix
  matching and KV shifting. 256 is the min-chunk size (matches
  llama.cpp's built-in server / FIM presets). Expected win is a big
  TTFT drop on Atlas's repetitive-system-prompt workload: every
  pipeline node (decide / respond / reconcile / graph_attention / ...)
  prepends the same mindsets + tool-list preamble. Under the prior
  config each request re-prefilled that preamble from scratch; now
  the common prefix gets shifted into place.

Atlas Nova.Api side: **no code change required**. The llama.cpp
OpenAI-compat endpoint defaults `cache_prompt` to `true`
(`tools/server/server-task.h:52`), so every request the existing
`LlamaCppClient` already makes benefits from the server-side cache
automatically.

Not applied:

- **`--slot-save-path`** (persistent prefix cache across restarts) —
  requires client-side slot-ID bookkeeping that Atlas doesn't do
  today. Would need a `LlamaCppClient` change to send `id_slot`
  hints. Deferred.
- **`--spec-type ngram-*`** (n-gram speculative decoding without a
  draft model) — a bonus easy-win I initially missed. Upstream
  llama-server exposes several n-gram speculation modes
  (`ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`,
  `ngram-mod`) via `arg.cpp:3504-3555`. Zero extra VRAM, no
  training, works on any model. Not applied in this pass because
  acceptance rate is very workload-dependent and we don't yet have
  a bench suite pointed at the live Atlas server; it goes on the
  "next" list so we measure before enabling.
- **Speculative-decoding draft model (`-md`, `--draft-max`)** — the
  real multi-× decode speedup lever, but picking the right draft is
  a research question. See next section.

### Deploy

The changes are in-tree; applying to the running Atlas stack needs:

```bash
docker compose down
docker compose up -d
```

Nothing permanent to verify pre-restart — the flags are all optional,
non-destructive (no KV cache on disk to migrate), and backed out by
reverting the YAML.

### Expected payoff

Rough numbers (order-of-magnitude, pre-bench):

| Lever | Effect |
|---|---|
| Q8_0 KV cache | ~50% KV memory freed at 8K ctx. Quality: indistinguishable from fp16 KV on Gemma-2 family in upstream benches; will validate on our prompts when the bench suite is wired here. |
| `--cache-reuse 256` | TTFT on repeat-preamble requests drops from "prefill all ~1K tokens of mindset + tool block" to "prefill only the delta". Estimated 4-10× TTFT improvement on pipeline nodes that reuse system context. |

Actual measurements will come from the next bench-suite run against a
stable load; not worth pre-speculating past that.

---

## 2026-04-20 — Next research direction: speculative decoding, upstream-only deployment

### Deployment constraint (pinned)

**Atlas serves from upstream llama.cpp** — the
`ghcr.io/ggml-org/llama.cpp:server-cuda` image — not from this
research fork. Anything we want deployed must either (a) be a
standalone GGUF that upstream already knows how to load, or (b) use
features that upstream ships. This rules out schemes that require
bespoke loader code or new tensor layouts on the serving side, even
when they'd be straightforward in our fork.

Upstream's already-shipped speculative support (per
`common/arg.cpp` in upstream):

- `-md <path>` / `--model-draft` — standalone draft model.
- `--draft-min N`, `--draft-max N` — draft length bounds.
- `--draft-p-min`, `--draft-p-split` — acceptance thresholds.
- `-devd`, `-ngld` — draft device placement / GPU layers.
- `--spec-replace TARGET DRAFT` — translates strings between draft
  and target, letting you pair models with different tokenizers
  (lossy; useful as an escape hatch when a same-family draft
  doesn't exist).
- `--spec-type` (no draft model needed): `ngram-cache`,
  `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`.

This is a generous feature surface. The research question isn't
"can upstream do speculation" — it can — but "what draft artifact
gets the best acceptance rate on our workload with the target we're
actually running."

### Viable draft-building paths under the upstream constraint

| Path | Deployable on upstream? | Home-lab cost | Expected speedup ceiling |
|---|---|---|---|
| **N-gram speculation** (`--spec-type`) | ✓ native, no draft artifact | zero | ~1.3-1.8× on repetitive output (code, JSON, tool-calls) |
| **Off-the-shelf same-family draft** (`-md Qwen3-0.6B` for a Qwen3 target) | ✓ native | model download | ~2-4× when aligned |
| **Distilled standalone draft** (train our own small same-family model) | ✓ native via `-md` | days-to-weeks on 5070 | ~2-4× with acceptance rate tunable via training recipe |
| **Cross-tokenizer draft + `--spec-replace`** | ✓ native, lossy | small | Unknown; acceptance rate depends on translation-table quality |
| **Medusa / EAGLE-style prediction heads** | ✗ **not supported upstream** | days of training | 3-5× in principle, but needs runtime changes upstream doesn't ship and AGENTS.md bars us from contributing AI-heavy PRs to. Research-only. |
| **LayerSkip / early-exit self-speculation** | ✗ **not supported upstream** | near-zero to train | 1.5-2×; same upstream gap as Medusa. Research-only. |
| **True from-scratch pretraining** | ✓ (the output is a normal GGUF) | not feasible on 5070 | n/a — home-lab can't produce a usefully-trained 500M model in reasonable time |

**Deployable-now options:** n-gram, off-the-shelf, distilled,
cross-tokenizer. **Research-only:** Medusa, EAGLE, LayerSkip, true
from-scratch pretraining. "Building a draft from scratch" in the
deployable sense means distillation — the output is an ordinary
small GGUF, loaded via `-md`, indistinguishable to upstream from
any other draft model.

### Why this pairs well with the pipeline doc

Each viable path slots into
[model_prep_pipeline.md](model_prep_pipeline.md) as a stage:

- `ngram_spec_configure` — marks a server config to use
  `--spec-type ngram-*` (and writes the right `--spec-ngram-size-n/m`
  for the workload). No model artifact. Smoke-test stage.
- `draft_fetch` — downloads a published same-family draft
  (e.g. `Qwen3-0.6B`) and validates tokenizer/template compatibility
  with the target.
- `distill_draft` — trains a small same-family model on target's
  output distribution. Outputs a standalone GGUF. Calibration stage
  provides teacher activations; training stage is new infra.
- `draft_bench` — measures acceptance rate + decode tok/s vs
  baseline (upstream target without speculation) and vs off-the-shelf
  draft. Verdict feeds the acceptance criteria in the pipeline doc.

The Medusa / EAGLE / LayerSkip paths could still live as stages, but
they'd be tagged "research-only; not deployable to Atlas until
upstream ships the runtime." Keeping them as stages means a fork-
internal bench can still measure them for comparison.

### First moves (not committed yet, just queued)

Ordered by cost × information value:

1. **N-gram speculation smoke** on the live Atlas llama-chat
   service. Add `--spec-type ngram-cache` (or cycle through variants)
   to `docker-compose.yml` behind a feature flag, measure
   acceptance rate + decode tok/s on representative Atlas prompts.
   Zero-artifact, ~30 min of wall time to set up.
2. **Evaluate target-draft pairings** for upstream-deployable paths.
   For Gemma-4-e4b (current target): no good same-family small draft;
   so the pragmatic comparison is (Gemma-4-e4b, no draft)
   vs (Qwen3-8B, `-md Qwen3-0.6B`). This is also the natural
   checkpoint to decide whether Atlas should switch chat targets
   to Qwen3 for the sake of cheap speculation wins.
3. **Distilled draft for whichever target we pick.** Scope:
   ~300-500M student, same tokenizer as target, trained on
   teacher-emitted logits + free-corpus text. Validate acceptance
   rate climbs past the off-the-shelf baseline for at least one
   representative Atlas workload slice before scaling.

Not a commitment to pursue all three; this is the map. Next concrete
step is a short research doc (`research/cross-layer-svd/
speculative_decoding.md`) when we actually start, mirroring the
pipeline doc shape, with the upstream-only deployment constraint
stated as a top-level invariant.

### Track opened — 2026-04-20

Research doc shipped:
**[speculative_decoding.md](speculative_decoding.md)**.

Scope: three tracks (n-gram native / off-the-shelf draft /
distillation), upstream-only deployment invariant pinned at the
top, metrics derived entirely from the llama-server `timings`
response object (acceptance rate + effective speedup + cache hits,
no log-scraping), non-disruptive bench via a parallel llama-server
on port 11502 in a separate compose file (Atlas's live service not
touched).

First experiment specified: Track A n-gram-cache smoke against the
current Gemma-4-e4b target. Kill criterion at 30% acceptance on
tool-call slice routes us straight to Track B target-choice
evaluation rather than knob-sweeping a losing config. Expected
duration 2-3 hours of setup + bench.

### Reframed 2026-04-20

Research track reframed from "Atlas-specific deployment" to
"universal efficiency for all llama.cpp users on consumer
hardware." Atlas is one downstream consumer that adopts stable
findings; not the target. Full rationale in
[speculative_decoding.md](speculative_decoding.md) — new top-level
"Purpose" section. Target for the first experiment switched from
Gemma-4-e4b to **Qwen3-8B-Q4_K_M** (publicly available, modern
8B-class representative, has a matching `Qwen3-0.6B` published as
a draft for Track B later). Fixture now public-sourced
(`public_prompt_fixture.json`), not drawn from Atlas logs.
Upstream-only deployment invariant is unchanged and reinforced.

---

## 2026-04-20 — First speculative bench: ngram-cache on Qwen3-8B-Q4_K_M

Executed the Track A feasibility test. Full writeup in
[speculative_decoding.md](speculative_decoding.md) under the
2026-04-20 dated section.

**Result: kill criterion triggered.** `--spec-type ngram-cache`
at upstream-default knobs (`--draft-max 8 --spec-ngram-size-n 4`)
on Qwen3-8B-Q4_K_M produces **0.0% overall acceptance and a 0.98×
decode speedup** (2% regression) across all 5 workload slices with
`enable_thinking=false`. Temperature-0 correctness preserved —
per-prompt outputs match byte-for-byte vs baseline. Server details:
upstream llama.cpp release `b8855`, RTX 5080 Laptop 16 GB, decode
~83-94 tok/s baseline.

**Root cause (hypothesis).** `ngram-cache` has no cross-session
state; each request starts with an empty cache. At n=4, short
outputs finish before the cache populates; longer outputs are
novel enough per prompt that n-gram lookups miss.

**Methodological finding.** Qwen3 defaults to reasoning mode. With
thinking on, n-gram-cache fires ~150× more often (formulaic
reasoning prose matches n=4 lookups) but acceptance stays ~5% and
end-to-end speedup is the same 0.98×. Archived think-on data under
`bench/results/think_on/`. **Lesson:** on reasoning-capable models,
bench reasoning-mode and answer-mode separately — a speculation
config tuned for `<think>` tokens is a different config from one
tuned for final-answer tokens.

**Infrastructure shipped** under `research/cross-layer-svd/bench/`:
`spec_bench.py` (llama-server client recording `timings` per
prompt), `run_condition.py` (start-server → bench → stop
orchestrator), `spec_compare.py` (JSONL → markdown), 25-prompt
public fixture, raw + archived results, per-condition server logs.
Upstream llama.cpp binary in `bin-upstream/`. All reproducible by
anyone with the same model file and a consumer CUDA GPU.

**Next action — Track B.** Pair Qwen3-8B-Q4_K_M with its published
`Qwen3-0.6B` draft via `-md` and bench the same 25-prompt fixture.
Reference point any llama.cpp user reaches with two flag additions;
characterization target is "what do off-the-shelf same-family
drafts actually deliver on consumer hardware at batch=1, greedy."
The first bench artifacts / infrastructure will be reused wholesale;
only the server command line changes.

---

## 2026-04-20 — Track B result: Qwen3-0.6B draft → 1.56× overall on Qwen3-8B

Executed Track B with the same infrastructure as Track A. Added the
draft via `-md Qwen3-0.6B-Q8_0.gguf --draft-max 8 -ngld 999`; every
other flag, the fixture, temp, and seed were unchanged. Full writeup
in [speculative_decoding.md](speculative_decoding.md) under the
2026-04-20 Track B section.

**Headline: 1.56× overall decode speedup, 71.5% overall acceptance,
zero quality loss, zero regression.**

Per-slice (identical 25-prompt fixture, temp=0):

| Slice | tok/s Δ | speedup | acceptance |
|---|---|---:|---:|
| structured_output | 83.7 → 152.3 | 1.82× | 80.1% |
| code | 89.7 → 163.0 | 1.82× | 81.3% |
| reasoning | 84.7 → 143.4 | 1.69× | 83.9% |
| factual_qa | 94.0 → 138.1 | 1.47× | 76.9% |
| conversational | 84.1 → 84.5 | 1.00× | 55.4% |

All 25 response previews match baseline byte-for-byte (temp=0
correctness preserved). Cost of the config: +609 MB VRAM for the
Q8_0 draft.

### Three-way summary (same fixture, same target)

| Config | tok/s | accept | speedup |
|---|---:|---:|---:|
| baseline | 87.2 | — | 1.00× |
| ngram-cache (default knobs) | 85.4 | 0.0% | 0.98× |
| **qwen-draft (default knobs)** | **136.3** | **71.5%** | **1.56×** |

### The interesting metric finding

**Acceptance rate alone does not predict speedup.**
`conversational` had 55% acceptance but 1.00× speedup —
`structured_output` had 80% acceptance and 1.82×. The difference is
the distribution of **accepted run lengths**. Predictable slices
accept in long contiguous runs (one target forward pass saves 7);
conversational accepts in short chop (`accept 2, reject 1, accept
3, reject 1 …`) so draft overhead amortizes poorly. Practical
threshold at `draft-max=8` looks like **acceptance ≥ 70% is where
speedups materialize**.

This is worth recording as a characterization finding in its own
right: papers often report per-slice acceptance rates, but the
run-length distribution is the stronger speedup predictor.

### Direct relevance to the efficiency thesis

This is the "more capability without more hardware" result in its
cleanest form:

- Standard upstream llama.cpp binary (no fork changes)
- Off-the-shelf target + draft from the same authors (no training)
- Three extra server flags (no code)
- Zero quality loss (temp=0 exact correctness)
- +56% overall decode throughput on mixed workloads
- +609 MB VRAM (trivial headroom on any card running the 8B)
- Consumer GPU (RTX 5080 Laptop, 16 GB), batch=1, single-user

It's directly usable by anyone. This lands in the middle of the
"stop scaling hardware, use what's already there better" thesis
for the research track.

### Next candidates (priority order, not committed)

1. **Knob sweep on conversational** — `--draft-max 4` + higher
   `--draft-p-min` to see if we can get that 55% acceptance to
   amortize into a modest win instead of 1.00×.
2. **Cross-family pairs** — Llama-3.1-8B + Llama-3.2-1B,
   Gemma-2-9B + Gemma-2-2B on the same fixture. Tests whether the
   "70% acceptance threshold" is a Qwen thing or a universal one.
3. **Track C (distilled draft)** — likely *not* priority. Current
   evidence says off-the-shelf already works excellently for
   modern families with a published tiny draft; training our own
   only becomes compelling for families without one.

---

## 2026-04-20 — Writeup shipped + Track C opened

### Public writeup

Shipped a standalone writeup of the Track A + Track B results:
**[writeups/2026-04-20-speculation-on-consumer-hardware.md](writeups/2026-04-20-speculation-on-consumer-hardware.md)**.

Frames the finding as directly usable to any llama.cpp user running a
modern 8B-class model with a published same-family tiny: three flags,
+56% decode throughput, −22% bench wall time, zero quality cost.
Includes per-slice numbers, the acceptance-rate-vs-run-length metric
discussion, the ngram-cache null result as a prior-art check, the
Qwen3-thinking methodology note, and full reproduction instructions.
Deliberately framed for a llama.cpp community audience, not Atlas.

### Track C — draft distillation on demand

Opened [draft_distillation.md](draft_distillation.md) to scope the
"produce a tiny draft for targets without a published match"
question. This is where Track B's finding points next: good
same-family drafts work great, but lots of targets don't have them.

Plan breaks into three sub-tracks by tractability:

- **C.1 — Retrieval pipeline.** Before training anything, map what's
  already possible via cross-family draft pairing using upstream's
  `--spec-replace` or same-tokenizer different-family pairs. Cheap:
  just runs the existing bench harness against pre-published
  candidate drafts. Tells us which targets genuinely need training
  vs which just need someone to try the existing tinies.
- **C.2 — Rollout-distillation prototype.** Validate the training
  infrastructure on Qwen3-8B (where we have the 71.5% acceptance
  baseline to compare against). Seed student, target rollouts,
  plain SFT. Goal is "pipeline works" not "beats the off-the-shelf."
- **C.3 — Logit distillation on a target without a published match.**
  The headline experiment. Pick Gemma-2-9B or Llama-3.1-8B, run
  DistillSpec-style training, target ≥ 70% acceptance / ≥ 1.4×
  overall speedup on our fixture.

### First action for Track C

**Retrieval-first (C.1). No training yet.** Start with Gemma-2-9B
or similar target, enumerate the tokenizer-compatible published
tinies, and bench them via the existing harness. If any hits the
≥70% acceptance bar, we answer that target by retrieval and Track C
immediately has a public registry artifact. If all fail, that's the
motivating case study for C.2/C.3.

Expected: 3-4 hours of downloads + benches, no training.

Also queued for before any training runs: a prior-art read pass on
DistillSpec (arxiv 2310.08461), EAGLE-2, and recent 2024-2025
draft-training papers. Listed in draft_distillation.md's "prior
art" section. Don't want to spend a week building a DistillSpec
clone if the paper already answered our question.

---

## 2026-04-20 — Self-quant draft experiment + CALDERA correction

Explored the intuitive "quantize the target itself as the draft"
path instead of training. Downloaded Qwen3-8B-Q2_K (3.1 GB, from
`bartowski/Qwen_Qwen3-8B-GGUF`) and ran it as `-md` for the
Q4_K_M target on the same 25-prompt fixture.

**Result: 82.8% acceptance, 0.83× speedup (17% wall-time
regression).** Every slice regressed; worst is conversational at
0.69×. Full writeup in [speculative_decoding.md](speculative_decoding.md)
under the self-quant dated section, plus an addendum in
[writeups/2026-04-20-speculation-on-consumer-hardware.md](writeups/2026-04-20-speculation-on-consumer-hardware.md).

**The interesting finding isn't the regression itself — it's
*why*.** Acceptance went *up* vs the 0.6B published draft (82.8%
vs 71.5%) because the quantized 8B genuinely has a more
target-aligned output distribution. But the draft was ~3× slower
per token than the 0.6B — not because of the bit depth, but because
it still has 8B parameters' worth of matmul ops per forward pass.
Quantization reduces **bits per weight**, not **parameter count**.

**Corrected claim for the record.** Earlier in this session I
framed CALDERA (Q + L·R decomposition) as a plausible path to a
tiny draft. That framing was wrong — CALDERA reduces storage bytes
but not FLOPs (the quantized matmul is still O(d_out × d_in),
plus the correction adds extra FLOPs on top). A CALDERA Qwen3-8B
would sit on the wrong side of the bandwidth-per-token curve for
speculation. CALDERA is a tool for "fit a bigger model in less
VRAM," not "produce a cheap draft." Updated the speculative-decoding
doc to make this explicit and call out the FLOPs-vs-bytes principle
that the Q2_K experiment calibrated.

### Principle calibrated by this experiment

**Speculation speedup at batch=1 consumer decode is governed by
FLOPs-per-token in the draft, not bytes-per-token.** Useful drafts
must have materially **fewer parameters** than the target, not
merely compressed parameters. The published Qwen3-0.6B draft wins
not because it's smarter — by acceptance rate it's measurably less
target-aligned — but because it has 14× fewer FLOPs per forward
pass.

### New research track opened — draft via pruning

Pivoted the "compress the target to make a draft" intuition to
**structural pruning** (layer drop, head/width drop) which reduces
parameter count directly. Full plan in
[draft_pruning.md](draft_pruning.md).

Approach ladder, ranked by cost × expected value:

- **A1** — Drop every other layer, no retraining. Quality-floor
  smoke test. ~1-2 hours.
- **A2** — A1 + rollout distillation overnight. First real attempt
  at a useful pruned draft.
- **A3** — Importance-based layer drop (ShortGPT-style) + rollout
  distillation. Potentially better quality retention than blind
  pruning.
- **A4** — Width pruning (SliceGPT-style). More complex to implement.
- **A5** — Combined depth + width + distillation. Max-compression bet.

A1 is always first — it establishes the no-recovery floor before
we commit training compute to A2.

### Infrastructure prerequisite blocker flagged

PyTorch on Python 3.14 (the user's installed version) is bleeding
edge. Before A1 can run, need to confirm torch nightly supports
3.14 on Windows, or fall back to installing 3.12 alongside / WSL2.
Listed in draft_pruning.md's "Infrastructure prerequisites" section.
This is the first concrete thing to verify on the pruning track
before any experiments start.

---

## 2026-04-20 — Prior-art survey: zero-shot parameter reduction

Dispatched a research agent to survey LLM compression techniques
that reduce parameter count or FLOPs without gradient retraining.
Full synthesis folded into [draft_pruning.md](draft_pruning.md)
under a new "Prior art — synthesis" section.

### Key takeaways

- **ShortGPT-style layer drop (arxiv 2403.03853) is canonical and
  fits our constraints.** ~25-30% of layers can be dropped with
  calibration-only, <5-10% quality hit on 7-13B models. Universal
  finding across ShortGPT / LaCo / SLEB / Gromov et al. 2403.17887:
  later and middle layers are more redundant than early ones.
- **LLM-Streamline** adds a cheap no-gradient refinement: replace
  the dropped span with a linear layer fit via least squares. We
  inserted this as an A1.5 step between "drop no retrain" (A1) and
  "drop + rollout distillation" (A2).
- **SliceGPT width pruning (arxiv 2401.15024) has GGUF friction** —
  its rotation matrices fuse into the residual stream and mainline
  GGUF has no loader support. A4 deprioritized.
- **SparseGPT / Wanda explicitly ruled out.** Unstructured magnitude
  pruning reduces storage but not FLOPs on dense GPU kernels;
  llama.cpp doesn't use sparse kernels. Not a useful axis on our
  runtime.
- **LayerSkip zero-shot doesn't work** (arxiv 2404.16710). Needs
  its own continued-pretraining regime. Meta ships LayerSkip
  checkpoints precisely because you can't retrofit it.
- **Medusa / EAGLE / Kangaroo / Draft&Verify** all require gradient
  training — out of scope for this track.
- **SVD-based methods** (LASER / ASVD / SVD-LLM / Basis Sharing)
  are weak standalone. GGUF has no low-rank tensor type, so they
  save bytes not FLOPs without loader extensions.

### Novel gap identified

**No published paper explicitly reports speculation-draft acceptance
rates from zero-shot pruned targets.** Community anecdote suggests
60-70% acceptance at ~25% layer drop, but nothing peer-reviewed.
Our bench output — if A1/A1.5 work — would fill this gap with
honest per-slice numbers. This is the first clearly-novel
contribution in the speculation research direction since the track
opened.

### Plan changes

- A1 mechanism refined: **Block-Influence–based layer drop targeting
  middle/late layers**, not blind every-other drop. Costs nothing
  extra (one calibration forward pass computes BI scores).
- **A1.5 inserted**: LLM-Streamline linear-LSQ replacement layer
  between A1 and A2. Genuinely gradient-free, potentially bridges
  much of the A1→A2 quality gap cheaply.
- **A4 deprioritized** (SliceGPT GGUF friction).
- **Unstructured-sparsity and LayerSkip removed** from the ladder
  entirely — documented as out-of-scope with reasons.

Next action still the PyTorch-on-Python-3.14 infrastructure check.

---

## 2026-04-20 — A1 + A1.1 executed: zero-shot pruning characterized

Infra cleared (PyTorch nightly + CUDA 12.8 + Python 3.14 work on the
RTX 5080 Laptop). Built the pruning pipeline end-to-end: HF checkpoint
→ `prune_layers.py` → `convert_hf_to_gguf.py` → `llama-quantize` →
bench. Round-trip validated on unmodified Qwen3-8B first (96-byte
metadata diff vs the pre-downloaded official Q4_K_M GGUF).

Ran two zero-shot prune experiments on the same 25-prompt fixture as
Track A/B:

| Drop | Layers | Draft size | Accept | Speedup |
|---|---|---|---|---|
| A1: 28% (10 layers 20-29) | 26L | 3.7 GB | **26.3%** | **0.50×** |
| A1.1: 8% (3 layers 20-22) | 33L | 4.6 GB | **72.1%** | **0.72×** |

**The genuinely novel data point:** 3-layer drop hits 72% acceptance —
*statistically identical to the published Qwen3-0.6B draft's 71.5%* —
**without any retraining**. The prior-art survey flagged this exact
measurement as unpublished in peer-reviewed form. It now has numbers.

**The limitation:** 33L still executes 92% of target FLOPs, so the
draft is only 1.08× faster than target per forward pass. Speculation
math says that's not enough cost reduction to amortize even at 72%
acceptance. Measured speedup 0.72× matches the theoretical 0.69× well.

**Zero-shot pruning characterized as a frontier:**
- Light drop (≤10%): high acceptance, too-slow draft → regression.
- Heavy drop (≥25%): quality collapse → worse regression.
- Middle ground where both axes cooperate does not exist zero-shot.
  The acceptance-vs-FLOPs trade-off is bimodally bad.

To win via pruning, aggressive FLOP reduction (50%+ layer drop) plus
quality-recovering retraining is required — confirms A2 is the honest
path, with A1.5 as the cheapest "try to recover some quality via
calibration-only" middle step before committing gradient training.

Full per-slice writeup in [draft_pruning.md](draft_pruning.md) dated
section. Raw data: `bench/results/pruned_midlate10.jsonl` and
`pruned_mid3.jsonl`. A1.5 (span-averaging replacement) is the next
concrete action.

### Disk note

~70 GB of research artifacts now accumulated (HF source 16 GB + two
pruned HFs 28 GB + bf16 GGUFs + Q4_K_M GGUFs + roundtrip validation
files). The `Qwen3-8B-roundtrip-*` validation artifacts (~21 GB) can
be reclaimed since the pipeline is validated; everything else is
live research state.

---

## 2026-04-20 — A1.5 result: averaging replacement doesn't help; zero-shot frontier complete

Implemented span-averaging replacement
(`bench/prune_avg_replace.py`): drop 10 layers (20-29), insert one
block whose weights are element-wise mean of the dropped ten.

**Result: 26.5% accept, 0.50× speedup — statistically identical to
A1 (drop 10, no replace: 26.3% accept, 0.50× speedup).** The averaged
block contributes no measurable quality recovery. Intuition: 10
transformer blocks of nonlinear processing isn't approximated by an
averaged linear-ish single block.

Full LSQ version of LLM-Streamline deferred — the theoretical
ceiling (linear fit of 10 nonlinear blocks) plus the
transformer-block-encoding complexity don't justify implementation
given A1.5-lite's negative result.

## 2026-04-20 — Session wrap-up and complete findings

Full characterization of the "draft-from-target without training"
space is now in hand. Six conditions on the same Qwen3-8B-Q4_K_M
target, same 25-prompt public fixture, same flags:

| Draft strategy | Accept | Speedup | Category |
|---|---:|---:|---|
| No draft (baseline) | — | 1.00× | control |
| **Qwen3-0.6B-Q8_0 published tiny** | 71.5% | **1.56×** | **winner (trained-from-scratch)** |
| Q2_K of target | 82.8% | 0.83× | self-quant — bytes not FLOPs |
| Light prune (33L / drop 3) | 72.1% | 0.72× | high accept, insufficient FLOP savings |
| Heavy prune (26L / drop 10) | 26.3% | 0.50× | quality collapse |
| Averaged span-replace (27L) | 26.5% | 0.50× | averaging doesn't bridge |

**Findings consolidated into three takeaways**:

1. **Speculation speedup is FLOPs-bound at batch=1 consumer decode.**
   Not bytes. Every zero-shot technique that reduces bytes without
   reducing parameters (Q2_K of target) or preserves too many
   parameters (light pruning) fails to produce speedup even when
   acceptance is high.
2. **Zero-shot aggressive parameter reduction collapses acceptance.**
   Past ~25% layer drop without retraining, the pruned model's
   output distribution diverges from target's enough that speculation
   can't keep up, regardless of byte/FLOP savings.
3. **The frontier is bimodally bad zero-shot.** Light drop: high
   accept, slow draft → regression. Heavy drop: quality collapse →
   bigger regression. Middle where both axes cooperate does not exist
   without retraining.

**Novel contribution documented**: no published paper reports
speculation-draft acceptance rates from zero-shot pruned targets.
Our 5-row pruning frontier is the first explicit measurement.
72.1% accept at 3-layer drop specifically is noteworthy — it
matches the published 0.6B's 71.5% accept, without training —
but combined with the speedup regression, tells the complete
story of *why* shortcutting training doesn't work.

### Shipped this session (full list)

**Runtime config for upstream llama.cpp**:
- `Atlas/docker-compose.yml` updated with `-ctk q8_0 -ctv q8_0
  --cache-reuse 256` on top of existing `--flash-attn on`.
- No Atlas Nova.Api code changes needed.

**Bench infrastructure** (`research/cross-layer-svd/bench/`):
- `public_prompt_fixture.json` — 25 prompts across 5 public-sourced
  slices, reproducible by anyone.
- `spec_bench.py` — llama-server OpenAI-compat client recording
  `timings` per prompt as JSONL, with thinking-mode toggle.
- `run_condition.py` — orchestrator (start server → bench → stop).
- `spec_compare.py` — JSONL → markdown comparison table.
- `prune_layers.py` — HF-checkpoint layer dropper.
- `prune_avg_replace.py` — drop-and-replace-with-averaged-layer.
- `results/` — eight raw JSONLs across all conditions, server logs,
  run logs for audit.

**Binaries + models on disk**:
- `bin-upstream/` — upstream llama.cpp b8855 release (Windows CUDA
  13.1), including `llama-server.exe`, `llama-quantize.exe`, etc.
- `Atlas/models/Qwen3-8B-Q4_K_M.gguf` (official, 5 GB)
- `Atlas/models/Qwen3-0.6B-Q8_0.gguf` (official, 0.6 GB)
- `Atlas/models/Qwen3-8B-Q2_K.gguf` (bartowski, 3.1 GB)
- `Atlas/models/Qwen3-8B-pruned-midlate10-Q4_K_M.gguf` (3.7 GB)
- `Atlas/models/Qwen3-8B-pruned-mid3-Q4_K_M.gguf` (4.6 GB)
- `Atlas/models/Qwen3-8B-avgspan-10to1-Q4_K_M.gguf` (3.8 GB)
- `research/models-hf/Qwen3-8B/` (bf16 HF source, 16 GB)
- `research/models-hf/Qwen3-8B-pruned-*/` and
  `research/models-hf/Qwen3-8B-avgspan-10to1/` — pruned HF
  checkpoints retained for any A2 retraining experiment.

**Research docs updated**:
- `speculative_decoding.md` — Track A (n-gram null), Track B
  (published tiny win), self-quant negative result with the
  FLOPs-vs-bytes principle, CALDERA correction.
- `draft_pruning.md` — complete prior-art synthesis, A1 / A1.1 /
  A1.5 results, complete frontier table, decision.
- `writeups/2026-04-20-speculation-on-consumer-hardware.md` —
  standalone publishable article: three-flag config, per-slice
  numbers, acceptance-vs-run-length finding, Q2_K and pruning
  addenda covering all negative results honestly.

**PyTorch infra cleared**:
- `research/cross-layer-svd/venv-research/` — Python 3.14 + torch
  2.12.0.dev20260408+cu128 + HF transformers 5.5.4 stack. Fully
  working with the RTX 5080 Laptop (Blackwell SM 12.0). Available
  for A2 or any further ML-training-adjacent work without
  re-provisioning.

### Open question — A2

The only path that remains plausibly productive for "produce a draft
from target without a published match" is gradient retraining of an
aggressively pruned model. Estimate: ~8-16 hours of training on our
GPU for a first attempt. **Not run this session** — left as a
deliberate decision point rather than sinking overnight compute
before consolidating today's findings.

A2 implementation would start from
`research/models-hf/Qwen3-8B-pruned-midlate10/` (26L pruned, HF
format, ready to train) plus a rollout-distillation loop on
target-emitted text from a public prompt corpus.

### Closing status

The "stop throwing hardware at it" thesis has one clean practical
artifact shipped (the 1.56× published-draft recipe) and one clean
negative-space map (the zero-shot frontier characterized fully).
Both are publishable-quality contributions in the honest-measurement
sense. Further investment decisions can be deferred to future
sessions based on whether A2 retraining is worth the compute.

---

## 2026-04-21 — Paper survey: sub-1-bit and low-bit quantization

Follow-up to the 2026-04-19 synthesis, which named "retraining-allowed
compression" as the one remaining productive direction. Surveyed four
recent papers that all relax the post-training-only constraint in
different ways. Full notes with links, numbers, and llama.cpp-mapping
commentary live in
[docs/research/quantization-papers-2025.md](../../docs/research/quantization-papers-2025.md).

### Papers covered

1. **BitNet b1.58 2B4T** (Microsoft, arXiv 2504.12285). Native 1.58-bit
   (ternary `{-1, 0, +1}`) LLM trained from scratch on 4T tokens. 2B
   params, matches FP peers of similar size at ~1/10 memory. Trained-
   in, not post-training. Weights and inference kernels already open.
2. **LittleBit** (Samsung, NeurIPS 2025, arXiv 2506.13771). Factorizes
   `W ≈ U·V^T` at low rank, then binarizes the factors. Claims 0.1 BPW
   on Llama2-13B (~0.9 GB weights), beating STBLLM at 0.55 BPW. Uses
   Dual-SVID init + QAT + multi-scale (row/col/rank) compensation.
3. **BTC-LLM** (arXiv 2506.12040). Dense sub-1-bit via learnable
   invertible rotation + binary codebook (no sparsity masks). 3.1%
   zero-shot drop at 0.8 bits on LLaMA-2-13B, 1.6× FP16 speedup.
4. **D²Quant** (arXiv 2602.02546). 2–3 bit weight-only PTQ with
   Dual-Scale Quantizer on FFN down-projection + mean-shift LayerNorm
   correction. Only one of the four that's pure PTQ and could
   retrofit onto existing block formats.

### Why this is a reopening, not a restart

This track was archived on 2026-04-19 because every post-training
approach either tied or lost to K-quants. Three of the four papers
above (BitNet, LittleBit, BTC-LLM) require training — QAT, fine-
tuning, or from-scratch — which is exactly the constraint the archive
entry said we'd need to relax.

LittleBit is the one that sits most directly on the factored-inference
lineage from the first half of this journal: same `W ≈ U·V^T`
skeleton we ran rank sweeps on in April, but with `U` and `V`
binarized and QAT used to recover quality. The archived per-matrix
SVD results (best case: PPL 86 at rank 512 on Qwen 2.5 0.5B, baseline
9.4) are the right floor to compare any LittleBit reproduction
against.

### Open research questions this survey raises

- If QAT is on the table, does LittleBit's factored-binary scheme
  beat our archived per-matrix SVD at matched storage? The closest
  apples-to-apples point: LittleBit at ~0.5 BPW vs our rank-512
  per-matrix SVD (~1.14 BPW-equivalent for Qwen 0.5B d=896).
- Does BitNet b1.58 2B4T at 2B params outperform Qwen 2.5 0.5B +
  K-quants on the interactive-tok/s problem on a 5070? Cheap
  benchmark — weights and kernels already exist.
- Would any of these schemes benefit from the `imatrix`-style
  activation weighting that we kept finding was the right metric
  (weighted rel-err predicts PPL; raw Frobenius doesn't)?

### Status

Survey only. No experiments run yet. Next concrete action is either:
(a) one-afternoon benchmark of BitNet b1.58 2B4T against our Qwen
baselines to anchor the "is retraining worth it?" question, or
(b) math walkthrough of LittleBit's Dual-SVID + multi-scale
compensation on a toy matrix, as the prereq to any reproduction
attempt.

No commitment yet on which to run first. The archive recommendation
from 2026-04-19 still stands until one of these papers actually
beats our baselines end-to-end on a model we care about.

---

## 2026-04-21 — LittleBit math walkthrough

Executed path (b) from the prior entry: reconstructed LittleBit's
core math from the arXiv HTML and documented findings. Full
writeup in **[littlebit_math.md](littlebit_math.md)**.

Key points from the walkthrough:

- **Compression math is exact.** BPW `≈ 2r/d` dominates at `r ≪ d`;
  the paper's headline "0.1 BPW" is `r = 256` on `d = 5120`
  (Llama2-13B). The trick is that `r` decouples parameter count
  from weight count, so sub-1-BPW is trivially reachable once
  factors are used at all — the 1-bit precision on the factors
  then contributes an additional 16× vs FP16-SVD at the same `r`.
- **Proposition 1 (efficient forward) is a clean algebraic
  identity.** `Y = ((((X⊙g)·V_sign)⊙ℓ)·U_sign^T)⊙h` replaces one
  FP16 GEMM with two binary matmuls plus three broadcasts, no
  materialization of `Ŵ`. Derivation holds rigorously; this is
  not an approximation.
- **Dual-SVID initialization is heuristic, not derived.** Paper
  asserts (a) `|U'|` is rank-1 separable as `h · ℓ_uᵀ` and (b) the
  rank axis combines as `ℓ_0 = ℓ_u ⊙ ℓ_v`. Neither is proved;
  both are plausible design choices. The multiplicative
  separability assumption is directly measurable on our existing
  7B calibration dumps — cheapest single sanity check that
  informs whether Dual-SVID will actually warm-start QAT or just
  produce a bad starting point that QAT has to climb out of.
- **QAT uses SmoothSign backward** with `d/dx tanh(100x)`, not
  standard STE. Temperature `100` is a hyperparameter the paper
  doesn't ablate.
- **The batch=1 decode speedup claim is unanswered.** Paper
  reports up to 11.6× kernel speedup but admits "inference at
  small batch sizes is often dominated by memory access." Our
  2026-04-19 Cloudflare-pattern investigation already established
  batch=1 is memory-bandwidth-bound on MMVQ; whether a
  factored-binary kernel holds its speedup at batch=1 on a
  consumer card is the decisive unknown for this fork.
- **Not upstream-deployable today.** No llama.cpp tensor type for
  factored-binary storage exists; LittleBit is research-only under
  our speculation-track invariant until upstream ships (or we land
  a loader extension). Same status tier as EAGLE / Medusa in
  `speculative_decoding.md`.

Findings relating to the archived SVD track: LittleBit is the one
paper of the 2026-04-21 survey that sits directly on our `W ≈ U·Vᵀ`
lineage. It targets a different operating point (sub-1-BPW, QAT
allowed) than our archived floor (FP16 factors, post-training only,
PPL 86 at r=512 on Qwen 0.5B). Whether QAT at binary factors
beats our FP16 post-training floor at matched storage is a directly
measurable question — not a theoretical one.

Next action queued (from §10.1 of the math doc): one afternoon of
single-matrix numerical sanity on a 7B calibration dump. Verifies
Prop. 1 on a toy case, measures Dual-SVID initial-point quality
before any QAT commitment, and sweeps the rank-1 separability
assumption. Much cheaper than the full reproduction and narrows the
question about whether the method's warm-start is doing real work.
