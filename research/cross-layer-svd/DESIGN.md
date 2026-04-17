# Factored Inference — Phase 0 Design Document

**Status:** Phase 0 (investigation + design). No kernel code yet.
**Date:** 2026-04-17
**Supersedes decisions in:** earlier sections of `JOURNAL.md`.

---

## 1. Executive summary

We're building a post-training, model-agnostic weight decomposition + streaming inference runtime in this `llama.cpp` fork. The decomposition math borrows established techniques from the 2024 LLM-compression literature. The runtime — specifically a CUDA kernel that executes `y = B(A_i·x)` with PCIe-overlapped coefficient streaming — is the novel contribution; nobody in the open-source ecosystem has shipped one.

**Chosen decomposition stack:** windowed Basis Sharing (window=2) + balanced-truncation weighting + SparseGPT-style OBS closed-form repair. Fallback for extreme compression: CALDERA (2-bit LDLQ + 4-bit rank-k correction).

**Chosen runtime architecture:** custom `GGML_OP_FACTORED_LINEAR` op, two CUDA streams (compute + transfer), pinned host memory with write-combined flag, L2 persistent window for the shared basis. GGUF naming convention (`.basis` / `.coeffs` sibling tensors), no GGUF version bump.

**Fork strategy:** stays private. Upstream maintainer stance is hostile to streaming runtimes (see §7). We benchmark against vanilla llama.cpp rather than attempting to merge.

---

## 2. Decomposition math (what goes into the converter)

### 2.1 Basis Sharing (arxiv:2410.03765)

> Note: earlier notes in this repo cited arxiv:2410.07383 — that was a different paper (SparseGrad). Correct ID is **2410.03765**.

For each weight role R across L layers, form a group of **window_size = 2** adjacent layers and compute one shared basis:

```
W_stack   = [W^(1) | W^(2)] ∈ R^(d_out × 2·d_in)       (horizontal concat)
X_stack   = [X^(1); X^(2)] ∈ R^(2·m × d_in)             (vertical concat of activations)
S         = cholesky(X_stack^T · X_stack)               (whitening factor, FP64)
U, Σ, V^T = svd(W_stack · S)                            (activation-weighted SVD)
B_shared  = (S^-1) · U_k                                 (basis, d_out × k, resident on GPU)
C^(i)     = Σ_k · V_k^T[:, i·d_in : (i+1)·d_in]         (per-layer coeffs, k × d_in)
W^(i) ≈ B_shared · C^(i)
```

**Critical details from the paper:**
- `window_size = 2` is the only safe choice. Paper's ablation on LLaMA-7B at 50% compression: window=1 (no share) = PPL 23.97, **window=2 = 19.99**, window=8 = 27.92, window=32 = 85.24. The sweet spot is narrow.
- **Roles to share:** `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`. **Do NOT share:** `o_proj` and `down_proj` — the paper's Fig. 4b shows sharing hurts on these. Those stay per-matrix (use balanced-truncation SVD there, see §2.2).
- **FP64 throughout the decomposition.** Cholesky of the activation covariance is numerically unstable in FP32 on 7B-scale matrices.
- **Sequential recalibration:** after compressing group i, re-run calibration for group i+1 using the already-compressed upstream weights. Skipping this is meaningfully worse.
- Paper uses 256 samples × 2048 tokens from WikiText-2 for calibration.
- Published results on LLaMA-7B at 50% compression: Wiki2 PPL 19.99 vs unquantized 5.68.

### 2.2 Balanced truncation for non-shared weights

For `o_proj` and `down_proj` (per-matrix, no cross-layer sharing), use the balanced-truncation weighting from our whole-model experiments: SVD of `S_out · W · S_in` where `S_in = (X^T X)^(1/2)` and `S_out = (G^T G)^(1/2)` from a backward calibration pass. Our 0.5B results showed 5-20% PPL improvement over pure ASVD at matched rank. (See `balanced_test.py`.)

### 2.3 SparseGPT-style OBS repair (our extension)

SparseGPT (arxiv:2301.00774) gives a closed-form weight update that, after pruning entry `W[i,j]`, analytically redistributes the error into the remaining weights to minimize `||W·X - W'·X||`. The math extends cleanly to SVD truncation:

For a truncated direction `v ∈ R^(d_in)` (a discarded right singular vector) and Hessian `H = X·X^T + λI`:

```
W' = W − (W·v) · (v^T · H^-1 · v)^-1 · (H^-1 · v)^T
```

For k dropped directions stacked as `V_k ∈ R^(d_in × k)`:

```
W' = W − (W · V_k) · (V_k^T · H^-1 · V_k)^-1 · (H^-1 · V_k)^T
```

Cost per matrix: one Cholesky of `H` (shared across all output rows), one k×k solve per truncation batch. The paper's magnitude-pruning vs OBS-pruning gap on OPT-175B was near-random vs PPL+0.3 — suggesting the "project and repair" upgrade over our current "project and pray" SVD could be significant. Reference implementation: [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt).

**Open question:** the paper only validates OBS on entry-wise pruning, not SVD truncation. The derivation above is ours. Validate empirically on a single `gate_proj` before relying on it.

### 2.4 CALDERA (arxiv:2405.18886) — secondary path

For extreme compression beyond Basis Sharing's reach, `W ≈ Q(W) + L·R` where:
- `Q(W)` uses **LDLQ** quantization (from QuIP) with the **E8 lattice codebook** (from QuIP#) — NOT GPTQ, NOT plain per-channel int4
- `L·R` is rank 64-256, stored at 4 bits (also E8-lattice-quantized)
- Randomized Hadamard transform is applied as incoherence preprocessing

**Reported numbers on LLaMA-2-7B with `k=256`, 2.4 avg bits:** WikiText2 PPL 6.19 (unquantized: 5.12). On LLaMA-2-70B `k=256, 2.2 bits`: PPL 3.98 (unquantized: 3.12).

**Bit/VRAM math for 70B:**
- CALDERA's lowest validated config is ~1.9 avg bits (`k=64, B_Q=2, B_{LR}=4`) → ~16.6 GB resident
- **Still doesn't fit 12 GB** — even CALDERA requires streaming for our target
- At ~1.37 bits we'd fit; that's outside their validated envelope

**Decision:** CALDERA is a Phase 4 investigation if Basis Sharing falls short. Not on the critical path. Implementation complexity is significantly higher (LDLQ + E8 lattice + RHT are all non-trivial).

---

## 3. Runtime architecture

### 3.1 Abstraction: `FactoredLinear`

A new ggml op `GGML_OP_FACTORED_LINEAR` with three inputs:
```cpp
// Signature (mirrors existing ggml ops)
ggml_tensor * ggml_factored_linear(
    ggml_context * ctx,
    ggml_tensor * basis,    // d_out × k, fp16, resident on GPU
    ggml_tensor * coeffs,   // k × d_in, fp16 or int8, pinned host RAM
    ggml_tensor * x         // activation, seq × d_in
);
```

**Why a custom op, not a new tensor type:** the Basis/coeffs are two separate tensors with different lifecycles (one resident, one streamed), and we need the op to receive them both explicitly to coordinate the transfer stream. From the ggml recon (see §9): `GGML_OP_FACTORED_LINEAR` is ~100 lines of boilerplate vs ~30 for a new `GGML_TYPE_*`, but the type approach can't express "this weight has a streaming component."

The op dispatches to `factored-linear.cu` (new file), which does:
```
  y_tmp = cuBLAS gemm(A_i, x)           // two GEMMs: first into rank-k space
  y     = cuBLAS gemm(B_shared, y_tmp)  // then back to d_out
```
Reuses existing `ctx.cublas_handle()` and `ctx.stream()` infrastructure. ~150 lines of CUDA total.

### 3.2 Streaming state machine

Two CUDA streams:
- **`compute`** — main stream, runs the two-GEMM forward
- **`transfer`** — created with `cudaStreamNonBlocking`, dedicated to `cudaMemcpyAsync` of per-layer coeffs

Ring buffer on GPU: **3 landing zones** per shared-basis group (triple-buffering handles layer-compute-time variance; paper's CUDA C++ Programming Guide §6.2.8 + Mark Harris's canonical post confirm triple is the safe default).

**Per-layer sequence:**
```
layer N compute launches on compute stream
   └─ reads basis (resident) + coeffs[slot N mod 3] (already DMA'd)
   └─ records event_compute_done_N on compute stream

layer N+1 transfer launches on transfer stream
   └─ waits on event_compute_done_(N-2)  (slot freed after triple-buffer shift)
   └─ cudaMemcpyAsync: host coeffs → coeffs[(N+1) mod 3]
   └─ records event_transfer_done_(N+1)

layer N+1 compute launches on compute stream
   └─ waits on event_transfer_done_(N+1)
   └─ (proceeds)
```

All events created with `cudaEventDisableTiming` — materially cheaper.

### 3.3 Host-side memory

**Pinned host pool** allocated once at model load via `cudaHostAlloc(size, cudaHostAllocWriteCombined)`. Write-combined is faster H2D and we never read from it host-side. Pool size:
```
  pool = (largest coeff tensor bytes) × (num layers per role) × (roles sharing)
  roughly 70B case: ~3 GB × 5 shared roles ≈ 15 GB pinned
```

Fallback to pageable malloc if pinning fails (existing ggml-cuda already has this fallback at `ggml-cuda.cu:1334`).

### 3.4 L2 cache residency (Blackwell)

RTX 5070 L2 = 48 MB. The shared basis is small enough (e.g., 2048×k fp16 at k=512 = 2 MB per role) that 5-10 roles can fit. Use `cudaStreamSetAttribute(compute_stream, cudaStreamAttributeAccessPolicyWindow, {basis_ptr, basis_size, 1.0, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming})` to hint persistence. Budget via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 16 * 1024 * 1024)`.

### 3.5 PCIe math on RTX 5070

Gen5 x16 sustained H2D ≈ **55-60 GB/s** (per 2024-25 NVIDIA Blackwell docs and empirical reports). For Basis Sharing at window=2, rank=512 on a 70B model, per-layer coefficient transfer ≈ 512 × 8192 × 2 bytes = **8 MB per shared role per layer pair**. 5 shared roles × 80 layers / 2 window = 1600 MB total stream per token pass. At 60 GB/s sustained that's **27 ms of transfer** — hideable under the ~200-400 ms of compute for a 70B decode token. PCIe overlap is not the constraint here; kernel launch overhead on small k might be.

### 3.6 Known bugs to avoid

From the upstream archaeology, these are directly on our path and **closed as "not planned"** (we must work around, not rely on fix):
- [#18310](https://github.com/ggml-org/llama.cpp/issues/18310) — missing `synchronize()` after async tensor copy causes race on defrag/realloc
- [#18313](https://github.com/ggml-org/llama.cpp/issues/18313) — `ggml_backend_cuda_get_tensor_async` missing `ggml_cuda_set_device()` + scheduler reset sync

Reading these threads before writing code is mandatory.

---

## 4. GGUF format and loader changes

### 4.1 Naming convention (no version bump)

For each weight that's been factored, emit sibling tensors:
```
blk.12.attn_q.weight           → [d_out × k]  fp16   basis (shared across window)
blk.12.attn_q.weight.coeffs    → [k × d_in]   fp16   per-layer coefficients
blk.12.attn_q.weight.group     → scalar int   window group ID (layers with same group share basis)
```

**Why sibling tensors instead of new GGUF type:**
- GGUF's tensor type enum is fixed (42 entries); adding requires both loader and writer changes plus compatibility breakage
- Old loaders without factored-weight support will load the basis as a normal tensor and ignore `.coeffs` → no crash, just won't use compression
- No GGUF_VERSION bump needed

### 4.2 Python converter

New script: `gguf-py/scripts/convert_factored.py`. Reads a standard GGUF, applies our decomposition pipeline (Basis Sharing with window=2 on Q/K/V/Gate/Up, balanced-truncation per-matrix on O/Down, OBS repair as final pass), writes a factored GGUF.

### 4.3 Model loader changes

In `src/llama-model-loader.cpp`:
- After normal tensor load loop, scan for tensors with `.coeffs` suffix
- For each, find its sibling `.basis` and `.group`
- Emit a `factored_weight_t` struct: `{basis_tensor*, coeffs_tensor*, group_id}`
- Expose via existing tensor-lookup API so model graph builder can find it

In `src/llama-model.cpp`:
- In each `build_*` function that does `ggml_mul_mat(ctx, weight, act)`, check if `weight` has a factored representation; if so, emit `ggml_factored_linear(ctx, basis, coeffs, act)` instead
- Only `ffn_gate`, `ffn_up`, `attn_q/k/v` paths need this — the rest keep ordinary MUL_MAT

---

## 5. Phases and sequencing (revised)

| Phase | Scope | Duration | Validation gate |
|-------|-------|----------|-----------------|
| **0** (this doc) | Design, investigation, research aggregation | 1-2 days | this document exists + reviewed |
| **1** | Python converter: standard GGUF → factored GGUF via Basis Sharing + balanced + OBS | 4-6 days | reproduces paper's LLaMA-7B @ 50% PPL 19.99 within tolerance |
| **2** | Loader + model graph builder recognize factored weights; fall back to "materialize dense" for correctness baseline | 2-3 days | end-to-end runs with factored GGUF producing identical tokens vs vanilla llama.cpp at FP16 |
| **3** | **Streaming runtime:** pinned pool, two CUDA streams, `GGML_OP_FACTORED_LINEAR` kernel, double-buffered transfer. Initially with factored weights that are just identity-split (correctness only) | 1.5-2 weeks | end-to-end tokens match Phase 2 baseline within fp16 tolerance |
| **4** | Plug in real Basis Sharing coefficients; benchmark PPL + tok/s vs vanilla `-ngl` split on 70B | 3-5 days | PPL within paper's bounds, tok/s comparable or better than vanilla offload |
| **5** | Performance tuning: nsys profiling, stream/event placement, kernel fusion, CUDA graph capture for decode loop | 1-2 weeks | demonstrated overlap (PCIe + compute concurrent on nsys timeline), throughput target met |
| **6** (optional) | CALDERA path: Q+LR decomposition for sub-2-bit envelope; additional kernels | 2-3 weeks | 70B runs on 12 GB VRAM with usable PPL |

Total through Phase 5: **~5-7 weeks** of focused work. Phase 6 deferred unless Basis Sharing doesn't compress enough.

---

## 6. Open questions (must resolve before starting Phase 1)

1. **OBS repair on SVD truncation — does it help empirically?**
   Our derivation of the SVD-truncation analog (§2.3) isn't in the SparseGPT paper. Validate on a single `gate_proj` before committing to the full pipeline — it may already be captured by balanced truncation, or may add meaningful lift.

2. **Window=2 + which layers pair together?**
   Paper does contiguous grouping (layers 0+1, 2+3, ...). Does learned pairing by weight similarity help? *Probably not worth exploring; contiguous is the published recipe.*

3. **Calibration corpus — WikiText-2 or something broader?**
   Paper uses WikiText-2 × 256 × 2048. In-domain data usually wins. For Atlas's use case (tool-calling, agent chat), is calibration on WikiText biased in ways that hurt? Probably worth a calibration-sensitivity ablation at some point.

4. **Windows 5070 + CUDA 12.8 + PyTorch nightly stability.**
   Our 3B runs have been hitting cuSolver errors intermittently. These are *Python-side* bugs (won't affect the C++ kernel), but we need to understand them well enough to trust the calibration pipeline. The full CPU fallback works but adds ~45 min per run.

5. **GGUF tool ecosystem compatibility.**
   Does `gguf-py` writer round-trip our naming convention cleanly? Does the HuggingFace `gguf` library ignore unknown tensors gracefully?

---

## 7. Strategic decisions

### 7.1 Fork, don't merge

Upstream archaeology (summarized in JOURNAL.md) shows every streaming/offloading proposal since 2023 has been rejected or closed as stale. Maintainer stance is that whole-weight streaming is bandwidth-bounded — a valid critique that our *coefficient*-streaming design specifically sidesteps, but the signal is that runtime-policy changes aren't welcome. Keep this in `Atlas---llama.cpp/` indefinitely.

### 7.2 Adopt proven math, innovate on runtime

The decomposition space is crowded with 2023-2024 papers. Our contribution is the overlapped-streaming runtime, not new math. Adopt Basis Sharing wholesale; balanced truncation is our one empirical improvement; OBS repair is a direct port; CALDERA stays on deck.

### 7.3 Gen5 PCIe changes the economics

RTX 5070's ~55-60 GB/s sustained H2D is ~2× Gen4. This means:
- Compute/transfer overlap is easier to achieve
- Small-coefficient streams (rank 64-128) become nearly free
- For the 70B target, we're no longer PCIe-bound — kernel launch overhead on small k is now the dominant concern

Design the ring buffer and event discipline carefully, but don't over-engineer for PCIe throughput.

---

## 8. References

### Decomposition math
- Basis Sharing: https://arxiv.org/abs/2410.03765
- SparseGPT: https://arxiv.org/abs/2301.00774 — repo https://github.com/IST-DASLab/sparsegpt
- CALDERA: https://arxiv.org/abs/2405.18886 — repo https://github.com/pilancilab/caldera
- ASVD (baseline): https://arxiv.org/abs/2312.05821
- SVD-LLM V2: https://arxiv.org/abs/2503.12340
- Scaling laws for precision: https://arxiv.org/abs/2411.04330

### CUDA runtime
- NVIDIA streams overlap: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- Per-thread default stream: https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
- CUDA graphs: https://developer.nvidia.com/blog/cuda-graphs/
- Ampere L2 cache control: https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/
- CUDA 12.8 release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

### llama.cpp — bugs to avoid
- Async copy race: https://github.com/ggml-org/llama.cpp/issues/18310
- get_tensor_async illegal access: https://github.com/ggml-org/llama.cpp/issues/18313
- Non-P2P PCIe corruption: https://github.com/ggml-org/llama.cpp/issues/20052

### Existing related work (all private forks)
- PowerInfer: https://github.com/SJTU-IPADS/PowerInfer
- ik_llama.cpp: https://github.com/ikawrakow/ik_llama.cpp

---

## 9. File map of intended changes

```
gguf-py/scripts/convert_factored.py          (NEW)  — converter
src/llama-model-loader.cpp                   (MOD)  — detect .basis/.coeffs siblings
src/llama-model.cpp                          (MOD)  — build_factored_linear() helpers
ggml/include/ggml.h                          (MOD)  — GGML_OP_FACTORED_LINEAR enum
ggml/src/ggml.c                              (MOD)  — op dispatch, shape inference
ggml/src/ggml-cpu/ops.cpp                    (MOD)  — CPU stub (can assert unsupported)
ggml/src/ggml-cuda/factored-linear.cu        (NEW)  — main kernel
ggml/src/ggml-cuda/factored-linear.cuh       (NEW)  — header
ggml/src/ggml-cuda/ggml-cuda.cu              (MOD)  — dispatch for the new op
ggml/src/ggml-backend.cpp                    (MOD)  — pinned-pool coord + transfer stream exposure
research/cross-layer-svd/                    (existing prototype dir — stays)
```

~15 files touched, ~2500-3500 net lines added. Most of the bulk is the runtime scheduler integration in `ggml-backend.cpp`.

---

## 10. What Phase 0 delivers

- ✅ This document
- ✅ Updated `JOURNAL.md` pointing here
- Pending: OBS-on-SVD empirical validation (small Python test on one matrix) — one-afternoon task before Phase 1 starts

**Phase 1 starts as soon as the OBS empirical check passes.**
