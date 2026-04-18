# Factored Inference — Optimization Path

**Status:** Research aggregation, post-Phase 3.2.
**Date:** 2026-04-17.
**Relates to:** [DESIGN.md](DESIGN.md), [JOURNAL.md](JOURNAL.md).

Purpose: catalog related published work and concrete optimizations that map
onto the factored-weights + streaming runtime we started in Phase 1–3.2, with
file/line anchors into the current code so each item is actionable.

Two separate optimization tracks:

1. **Runtime (C++/CUDA)** — getting the factored model to decode fast
   (current Qwen 3B @ 20% shows −27% decode throughput; dominated by
   kernel-launch overhead per [JOURNAL.md](JOURNAL.md)).
2. **Decomposition pipeline (Python)** — getting factored weights produced
   fast enough to iterate at 7B+ scale
   (current 3B ≈ 3 min; CPU gramian is the main cost in
   `basis_sharing.py:356-374`).

---

## 1. Runtime (C++/CUDA) optimizations

### 1.1 Kernel fusion — fix the two-matmul anti-pattern

`ggml_factored_linear` (`ggml/src/ggml.c:3269`) emits two sequential
`ggml_mul_mat` nodes. Qwen 3B has 7 factored linears × 36 layers = 252
factored-linear sites per token; at 2 launches each that's **504 total
launches, 252 extra over the dense baseline**. This matches the measured
2.5 ms decode regression.

- **FlashSVD** ([arxiv:2508.01506](https://www.arxiv.org/pdf/2508.01506)) —
  end-to-end rank-aware streaming inference for SVD-compressed models. Fuses
  low-rank projection kernels *into* attention/FFN pipelines with SRAM tiling.
  **Applicability caveat (see JOURNAL Background reading):** FlashSVD has two
  variants. V1 (cuBLAS GEMM + fused second projection + activation) wins on
  latency at long sequence; V2 (fully fused GEMM-Activation-GEMM, zero HBM
  I/O) pays a **60% latency penalty** and benchmarks **0.62× slower than
  dense at M=128**. Decode sits at M=1 where a naïve port would *regress*
  tok/s. Memory wins (−70% peak activation, −75% transient) apply
  unconditionally. Treat as the **prefill-only** target, gated on a batch/seq
  threshold that falls back to the dense path for decode. Also a prerequisite
  for making Phase 3.4 streaming tractable (per-tile working set instead of
  full coefficient residency).
- **SVDQuant / Nunchaku**
  ([arxiv:2411.05007](https://arxiv.org/abs/2411.05007),
   [deepwiki](https://deepwiki.com/mit-han-lab/nunchaku/2-svdquant-quantization))
  — identical "two-branch" structure to ours; their fix is to fuse the
  low-rank branch kernels into the low-bit branch to eliminate redundant
  activation memory traffic. 3× decode speedup on RTX 4090. Unlike FlashSVD
  this one does claim wins at batch-1 decode — the more relevant reference
  for our regime.
- **BLR on Resource-Constrained GPUs**
  ([arxiv:2512.20861](https://arxiv.org/abs/2512.20861), Dec 2025) —
  same problem, roofline analysis showing multi-token inference stays
  memory-bound under low-rank. Custom **Triton kernels with partial fusion**
  give 3.76× speedup, 3× compression on Llama-7B/1B on Jetson Orin Nano and
  A40. Open-source reference implementation; `factored-linear.cu` can crib
  the layout directly.

### 1.2 Kernel-launch reduction

Three attacks, weakest → strongest:

1. **CUDA Graph capture.** llama.cpp already has this path
   ([NVIDIA blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)).
   ~1.2× on Llama-7B/H100. Verify our factored-linear emission doesn't break
   the `update_cuda_graph_executable` path before Phase 3.4.
2. **cuBLAS Grouped GEMM (12.5+)** — one kernel for N different-shaped
   matmuls
   ([NVIDIA blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)).
   Collapse same-`d_in` basis matmuls per layer into one grouped launch
   (q/k/v/o share `d_in=hidden`; gate/up share `d_in=hidden`; down stands
   alone at `d_in=intermediate`). Realistic target per layer: 3 grouped
   basis launches + 1 grouped coefficient launch = ~4, so **504 → ~150
   launches/token**, or ~100 if the coefficient matmuls also group cleanly.
   No custom CUDA.
3. **Megakernels** — compile the whole forward pass into one kernel.
   [Mirage MPK](https://arxiv.org/html/2512.22219v1) (14.5 → 12.5 ms on
   A100), [Hazy "No Bubbles" Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
   (exactly batch-1 decode, our target). Large engineering cost; only revisit
   after items 1–2 exhaust headroom.

### 1.3 Structured-matrix alternatives (strictly generalize low-rank SVD)

- **BLAST** ([arxiv:2410.21262](https://arxiv.org/html/2410.21262v1), ICLR 2025) —
  block-level adaptive structured matrix that *subsumes* low-rank, Monarch,
  and block-diagonal. Our Basis Sharing + per-matrix SVD is a special case.
  50% compression with minimal recovery.
- **Monarch** ([arxiv:2204.00595](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf))
  — structured batched GEMMs, 2× over dense. Worth trying on `o_proj` /
  `down_proj` where Basis Sharing says cross-layer sharing hurts.
- **Kronecker-sparse / butterfly**
  ([arxiv:2405.15013](https://arxiv.org/abs/2405.15013)) — 16–22% end-to-end
  latency reduction on GPT-2-medium and ViT-S/16.

### 1.4 Transport — beyond pinned host memory

**Bandwidth ceiling first (see JOURNAL Background reading).** Naïve "stream
every coefficient every token" for Qwen 2.5 7B @ r=0.5 needs
**220-440 GB/s** vs ~55 GB/s from a PCIe Gen5 x16 link. Transport upgrades
alone do not close this — they have to combine with tile-fused kernels
(§1.1, FlashSVD V1-style) so the working set in VRAM drops from 100s of
MB/layer to O(MB). Without fusion, no transport option makes pure per-token
streaming viable.

- **GPUDirect Storage**
  ([NVIDIA blog](https://developer.nvidia.com/blog/gpudirect-storage/)) —
  DMA engine moves coefficients NVMe → GPU, bypassing CPU RAM. Gets
  streaming from ~6-8 GB/s (CPU-mediated) to ~25-45 GB/s on Gen5 NVMe.
  **Platform blocker for us:** Linux-only, no Windows/WSL2 driver, so our
  current dev box can't use it. The practical Windows substitute is pinned
  host staging + dual-stream double-buffering, which captures ~70% of GDS's
  benefit (gap is ~2-3× for our workload, not 10×). Revisit GDS only if
  Phase 3.4 is measurably PCIe-bound in practice and worth a Linux dual-boot.
- **HMM + `cudaMemPrefetchAsync`**
  ([CUDA docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html))
  — simpler than hand-coded pinned double-buffering. Less control, 80% of
  the benefit for 10% of the code. Linux kernel ≥ 6.1, Blackwell-ready.
  Same Windows caveat as GDS.
- **Pinned host + double-buffered streams (Windows-viable baseline).**
  Pre-allocate pinned staging buffers once, one CUDA stream for compute and
  one for copy, 2-16 MB chunk granularity tuned to SSD queue depth. This is
  the design we can actually ship on the current dev box. Everything else
  in this section is conditional on Linux.
- **PIPO's tensor-merging pattern**
  ([arxiv:2504.03664](https://arxiv.org/abs/2504.03664)) — concat all layer
  coeffs, one `memcpyAsync`, block-subdivide on GPU. 40→90% GPU util on
  RTX 3060 6 GB. Platform-agnostic; retrofit into the pinned-host-pool in
  DESIGN.md §3.3 regardless of whether we get GDS.

### 1.5 Coefficient quantization (halve PCIe bytes/layer)

The `.coeffs` stream is currently FP16. Options:

- **IntLoRA** ([arxiv:2410.21759](https://arxiv.org/html/2410.21759v3)) —
  integer-only low-rank branch that merges cleanly into a quantized base.
  Direct retrofit for the `basis` (resident, FP16) + `coeffs` (streamed,
  INT4/INT8) split.
- **FireQ** ([arxiv:2505.20839](https://arxiv.org/html/2505.20839v2)) —
  W4A8 kernels; if we stream INT4 coeffs this is the kernel.
- **QA-LoRA** ([arxiv:2309.14717](https://arxiv.org/html/2309.14717)) —
  quantization-aware adaptation; merges into quantized weights without
  accuracy loss.

### 1.6 Orthogonal multipliers (stack on top of streaming)

- **Activation sparsity** — [SparseInfer](https://arxiv.org/html/2411.12692v1)
  (training-free ReLU predictor), [WiSparse](https://arxiv.org/html/2602.14452)
  (+21.4% on Llama-3.1 at 50%). Skip `A·x` rows where activations are
  near-zero. PowerInfer's hot/cold-neuron 80/20 rule is the same phenomenon.
- **Speculative decoding** — [SpecEE](https://dl.acm.org/doi/10.1145/3695053.3730996)
  (2.43× on RTX 4060 8 GB), [Dovetail](https://arxiv.org/html/2412.18934v1)
  (CPU/GPU draft-model). Multiplicative with streaming for batch-1 decode.

### 1.7 Rank-allocation refinement

- **SVD-LLM V2** ([arxiv:2503.12340](https://arxiv.org/html/2503.12340v1),
  NAACL 2025) — per-matrix compression ratios from theoretical truncation
  loss, vs our single global `target_ratio`. Free PPL at matched compression.
  Small change in `rank_shared` / `rank_permatrix` (`basis_sharing.py:406-414`).
- **Zero-Sum SVD** ([arxiv:2602.02848](https://arxiv.org/html/2602.02848)) —
  loss-sensitivity balanced allocation; complementary to SVD-LLM V2.

### 1.8 Other 2025 systems worth reading

- **KTransformers** ([SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf))
  — Expert Deferral async CPU/GPU scheduling. MoE-focused but the scheduler
  pattern is analogous to our Phase 3.4 ring buffer.
- **NEO** ([MLSys'25](http://minlanyu.seas.harvard.edu/writeup/mlsys25.pdf)),
  **KVPR** ([ACL findings'25](https://aclanthology.org/2025.findings-acl.997.pdf))
  — KV-cache-side streaming overlap patterns; same pipeline discipline
  applies to weight-coefficient streaming.

---

## 2. Decomposition pipeline (Python) speedups

Grounded in actual code: calibration time on Qwen 2.5 3B is ~3 min with
32 samples; at 7B this scales to ~10–20 min and at 13B to ~30–60 min, by
which point the iteration loop is too slow for ablation work.

### 2.1 GPU-resident, role-sequential gramian (highest payoff)

`basis_sharing.py:356-374` D2H-copies every activation and computes the
`d×d` gramian on CPU. This trades PCIe bandwidth for CPU FLOPs — currently
the dominant calibration cost.

Fix: process one role at a time, keep the gramian GPU-resident, `addmm_`
accumulate there, D2H once at end-of-role. Run the forward 7× instead of
1× — each pass is far cheaper than the current CPU hotspot.

**Memory per role on Qwen 2.5 3B** (hidden=2048, intermediate=11008, L=36),
keeping one gramian per layer GPU-resident during the pass:

| role | d_in | L × d² × 4 | fits 12 GB? |
|---|---|---|---|
| q/k/v/o, gate/up | 2048 | 0.6 GB | ✓ |
| **down_proj** | **11008** | **17.4 GB** | **✗** |

The `down_proj` role doesn't fit in 12 GB on 3B, and gets worse at 7B
(intermediate=18944 → 52 GB) and 13B. Concrete options:

1. **Row-chunked accumulation for the wide role** — hold only `L × d × chunk × 4`
   at a time, iterate chunk tiles of the gramian across the corpus. 3B
   down_proj at chunk=1024: 36 × 11008 × 1024 × 4 ≈ 1.6 GB per chunk slice.
   Adds ~11 passes over the corpus for that role, still far under the CPU
   hotspot wall-time.
2. **Per-layer evict-and-spill** — compute each layer's gramian fully
   GPU-resident for the current layer, addmm, then D2H to pinned host
   between layers. Single-layer working set: d² × 4 = 0.48 GB for down on 3B,
   1.4 GB on 7B. Cheap, loses cross-layer overlap.
3. **24 GB GPU floor** — brute-force fix; doesn't help at 13B+.

Go with option 1 as the default; fall back to option 2 when
`L × d² × 4 > 0.7 × free_vram`. Expected end-to-end: **10–20× on
`collect_stats`** (the ratio survives either workaround because the CPU path
remains the bottleneck we're removing).

### 2.2 Randomized / truncated SVD

`_stable_svd` (`basis_sharing.py:71`) uses classical Golub-Reinsch. For
`rank ≪ min(d_out, n·d_in)` — i.e. every window — this is wasted work.

Replace with `torch.svd_lowrank` (Halko–Martinsson–Tropp) when rank < dim/4.
For Qwen 2.5 3B window=2 `gate_proj` (m = 2×11008 = 22016 stacked rows,
n = 2048, rank 655): full SVD ≈ 4 `m n²` ≈ 370 GFLOPs, randomized at
oversample=10 ≈ `m n (r+o)` ≈ 30 GFLOPs → ~10× faster per matrix, no
accuracy loss for low-rank approximation. (Order-of-magnitude numbers; exact
depends on how the window is stacked.)

Keep `gesvd` as fallback for rank-close-to-min cases and numerical escalation.

### 2.3 Gate `refit` + backward pass

The paper doesn't refit; our WLS refit is the extension (DESIGN.md §2.3).
Backward-pass gramian collection roughly doubles calibration time.

- Default `--refit=False`. Validate PPL gap on 3B is worth the 2× cost
  before making it the default.
- If refit stays: restrict `ggt` hooks to roles where the single-matrix
  ablation shows >20% weighted-error improvement (gate_proj, down_proj
  likely; Q/K/V marginal).

### 2.4 Batched calibration samples

`collect_stats` runs `batch=1`. Batch 4–8 padded samples per forward:
- 4–8× fewer hook invocations
- Better SM utilization on a batch-1-starved 3B/7B model

Expected: 2–4× additional on calibration wall time.

### 2.5 Gramian caching across `target_ratio` sweeps

Per-phase checkpointing exists but the JOURNAL's 3B quality-curve table
(ratios 0.8 / 0.7 / 0.5) looks like it paid calibration 3×. Ensure
`xtx` / `ggt` are in the Phase-1 checkpoint and the decomposition step is a
pure function of those plus `target_ratio`. Enables cheap ratio sweeps and
method ablations.

### 2.6 Cholesky ladder pruning

`sqrt_and_inv` (`basis_sharing.py:103`) ladders GPU-fp32 → GPU-fp64 →
CPU-fp64. DESIGN.md §2.1 says FP64 is "mandatory" (sourced from Basis
Sharing paper's 7B scale). Verify `_CUDA_FALLBACK_LOGGED["eigh"]` is
actually hit in practice; if it's 0 on our models, the FP64 path is dead
code and we can drop it or gate behind a flag.

### 2.7 Forward/gramian stream overlap

In the role-sequential variant, use a `torch.cuda.Stream` for gramian
`addmm_` so the next sample's forward starts while the current sample's
hook is still accumulating. Modest gain (~15–20%) once 2.1 lands.

### 2.8 Decouple gramian collection from decomposition

Turn the pipeline into two pure stages:

```
calibrate(model, corpus) -> cache/{model}/gramians.safetensors
decompose(gramians, target_ratio, method, refit) -> factors
```

Primary value isn't speed; it's enabling fast method-and-ratio ablation.
Plugs into item 2.5 and makes decomposition reproducible.

### 2.9 Calibrate through llama.cpp (structural, long-term)

Instead of HF transformers, instrument `build_lora_mm` (or a sibling
calibration-only path) in `src/llama-graph.cpp` to dump per-layer gramians
to a pinned buffer, stream to a Python consumer. Benefits:

- Calibrate on the deployed quantized model's actual activations (more
  correct)
- Avoid HF model-load entirely
- Fit a 70B model in 12 GB GPU + 64 GB RAM, where HF doesn't

AutoAWQ and SqueezeLLM have moved in this direction.

---

## 3. Priority ordering

**Cross-track dependency:** custom-CUDA work (Runtime items 3+) and Phase
3.4 are gated on the Model-quality track clearing the scale-gate. No point
tuning a fused factored-linear kernel for a decomposition that hasn't
reached paper PPL at 7B. Pipeline track is the near-term unblocker for that
gate.

### Pipeline track — this week, unblocks 7B+ scale validation

1. **GPU-resident role-sequential gramian** (§2.1), with row-chunked
   accumulation for `down_proj` (the d=11008 role doesn't fit 12 GB
   otherwise).
2. **Randomized SVD** via `torch.svd_lowrank` (§2.2).
3. **Gramian caching across ratios** (§2.5).
4. Then 2.3, 2.4 as needed for ablation velocity.

### Model-quality track — gates Runtime items 3+ and Phase 3.4

1. **Scale-gate: 7B at target_ratio=0.5 must land near paper territory
   (PPL ~19–25).** This is the go/no-go for custom-CUDA kernel work and
   Phase 3.4 streaming. Blocked on Pipeline track item 1.
2. **Per-matrix rank allocation** (SVD-LLM V2) — §1.7. Small change in
   `rank_shared` / `rank_permatrix`; run after the scale-gate passes.

### Runtime track — unblock Phase 3.3 / 3.4

1. **cuBLAS Grouped GEMM** in `ggml-cuda` — ~1 day, collapses kernel
   launches without custom CUDA. If the −27% decode regression closes here,
   custom-kernel and megakernel work gets deprioritized. Safe to do in
   parallel with the Pipeline track.
2. **CUDA Graph capture verification** — confirm factored-linear doesn't
   break the existing `update_cuda_graph_executable` path; 1–2 days. Also
   parallelizable.
3. **Custom fused factored-linear kernel (prefill-only)** cribbing the
   arxiv:2512.20861 Triton layout and FlashSVD V1-style partial fusion.
   Gated on the quality scale-gate. Gate on an M threshold at runtime so
   decode falls back to the dense path — FlashSVD-style V2 fusion regresses
   at M=1, so shipping it blanket would hurt the chat path.
4. **Quantize streamed coeffs to INT4/INT8** via IntLoRA pattern. Halves
   PCIe bytes/layer, aligns with DESIGN.md §3.3 "int8 streamed
   coefficients" goal. Order-independent with item 3.
5. **Defer megakernel / GPUDirect Storage to Phase 5+** unless 1–4 leave
   measurable headroom. GDS in particular needs a Linux dev box we don't
   currently have.

---

## 4. References (consolidated)

### Factored / low-rank compression
- Basis Sharing: [arxiv:2410.03765](https://arxiv.org/abs/2410.03765)
- SVD-LLM V2: [arxiv:2503.12340](https://arxiv.org/html/2503.12340v1)
- FlashSVD: [arxiv:2508.01506](https://www.arxiv.org/pdf/2508.01506)
- BLR on constrained GPUs: [arxiv:2512.20861](https://arxiv.org/abs/2512.20861)
- BLAST: [arxiv:2410.21262](https://arxiv.org/html/2410.21262v1)
- Monarch: [arxiv:2204.00595](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf)
- Kronecker-sparse: [arxiv:2405.15013](https://arxiv.org/abs/2405.15013)
- Zero-Sum SVD: [arxiv:2602.02848](https://arxiv.org/html/2602.02848)
- SparseGPT: [arxiv:2301.00774](https://arxiv.org/abs/2301.00774)
- CALDERA: [arxiv:2405.18886](https://arxiv.org/abs/2405.18886)

### Quantization + low-rank
- SVDQuant / Nunchaku: [arxiv:2411.05007](https://arxiv.org/abs/2411.05007)
- IntLoRA: [arxiv:2410.21759](https://arxiv.org/html/2410.21759v3)
- FireQ: [arxiv:2505.20839](https://arxiv.org/html/2505.20839v2)
- QA-LoRA: [arxiv:2309.14717](https://arxiv.org/html/2309.14717)
- AWQ: [arxiv:2306.00978](https://arxiv.org/abs/2306.00978)

### Streaming / offload systems
- PIPO: [arxiv:2504.03664](https://arxiv.org/abs/2504.03664)
- KTransformers (SOSP'25): [paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- NEO (MLSys'25): [paper](http://minlanyu.seas.harvard.edu/writeup/mlsys25.pdf)
- KVPR (ACL findings'25): [paper](https://aclanthology.org/2025.findings-acl.997.pdf)
- FlexGen: [arxiv:2303.06865](https://arxiv.org/abs/2303.06865)
- PowerInfer: [arxiv:2312.12456](https://arxiv.org/html/2312.12456v2)

### Kernel / runtime
- llama.cpp + CUDA Graphs: [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
- cuBLAS Grouped GEMM: [NVIDIA blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- CUTLASS grouped example: [GitHub](https://github.com/NVIDIA/cutlass/blob/main/examples/24_gemm_grouped/gemm_grouped.cu)
- Mirage MPK: [arxiv:2512.22219](https://arxiv.org/html/2512.22219v1)
- Hazy "No Bubbles" megakernel: [blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- CUTLASS Blackwell tcgen05: [Colfax tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- CUTLASS Hopper WGMMA: [Colfax tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- GPUDirect Storage: [NVIDIA blog](https://developer.nvidia.com/blog/gpudirect-storage/)
- CUDA Unified Memory / HMM: [docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)

### Orthogonal
- SparseInfer: [arxiv:2411.12692](https://arxiv.org/html/2411.12692v1)
- WiSparse: [arxiv:2602.14452](https://arxiv.org/html/2602.14452)
- SpecEE (ISCA'25): [paper](https://dl.acm.org/doi/10.1145/3695053.3730996)
- Dovetail: [arxiv:2412.18934](https://arxiv.org/html/2412.18934v1)
