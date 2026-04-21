# Low-Bit LLM Quantization: 2025–2026 Research Notes

Research notes on four recent papers pushing LLM weight quantization into the
1-bit and sub-1-bit regime. Collected for reference when evaluating which
schemes might be relevant to `llama.cpp` quantization formats (`Q1_*`, `TQ*`,
`IQ*`) and to the BitNet / ternary path already in tree.

All four papers target the same underlying problem — preserving model quality
at increasingly aggressive bit-widths — but attack it from three different
angles: native 1.58-bit pretraining (BitNet b1.58), sub-1-bit post-training
compression via factorization or codebooks (LittleBit, BTC-LLM), and
accuracy-first 2-bit PTQ that stays friendly to existing inference stacks
(D²Quant).

---

## 1. BitNet b1.58 2B4T — Microsoft Research

- arXiv: https://arxiv.org/abs/2504.12285
- Weights: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- Reference inference: https://github.com/microsoft/BitNet
- Authors: Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu,
  Ting Song, Yan Xia, Furu Wei

### What it is

First open-weights, native 1-bit (technically 1.58-bit ternary `{-1, 0, +1}`)
LLM at the 2B parameter scale, trained from scratch on 4T tokens. This is a
"trained-in", not post-training, quantization story — the model is never FP16
in deployment.

### Architecture

- Standard Transformer backbone with full-precision linears replaced by
  `BitLinear` layers (ternary weights, 8-bit activations — "W1.58A8").
- `absmean` quantization for weights, per-token `absmax` for activations.
- Uses `subln` normalization instead of RMSNorm at the BitLinear boundary to
  keep activation variance bounded during training.
- Rotary positional embeddings; squared-ReLU FFN.

### Training recipe

- Pretraining on 4T tokens in two phases (general + higher-quality data),
  followed by SFT and DPO. Straight-through estimator for the weight
  quantizer during training.

### Numbers worth remembering

- Matches FP LLMs of comparable size (LLaMA 3.2 1B, Qwen 2.5 1.5B,
  Gemma-3 1B) on average across reasoning, math, code, conversational
  benchmarks, at roughly a 1/10th memory footprint.
- Non-embedding weight memory ≈ 0.4 GB vs ~2–5 GB for FP peers.
- Decoding energy and latency claims rely on kernels that assume ternary
  matmul support (e.g. the `bitnet.cpp` T-MAC / LUT-based GEMM path).

### Relevance

This is the reference point for anyone doing native 1-bit training. For
`llama.cpp`, the practical question is kernel support for packed ternary
weights on CPU and GPU — see `ggml-cpu/arch/*/repack.cpp` and the `TQ1_0`
/ `TQ2_0` paths.

---

## 2. LittleBit — Samsung Research (NeurIPS 2025)

- arXiv: https://arxiv.org/abs/2506.13771
- Authors: Banseok Lee, Dongkyu Kim, Youngcheon You, Youngmin Kim

### What it is

Post-training compression framework that pushes weights down to ~0.1 bits
per weight (BPW) — roughly 31× smaller than FP16 — by factorizing each
weight matrix into low-rank latent factors and binarizing the factors, not
the original weights.

### Technique (three moving parts)

1. **Latent factorization + binarization.** `W ≈ U · V^T` with low rank `r`.
   Instead of binarizing `W` directly, `U` and `V` are binarized. Storage
   collapses to `(m + n) · r` bits plus small scale tensors, so total BPW is
   set by `r / min(m, n)`, which is how sub-0.5 BPW becomes feasible.
2. **Multi-scale compensation.** Learnable per-row, per-column, and
   per-latent-rank scale vectors multiply the binarized factors back up.
   The latent-rank scale is the novel piece — it lets the model learn
   which of the `r` ranks matter.
3. **Dual Sign–Value-Independent Decomposition (Dual-SVID).** Initialization
   scheme for QAT that decouples the sign and magnitude of the SVD factors
   so the binarization step starts close to the full-precision operator,
   plus a residual-compensation loop during fine-tuning.

### Numbers

- Llama2-13B at 0.1 BPW fits in under 0.9 GB of weights.
- At 0.1 BPW beats STBLLM (prior sub-1-bit SOTA) running at 0.55 BPW on
  Llama2-13B.
- Reports a potential 11.6× inference speedup vs FP16, assuming kernels
  that can exploit the factored binary form.
- Tested on models from 1.3B to 32B.

### Relevance

The factored-binary storage layout is different enough from existing
`llama.cpp` block quant formats that it would need a new tensor type, not
just a new quantizer. The interesting part is that storage drops below
what a fixed per-weight bit-width could ever deliver, because rank `r`
decouples parameter count from weight count.

---

## 3. BTC-LLM — HKUST / collaborators

- arXiv: https://arxiv.org/abs/2506.12040
- OpenReview: https://openreview.net/forum?id=yBDBCpEzsO
- Authors: Hao Gu, Lujun Li, Hao Wang, Lei Wang, Zheyu Wang, Bei Liu,
  Jiacheng Liu, Qiyuan Zhu, Sirui Han, Yike Guo

### What it is

Sub-1-bit PTQ that explicitly avoids sparsity masks (the typical way to
drop below 1 BPW in earlier work like STBLLM). Two components:

1. **Learnable Transformation.** Invertible scale + rotation applied to
   each weight matrix before binarization, learned end-to-end. Acts as
   incoherence processing — outliers are smeared out and shared sign
   patterns emerge, so the subsequent binary approximation is tighter.
2. **Flash and Accurate Binary Codebook.** After binarization, repeated
   binary vectors across rows are clustered into a small codebook;
   each row stores an index into the codebook. Uses a custom Hamming-like
   distance and sign-based centroid updates.

The combination gives dense (not sparse) sub-1-bit weights, which matters
because dense formats map cleanly onto existing GEMM kernels.

### Numbers

- LLaMA-2-13B at ~0.8 bits: 3.1% zero-shot accuracy drop vs FP16 baseline.
- 1.6× end-to-end speedup over FP16.
- Evaluated across LLaMA, Qwen, and FBI-LLM families in the 1.11–0.7 bit
  range.

### Relevance

Codebook-indexed binary weights are the scheme closest in spirit to the
IQ2/IQ3 k-quant family already in `llama.cpp` (grid lookups with learned
codebooks). A GGUF-side implementation would need a new block layout plus
the learned rotation stored per layer; the rotation is a fixed matrix at
inference time, so it can be fused into adjacent linears or kept as a
side tensor.

---

## 4. D²Quant — Accurate Low-Bit Weight-Only PTQ

- arXiv: https://arxiv.org/abs/2602.02546
- Code (pending): https://github.com/XIANGLONGYAN/D2Quant
- Authors: Xianglong Yan, ChengZhu Bao, Zhiteng Li, Tianao Zhang,
  Shaoqiu Zhang, Ruobing Xie, Samm Sun, Yulun Zhang

### What it is

Weight-only PTQ targeting the sub-4-bit regime (2–3 bit), focused on two
specific pain points rather than on ultra-low bit-widths:

1. **Dual-Scale Quantizer (DSQ).** Targets FFN down-projection matrices,
   which have consistently been the hardest layers to quantize. Uses two
   scale factors per group — one of which is "absorbable" into the
   neighboring linear / norm so it costs nothing at inference — and gives
   the quantizer extra degrees of freedom without inflating the bit budget.
2. **Deviation-Aware Correction (DAC).** A mean-shift term folded into
   LayerNorm that cancels the activation-distribution drift introduced by
   weight quantization. Pure math fix, no extra runtime cost.

### Numbers

- Claims SOTA for weight-only PTQ in the sub-4-bit regime across multiple
  LLM families (LLaMA, LLaMA-2, LLaMA-3 reported in the paper).
- Specific perplexity/accuracy tables vs GPTQ, AWQ, and OmniQuant are in
  the paper; headline claim is consistent improvement at 2–3 bit without
  additional runtime overhead.

### Relevance

Lowest-friction of the four for `llama.cpp`. DAC is a one-shot offline
transform on the model weights (merge a bias into LayerNorm); DSQ is a
choice of scale storage that fits inside existing group-wise block quant
formats. Could likely be retrofitted onto `Q2_K` / `IQ2_*` as a quantizer
option rather than a new type.

---

## Cross-paper summary

| Paper          | Bit width       | Type                       | PTQ or trained-in | Needs new kernel? |
|----------------|-----------------|----------------------------|-------------------|-------------------|
| BitNet b1.58   | 1.58 (ternary)  | Native ternary             | Trained from scratch | Yes (ternary GEMM) |
| LittleBit      | ~0.1 BPW        | Binary factorization       | PTQ + QAT recovery | Yes (factored binary) |
| BTC-LLM        | 0.7–1.11 BPW    | Binary + learned rotation + codebook | PTQ | Yes (binary + codebook lookup) |
| D²Quant        | 2–3 BPW         | Group-wise weight-only PTQ | PTQ | No (fits existing block formats) |

### Takeaways for `llama.cpp`

- **D²Quant** is the only one of the four that could plausibly ship as a
  quantizer tweak without new tensor types. Worth prototyping on
  `llama-quantize` as an alternative scale-search mode for `Q2_K` /
  `IQ2_XS`.
- **BitNet b1.58** is already partially landed via the ternary (`TQ*`)
  formats and the upstream `bitnet.cpp` work; the 2B4T release is the
  largest native 1-bit checkpoint to evaluate against those kernels.
- **LittleBit** and **BTC-LLM** both need a new on-disk layout. BTC-LLM's
  codebook-indexed binary is the closer match to existing `llama.cpp`
  idioms; LittleBit's factored form is more disruptive but offers the
  lowest BPW in the literature.
- Across all four, the accuracy story at <1 BPW is still "a few points off
  FP16 on zero-shot, worse on reasoning-heavy tasks." Native-trained
  ternary (BitNet) currently has the cleanest quality/efficiency tradeoff
  at the 2B scale.
