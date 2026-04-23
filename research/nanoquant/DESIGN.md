# NanoQuant Replication

**Status:** Phase 0 (environment setup). No quantization code yet.
**Date:** 2026-04-23
**Paper:** [NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](https://arxiv.org/html/2602.06694v1) (Chong, Kim, Kim, Choi — v1, Feb 2026)
**Relates to:** [extreme-quantization](../extreme-quantization/) — a candidate stage in that pipeline *if* replication succeeds.

---

## 1. Goal

Faithful reimplementation of NanoQuant from the paper alone (no code release). Reproduce the quantization-quality numbers from Tables 2, 3, 5, 6 on models that fit local hardware, then decide whether the method is worth productionizing into the extreme-quantization stack.

Explicitly **not** the goal for now:

- Matching their RTX 3050 inference throughput (Table 4) — requires custom binary GEMV/GEMM kernels they didn't release. Deferred until quality replicates.
- Matching their 70B results — 64 GB system RAM can't hold Llama2/3-70B FP16 (~140 GB) even with block-wise CPU offload.
- Improving on NanoQuant. Reimplement first, innovate later.

---

## 2. Method summary

NanoQuant represents each weight matrix as

```
W ≈ s₁ ⊙ (U_{±1} V_{±1}^T) ⊙ s₂^T
    U_{±1} ∈ {−1,+1}^{d_out × r},  V_{±1} ∈ {−1,+1}^{d_in × r}
    s₁ ∈ ℝ^{d_out},  s₂ ∈ ℝ^{d_in}
```

Applied block-wise through the transformer. Four stages per block:

1. **Error propagation mitigation.** Adjust FP weights of the current block to compensate for quantization error accumulated in earlier blocks.
2. **LB-ADMM initialization.** K-FAC Hessian preconditioning with Ledoit–Wolf shrinkage γ → ADMM solve with SVID rank-1 Z-update → magnitude balancing extracts s₁, s₂ and latent U, V.
3. **Factorized refinement.** Straight-Through Estimator fine-tune of latent U, V and scales against FP block output on calibration activations.
4. **Global scale optimization.** Freeze binary signs, tune scales end-to-end by KL against FP16 logits.

Calibration: 128 × 2048 tokens WikiText-2 (~0.26 M).

---

## 3. Target outcomes — quality only

| Tier    | Model       | Bits | Metric                    | Paper value       | Our target   |
|---------|-------------|------|---------------------------|-------------------|--------------|
| Core    | Qwen3-8B    | 1.0  | WikiText-2 PPL            | 25.31 @ 0.8 bpw   | within 10%   |
| Core    | Llama2-7B   | 1.0  | WikiText-2 PPL            | 10.34             | within 10%   |
| Core    | Llama3-8B   | 1.0  | 6-task zero-shot avg      | 45.95%            | within 2 pp  |
| Stretch | Gemma3-12B  | 1.0  | WikiText-2 PPL            | (paper)           | within 15%   |
| Stretch | Llama2-13B  | 1.0  | WikiText-2 PPL            | (paper)           | within 15%   |

Qwen3-8B and Llama2-7B are the proving ground. Llama2-7B because it's the paper's strongest advertised gap (dominates BiLLM 19.87 → 10.34). Qwen3-8B because it's the Atlas daily driver and where the paper's claims matter most to us.

Ablations to replicate:
- **Table 5** — initialization comparison (LB-ADMM vs Dual-SVID vs DBF ADMM).
- **Table 6** — Qwen3-8B per-stage contribution (init → +error mit → +refinement → +global scale).

---

## 4. Hardware and data

- **GPU:** RTX 5080 Laptop, 16 GB VRAM, Blackwell cc 12.0
- **RAM:** 64 GB
- **No cloud compute** — invariant, see user memory.

Block-wise quantization fits the largest targets because only one transformer block sits on GPU at a time; the rest lives on CPU RAM. Calibration activations stored on disk.

| Model        | FP16 weights | Fits in RAM? |
|--------------|--------------|--------------|
| Llama2-7B    | ~14 GB       | yes          |
| Llama3-8B    | ~16 GB       | yes          |
| Qwen3-8B     | ~16 GB       | yes          |
| Gemma3-12B   | ~24 GB       | yes          |
| Llama2-13B   | ~26 GB       | yes          |
| Llama2-70B   | ~140 GB      | **no** — out of scope |

Calibration: WikiText-2 train split, 128 samples × 2048 tokens, deterministic seed. Eval: WikiText-2 test split PPL + the six zero-shot tasks via `lm-eval-harness` (pinned version, recorded in `requirements.txt`).

---

## 5. Known ambiguities in the paper

Resolved case-by-case in JOURNAL.md as we hit them. Initial defaults, taken from related work where possible:

- **τ, T_pre, T_post, T_glob** (iteration budgets) — infer from the 1.7 H100-hour Llama2-7B budget backwards.
- **λ** (ridge reg), **ρ** (ADMM penalty) — start with LittleBit's / DBF's published values, sweep narrowly if numerics are unstable.
- **K** (ADMM iterations) — start at 50, measure convergence.
- **r** (rank) — paper mentions r=2–4 for sub-1-bit. Start with r=2 for primary runs, sweep later.
- **Bit-packing layout** — not needed for Phase 0–3 (PyTorch sign tensors are bit-identical).
- **lm-eval-harness version** — pin current stable; record SHA.
- **γ** (Hessian shrinkage) — paper specifies 0.2 for Llama/Qwen, 0.6 for Gemma/Rnj.
- **Author contact** — not planned yet. Enough latitude in related work to proceed honestly.

---

## 6. Phased plan

Phase numbering matches the conversation that opened this vein.

- **Phase 0** — Environment + baselines. Pinned PyTorch + HF stack; WikiText-2 loader; FP16 PPL and zero-shot baselines for target models; reproduce one public BiLLM number to validate the eval harness.
- **Phase 1** — Skeleton quantizer. Binary factorization + STE refinement only. No LB-ADMM, no K-FAC, no error propagation. Establishes the worst-case baseline for the method, independent of the paper's recipe.
- **Phase 2** — LB-ADMM init. Hessian preconditioning + ADMM + SVID. Isolates the value of the paper's main algorithmic contribution vs Phase 1.
- **Phase 3** — Full pipeline. Error propagation + global scale optimization. Replicate Table 2 / 3 / 6 on the core targets. Decision point on whether quality holds.
- **Phase 4** — Kernels and deployment. Out of scope until Phase 3 succeeds.

---

## 7. Non-goals

- Outperforming NanoQuant, mixed-method stacks, QAT (belong in extreme-quantization).
- Upstream merge.
- Matching the paper's 70B headline numbers.
