# Memory-efficient training on resource-limited GPUs — research survey

State-of-the-art (April 2026) techniques for training transformer
models under tight VRAM budgets. Focused on the 16 GB consumer-GPU
regime that this fork operates in. Organized by mechanism; each
section cites the primary source and rates applicability to our
LittleBit QAT pipeline.

Our specific pain points from Phase A/B work:
- **Vocabulary-sized logits** at `batch × seq × vocab=152k × 4 bytes`
  dominate loss-region memory (~625 MB per instance at seq=512 fp32).
- **Per-layer hidden states** held for intermediate-MSE loss —
  ~90 MB × 2 copies (student + teacher) × 24 layers = ~4.3 GB
  at batch=2 seq=512.
- **Optimizer state** dominates at larger scales; bitsandbytes
  already knocks this down ~4×.
- **Teacher forward** holds full fp16/bf16 model even though it's
  frozen — 1 GB at 0.5B scaling to 14 GB at 7B.

Current Phase A stack handles some of these but leaves significant
VRAM on the table. Below are techniques we haven't yet used.

## 1. Liger Kernel — the highest-impact unused tool

**Source**:
[linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) /
[arXiv:2410.10989](https://arxiv.org/pdf/2410.10989) /
[LinkedIn Engineering writeup](https://www.linkedin.com/blog/engineering/open-source/liger-kernel-open-source-ecosystem-for-efficient-llm-training)

Collection of Triton kernels that fuse common transformer operations.
Claims 20% throughput improvement and **60% memory reduction** on
multi-GPU LLM training. For post-training (including distillation)
specifically, up to **80% memory savings**.

### The relevant piece: FusedLinearCrossEntropy

Standard KD/QAT pipeline materializes the full logits tensor then
computes softmax/cross-entropy. At `vocab=152k`, this is the
dominant memory cost for our pipeline.

Liger's `FusedLinearCrossEntropy` fuses the final linear projection
(`lm_head`) with the loss computation, processing the vocab in
**blocks** so the full logits tensor is never materialized.

Applied to our case:
- Current peak: logits (625 MB) + softmax (625 MB) = 1.25 GB per
  instance
- Liger-fused: ~80 MB per block × 1 block at a time = 80 MB peak
- **Savings ~1.2 GB per forward pass** at seq=512, more at longer seq

### Compatibility concern

FusedLinearCrossEntropy expects CrossEntropy-style loss (one-hot
targets). Our loss is KL divergence against soft targets (teacher's
softmax). We'd need to either:
- Adapt the kernel to KL-div-with-soft-target form (non-trivial)
- Chunk the KL loss manually over the vocab dim (simpler, similar
  effect)

The chunked KL approach is ~30 lines of Python; the pattern is:

```python
chunk_size = 8192  # smaller than full vocab
total_kl = 0.0
for chunk_start in range(0, vocab, chunk_size):
    chunk_end = min(chunk_start + chunk_size, vocab)
    s_chunk = s_logits[..., chunk_start:chunk_end]
    t_chunk = t_logits[..., chunk_start:chunk_end].to(s_chunk.dtype)
    # Partial log-softmax / softmax are NOT valid chunkwise; we'd
    # need to pre-compute the denominator across all chunks first.
```

Actually KL isn't naively vocab-chunkable because softmax requires
the full vocab denominator. **Correct chunked form** uses the log-
sum-exp trick with two passes:

1. Pass 1: compute `max` and `sum(exp(...))` across vocab in chunks,
   holding only chunk at a time.
2. Pass 2: compute per-chunk KL contribution using the pre-computed
   normalizers.

Same total math, half the peak memory if done in two passes.
Liger's kernel does this internally as one fused Triton pass.

### Other Liger kernels worth pulling in

- `RMSNorm` — 2-4× faster than torch native
- `RoPE` — fused
- `SwiGLU` — fused (matches Qwen activation)

**Impact estimate**: Adding Liger to our pipeline could save ~2-3 GB
of loss-region + norm memory, free up budget for larger seq or
batch. At 7B scale, the logits savings alone would be ~20 GB+.

## 2. GaLore — gradient low-rank projection

**Source**: [GaLore paper](https://huggingface.co/papers/2403.03507)

Trains a low-rank projection of gradients instead of full gradients.
Enables **pretraining** 7B on consumer GPUs (not just fine-tuning —
actually from-scratch training).

Mechanism: gradient matrix `G ∈ R^{m×n}` projected to low-rank
`P G Q^T` via learned projection matrices, reducing effective grad
memory by `rank/min(m,n)`.

### Applicability to QAT

For our LittleBit setup:
- Trainable params are `U_fp`, `V_fp` (themselves low-rank) + scales
- U_fp and V_fp gradients are already rank-bounded by `r`
- Applying GaLore on top seems redundant — gradient is already
  low-rank structurally.

**Verdict**: low priority for our specific QAT pipeline; high value
for standard full-param training.

## 3. Unsloth — Triton kernel rewrites, QLoRA-focused

**Source**:
[Unsloth docs](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) /
[Red Hat guide](https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning) /
[Craftrigs 7B benchmark](https://craftrigs.com/guides/fine-tuning-7b-llm-consumer-gpu-unsloth-lora/)

Rewrites LoRA/QLoRA training in Triton. Claims:
- ~70% less VRAM than full fine-tuning
- 2× faster than standard LoRA/PEFT
- 7B QLoRA fits in 8-10 GB VRAM

### Applicability to QAT

Unsloth is LoRA/QLoRA-focused. Our setup isn't LoRA — LittleBit
replaces entire linear layers with a different parameterization, not
adding low-rank adapters on top of frozen weights. Adopting
Unsloth's infrastructure directly doesn't map.

**However**, Unsloth's low-level kernel rewrites (fused
RMSNorm, fused RoPE, memory-efficient attention) overlap heavily
with Liger's. If Liger gets us the same gains on a non-LoRA
workflow, Unsloth is redundant.

**Verdict**: use Liger instead.

## 4. COAT — FP8 training with dynamic range expansion

**Source**: [arXiv:2410.19313](https://arxiv.org/html/2410.19313v1)

Native FP8 training (not just fp8 inference). Halves memory vs bf16
while preserving training stability via:
- **Dynamic Range Expansion**: rescales optimizer states into the
  FP8 representable range before quantization
- **Mixed-Granularity Activation Quantization**: per-tensor for
  stable layers, per-group for sensitive ones

Requires Hopper (H100+) or Blackwell (RTX 5090/B100) hardware for
native FP8 compute. RTX 5080 Laptop is Blackwell-based, so
hardware-supported.

### Applicability to QAT

Our SmoothSign forward/backward surrogate in FP8 is unvalidated —
tanh(100·x) saturates fast; FP8's limited range could cause
issues near x=0 where the surrogate matters. But the paper shows
FP8 works for non-QAT training up to 7B scale.

**Impact estimate**: halving bf16 memory would bring our 7B estimate
from ~35 GB to ~18 GB — within reach of consumer GPUs with other
optimizations.

**Verdict**: speculative but high upside if SmoothSign is stable in
FP8. Worth an ablation experiment post-Phase B.

## 5. ZenFlow — selective gradient offload

**Source**: DeepSpeed extension, 2026 addition

Key insight: **a small subset of gradients contributes most of the
update signal**. ZenFlow keeps the top-N gradients (by norm) on GPU
and processes them immediately; offloads the rest to CPU for
asynchronous update.

Applied to our QAT:
- Most gradient mass in LittleBit flows through `U_fp`, `V_fp` sign
  parameters (via SmoothSign surrogate)
- Scale vectors (`h`, `g`, `ell`) have much smaller gradients per
  step (shown empirically in our §13 scales-only ablation)
- Could priority-keep sign grads on GPU, offload scale grads to CPU

**Verdict**: conceptually fits but requires DeepSpeed integration
overhead; better return is probably from Liger.

## 6. Activation offloading to host memory

**Source**:
[Anyscale LLM fine-tuning docs](https://docs.anyscale.com/llm/fine-tuning/speed-and-memory-optimizations) /
[Google Open Source Blog](https://opensource.googleblog.com/2026/04/leveraging-cpu-memory-for-faster-cost-efficient-tpu-llm-training.html)

Keep most-recently-used activations on GPU; spill older ones to CPU
RAM via PCIe. Per-step overhead depends on bandwidth but typically
20-50%.

### Applicability

Our Phase A already has gradient checkpointing, which recomputes
activations. Activation offloading is an alternative: don't
recompute, just move to CPU. Tradeoff: CPU-GPU transfer time vs
recomputation time.

For our short-seq (512) workload, recomputation is cheap; offloading
might not pay. For longer seq (2048) at larger scales, offloading
typically wins.

**Verdict**: later optimization, once we're on longer seq.

## 7. MemAscend / MLP-Offload — SSD offload for optimizer state

**Source**:
[arXiv:2505.23254](https://arxiv.org/html/2505.23254v1) (MemAscend) /
[ACM paper](https://dl.acm.org/doi/10.1145/3712285.3759864) (MLP-Offload)

Offload optimizer state to NVMe SSD when even system RAM is
constrained. System RAM → SSD paging managed by framework.

### Applicability

Our system RAM is plenty (likely 32-64 GB at least). 8-bit Adam at
7B is only ~7 GB, easily fits. SSD offload would be required only
for 30B+.

**Verdict**: not needed at our scales.

## 8. Chunked KL loss (DIY)

**Source**: standard softmax-chunked implementations, e.g. in
[vLLM](https://arxiv.org/pdf/2309.06180) KV cache management,
backed by the log-sum-exp formulation.

If we don't want a full Liger integration, a minimal chunked-KL
implementation gives us a slice of the gains:

```python
def chunked_kl_div(s_logits, t_logits, vocab_chunk=16384):
    """KL(softmax(s) || softmax(t)), processed in vocab chunks.

    Two-pass: first get normalizers, then sum per-chunk KL.
    Peak intermediate memory is chunk-sized, not vocab-sized.
    """
    # Pass 1: log-sum-exp for both student and teacher
    s_max = None
    s_sum = torch.zeros_like(s_logits[..., :1])
    t_max = None
    t_sum = torch.zeros_like(t_logits[..., :1])
    for start in range(0, s_logits.shape[-1], vocab_chunk):
        end = min(start + vocab_chunk, s_logits.shape[-1])
        # Update running max / sum_exp via log-sum-exp trick
        ...
    # Pass 2: sum per-chunk KL contribution using normalizers
    kl = torch.zeros_like(s_logits[..., :1])
    for start in range(0, s_logits.shape[-1], vocab_chunk):
        ...
    return kl.mean()
```

**Impact**: could save 1-2 GB at batch=1 seq=512 without new
dependencies. Fits in existing codebase. ~50 lines of code plus
testing.

## 9. Technique stack priority for our pipeline

Ranking by `(savings × applicability) / effort`:

| Rank | Technique | Savings (7B est) | Integration effort | Notes |
|---:|---|---:|---|---|
| 1 | **Liger Kernel** incl. FusedLinearCrossEntropy | ~20 GB | One pip install + model wrap | Drop-in via HF |
| 2 | **nf4 teacher via bitsandbytes** | ~10 GB | 5 lines | Already planned |
| 3 | **bf16 student + 8-bit Adam** | ~14+7 GB | ~20 lines (autocast) | Already planned |
| 4 | **Chunked KL (DIY)** | ~3 GB | ~50 lines | Alternative to Liger if we don't want the dep |
| 5 | **COAT FP8 training** | ~9 GB | new training loop | High risk on SmoothSign; experimental |
| 6 | **Hidden-state hooks (skip output_hidden_states)** | ~2 GB | ~30 lines | Needed for KD + checkpoint |
| 7 | **Activation offload to CPU RAM** | ~3-5 GB | framework dep | Not critical at short seq |
| 8 | **ZenFlow** | ~2-3 GB | DeepSpeed integration | High complexity |
| 9 | **GaLore** | — | — | Already low-rank via LittleBit |
| 10 | **SSD offload** | — | — | Only at 30B+ |

## 10. Revised feasibility for 7B local

Stacking techniques 1-6 (excluding speculative FP8 for now):

| Component | Baseline at 7B | With stack 1-6 |
|---|---:|---:|
| Student params + grads | 28 GB fp32 | **11 GB** (bf16) |
| Optimizer state | 14 GB | **2 GB** (8-bit Adam) |
| Teacher | 14 GB | **3.5 GB** (nf4) |
| Logits region | 5 GB peak | **~0.5 GB** (Liger fused CE) |
| Activations (grad ckpt) | ~5 GB | **~5 GB** (same) |
| Total | **~66 GB** | **~22 GB** |

22 GB is still over the 13.7 GB cap but **close enough that**:
- **One 24 GB GPU (RTX 4090, 5080 Desktop) fits comfortably**
- **Laptop 16 GB is still marginal** — but adding chunked KL (down
  to seq=256 or r=256) could squeeze it

## 11. Honest conclusion

Three clean paths emerge:

**A. Stay at ≤3B locally, adopt Liger + Tier 1 stack.**
Fully fits 16 GB with headroom. Establishes scaling story.
Time: ~2 hours coding + normal training time per run.

**B. Integrate full stack (Liger + nf4 teacher + bf16 student +
8-bit Adam + chunked KL) + cloud for 7B.**
~$15-25 per cloud run. Total stack fits 24 GB desktop GPU if we
eventually buy one. Time: ~1 day coding + $25 cloud.

**C. Go speculative with COAT FP8 training on laptop for 7B.**
Research-grade, high variance. Would be a legitimate contribution
if it works. Time: days of coding + debugging, outcome uncertain.

**Recommendation**: Path A during Phase B's completion. If Phase B
is clean, do Path B for 7B. Keep Path C as an optional follow-on.

## Sources

All references from searches April 2026. Links are markdown-
formatted so they're clickable.

- [GoCkpt: Gradient-Assisted Multi-Step overlapped Checkpointing](https://arxiv.org/html/2511.07035v1)
- [MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning](https://arxiv.org/html/2505.23254v1)
- [Does AI model offloading matter? Xinnor benchmark](https://xinnor.io/blog/does-ai-model-offloading-matter-benchmarking-ram-and-nvme-strategies-for-llm-training/)
- [OFFMATE: full fine-tuning of LLMs on a single GPU](https://hal.science/hal-04660745v1/document)
- [MLP-Offload: Multi-Level, Multi-Path Offloading](https://dl.acm.org/doi/10.1145/3712285.3759864)
- [Google: Leveraging CPU memory for TPU LLM training](https://opensource.googleblog.com/2026/04/leveraging-cpu-memory-for-faster-cost-efficient-tpu-llm-training.html)
- [Liger Kernel repo](https://github.com/linkedin/Liger-Kernel)
- [Liger Kernel paper](https://arxiv.org/pdf/2410.10989)
- [Unsloth docs](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
- [Red Hat: Unsloth + Training Hub](https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning)
- [COAT: FP8 training with dynamic range expansion](https://arxiv.org/html/2410.19313v1)
- [PyTorch FSDP efficient training](https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/)
- [DeepSpeed vs FSDP 2026](https://vrlatech.com/deepspeed-vs-pytorch-fsdp-which-distributed-training-framework-in-2026/)
- [GaLore paper](https://huggingface.co/papers/2403.03507)
- [Efficient Memory Management for LLM Serving (vLLM / PagedAttention)](https://arxiv.org/pdf/2309.06180)
- [GRASS: Compute Efficient Low-Memory LLM Training](https://aclanthology.org/2024.emnlp-main.835.pdf)
- [Craftrigs: Fine-Tuning a 7B LLM on a Consumer GPU with Unsloth](https://craftrigs.com/guides/fine-tuning-7b-llm-consumer-gpu-unsloth-lora/)
