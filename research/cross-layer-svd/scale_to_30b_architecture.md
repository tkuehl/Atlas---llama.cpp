# Scaling LittleBit QAT to 30B+ on consumer hardware

Architectural plan for training 30B-70B models with LittleBit QAT on
a 16 GB laptop GPU + consumer system RAM. Built on two core ideas:

1. **Offline teacher extraction** — teacher forward runs once,
   results cached to disk. No teacher in GPU during training.
2. **Streamed student training** — student layers live on CPU,
   stream to GPU one at a time.

Neither idea is novel individually. The combination for QAT at 30B+
scale on consumer hardware appears not to be published, which is
mildly exciting research-wise.

Status: **scoping plan**. Prerequisites: Phase B result + Runs A/C/D
from [savings_exploration_plan.md](savings_exploration_plan.md).

## 1. The core insight

A QAT student needs three things during training:
- **Itself** (params, grads, optimizer state, activations) — must be writable, so must be in fast memory during its layer's forward/backward
- **Teacher's logits** at each position for KL loss — just a constant
- **Teacher's hidden states** at each layer for MSE loss — just a constant

Teacher's contribution is **static**. If we pre-compute it once, the
student's training loop can read from disk instead of holding
teacher in GPU. Same math, no teacher memory.

This is obvious in hindsight. It's not what the paper does.

## 2. Offline teacher extraction

### 2.1 What to cache per training token

| Item | Shape | Precision | Bytes/token |
|---|---|---|---:|
| Top-k logit values | `k=256` | fp16 | 512 |
| Top-k logit indices | `k=256` | int32 | 1024 |
| Hidden states at selected layers | `L_sel × d_hidden` | bf16 | `L_sel × d_hidden × 2` |

For a 30B-class model (~80 layers, d_hidden=5120):
- Full hidden states every layer: `80 × 5120 × 2 = 820 KB/token`
- Hidden states every 4th layer: `20 × 5120 × 2 = 205 KB/token`
- Hidden states at 3 key layers (input, mid, output): `3 × 5120 × 2 = 31 KB/token`

Quality-impact ordering (expected):
- Every layer > every 4th layer > 3 key layers
- Paper uses every layer via MSE; skipping layers changes the recipe

### 2.2 Top-k logit approximation

Standard KL distillation uses full-vocab softmax. Top-k truncation
keeps only the top-k logits per position; remaining vocab is
treated as "other" mass.

Literature (MiniLLM, DistillationSurvey 2024) shows:
- **k=32** preserves >99% of KL signal
- **k=256** is essentially lossless
- Below k=16, tail mass matters

Our validation plan:
1. Generate top-256 cache
2. Compute live KL on a handful of batches with full teacher
3. Compute cached KL with k=256 on same batches
4. Confirm KL values match within ~1-3%
5. If yes, k=256 is safe. If not, cache top-1024.

### 2.3 Storage math for paper-scale corpus (500M tokens)

| Config | Storage |
|---|---:|
| Top-256 logits only | ~0.75 TB |
| + hidden states every 4th layer (30B) | ~102 TB |
| + hidden states at 3 key layers (30B) | ~16 TB |
| Reduced corpus to 50M tokens + top-32 + 3 key layers | ~1.6 TB |

**Realistic config for local 30B training**:
- Corpus: 50M tokens (still 12× Phase B's size, 10× less than paper)
- Top-32 logits per token: `50M × 256B = ~13 GB`
- Hidden states at 3 layers of 30B: `50M × 3 × 5120 × 2 = 1.5 TB`

1.5 TB on NVMe SSD: achievable on any reasonable workstation disk.

### 2.4 Where to run teacher forward

Teacher extraction is **embarrassingly parallel** (each token
independent) and **one-time per corpus**. Options by scale:

| Teacher size | Where to run | Wall clock | Cost |
|---|---|---:|---:|
| 0.5B — 7B | Local GPU, bf16 | 0.5 — 4 hours | $0 |
| 13B — 30B | **Rented cloud GPU** (single A100 for hours) | 2 — 6 hours | **~$5-15** |
| 70B+ | Rented H100 or A100 x2 | 6 — 20 hours | ~$25-80 |

Cloud for teacher is **much cheaper** than cloud for student
training because:
- It's forward-only, no backward
- No optimizer, no Adam state
- No gradient checkpointing overhead
- Larger batches easily fit → better GPU utilization

**30B teacher extraction on one cloud A100**: ~$8 one-time, enables
unlimited local student training runs.

### 2.5 Caching format

Binary packed files, one per training sample:

```
/cache/teacher/
  ├── metadata.json        # vocab_size, k, cached_layer_indices, token_count
  ├── logits/
  │   ├── 00000000.bin     # top-k values + indices, aligned
  │   ├── 00000001.bin
  │   └── ...
  └── hidden/
      ├── layer_04/
      │   ├── 00000000.bin
      │   └── ...
      ├── layer_40/
      │   └── ...
      └── layer_76/
          └── ...
```

Accessed at training time via `mmap` for zero-copy reads.

## 3. Streamed student training

### 3.1 The tight inner loop

Per training step with student on CPU, teacher cached on disk:

```python
for step in range(num_steps):
    for micro_batch in accumulation_steps:
        tokens = next(loader)                    # from train stream
        teacher_topk = logit_cache[tokens]       # mmap'd from disk
        teacher_hidden = hidden_cache[tokens]    # mmap'd from disk

        # Streamed student forward
        h = embed(tokens)
        for layer_idx in range(num_layers):
            layer = student_layers[layer_idx].to(device)
            h = layer(h)
            if layer_idx in hidden_mse_layers:
                student_hidden[layer_idx] = h
            student_layers[layer_idx].to("cpu")   # kick back

        logits = lm_head(h)
        loss = kl_topk(logits, teacher_topk) \
             + 10 * sum(mse(student_hidden[i], teacher_hidden[i])
                        for i in hidden_mse_layers)
        loss.backward()

        # Streamed backward — symmetric layer loading during
        # gradient computation
        ...

    opt.step()  # 8-bit Adam, state on CPU
```

### 3.2 Memory profile during one step

| Component | GPU memory |
|---|---:|
| Current active layer (fp32 params + grads + activations) | ~3-8 GB depending on scale |
| Student embeddings + lm_head (fp32) | ~2 GB at 30B |
| Teacher top-k logits (loaded for this batch) | ~50 MB |
| Teacher hidden states (loaded for this batch) | ~500 MB |
| KL + MSE loss workspace | ~1 GB |
| **Total peak GPU** | **~7-12 GB** |

Independent of model size! The active layer size scales with
hidden-dim but not with num_layers.

### 3.3 CPU / disk requirements

| Component | System RAM | Disk |
|---|---:|---:|
| Student params (bf16) | `model_size × 2` | — |
| Student gradients (bf16) | `model_size × 2` | — |
| 8-bit Adam state | `model_size × 1` | — |
| Teacher cache logits | — | ~15 GB |
| Teacher cache hidden states | — | ~1.5 TB |
| **Total for 30B** | **~150 GB RAM** | **~1.5 TB disk** |

150 GB RAM is a workstation, not a laptop. That's the real cost —
consumer desktops max at 128 GB DDR5 typically; 192 GB / 256 GB
needs high-end board.

For 70B: ~350 GB RAM. Out of consumer range; server hardware.

### 3.4 Wall clock estimate

Per-step cost breakdown:
- Activation size per layer (bf16): 3-5 MB at 30B batch=1 seq=512
- PCIe transfer time per layer: 3-5 MB / 16 GB/s = 0.3 ms per layer
- Forward + backward over 80 layers: 80 × 2 × 0.3 = 48 ms transfer
- Plus compute: ~100-300 ms per layer backward pass
- Per micro-step total: ~10-30 seconds for 30B
- Per opt-step (grad_accum=4): ~40-120 seconds
- 8000 opt-steps: **~90-270 hours for 30B**

**That's 4-11 days per training run.** Long but possible. Compare:
- Current Phase B 0.5B on GPU: ~10 hours
- 30B via this architecture: ~100-200 hours
- 30B via cloud A100x4: ~30 hours, $180

Cloud still wins on wall time. Local wins on total cost if you run
multiple ablations (amortize teacher extraction cost).

## 4. Quality considerations

### 4.1 What this architecture preserves vs full training

✅ **Preserved**:
- Same teacher (logits + hidden states) — identical distillation target
- Same LittleBit format — same compression mechanism
- Same end-to-end gradient flow through student — not block-wise local
- Same KL + MSE loss formulation

⚠️ **Approximated**:
- Top-k logit truncation (lose tail mass) — expected <3% PPL impact
- Reduced layer set for MSE — paper uses every layer, we use subset
- Reduced corpus size — 50M vs paper's ~500M+ tokens

❌ **Changed vs paper**:
- Still nothing that invalidates the methodology

### 4.2 Not block-wise — that's important

This is NOT GPTQ-style per-block local optimization. The student
sees the full loss through the full forward + backward. Signs flip
in layer 5 based on gradient signal that propagated from layer 79's
logits. Same dynamics as full-GPU training.

The only "streaming" is **memory management**: layers move between
CPU and GPU during their active window. Gradient flows are
unchanged.

## 5. Concrete implementation plan

### Phase I — cache infrastructure (~2 days)

1. **Cache format + reader**: define binary layout, mmap loader,
   metadata validator
2. **Extraction script** `littlebit_teacher_extract.py`:
   - Load teacher model (GPU if fits, CPU otherwise)
   - Iterate corpus, compute top-k + hidden states
   - Write to cache
   - Validate readback
3. **Top-k KL loss function** `chunked_kl_topk(logits, topk_cache)`
4. **Unit test**: cached KL matches live KL within 2% on 100 batches

### Phase II — CPU-offloaded student (~2 days)

1. **Layer streaming** via PyTorch `.to()` hooks per layer
2. **Gradient handling**: accumulate in fp32 on CPU, apply in
   micro-steps
3. **8-bit Adam on CPU**: bitsandbytes CPU variant
4. Smoke test at 1.5B locally (known to fit on GPU): does streamed
   version produce same PPL trajectory?

### Phase III — scale tests

1. **3B** with streamed student + cached teacher: validate the full
   pipeline end-to-end
2. **7B** with same, ~2-3 days wall
3. **13B** next, ~4-6 days wall
4. **30B** when everything works, ~5-11 days wall

### Phase IV — publication (optional)

Write up the architecture. This is genuinely novel for QAT at
consumer scale.

## 6. Comparison: this architecture vs cloud

| Path | Wall clock | Cash | Per-run cost after setup |
|---|---:|---:|---:|
| 30B cloud (4× A100, Vast.ai) | ~30 hours | $180 | $180 |
| 30B local, this architecture | ~100-200 hours | ~$10 (teacher) | ~$10 |
| 70B cloud | ~100 hours | $1500 | $1500 |
| 70B local, this architecture | ~300-500 hours | ~$25 | $25 |

Local wins on cash but **loses decisively on wall clock**. The
economics favor local only if:
- You plan to run many ablation variants (cache once, reuse many
  times) — amortize teacher extraction across 10+ runs
- You have patience / don't need fast turnaround
- You don't want to set up cloud workflow

## 7. Key risks

| Risk | Impact | Mitigation |
|---|---|---|
| Top-k truncation degrades quality | +3-5% PPL vs full | Run validation comparison before committing |
| Hidden state subset breaks MSE signal | Student fails to align | Ablate layer count: 24 → 6 → 3 at 0.5B first |
| Disk I/O bottlenecks student training | ~2× slowdown | Use NVMe; prefetch next batch while computing current |
| CPU-GPU PCIe saturates | ~3-5× slowdown | Already factored into estimates |
| Student layer CPU transfer races with next layer's prefetch | Stalls | Manual pipelining, buffer layer N+1 during N's compute |
| Cached hidden states mismatch live teacher at inference | Irrelevant — inference uses student alone | — |

## 8. When NOT to do this

- If cloud 30B for $180 once is an acceptable cost
- If you don't need ablation iteration (one-and-done training)
- If disk space (1.5+ TB) is constrained
- If system RAM is ≤64 GB (hard limit for streamed 30B)

## 9. Open questions

- **Exactly how many hidden-state layers are required for MSE to
  work at scale?** Paper uses every layer. Our §15 eval showed
  KL-only breaks gen even with hidden states implicitly available.
  Would 3 cached layers suffice? 6? 12? **Worth ablating on 0.5B
  before committing to 30B disk space.**
- **Can we get away with `int8` hidden state cache?** Would halve
  storage. Precision loss unclear.
- **What's the right corpus size?** 50M tokens is a guess. Could be
  20M or 100M. Should ablate.
- **Teacher extraction on CPU vs GPU for 30B**: haven't benchmarked.
  CPU int4 30B forward might take 10-30 days for 50M tokens. Cloud
  GPU cheaper in wall time.

## 10. Relation to Atlas deployment

If we pull this off locally:
- We can produce LittleBit-compressed Atlas models in-house
- No cloud dependency for reproduction / iteration
- Faithful to the "llama.cpp fork, local everything" ethos of this
  repository
- Downstream: compressed GGUF would ship as an upstream-loadable
  tensor type (separate engineering project)

Worth pursuing as a longer-term architecture whether or not the
short-term 7B work goes cloud vs local.

## 11. Execution gating

**Don't start this work until**:
1. Phase B finishes and we have a trusted 0.5B baseline
2. [savings_exploration_plan.md](savings_exploration_plan.md)
   Runs A-G validate which savings stack safely
3. We've decided 7B local with the stack (from plan above) is the
   right next target

If Runs A-G kill the approach (generation still broken even with
full paper recipe at 0.5B), this 30B plan is on hold until we know
what's wrong with our smaller-scale reproduction.
