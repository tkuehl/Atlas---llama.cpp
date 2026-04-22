# Scaling LittleBit QAT to 30B+ on consumer hardware

> **Part of the LittleBit plan set.** See [README.md](README.md)
> for the index and
> [consolidated_implementation_roadmap.md](consolidated_implementation_roadmap.md)
> for Sprint ordering.  Gated on Sprint 2 (teacher cache).
> Related: [savings](savings_exploration_plan.md) ·
> [wall-time](wall_time_reduction_plan.md) ·
> [inference runtime](inference_runtime.md) ·
> [memory research](memory_efficient_training_research.md).

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
independent) and **one-time per corpus**. All extraction runs
locally — cloud compute is not an option for this project. Options
by scale:

| Teacher size | Local approach | Wall clock |
|---|---|---:|
| 0.5B — 7B | Full model on 16 GB GPU, bf16, large batch | 0.5 — 4 hours |
| 13B — 30B | Streamed layers (same CPU↔GPU architecture as student §3) | 1 — 5 days |
| 70B+ | Streamed layers + NVMe tier (§11) | ~1-3 weeks |

Teacher extraction is faster per token than student training at the
same scale because:
- It's forward-only, no backward
- No optimizer, no Adam state
- No gradient checkpointing overhead
- Larger batches easily fit → better GPU utilization per layer swap

**30B teacher extraction locally**: multi-day one-time run via the
same streaming architecture as student training. The cost is wall
clock, not cash. Amortizes across every subsequent student
ablation on that teacher + corpus — run once, reuse forever.

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

**That's 4-11 days per training run.** Long but possible. Compare
against our existing local baseline:
- Current Phase B 0.5B on GPU: ~10 hours
- 30B via this architecture: ~100-200 hours (4-8 days)

That wall clock is the cost of this project's local-only
constraint. Accept it, plan for it, amortize it across ablations.

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

## 6. Wall-clock expectations (local-only)

| Path | Teacher extraction | Per student run |
|---|---:|---:|
| 7B local (post Sprint 3 teacher cache) | ~2-4 hours | ~3-6 hours |
| 13B local, this architecture | ~1-2 days | ~1-2 days |
| 30B local, this architecture | ~3-5 days | ~4-8 days |
| 70B local, NVMe tier (§11) | ~1-3 weeks | ~14-21 days |

Teacher extraction is paid once per teacher + corpus combo and
amortized over every subsequent student ablation. The per-run
wall clock is the cost of this project's local-only constraint.
Plan ablation batches so each run has time to complete overnight
or over a weekend.

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

- If you don't need ablation iteration (one-and-done training) —
  the one-time wall clock is hard to justify without reuse
- If disk space (1.5+ TB) is constrained
- If system RAM is ≤64 GB (hard limit for streamed 30B — see §11
  NVMe tier for a workaround at cost of additional wall clock)
- If wall-clock multi-day training is incompatible with the project
  timeline — in which case, stop at whatever scale fits in a
  single overnight run (7B is comfortably in range)

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
- **Teacher extraction for 30B — streamed GPU vs CPU int4**: haven't
  benchmarked. CPU int4 30B forward might take 10-30 days for 50M
  tokens. Streamed-layer GPU forward (same architecture as student
  §3) likely lands in the 1-5 day range per the table in §2.4, but
  needs measurement. One-time cost either way; benchmark before
  committing to which path.

## 10. Relation to Atlas deployment

If we pull this off locally:
- We can produce LittleBit-compressed Atlas models in-house
- Zero external dependency for reproduction / iteration
- Faithful to the "llama.cpp fork, local everything" ethos of this
  repository
- Downstream: compressed GGUF would ship as an upstream-loadable
  tensor type (separate engineering project)

Worth pursuing as a longer-term architecture once short-term
7B local work lands.

## 11. NVMe tier — extending to 70B+ on consumer hardware

Previous sections assumed student state lives in **system RAM**.
That caps us at workstation-class hardware (150 GB RAM for 30B,
350 GB for 70B — the latter is out of consumer range).

Adding a third tier — **NVMe SSD as cold storage** — removes the
RAM cap.

### 11.1 Memory hierarchy

```
  GPU VRAM  (16 GB):  active layer + activations + I/O buffers
     ↕  PCIe Gen4 x16, ~32 GB/s
  System RAM (32-64 GB):  hot-layer cache + grads + Adam state
     ↕  M.2 NVMe Gen4-5, ~7-14 GB/s
  NVMe SSD  (2-4 TB):  full student params (cold)
```

### 11.2 Bandwidth math

Per-layer transfer times for 30B (~750 MB bf16 per layer at d=5120)
and 70B (~1.75 GB bf16 per layer at d=8192):

| Storage | 30B layer | 70B layer |
|---|---:|---:|
| RAM (DDR5) | 10 ms | 25 ms |
| NVMe Gen4 | 100 ms | 250 ms |
| NVMe Gen5 | 50 ms | 125 ms |

NVMe is 7-10× slower than RAM but *can be hidden* if prefetching
layer N+1 during layer N's compute.  Effective wall-clock penalty:
~0-20% depending on whether compute dominates I/O.

### 11.3 Wall-clock estimates for prefetched NVMe tier

| Model | RAM-only architecture | NVMe-backed architecture |
|---|---:|---:|
| 7B | ~30 hours | ~35 hours |
| 30B | ~100-200 hours | ~130-250 hours |
| **70B** | **infeasible (350 GB RAM)** | **~330-500 hours (~14-21 days)** |

At 70B scale, per-layer compute dominates I/O enough that NVMe is
only marginally slower than hypothetical-if-you-had-the-RAM.

### 11.4 System RAM requirements drop dramatically

With NVMe backing the cold store, RAM just needs to hold 2-3 hot
layers plus the teacher cache page + optimizer workspace:

| Model | RAM (tier-only) | RAM (NVMe-tiered, 2-layer hot cache) |
|---|---:|---:|
| 7B | 35 GB | ~8 GB |
| 30B | 150 GB | ~24 GB |
| 70B | 350 GB | ~40-64 GB |

**Consumer hardware (64 GB RAM + 2 TB NVMe + 16 GB GPU) becomes
viable for 70B** under this architecture.

### 11.5 Critical caveat: SSD write wear

Naive NVMe-offload burns drive life fast:
- Adam state updated every step: read + write per step
- For 30B with 16 GB Adam state × 8000 steps: **~128 TB of writes
  per training run**

Consumer 2 TB NVMe is rated 600-1200 TBW.  **One 30B training run
consumes 10-20% of drive life.**  Unacceptable for iteration.

Three mitigations, in order of preference:

**Option A: Write-through-param-only architecture** (recommended)
- NVMe: student params (cold, bf16, written once at checkpoint,
  otherwise read-only during training)
- RAM: gradients, 8-bit Adam state, 2-layer hot cache
- Writes to NVMe happen only at checkpoint boundaries (e.g. every
  500 steps).  Adam state never touches NVMe.
- Total writes per 30B run: ~30 GB × 16 checkpoints = ~0.5 TB.
  Well under drive wear budget.

**Option B: Enterprise SSD**
- 10-30 PBW rated (vs 0.6-1.2 PB consumer)
- $500-1500 for 2 TB
- Lasts years of training

**Option C: RAM-disk / tmpfs for hot state**
- Allocate ~16-32 GB of RAM as tmpfs
- Adam state lives on tmpfs (backed by RAM, persists through
  process restart if machine doesn't reboot)
- NVMe holds params only

Option A is cleanest and fits our architecture naturally.

### 11.6 Adopted architecture for 30B-70B local

```
NVMe (read-mostly):
  student_params.safetensors  [140 GB for 70B]
  teacher_cache/
    logits_topk/              [15 GB]
    hidden_layers_3/          [1.5 TB for 50M token corpus at 30B]

RAM (working memory):
  hot_layers[2-3]             [~5 GB at 70B]
  gradients                   [~140 GB at 70B — needs 128 GB+ DDR5]
  adam_state_8bit             [~35 GB at 70B]
  prefetch_buffer_next_layer  [~2 GB]

GPU VRAM:
  current_layer_compute       [~4-8 GB]
  activations                 [~3 GB]
  loss workspace              [~1 GB]
  teacher_cache_page          [~500 MB]
```

At 70B: requires 128 GB+ RAM because gradients must be in RAM
(can't offload grads to NVMe and still do backward efficiently).
If we went further and offloaded grads to NVMe too (with Gen5 and
careful write batching), 64 GB RAM becomes enough — but wall clock
approaches 3-4 weeks per run.

### 11.7 Implementation sketch

Add a `NVMeLayerStore` class to the student:

```python
class NVMeLayerStore:
    def __init__(self, model_path, num_layers, device):
        self.path = model_path  # directory of layer_N.safetensors
        self.device = device
        self.hot_cache = OrderedDict()  # layer_idx → GPU tensor
        self.prefetch_stream = torch.cuda.Stream()

    def get(self, layer_idx):
        if layer_idx in self.hot_cache:
            return self.hot_cache[layer_idx]
        tensor = safetensors.torch.load_file(
            f"{self.path}/layer_{layer_idx}.safetensors",
            device=self.device,
        )
        self.hot_cache[layer_idx] = tensor
        # Evict if over capacity
        while len(self.hot_cache) > 3:
            self.hot_cache.popitem(last=False)
        return tensor

    def prefetch(self, layer_idx):
        # Async load on prefetch stream
        with torch.cuda.stream(self.prefetch_stream):
            _ = self.get(layer_idx)

    def commit_layer_params(self, layer_idx, updated_params):
        # Write-through to NVMe (at checkpoint only; for within-
        # step, keep in hot cache)
        safetensors.torch.save_file(
            updated_params,
            f"{self.path}/layer_{layer_idx}.safetensors",
        )
```

~200 lines. Plugged into the training loop, replaces
`student_layers[i].to(device)` calls.

### 11.8 DeepSpeed ZeRO-Infinity alternative

Microsoft DeepSpeed's `zero_infinity` config does all of this
automatically:

```python
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "nvme", "nvme_path": "/mnt/nvme"},
    "offload_param": {"device": "nvme", "nvme_path": "/mnt/nvme"},
  },
  "aio": {"block_size": 1048576, "queue_depth": 8, "thread_count": 4},
}
```

Tradeoffs vs rolling our own:
- Pro: production-tested, handles pipelining automatically
- Pro: handles write-wear via aio block-size tuning
- Con: Linux-primary (Windows support is partial via WSL2)
- Con: adds DeepSpeed dependency and config complexity
- Con: Our custom LittleBit modules need to be registered with
  DeepSpeed's parameter registry

For a first attempt, **DeepSpeed ZeRO-Infinity is the faster path**.
If we hit compatibility issues with our custom `SmoothSignEfficient`
autograd, we fall back to the hand-rolled `NVMeLayerStore`.

### 11.9 Full hardware picture for 70B local

| Component | Min for 70B |
|---|---|
| GPU | 16 GB (RTX 4080/5080 laptop works) |
| RAM | 128 GB DDR5 |
| NVMe | 2 TB Gen4 or Gen5 (enterprise preferred) |
| CPU | 8+ cores, matters less than RAM |
| Disk | additional 2-4 TB for teacher cache |

Total build: **~$3000-4000 one-time** for a dedicated workstation.

This is the only path to 70B training under the project's
local-only constraint. A workstation upgrade is the prerequisite;
without it, 70B stays out of reach regardless of software
architecture.

---

## 12. Execution gating

**Don't start this work until**:
1. Phase B finishes and we have a trusted 0.5B baseline
2. [savings_exploration_plan.md](savings_exploration_plan.md)
   Runs A-G validate which savings stack safely
3. We've decided 7B local with the stack (from plan above) is the
   right next target

If Runs A-G kill the approach (generation still broken even with
full paper recipe at 0.5B), this 30B plan is on hold until we know
what's wrong with our smaller-scale reproduction.
