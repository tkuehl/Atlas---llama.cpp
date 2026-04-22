# Wall-time and resource reduction plan

> **Part of the LittleBit plan set.** See [README.md](README.md)
> and [consolidated_implementation_roadmap.md](consolidated_implementation_roadmap.md)
> for context.  Related plans:
> [savings](savings_exploration_plan.md) (memory) ·
> [unexplored gains](unexplored_efficiency_gains.md) (quality) ·
> [scale-to-30B](scale_to_30b_architecture.md) (architecture) ·
> [memory research](memory_efficient_training_research.md) (external
> survey).

Companion to [unexplored_efficiency_gains.md](unexplored_efficiency_gains.md)
(quality combinations) and [savings_exploration_plan.md](savings_exploration_plan.md)
(memory ablations). **Focus here is wall-clock reduction**: get to a
result faster, per run, without worsening the result.

Phase B baseline: ~8-10 hours wall clock for 20,000 opt-steps at
~1.78s/step. Question: **how low can we push that?**

## 1. Per-step cost breakdown (measured in Phase B)

Rough decomposition of a single opt-step (4 micro-steps × forward +
backward + gradient reduce + opt.step):

| Component | Est. time | % of step |
|---|---:|---:|
| Teacher forward (bf16, no_grad) × 4 | ~0.3s | 17% |
| Student forward × 4 | ~0.5s | 28% |
| Loss computation (KL + MSE) × 4 | ~0.2s | 11% |
| Student backward × 4 | ~0.6s | 34% |
| Gradient clip + opt.step | ~0.1s | 5% |
| Python + I/O overhead | ~0.08s | 5% |
| **Total per opt-step** | **~1.78s** | **100%** |

Biggest reduce-targets:
1. Student backward (34%) — dominated by matmul + SmoothSign
2. Student forward (28%) — matmul + output_hidden_states overhead
3. Teacher forward (17%) — cacheable offline

## 2. Wall-time reductions, ranked

### 2.1 Teacher forward cache [estimated -17% per step, 100% of teacher eliminated]

**Already covered** in [scale_to_30b_architecture.md §2](scale_to_30b_architecture.md).
Relevant here too: for our current 0.5B runs, teacher forward is
~17% of step time. Caching eliminates it entirely after one
one-time offline run.

For 0.5B: teacher extraction takes ~30 min. Save across every
subsequent ablation: 17% × 8h = **~80 minutes saved per future
training run**. Break-even after 1 run.

**For larger models**: teacher forward % scales with model size;
at 7B teacher forward is proportionally larger (maybe 30-40% of
step time at matched student size).

**Effort**: ~2 days to build cache infrastructure once. Already
scoped in §2 of the 30B architecture doc.

### 2.2 `torch.compile(student)` [estimated -30-50% per step]

Compile the student's forward/backward graph. Removes Python overhead,
enables kernel fusion, should hide some launch latency.

```python
if args.compile:
    student = torch.compile(student, mode="reduce-overhead")
```

**Risk**:
- Our `SmoothSignEfficient` custom autograd function may break
  graph compilation (graph break). If it does, `reduce-overhead`
  mode silently degrades to partial compile.
- `gradient_checkpointing` and `torch.compile` historically don't
  compose perfectly. HF has been improving this.
- 8-bit Adam's compute may not be compile-friendly.

**Mitigation**: try `mode="default"` first (less aggressive), add
fallback path if it fails at startup.

**Effort**: 10 lines + smoke test.

**Expected savings**: 30-50% per step if it compiles. For Phase B's
8h: **2-4 hours saved**.

### 2.3 Early-stopping on PPL plateau [estimated -20-40% on total time]

Run 3's trajectory: PPL hit 93 at step 6000, final was 93 at step
8000. Last 2000 steps were pure LR-decay noise, no real improvement.

**Fix**: detect plateau and stop. Criterion:

```python
def should_stop(history, window=5, min_delta=2.0, min_steps=4000):
    if len(history) < window + 1 or history[-1]["step"] < min_steps:
        return False
    recent_ppls = [h["ppl"] for h in history[-window:]]
    return (max(recent_ppls) - min(recent_ppls)) < min_delta
```

Meaning: if PPL has moved less than 2 points over the last 5 evals
and we're past step 4000, stop. Saves the tail 20-40% of training
that isn't improving quality.

**Effort**: 20 lines.

**Expected savings**: 20-40% on runs that plateau early; 0% on runs
that are genuinely still improving. Pure upside.

### 2.4 Reduce eval cadence [estimated -1% per step, -60s total]

Currently every 500 steps, 20 evals × 5-7s each = 100-140s wall
time. Change to every 2000 steps:
- 5 evals × 5-7s = 25-35s saved
- Fewer intermediate PPL curves but final result unchanged

**Effort**: one CLI value change.

**Expected savings**: ~60-100s per 20k-step run. Small but free.

### 2.5 Liger fused kernels (RMSNorm + RoPE + SwiGLU) [estimated -10-15% per step]

Already referenced in [memory_efficient_training_research.md](memory_efficient_training_research.md).
These three ops are ~15% of student forward time combined. Liger's
fused Triton kernels are 2-4× faster per op.

**Compatibility**: Qwen2's RMSNorm / RoPE / SwiGLU are in
`transformers.models.qwen2.modeling_qwen2`. Liger patches them via
`apply_liger_kernel_to_qwen2(student)`. Our LittleBit wrapping
doesn't touch those modules, so there's no direct conflict.

**Effort**: ~10 lines + `pip install liger-kernel`.

**Expected savings**: 10-15% per step. For Phase B: **~50-70 min
saved per run**.

### 2.6 CUDA graphs [estimated -5-10% per step at small scale]

PyTorch's `torch.cuda.graph()` captures a sequence of GPU operations
and replays them with minimal kernel-launch overhead. Helpful when
Python/driver overhead is a significant share of per-step time.

**Best for**: small batch sizes with many small kernels.
**Our regime**: batch=1 seq=512 has lots of small kernels → maybe
~5-10% gain.

**Risk**: CUDA graphs don't tolerate dynamic shapes or data-dependent
control flow. Our SmoothSign.backward has `torch.where` but shapes
are fixed. Probably works.

**Effort**: significant. CUDA graphs require refactoring the training
step to be replayable. ~200 lines of code + careful testing.

**Verdict**: defer until after lower-effort wins.

### 2.7 Parallel teacher + student forward via CUDA streams [estimated -10-15% per step]

Currently sequential: teacher forward runs first (under `no_grad`),
then student forward. On a single GPU, these use the same compute
resources but can be interleaved via separate CUDA streams.

```python
teacher_stream = torch.cuda.Stream()
student_stream = torch.cuda.Stream()
with torch.cuda.stream(teacher_stream):
    t_out = teacher(batch, ...)  # no_grad, bf16
with torch.cuda.stream(student_stream):
    s_out = student(batch, ...)
# Wait for both to complete before loss
torch.cuda.synchronize()
```

**Theory**: teacher is bf16 (tensor cores happy), student is fp32
(different kernels). Might interleave on SM occupancy.

**Practice**: single-GPU concurrent streams rarely hit 2× speedup
because kernels share SM resources. Realistic: 10-15%.

**Risk**: synchronization bugs, harder to debug.

**Effort**: ~50 lines + testing.

### 2.8 Skip `output_hidden_states=True`, use forward hooks [estimated -5-10% per step]

Already called out in [savings_exploration_plan.md](savings_exploration_plan.md)
Run D. HF's `output_hidden_states=True` path retains a list of tensors
across the model and defeats some gradient-checkpointing savings.
Hooks that capture and release immediately are more memory-friendly
AND slightly faster (less tensor tracking).

**Effort**: ~30 lines.

**Expected**: 5-10% per step savings in our MSE-enabled runs.

### 2.9 Reduced grad_accum steps with larger physical batch [neutral-to-positive]

Currently: `batch=1, grad_accum=4` = effective batch 4. Per opt-step:
4 forward/backward passes.

If memory optimizations allow `batch=2, grad_accum=2`: same effective
batch, but now 2 forward/backward passes per opt-step. Per-step
latency halved at cost of 2× memory usage.

**Tradeoff**: memory (doubled) vs wall clock (halved per opt-step).
Opens up only with stacked memory optimizations from the savings
plan.

**Expected savings once we have the memory**: **35-40% per opt-step**
at same effective batch size.

### 2.10 TF32 on for matmuls [estimated -10-20% per matmul on Ampere+]

```python
torch.set_float32_matmul_precision("high")  # TF32
torch.backends.cuda.matmul.allow_tf32 = True
```

TF32 uses 10-bit mantissa for fp32 matmuls on Ampere/Hopper/Blackwell.
Near-zero quality impact, ~15% speedup on matmul-heavy workloads.

**RTX 5080 Laptop**: Blackwell, supports TF32 natively.

**Effort**: 2 lines.

**Expected savings**: 10-20% on student forward/backward (matmul
dominated). For Phase B: **~50-90 min saved**.

### 2.11 Fused LittleBit-linear Triton kernel [speculative, -20-30%]

Our Eq. 5 efficient form:

```python
y = x * g                    # broadcast
y = y @ V_sign               # matmul
y = y * ell                  # broadcast
y = y @ U_sign.T             # matmul
y = y * h                    # broadcast
```

5 separate kernel launches. A fused Triton kernel could merge them
into one, eliminating 4 kernel-launch overheads and intermediate
tensor allocations.

**Risk**: Triton kernels need careful debugging. Also: the SmoothSign
backward would need a matching fused kernel, doubling the engineering
work.

**Effort**: 1-2 days Triton development + validation.

**Expected savings**: 20-30% per student forward/backward. Only
worth it after simpler wins are in.

### 2.12 CPU-side data prefetch [estimated -5-10%]

Currently `batch = next(it).to(device)` is synchronous. A proper
`DataLoader` with `num_workers > 0` and `pin_memory=True` overlaps
tokenization + device copy with compute.

Our `prepare_train_stream` already pre-tokenizes the full corpus.
The overhead is just `tokens[start:start+seq_len]` slicing +
`.to(device)`. Still, async prefetch hides the ~50-100ms latency.

**Effort**: refactor sampler to proper DataLoader. ~50 lines.

**Expected savings**: 3-7%.

### 2.13 Remove unnecessary `output_hidden_states=True` when MSE weight=0

Our code already gates `output_hidden_states=bool(args.inter_mse_weight)`.
Not a gain — just a note that it's correctly implemented.

### 2.14 Shorter initial sequences (curriculum for wall time, not just quality)

From [unexplored_efficiency_gains.md §4.1](unexplored_efficiency_gains.md)
— curriculum learning. Wall-time angle:
- seq=128 for first 2000 steps → 4× faster per step
- seq=256 for next 4000 → 2× faster
- seq=512 for remaining 14000 → full speed

Total wall clock: **~20% reduction** if we do this. Quality impact
is the open question — short sequences give weaker long-range signal
but earlier convergence.

**Effort**: moderate. Requires sampler support for variable seq_len
across training phases.

## 3. Resource reductions (GPU + RAM + disk)

### 3.1 Eval-only harness decouples teacher from training

Our eval script currently loads teacher alongside student. But for
evaluating a saved checkpoint, teacher is only needed for PPL
comparison — which is just a fixed number per corpus.

**Fix**: cache teacher PPL once per (model, dataset, seq_len)
combo. Eval script reads cached value, only needs student in memory.

**Savings**: eval at 30B scale wouldn't need 60 GB of teacher RAM.

### 3.2 Shared calibration data across runs

Multiple ablation techniques (Dual-SVID init, CALDERA init, XTX for
activation-weighted QAT) want similar calibration activations. One
extraction run, saved to disk, reused across initialization variants.

**Effort**: small refactor of the extraction script to save
calibration data as reusable artifacts. ~30 lines.

**Savings**: ~10-30 minutes per init variant, compound across
ablations.

### 3.3 In-place activation accumulation for MSE

Current MSE computes `l_inter = l_inter + mse_loss(sh, th)` in a
Python for-loop, allocating an intermediate per iteration. For 24
hidden-state pairs at batch=1 seq=512, that's ~24 small loss tensors
accumulated.

**Fix**: stack and mean:

```python
sh_stack = torch.stack(s_hidden)  # (n_layers, batch, seq, hidden)
th_stack = torch.stack([th.to(sh_stack.dtype)
                        for th in t_hidden_list])  # same
l_inter = F.mse_loss(sh_stack, th_stack) * n_layers
```

Same math as sum, single loss computation. Fewer Python iterations.

**Effort**: ~10 lines.

**Savings**: minor per-step, but freeze-flash memory profile.

### 3.4 Rolling checkpoint write-efficiency

Our rolling checkpoint saves full optimizer state each time. At 0.5B
that's ~2 GB per save. Every `--ckpt-every 1000` steps: ~40 MB/s
write bandwidth for the save.

**Optimization**: save optimizer state in bf16 instead of fp32.
Halves write size. AdamW `m` and `v` in bf16 may cost minor
precision but bitsandbytes 8-bit Adam is already coarser than bf16
without issues.

**Effort**: ~20 lines.

**Savings**: halves checkpoint-save time (few seconds per save).

## 4. Stacked wall-time projection

Assuming all low-risk gains stack without interference:

| Gain | Mult factor | Cumulative |
|---|---:|---:|
| Baseline (Phase B) | 1.00× | 100% |
| + TF32 matmul (§2.10) | 0.87× | 87% |
| + Liger fused kernels (§2.5) | 0.88× | 77% |
| + torch.compile (§2.2) | 0.70× | 54% |
| + Teacher cache (§2.1) | 0.90× | 49% |
| + Forward hooks (§2.8) | 0.93× | 45% |
| + Larger physical batch (§2.9) | 0.62× | 28% |
| + CPU data prefetch (§2.12) | 0.95× | 26% |
| + Plateau early-stopping (§2.3) | 0.80× | 21% |

**~79% wall-time reduction** if all stack cleanly. Phase B's 8h →
**~1.7h**. Even if only half the gains materialize, 8h → **~3h**.

The biggest single wins are `torch.compile` and larger physical
batch (which requires the memory savings from the other plan first).

## 5. Realistic ordering for adoption

Same **effort × impact** calculus:

| Order | Technique | Effort | Wall-time gain | Correctness risk |
|---:|---|---|---:|---|
| 1 | TF32 matmul | 2 lines | ~13% | None |
| 2 | Eval cadence (500 → 2000) | 1 flag | ~1% | None |
| 3 | Plateau early-stop | ~20 lines | ~20-30% | None |
| 4 | Liger norm/RoPE/SwiGLU | ~10 lines + pip | ~10-15% | Low |
| 5 | `torch.compile(student)` | ~10 lines | ~30-50% | Medium |
| 6 | Forward hooks for MSE | ~30 lines | ~5-10% | Low |
| 7 | Teacher cache | ~300 lines | ~17% + scale unlock | Low |
| 8 | CPU data prefetch | ~50 lines | ~5-7% | Low |
| 9 | Larger physical batch | depends on memory | ~35% | Medium |
| 10 | CUDA streams for teacher/student | ~50 lines | ~10-15% | Medium |
| 11 | CUDA graphs | ~200 lines | ~5-10% | High |
| 12 | Fused Triton LittleBit kernel | ~1 week | ~20-30% | High |

**Adopt 1-5 next session (~1 hour of coding)**: cumulative ~50-60%
wall-time reduction. Drops Phase B-class runs to ~4-5 hours.

**Adopt 6-9 within one more session (~3 hours of coding)**: ~70%+
cumulative. Phase B to ~2.5-3 hours.

**10-12 are for when we really care** — one-time engineering cost
for ongoing productivity gain across many runs.

## 6. Resource budget at each level

If we commit to the full stack (items 1-10):

| Resource | Baseline | Full stack |
|---|---:|---:|
| GPU VRAM (peak) | 12.7 GB | ~8-10 GB |
| System RAM (training) | ~8 GB | ~8 GB |
| Disk (checkpoints + cache) | ~3 GB | ~15 GB (teacher cache) |
| Wall clock per 20k step | 8-10 h | **~2.5-3 h** |
| Engineering time to adopt | 0 | ~6-8 hours |

## 7. What this enables

### For 0.5B ablations
Each ablation from [savings_exploration_plan.md](savings_exploration_plan.md)
drops from ~4h (Short tier) to ~1.5h with full stack. **Enables 5-8
ablations per day instead of 2-3.**

### For 7B local
Memory-stacked 7B (from the savings plan) was estimated at 7-10
hours. With wall-time stack, potentially **~3-4 hours for a 7B
training run** — fits comfortably in an overnight or single-day
turnaround once Sprint 3 (teacher cache) removes teacher from VRAM.

### For 30B local
Already estimated at 100-200 hours under old architecture.
Wall-time stack cuts this to maybe **~40-60 hours** — still very
long but feasible for overnight runs over a weekend.

## 8. Low-hanging fruit checklist

Before the next serious ablation, adopt:

- [ ] `torch.set_float32_matmul_precision("high")` at script start
- [ ] `torch.backends.cuda.matmul.allow_tf32 = True`
- [ ] `torch.backends.cudnn.allow_tf32 = True`
- [ ] `torch.backends.cudnn.benchmark = True` (for fixed shape workloads)
- [ ] `--eval-every` default bumped to 2000 (currently 500)
- [ ] Plateau early-stop (min_delta=2, window=5, min_steps=4000)
- [ ] `torch.compile(student)` with graceful fallback on failure
- [ ] Smoke test: verify all above don't break Run 3's reproducibility

That's a **morning of work** for roughly half the wall-time
reduction of the full stack. Everything else is incremental.

## 9. Engineering effort vs saving tradeoffs

For items requiring >4h engineering (teacher cache, fused Triton,
CUDA graphs), adoption should be gated on actually running at least
5 more full-scale training runs. Amortized savings:

| Engineering hours | Break-even runs | Where we are |
|---:|---:|---|
| 4 | ~2 runs | Already justified by planned ablations |
| 16 | ~8 runs | Justified if we do 1.5B + 3B + 7B + ablations |
| 40 | ~20 runs | Only if we commit to 30B+ local |

Teacher cache is in the 16-hour tier → justified **after** we decide
to do the full scale ladder.

## 10. Relationship to other plans

- [savings_exploration_plan.md](savings_exploration_plan.md) is
  about memory (fitting larger models / batches). This doc is about
  wall time. They compose — memory savings enable larger physical
  batches, which reduce wall time.
- [scale_to_30b_architecture.md](scale_to_30b_architecture.md) is
  about scale. Teacher cache from §2.1 here is a building block for
  the 30B architecture.
- [unexplored_efficiency_gains.md](unexplored_efficiency_gains.md)
  is about quality combinations. Orthogonal to this but some of its
  items (e.g., staged training, incremental rank) also reduce wall
  time.

These four plans together describe a coherent roadmap for getting
LittleBit from "proof of concept on 0.5B in 10 hours" to "scaled
reproduction on 7B+ in reasonable time, on consumer hardware."
