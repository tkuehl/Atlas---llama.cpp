# LittleBit QAT — potential enhancements

> **Part of the LittleBit plan set.** See
> [README.md](README.md) for the index and
> [consolidated_implementation_roadmap.md](consolidated_implementation_roadmap.md)
> for the execution sequence.  Baseline and math foundation:
> [littlebit_math.md](littlebit_math.md).
> Related: [savings](savings_exploration_plan.md) ·
> [unexplored gains](unexplored_efficiency_gains.md) ·
> [wall-time](wall_time_reduction_plan.md).

Catalogue of speed / memory / accuracy improvements identified during
the implementation review of our QAT pipeline
([`littlebit_qat_model.py`](littlebit_qat_model.py),
[`littlebit_qat_single.py`](littlebit_qat_single.py),
[`littlebit_qat_activation.py`](littlebit_qat_activation.py),
[`littlebit_eval.py`](littlebit_eval.py)).

Enhancements ranked by expected impact per unit effort. Status column
tracks what's been shipped vs what's open.

Baseline to beat: Qwen 2.5 0.5B, r=512, 8000 steps KL-only QAT,
**PPL 54.81** on wikitext-2 test, ~45 min on RTX 5080 Laptop 16 GB.

## Priority table

| # | Enhancement | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | 8-bit AdamW via bitsandbytes | Enables 7B locally | ~3 lines | **Open** |
| 2 | Intermediate-MSE loss (paper λ=10) | ~5-15% PPL improvement | 1 flag flip | **Open** |
| 3 | Gradient checkpointing on student | Enables seq=1024 at 7B | 1 line | **Open** |
| 3b | SmoothSign memory diet | -1 GB activations | ~6 lines | **Open** |
| 4 | `torch.compile(student)` | 30-50% faster/step | 5 lines + smoke | **Open** |
| 5 | Tau warmup schedule (10 → 100) | Sign-flip dynamics | ~10 lines | **Open** |
| 6 | KD temperature (T=4 softmax) | Softer targets | ~5 lines | **Open** |
| 7 | Per-layer activation-drift logging | Early-kill signal | ~30 lines | **Open** |
| 8 | Periodic checkpoint (not just end) | Crash recovery | ~10 lines | Partially open — end-save shipped |
| 9 | Fix `reduction=batchmean` semantics | Clarity | 3 lines | Open, cosmetic |

## Detailed rationale

### 1. 8-bit AdamW (critical-path for scale)

AdamW stores two fp32 moments (`m`, `v`) per parameter — 8 bytes × params.
At 7B student scale: **11.8 GB just for Adam state**. `bitsandbytes.optim.AdamW8bit`
drops this to ~3 GB at no measurable quality cost in practice
(established across QLoRA and derivative literature).

**Drop-in patch:**

```python
try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=args.lr)
except ImportError:
    opt = torch.optim.AdamW(params, lr=args.lr)
```

Expected saving: 600 MB @ 0.5B, 9 GB @ 7B, 25 GB @ 30B. This alone is
the difference between 7B fitting on a 16 GB GPU or not.

### 2. Intermediate-MSE loss (paper's full recipe)

Paper combines KL on logits with MSE on every hidden state, weighted
`λ=10`. We have the plumbing via `--inter-mse-weight` but it defaults
to `0.0` (KL-only). Flipping the default gives us the paper's recipe
at zero code cost.

**Expected gain**: Literature on LLM distillation consistently shows
hidden-state matching improves quality 5-15% on PPL-class metrics.
For us, this would be pure upside — we've already seen KL alone beat
the FP-SVD floor by 36%; adding MSE should tighten further toward
the paper's OPT-1.3B-at-0.1-BPW number of PPL 53.76.

**Caveat**: Stored hidden states double memory pressure during
training. Only add once gradient checkpointing (3) is in.

### 3. Gradient checkpointing on student

HF's `model.gradient_checkpointing_enable()` is a single line. Trades
~30% per-step compute for ~1 GB memory at seq=512, or ~4 GB at
seq=1024. Without it, we can't run seq=1024 at 0.5B on 16 GB — we
proved this the hard way (OOM paging stall in run 1 attempts).

### 3b. SmoothSign memory diet

Current implementation:

```python
class SmoothSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau):
        ctx.save_for_backward(x)       # <-- saves full x tensor
        ctx.tau = tau
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
```

`x` has shape of `U_fp` / `V_fp`. For our 168 layers at 0.5B:
- `U_fp` mean size ~2500 × 512 fp32 = 5 MB per layer
- `V_fp` mean size ~1500 × 512 fp32 = 3 MB per layer
- Across 168 layers: ~1.3 GB of saved-for-backward activations

At 7B this grows to ~5 GB. Three possible reductions:

**Option A**: Save only `tanh(tau·x)` instead of `x`:

```python
ctx.save_for_backward((tau * x).tanh())  # same shape, same dtype
# backward: surrogate = tau * (1 - tanh_tx ** 2)
```

No memory win but simpler backward; marginal speedup.

**Option B**: Save `tau · (1 - tanh²(tau·x))` directly (the full
surrogate gradient):

```python
surrogate = tau * (1 - (tau * x).tanh() ** 2)
ctx.save_for_backward(surrogate)
# backward: return grad_output * surrogate
```

Same tensor size, but we can also downcast to bf16 (surrogate values
are in [0, tau] range, easily representable):

```python
ctx.save_for_backward(surrogate.to(torch.bfloat16))
```

Halves activation memory for SmoothSign → **-650 MB @ 0.5B, -2.5 GB @ 7B.**

**Option C**: Recompute the surrogate in backward (don't save anything).
Trades compute for memory. Backward pass becomes 2× slower for the
sign op but activation memory goes to zero. Worth only if memory is
critical and compute budget allows.

Recommend Option B for the next iteration: best memory/compute tradeoff.

### 4. `torch.compile(student)`

PyTorch 2.x compile can yield 30-50% speedup on transformer forward
passes. One line:

```python
student = torch.compile(student, mode="reduce-overhead")
```

**Risk**: Our `SmoothSign` custom autograd function may trigger graph
breaks. In `reduce-overhead` mode graph breaks are silent but reduce
speedup. In `max-autotune` mode they can fail. Test with a smoke run;
if it compiles cleanly, keep it. Fallback path is trivial.

Not critical for correctness — purely a speed optimization.

### 5. Tau warmup schedule

The SmoothSign surrogate `tau · (1 − tanh²(tau·x))` is only
meaningfully nonzero when `|x| < ~0.02/tau`. At `tau=100`, only
entries within `|x| < 2e-4` receive backward signal. Since
Dual-SVID initializes `U_fp`, `V_fp` from the SVD factors scaled by
`sqrt(S_k)`, most entries have `|x| > 0.01` at init — **they receive
essentially zero gradient from step 1**. They're sign-frozen.

Proposed schedule:

```python
def tau_at(step):
    if step < warmup_tau_steps:
        return 10.0 + (100.0 - 10.0) * step / warmup_tau_steps
    return 100.0
```

At `tau=10` the active window is `|x| < 2e-3` — ~10x wider, and a
substantially larger fraction of entries can flip. As training
progresses, tau → 100 narrows the window to make refinement precise.

Matches the "coarse early, fine late" annealing pattern common in
straight-through estimators. Paper doesn't ablate this; we could
find a better operating point than their fixed `tau=100`.

**Expected gain**: Uncertain but plausible 5-10% PPL improvement.
Worth testing.

### 6. Knowledge-distillation temperature

Standard distillation softens the teacher's softmax with a
temperature T > 1:

```python
T = 4.0
p_t = F.softmax(t_logits / T, dim=-1)
log_p_s = F.log_softmax(s_logits / T, dim=-1)
loss = (T ** 2) * F.kl_div(log_p_s, p_t, reduction="batchmean")
```

The `T²` multiplier preserves gradient magnitude across temperature
choices (Hinton et al. 2015). At `T=1` (our current setting), teacher
probabilities are often near-one-hot — gradient signal collapses.
At `T=4`, low-probability tokens carry more weight, student learns
richer distributional structure.

**Expected gain**: Modest, 2-5% PPL. Small code change.

### 7. Per-layer activation-drift logging

Register forward hooks on every decoder layer of both teacher and
student, every N=200 training steps run a parallel teacher forward,
compute per-layer rel-err, log. Produces a table like:

```
step 1000 | L0: 0.08 | L6: 0.15 | L12: 0.22 | L18: 0.28 | L23: 0.31 | mean: 0.21
```

If any specific layer's rel-err explodes (say L12 → 0.8 while others
stay <0.3), that's a divergence signal we'd want to act on before
final PPL reveals a broken run.

Overhead: ~10% per-step when monitoring is active; log every 200
steps limits total overhead to ~0.5%. Basically free.

### 8. Periodic checkpointing

Already shipped: end-of-training `state_dict` save. Not yet shipped:
save every eval checkpoint, track best-PPL checkpoint separately.

**Proposed:**

```python
if ppl < best_ppl:
    torch.save(state, f"{ckpt_stem}_best.pt")
    best_ppl = ppl
torch.save(state, f"{ckpt_stem}_step{step}.pt")  # rolling
```

Keeps last 3 rolling checkpoints to cap disk (~6 GB at 0.5B). Useful
if training plateaus and we want to compare early-stop vs fully-
converged checkpoints.

### 9. Fix `reduction=batchmean` semantics

Current:

```python
kl = nn.KLDivLoss(reduction="batchmean", log_target=False)
loss = kl(
    log_softmax(s_logits).view(-1, vocab),
    softmax(t_logits).view(-1, vocab),
)
```

`batchmean` divides the sum by `input.size(0)`, which after `view(-1, vocab)`
is `batch * seq_len`, not `batch`. So it's actually per-token mean,
not per-sample mean. Works for LM training (per-token is what we want)
but the naming is misleading.

Fix to be explicit:

```python
loss = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1).mean()
```

Same math; clearer about what's being averaged over.

## Bundle recommendations

### Minimum viable upgrade for next iteration (≤30 lines, low risk)

1. `bnb.AdamW8bit`
2. `inter_mse_weight=10` default
3. `gradient_checkpointing_enable()`
4. SmoothSign Option B (save bf16 surrogate)

Unlocks: 7B training locally. Expected PPL improvement on 0.5B:
~5-15% (MSE term is the main contributor).

### Speed-focused upgrade (adds ~5 more lines)

Above + `torch.compile` smoke test. 30-50% faster per step if it
compiles.

### Accuracy-focused upgrade (adds ~20 more lines)

Above + tau warmup + KD temperature. Better-targeted gradient
signal through the first few thousand steps.

### Full stack (adds ~30-50 more lines)

All of 1-7 + periodic checkpointing (8) + cleaner reduction (9).
The "paper-faithful plus home-lab ergonomics" bundle.

## Validation protocol for any enhancement

Each proposed enhancement should be validated against the **baseline
PPL 54.81 on Qwen 0.5B r=512**. Specifically:

- Run with the enhancement, same hyperparameters otherwise.
- Compare final PPL: lower is better; flag regressions ≥2%.
- Compare wall time: lower is better for speed enhancements; some
  loss is acceptable for memory enhancements.
- Compare per-layer activation drift (measured via
  `littlebit_eval.py`): lower mean and worst-layer rel-err is
  better.

If an enhancement regresses PPL, roll it back or tune before
stacking.

## Non-enhancements (considered and rejected)

- **Custom binary GEMM kernel**: Inference-time concern, not training-
  time. Training is fp32-bound regardless. Paper's 11.6× claim is for
  their kernel at their config — not something we hit by writing a
  better trainer.
- **signSGD on U_fp / V_fp**: Potentially cleaner for sign parameters,
  but adds an optimizer-per-parameter-group complication. Only worth
  trying if plateau becomes the bottleneck. Plateau isn't the
  bottleneck yet — we're not at the paper's numbers because we ran a
  stripped-down recipe, not because the optimizer is wrong.
- **Move student to bf16**: `SmoothSign`'s `tanh(100x)` on bf16 has
  precision issues near zero (where the surrogate is biggest). Keep
  student params in fp32 until we've validated at larger scale.
- **Train only h/g/ell (freeze signs)**: Already tried in
  [`littlebit_math.md §13.1`](littlebit_math.md) — scales-only QAT
  recovered essentially zero Frobenius error (improvement +0.001).
  Dead end.
