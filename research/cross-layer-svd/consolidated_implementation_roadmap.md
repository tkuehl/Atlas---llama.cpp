# Consolidated implementation roadmap — overlap analysis

Cross-reference of techniques across our four plans to find
overlapping wins and build a single execution sequence.

Plans reviewed:
- [savings_exploration_plan.md](savings_exploration_plan.md) —
  memory
- [unexplored_efficiency_gains.md](unexplored_efficiency_gains.md)
  — quality
- [scale_to_30b_architecture.md](scale_to_30b_architecture.md) —
  scale
- [wall_time_reduction_plan.md](wall_time_reduction_plan.md) —
  wall time

**Question**: which techniques hit multiple axes? Those are the
highest leverage and should be implemented first.

## 1. Cross-plan overlap table

Techniques appearing in multiple plans, ranked by axis count.

### Triple-axis (memory + wall time + scale)

| Technique | Memory | Wall time | Scale | Quality |
|---|---|---|---|---|
| **Teacher cache (offline extraction)** | ✓ eliminates teacher VRAM entirely | ✓ removes 17% of per-step time | ✓ architectural enabler for 30B+ | neutral |

This is **the single highest-leverage technique in the entire plan set**. Appears in scale (§2), wall-time (§2.1), savings (Run I), unexplored (§1.6). If only one non-trivial thing is built, this should be it.

### Double-axis

| Technique | Memory | Wall time | Scale | Quality |
|---|---|---|---|---|
| **Liger Kernel (norm/RoPE/SwiGLU + FusedLinearCE)** | ✓ ~2-20 GB | ✓ 10-15% kernels, ~30% CE | – | neutral |
| **torch.compile(student)** | ✓ slight | ✓ 30-50% | – | neutral |
| **Forward hooks for MSE hidden states** | ✓ 2 GB | ✓ 5-10% | – | neutral |
| **Per-layer non-uniform rank** | ✓ proportional | ✓ smaller matmuls | – | potentially ✓ |
| **Staged KL-then-MSE training** | ✓ MSE deferred | ✓ faster stage 1 | – | potentially ✓ |
| **Incremental rank (r=128→256→512)** | ✓ smaller early | ✓ faster early | – | potentially ✓ |
| **CALDERA init + LittleBit** | – | ✓ faster convergence | – | ✓ via better init |
| **nf4 teacher** | ✓ 10 GB at 7B | ✓ slight | ✓ scale enabler | low risk |
| **Chunked KL (DIY or Liger)** | ✓ 3 GB | ✓ part of loss path | ✓ enables larger vocab at scale | neutral |

### Single-axis (still useful, just narrower)

| Technique | Axis | Gain |
|---|---|---|
| TF32 matmul precision | wall time | 13% |
| cuDNN benchmark mode | wall time | ~5% |
| Plateau early-stop | wall time | 20-30% |
| CPU data prefetch | wall time | 5-7% |
| CUDA streams for teacher/student | wall time | 10-15% |
| CUDA graphs | wall time | 5-10% |
| Relation-based MSE | quality | rotation tolerance |
| Lion optimizer for signs | quality | potentially ✓ |
| MSE λ decay schedule | quality | late-stage polish |
| KD temperature T=4 | quality | softer targets |
| tau warmup 10→100 | quality | gradient coverage |
| Post-training scale LSQ | quality | free 1-3% |
| NVMe offload | scale | 70B local |
| Fused Triton LittleBit kernel | wall time | 20-30% |

### Already shipped (Phase A)

| Technique | Axis |
|---|---|
| 8-bit AdamW (bitsandbytes) | memory |
| Gradient checkpointing | memory |
| SmoothSign memory diet | memory |
| GPU memory cap | safety |
| Grad accumulation (effective batch=4) | memory + quality |
| C4 data streaming | quality |
| Weight decay default 0.01 | quality |
| Checkpoint-resume infrastructure | engineering |

## 2. Synergies between techniques

Beyond what each does alone, several combinations compound.

### Liger FusedLinearCrossEntropy + Chunked KL
These overlap heavily — both solve the same problem (vocab-sized
logits tensor). Only one is needed. Decide:
- **Liger** if it ships a KL-against-soft-targets kernel (check first)
- **DIY chunked KL** if Liger's fused CE only supports hard targets

Pick one, skip the other.

### torch.compile + teacher cache
Compilation works better when there's a single Python loop running
student without teacher forward breaking the graph. Teacher cache
removes teacher from the training path entirely, **improving
compile effectiveness**.

### Per-layer rank + incremental rank
Both adjust rank but in orthogonal ways:
- Per-layer: different r for attention vs MLP
- Incremental: same r for all, grows over training
Can be combined — e.g. attention stays at r=64 throughout,
MLP grows r=128 → 256 → 512. Not tested.

### Forward hooks + gradient checkpointing
Currently our `output_hidden_states=True` partially defeats gradient
checkpointing. Forward hooks restore full checkpointing
effectiveness, giving both memory savings and wall-time gains that
don't add independently — they multiply.

### Memory savings → larger physical batch → wall-time savings
This is the **key memory-wall-time interconnect**. Memory techniques
(Liger, chunked KL, nf4 teacher, forward hooks) free up GPU budget.
That budget enables `batch_size=2` instead of `grad_accum=2`. Per
opt-step, `batch=2` is ~40% faster than `grad_accum=2` at same
effective batch. Memory savings → wall time reduction.

### CALDERA init + staged training + plateau early-stop
Three independent ways to shorten training:
- Better init (CALDERA): fewer steps needed from start
- Staged (KL first, MSE later): cleaner objective convergence
- Plateau stop: cuts the tail-of-diminishing-returns

Combined, training that takes 20k steps under paper recipe might
converge in ~8-10k. 2× wall-time reduction on top of per-step gains.

## 3. Priority matrix — multi-axis impact × effort

Sorting all overlapping techniques by combined impact and effort:

| # | Technique | Axes | Effort | Total impact |
|---:|---|---|---|---|
| **1** | **TF32 + cuDNN benchmark** | wall | 2 lines | **Free 15-18% wall time** |
| **2** | **Plateau early-stop** | wall | ~20 lines | **Free 20-30% wall time for runs that converge** |
| **3** | **torch.compile(student)** | memory + wall | ~10 lines + fallback | **30-50% wall time, likely works** |
| **4** | **Liger partial (non-CE kernels)** | memory + wall | ~10 lines + pip | **10-15% wall time, 1-2 GB memory** |
| **5** | **Forward hooks for MSE** | memory + wall | ~30 lines | **2 GB memory + 5-10% wall time** |
| **6** | **Teacher cache** | memory + wall + scale | ~300 lines + 1 day | **17% wall + teacher VRAM elimination + scale unlock** |
| **7** | **nf4 teacher (if no cache)** | memory + scale | ~5 lines | **10 GB at 7B, scale enabler** |
| **8** | **Chunked KL or Liger fused CE** | memory + wall | ~50 lines or Liger | **3 GB + logits ops speed** |
| **9** | **Staged KL-then-MSE** | quality + wall | ~20 lines | **Possibly faster convergence** |
| **10** | **CALDERA init** | quality + wall | ~100 lines | **Better init, fewer steps** |
| **11** | **Relation-based MSE** | quality | ~20 lines | **Addresses hidden rotation** |
| **12** | **Incremental rank** | quality + wall | ~200 lines | **Progressive training** |
| **13** | **Per-layer non-uniform rank** | memory + wall | ~50 lines | **Budget efficiency** |
| **14** | **CUDA streams teacher/student** | wall | ~50 lines | **Only if teacher not cached** |
| **15** | **NVMe offload** | scale | ZeRO-Inf | **70B local** |

## 4. Recommended execution sequence

### Sprint 1 — "Free wins morning" (2-3 hours)

Pure wall-time gains, zero correctness risk, 1-2 lines each:

- [ ] `torch.set_float32_matmul_precision("high")` + `cuDNN.benchmark`
- [ ] `torch.backends.cuda.matmul.allow_tf32 = True`
- [ ] Change default `--eval-every` from 500 to 2000
- [ ] Add `--early-stop-min-delta 2 --early-stop-window 5 --early-stop-min-steps 4000`
- [ ] Smoke test: reproduce Run 3's step-500 PPL within ±1%

**Validation**: runs are same quality, ~35% faster.

### Sprint 2 — "Compile + forward hooks" (half day)

Higher per-change effort but high leverage:

- [ ] `torch.compile(student, mode="reduce-overhead")` with try/except fallback
- [ ] Liger Kernel partial application (RMSNorm + RoPE + SwiGLU; skip FusedLinearCE for now)
- [ ] Forward hooks replace `output_hidden_states=True` for MSE capture
- [ ] Smoke test: reproduce Run 3's PPL trajectory (should be within 3%)

**Validation**: ~50-60% wall-time reduction vs Phase B. Memory down ~2-3 GB.

### Sprint 3 — "Teacher cache and soft-targets KL" (2-3 days)

The single biggest leverage piece:

- [ ] Binary cache format + mmap reader (see
  [scale_to_30b_architecture.md §2.5](scale_to_30b_architecture.md))
- [ ] Teacher extraction script (top-256 logits + 3 key hidden-state
  layers)
- [ ] Training loop change: read from cache instead of teacher
  forward when cache available
- [ ] Chunked KL against cached top-k logits
- [ ] Validation: reproduce Phase B's PPL within 5%, no teacher on GPU

**Validation**: Phase B 10h → 6h, teacher eliminated from training
VRAM. Unlocks 7B and 30B architecturally.

### Sprint 4 — "Quality combinations" (1-2 weeks, in parallel with user)

Now that ablations are cheap (~2h each instead of 8-10h), run the
combined-quality experiments from `unexplored_efficiency_gains.md`:

- [ ] Staged KL-then-MSE ablation
- [ ] CALDERA init ablation
- [ ] Relation-based MSE ablation
- [ ] Per-layer non-uniform rank ablation
- [ ] tau warmup ablation
- [ ] nf4 teacher ablation
- [ ] Decide: does any combination materially improve over Phase B?

**Validation**: each ablation against the Sprint 3 fast baseline.
Keep winners, drop losers.

### Sprint 5 — "Scale ladder" (1-2 weeks)

Apply the consolidated stack to progressively larger models:

- [ ] Qwen 2.5 1.5B with full stack → ~1.5h per run
- [ ] Qwen 2.5 3B with full stack → ~3-4h per run
- [ ] Qwen 2.5 7B with full stack → ~4-6h per run (fits with
  Sprint 3's teacher cache + memory savings from Sprint 2)
- [ ] Decide: is 7B result good enough to justify 30B?

### Sprint 6 — "30B (conditional)" (2+ weeks)

Only if Sprint 5 produces a paper-grade 7B result:

- [ ] NVMe layer storage (Sprint 6 or Sprint 3 depending on urgency)
- [ ] 30B teacher extraction (cloud, ~$8)
- [ ] 30B student training with NVMe + cached teacher (~40-60h)

## 5. Redundancies to drop

Some techniques in our plans are redundant with higher-priority items:

| Dropped | Why | Supersedes |
|---|---|---|
| **Fused Triton LittleBit kernel** (wall_time §2.11) | torch.compile achieves ~40% of its gain with 1% of the effort | Sprint 2 item |
| **CUDA graphs** (wall_time §2.6) | torch.compile captures similar benefit | Sprint 2 |
| **CUDA streams teacher/student** (wall_time §2.7) | Teacher cache removes teacher entirely | Sprint 3 |
| **KD temperature T=4** (savings Run F) | Marginal quality gain, tune after other stuff lands | Defer |
| **Unsloth** | Liger covers same ground | Sprint 2 uses Liger only |
| **Full Liger FusedLinearCrossEntropy** | KL against soft targets isn't standard Liger | Defer unless adapted |
| **Lion optimizer on signs** | Research-grade, too risky before establishing baseline | Defer to Sprint 4 if bored |

## 6. What becomes possible at each sprint boundary

After each sprint, check what's now within reach:

| After sprint | New capability |
|---|---|
| 1 | Faster training loop. Every future run benefits immediately. |
| 2 | Faster + slightly less memory. Ablations go 2h each. |
| 3 | **No teacher in training memory.** 7B local feasible. 30B architecturally possible. Teacher cache reusable across all future runs. |
| 4 | Quality-tuned recipe. Know which combinations help. |
| 5 | Scale-validated method. Clear signal on cloud vs local for 30B. |
| 6 | 30B locally produced. Publishable novelty. |

## 7. Gates and off-ramps

Not every sprint needs to succeed. Exit criteria:

**After Sprint 1**: If TF32 + early-stop breaks reproducibility →
revert, accept slower baseline.

**After Sprint 2**: If torch.compile silently degrades quality (PPL
worse at same step) → drop compile, keep Liger.

**After Sprint 3**: If teacher cache's top-k truncation materially
hurts PPL (more than 5%) → fall back to online teacher but keep the
infrastructure for future corpus expansion.

**After Sprint 4**: If no quality combinations meaningfully beat
paper-recipe baseline → we've shown paper's recipe is
locally-optimal, write it up as a reproduction study.

**After Sprint 5**: If 3B shows method hits quality ceiling — stop
scaling, report findings. Don't waste 30B compute on a method that
already plateaued.

**After Sprint 6**: If 30B local works → write up the architecture
as a novel contribution.

## 8. Effort / time summary

| Sprint | Engineering time | Wall-time gain | Memory gain | Scale unlock |
|---:|---|---|---|---|
| 1 | 2-3 hours | ~35% | - | - |
| 2 | half day | +25% (cumul 60%) | 2-3 GB | - |
| 3 | 2-3 days | +15% (cumul 75%) | 10 GB at 7B | **30B feasible** |
| 4 | 1-2 weeks | per-ablation faster | - | - |
| 5 | 1-2 weeks | - | - | 7B locally trained |
| 6 | 2+ weeks | - | - | 30B locally trained |

**Sprints 1-3 are the critical path.** They together:
- Cut wall time by ~75%
- Free ~10 GB of memory at 7B scale
- Enable 7B local training
- Build reusable teacher cache infrastructure

Everything after Sprint 3 uses this platform for experiments.

## 9. Risk budget

Things that could derail this plan:

| Risk | Sprint | Impact | Mitigation |
|---|---|---|---|
| `torch.compile` doesn't compose with our SmoothSign autograd | 2 | Lose ~30% wall-time gain | Fallback path, Liger only |
| Teacher cache top-k truncation hurts quality more than expected | 3 | Fall back to online teacher | Keep infrastructure, use online for small scales |
| NVMe wear rate higher than calculated | 6 | Drive fails | Enterprise SSD ($500-1000) |
| 1.5B or 3B ablations show method plateaus | 5 | No 30B justification | Write up as reproduction study |
| Local system RAM insufficient for 30B | 6 | 30B infeasible locally | Cloud $180 fallback |

## 10. When to revisit

Review this plan after:
- Phase B finishes (hours from now)
- Sprint 1 ships (same day)
- Each ablation's result
- Any sprint's exit criterion is triggered

The sprint structure is a proposal, not a contract. Techniques
within a sprint can be reordered; sprints themselves can be
reordered based on what we learn.

## 11. Bottom line

Four plans, one recommended path:

1. **Sprint 1 + 2 this afternoon** (~4 hours of coding) → 60%
   wall-time reduction, every future run 2.5× faster
2. **Sprint 3 next week** (~2-3 days) → teacher cache, 7B local
   feasible, 30B architecturally possible
3. **Sprint 4-5** (~2-4 weeks) → validated quality recipe +
   scaled reproductions
4. **Sprint 6** (conditional, 2+ weeks) → 30B locally produced,
   novel architecture contribution

Total: **~1 day of low-risk coding for most wins**, **~1 week for
scale unlocks**, **~4-6 weeks for end-to-end 30B local reproduction**.
