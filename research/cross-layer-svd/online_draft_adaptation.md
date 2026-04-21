# Online draft adaptation — LittleBit student learns from speculation rejections

Proposal for continuously updating the LittleBit-compressed draft
model during deployment, using teacher-vs-student rejections as
free training signal. Preserves speculation's correctness
guarantee; pure performance optimization.

Related threads:
- [consolidated_implementation_roadmap.md §Sprint 3.5](consolidated_implementation_roadmap.md)
  — measures baseline acceptance rate
- [unexplored_efficiency_gains.md §1.2](unexplored_efficiency_gains.md)
  — LittleBit-as-draft combination
- Atlas speculative decoding research track
  ([speculative_decoding.md](speculative_decoding.md))

## 1. The core insight

Speculation's protocol:
1. Draft proposes N tokens in one forward
2. Teacher verifies all N in one forward
3. Accept tokens up to first disagreement; replace rest with
   teacher's argmax

**Teacher's full forward already produced logits at every position.**
At positions where teacher disagrees, teacher's logit distribution
*is* the correct training target. We already paid the compute cost
to produce it — using it for a student gradient step is **free
training data**.

Update cost = student forward + backward for the rejected
positions. Teacher cost = **zero additional** beyond verification
we already ran.

## 2. Correctness preservation

Speculation's correctness guarantee is independent of student
quality:
- Accepted tokens = positions where `student.argmax == teacher.argmax`
- Rejected positions → replaced with teacher's argmax
- Final output matches teacher greedy decode regardless of student

Student drifting → fewer acceptances → slower, not wrong.
Student improving → more acceptances → faster, not different
output.

**Online adaptation cannot reduce output quality.** Worst case:
student gets worse at acceptance, end-to-end speed drops back
toward no-speculation baseline.

## 3. Adaptation levels

### Level 1: Deferred batch updates

```python
class DeferredAdapter:
    def __init__(self, student, opt, batch_size=100):
        self.student = student
        self.opt = opt
        self.batch_size = batch_size
        self.buffer = []  # (input_ids, position, teacher_logits)

    def record_rejection(self, input_ids, pos, t_logits):
        self.buffer.append((input_ids, pos, t_logits))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        # Batch update
        losses = []
        for inp, pos, tl in self.buffer:
            s_logits = self.student(inp).logits[0, pos]
            losses.append(F.kl_div(
                F.log_softmax(s_logits, -1),
                F.softmax(tl, -1),
                reduction="sum",
            ))
        loss = torch.stack(losses).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.buffer.clear()
```

**Pros**: Amortizes optimizer overhead, stable updates.
**Cons**: Slow adaptation — user might see 100 rejections before
benefit.

### Level 2: Online per-rejection update

```python
def on_rejection(input_ids, pos, teacher_logits):
    student.train()
    s_logits = student(input_ids).logits[0, pos]
    loss = F.kl_div(
        F.log_softmax(s_logits, -1),
        F.softmax(teacher_logits, -1),
        reduction="sum",
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    student.eval()
```

**Pros**: Fastest adaptation, per-token.
**Cons**: Higher per-rejection latency (~20-50 ms at 0.5B,
~200-500 ms at 7B without kernel).

### Level 3: Continuous fine-tuning

Every teacher verification is both verification AND training.
All positions, every step, contribute to student gradient.

**Pros**: Maximum adaptation rate.
**Cons**: Dominates inference latency if not carefully pipelined;
risk of instability.

## 4. Latency / break-even analysis

Rough numbers for 7B teacher + 7B LittleBit draft:

| Operation | Cost (pre-kernel, PyTorch eager) |
|---|---:|
| Teacher forward (bf16) | ~200 ms |
| Student forward (fp32 matmul on binarized sign) | ~200 ms |
| Student backward | ~400 ms |
| Optimizer step | ~50 ms |
| **Per-rejection update (Level 2)** | **~650 ms** |

Break-even acceptance improvement per update:
- If update raises acceptance from 60% → 65% over next 100
  positions, save 5 teacher forwards × 200 ms = **1 second**
- Net win per 100-position interval: ~350 ms

**Regime where it pays off**:
- Sustained-use workloads (chat with >100 tokens per session)
- Narrow-domain use (student specializes quickly)

**Regime where it loses**:
- One-shot completions (not enough amortization)
- Diverse workloads (student drifts faster than it adapts)

With the fused binary kernel (Sprint 7+), student forward/backward
drops ~10×, making updates cost ~65 ms. Break-even becomes
trivial — update pays off after first few more accepts.

## 5. LoRA-adapter variant (recommended)

Instead of modifying student's `U_fp`, `V_fp`, `h`, `g`, `ell`
directly, add a small LoRA adapter on top:

```
deployment_student = frozen_littlebit + LoRA(rank=8 or 16)
online updates → adapter parameters only
per-session / per-user: fresh adapter
persistent: save adapter to disk per user
```

### Advantages

1. **Tiny footprint**: rank-8 adapter on 0.5B is ~4 MB; 7B is ~50 MB
2. **Multi-user isolation**: each user gets their own adapter, no
   cross-contamination
3. **Easy rollback**: drop adapter, back to base student
4. **Safe default**: if adapter destabilizes, disable it and lose
   no ground
5. **Persistence**: save user-specific adapter to session storage;
   load on reconnect

### Adapter math

Output of each Linear with adapter:

```
y_adapted = LittleBit(x) + x @ A @ B * scale
  where A ∈ R^{d_in × r}, B ∈ R^{r × d_out}, r ≪ d_in, d_out
```

Only A and B are trainable during deployment. LittleBit's
internals (signs, scale vectors) stay frozen.

### Storage math for Atlas

Per-user adapter sizes:
- 0.5B student: ~4 MB
- 7B student: ~50 MB
- 30B student: ~200 MB

Atlas with 100 active users: ~50 MB - 20 GB depending on scale.
Store on NVMe, load on session start.

## 6. Where this helps vs hurts

### Strong fit (high expected benefit)

- **Atlas agent loops**: tool-calling templates repeat,
  student memorizes tool structures fast
- **Personalized chat**: student learns user's phrasing, topics
- **Narrow-domain deployments**: Atlas on specific task → rapid
  specialization
- **Long-form interactions**: 1000+ token conversations
- **Code completion**: highly repetitive patterns

### Weak fit (likely break-even or negative)

- **Diverse multi-user without isolation**: drift cancels gains
- **One-shot prompts**: no time to amortize update cost
- **Adversarial users**: could deliberately poison adapter
- **Very short session lifetimes**: adapter doesn't accumulate

## 7. Connection to Atlas specifically

Atlas has exactly the characteristics this technique suits:
- **Agent workflows** with repeated patterns (`DecideNode`,
  `ReasonNode`, `ExecuteNode`)
- **Per-user state** (profile selector, multi-user scoping
  already implemented)
- **Narrow domains** (HA control, memory queries, routine
  execution)

Integration point: extend `LlamaCppClient` to:
1. Load per-user adapter on connect
2. Stream rejections back to an adapter-update service
3. Persist adapter on disconnect

The Atlas project's per-user profile infrastructure is already in
place (see recent 66cab15 commit). Adapters slot cleanly into it.

## 8. Open research questions

- **Adapter warm-up**: at session start, should we warm up on
  Atlas's routine prompt templates to seed acceptance?
- **Global vs personal**: would a globally-adapted base + personal
  micro-adapter outperform personal-only?
- **Forgetting rate**: how fast does a personal adapter decay if
  not used? Do we need periodic re-training against base?
- **Exploration / exploitation**: do we ever want to reject
  teacher's answer to force student to try something?
- **Multi-token updates**: batch updates over the whole rejected
  suffix instead of single rejected position
- **Kernel support**: if we ship a binary-GEMM kernel, does it
  compose with LoRA delta? (Matmul with binary main + fp32 delta
  might be efficient.)

## 9. Prior art

- **Knowledge distillation**: extensively studied offline
- **Online continuous learning**: well-studied for generic
  neural nets
- **Test-time adaptation**: growing body of work for vision
  (TENT et al.)
- **EAGLE** / **EAGLE-2**: separately trained speculation heads
  — related but not online
- **Self-speculation / LayerSkip**: uses the same model as draft
  and verify via internal layers — orthogonal approach

No paper I'm aware of covers **online adaptation of a sub-1-bit
compressed draft during speculative deployment**. This would be a
publishable combination of:
1. Sub-1-bit compression (LittleBit)
2. Speculative decoding
3. Online distillation
4. Per-user LoRA adapters

## 10. Proposed experiment (Sprint 8+)

**Precondition**: Sprint 3.5 establishes baseline acceptance rate.

### Experiment A: offline simulation of online updates

Cheap preliminary (~2 hours):

1. Run speculation benchmark on Phase B checkpoint (Sprint 3.5)
2. Log all rejected positions + teacher logits
3. Offline: apply 100 gradient updates to student using logged data
4. Re-run speculation → measure acceptance rate delta

If acceptance improves >5 percentage points, Level 2/3 online
adaptation is worth implementing live.

### Experiment B: LoRA adapter variant

Same as A but updates go into a rank-8 LoRA on top of frozen
student. Tests that LoRA capacity is sufficient for meaningful
adaptation.

### Experiment C: live adaptation against Atlas workload

Only after A and B validate. Deploy the adapted draft + teacher
pair against a realistic Atlas prompt mix. Measure:
- Acceptance rate over session length
- Wall-clock speedup
- Adapter stability across sessions

## 11. Scope in the master plan

Adding to the sprint sequence:

| Sprint | Content | Gated on |
|---:|---|---|
| 3.5 | Baseline acceptance measurement | Phase B done |
| 8 | **Online adaptation research** | **Sprint 3.5 shows ≥50% acceptance** |
| 8a | Experiment A (offline simulation) | – |
| 8b | Experiment B (LoRA variant) | 8a positive |
| 8c | Experiment C (live Atlas) | 8b positive |

**Effort estimate**: Sprint 8a ~4 hours, 8b ~2 days, 8c ~1 week.

## 12. Summary

Using the LittleBit draft's rejections as free training signal
during deployment is:
- **Correctness-safe**: speculation guarantees identical output
- **Compute-efficient**: teacher's existing verification forward
  produces the training target
- **Well-suited to Atlas**: narrow domains, per-user isolation,
  agent loop repetition
- **Novel combination** of established techniques:
  sub-1-bit compression + speculation + online KD + LoRA

First experiment is an offline simulation that takes ~4 hours and
tells us whether online adaptation would meaningfully improve
acceptance rate. That's a very cheap Sprint 8a that gates the
more ambitious live-adaptation work.
