# LittleBit QAT on 30B — cloud GPU plan

Scoping doc for training a 30B-class model (e.g. Qwen 2.5 32B) under
LittleBit QAT on rented cloud GPUs, since local training at that
scale is infeasible on a 16 GB laptop (as established in
[`littlebit_math.md §12`](littlebit_math.md) discussion of scaling).

All pricing figures gathered April 2026; check each provider's live
page before committing.

## Compute requirement

### Memory budget at 30B

A 30B model needs ~60 GB just for bf16 weights. Full QAT adds:

| Component | Estimate (30B, fp32-master/bf16-compute) |
|---|---:|
| Student bf16 params | ~60 GB |
| Student fp32 master weights (for stable updates) | ~120 GB |
| Student bf16 grads | ~60 GB |
| 8-bit AdamW state | ~30 GB |
| Teacher bf16 (frozen) | ~60 GB |
| Activations (seq=2048, batch=1, grad-checkpointed) | ~3-5 GB |
| **Total** | **~335 GB** |

Even aggressively optimized, this does not fit on a single 80 GB
GPU. Required: **multi-GPU** (minimum 4× A100 80GB with ZeRO-3 sharding)
or single GPU with CPU offload (much slower).

### Compute time

Paper training recipe (inferred from
[docs/research/quantization-papers-2025.md](../../docs/research/quantization-papers-2025.md)):
5 epochs over wikitext-2 + C4 partitions, seq=2048, Adam, cosine LR.

Per-step compute scales roughly linearly with parameter count. From our
local 0.5B baseline (0.3 s/step at seq=512):

| Model | Seq | Est. step time (single A100) | 8000-step run |
|---|---:|---:|---:|
| 0.5B (measured, laptop 5080) | 512 | 0.3 s | 40 min |
| 7B | 512 | 4.6 s (linear scaling) | ~10 hours |
| 7B | 2048 | 18 s (seq² on attention) | ~40 hours |
| 30B | 2048 | 80 s (scale + seq) | ~180 hours |
| 30B (4× A100 ZeRO-3) | 2048 | 25 s (parallelism overhead) | ~55 hours |
| 30B (4× H100 ZeRO-3) | 2048 | 13 s (H100 ~2× A100) | ~29 hours |

5 epochs of wikitext-2 + C4 subset is ~10× more tokens than wikitext-2
alone; paper-faithful reproduction is closer to **~80000-step-equivalent**
(6-8× longer than our 8000-step local recipe). At 4×A100 that's ~400
wall-clock hours for a truly paper-faithful run. Most people don't do
the full 5 epochs — 1-2 epochs is typical for reproduction attempts.

**Realistic reproduction target**: 20k-30k steps on 4× A100 80GB ≈
**60-80 hours wall clock**.

## Cloud pricing snapshot (April 2026)

### A100 80GB

| Provider | On-demand $/hr | Spot/interruptible $/hr | Notes |
|---|---:|---:|---|
| Vast.ai | $0.67 (SXM, marketplace) | $0.30-0.50 | Marketplace; provider quality varies |
| RunPod (PCIe) | $1.19 | not available | Consistent hardware |
| RunPod (SXM) | $1.39 | not available | Faster interconnect |
| Lambda Labs | $1.99-$2.49 | not listed | Enterprise-grade |
| Paperspace | $3.09+ | — | Higher price, dedicated |

### H100 80GB

| Provider | On-demand $/hr | Spot $/hr |
|---|---:|---:|
| Vast.ai | $1.55 (mkt) | $1.87 lowest listed |
| RunPod | $1.50-$2.69 | — |
| Spheron | $2.01 | — |
| Paperspace | $5.95 | — |

H100 is ~2× compute of A100 for LLM workloads; price ratio ~1.2×
suggests H100 is the better $/flop on most providers.

## Cost scenarios for a 30B reproduction

Assuming a 30000-step "reasonable but not paper-faithful" run:

### Scenario A — 4× A100 80GB on RunPod (stable, simple)
- $1.39/hr × 4 GPUs = $5.56/hr
- ~55 hours wall clock
- **~$310 total**
- Pros: consistent hardware, easy multi-GPU orchestration, no spot preemption
- Cons: most expensive local option

### Scenario B — 4× A100 80GB on Vast.ai marketplace
- ~$0.67/hr × 4 = $2.68/hr (subject to availability)
- ~55 hours wall clock
- **~$150 total**
- Pros: ~half the price of RunPod
- Cons: marketplace — finding 4 matching-spec GPUs from one provider takes shopping; quality varies

### Scenario C — 4× H100 80GB on Vast.ai
- ~$1.55/hr × 4 = $6.20/hr
- ~29 hours wall clock (H100 faster)
- **~$180 total**
- Pros: H100 perf + marketplace pricing; lowest total cost for H100-speed
- Cons: H100 availability less consistent

### Scenario D — 8× A100 80GB ("throw hardware at it")
- $1.39/hr × 8 = $11.12/hr
- ~25 hours wall clock with 8-GPU parallelism (diminishing returns)
- **~$280 total**
- Pros: fastest wall clock for A100-class
- Cons: near-equal to Scenario A total cost; no strong reason over 4x

### Scenario E — Spot on Vast.ai with checkpoint-resume
- 4× A100 at ~$0.35/hr spot × 4 = $1.40/hr
- Factor in retries from preemption: ~70 hours of "useful" compute over ~90 hours wall
- **~$100-120 total**
- Pros: cheapest viable path
- Cons: **requires robust checkpoint-resume in the training code** (not yet shipped in our script); preemption recovery overhead

### Recommendation

**Scenario C (4× H100 on Vast.ai, on-demand not spot)** — best
$/performance, simplest orchestration, ~$180 for one 30B training run.
Checkpoint-resume is optional for this scenario since it's
uninterruptible on-demand.

If cost is the primary constraint, **Scenario E (4× A100 spot)** at
~$100-120 but needs checkpoint-resume infrastructure first.

## Prerequisites before spending

### 1. Upgraded QAT stack

Before cloud-burning money, validate the enhancements from
[`littlebit_enhancements.md`](littlebit_enhancements.md) locally on
a tractable scale. Specifically the memory-critical ones must be in:

- `bnb.AdamW8bit` optimizer
- `model.gradient_checkpointing_enable()`
- SmoothSign memory diet (Option B — save bf16 surrogate)

Validate each on the 0.5B baseline (PPL 54.81 target), then upscale.

### 2. Checkpoint-resume (required for spot)

Our current script saves one end-of-training checkpoint. For spot
GPUs we need:

- Periodic checkpoint every N steps (every eval)
- Resume from checkpoint on relaunch: reload student state_dict,
  optimizer state, scheduler state, step counter, train-stream RNG
- Rolling retention (keep last 3 to cap disk)

Estimated work: ~50 lines of code on top of current script.

### 3. Multi-GPU via DeepSpeed / FSDP

Our current script is single-GPU. For 30B we need either:

- **DeepSpeed ZeRO-3** — Samsung's repo already ships `configs/zero3.json`.
  Can use their `main.py` directly instead of porting.
- **PyTorch FSDP** — native to torch 2.x; cleaner but more
  integration work.

Recommended path: **use Samsung's `main.py` with their ZeRO-3 config**.
Minimal integration risk; matches paper hyperparameters exactly;
our value-add becomes dataset/model selection rather than framework.

### 4. Data and pipeline

- Wikitext-2 is available via HF datasets; no prep
- C4 partitions: streaming works, but selected partitions need
  specifying (paper doesn't fully document which)
- Teacher model: download Qwen 2.5 32B (~60 GB) to provider-side disk
  before training. Add ~30 min for download and verification.

## Alternative: smaller scales first

Before paying for 30B, validate the scaling story locally and with
smaller cloud runs:

| Model | Local or cloud | Est cost | Purpose |
|---|---|---:|---|
| Qwen 2.5 1.5B | Local (post-enhancements) | $0 | Validate optimized stack |
| Qwen 2.5 3B | Local (post-enhancements) | $0 | Confirm trend holds |
| Qwen 2.5 7B | Local (post-enhancements, aggressive) | $0 | Or $50-100 cloud fallback |
| Llama-2-7B | Cloud (match paper exactly) | $50-75 | Direct paper reproduction |
| Qwen 2.5 14B or 32B | Cloud (4× A100 or H100) | $150-300 | Scale-frontier reproduction |

The case for 30B spending **only becomes compelling after** 1.5B/3B/7B
results consistently beat their FP-SVD floors by ≥2×, matching the
paper's trend from Llama-2-7B (PPL gap 1.85× at 0.55 BPW) → 13B
(1.80× gap). Our 0.5B already hits 1.6× (PPL 54.8 / FP-SVD floor 86).
If that trend continues or tightens as we go up, 30B is high-EV to
reproduce.

## Checkpoint-resume code sketch

Minimum viable resume logic for spot-GPU-friendly training:

```python
# At init (load if present)
if resume_path and Path(resume_path).exists():
    ckpt = torch.load(resume_path, map_location="cpu")
    student.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    start_step = ckpt["step"]
    rng_state = ckpt["rng"]
    torch.set_rng_state(rng_state["torch"])
else:
    start_step = 0

# Every eval
if step % args.eval_every == 0:
    torch.save({
        "model": student.state_dict(),
        "opt":   opt.state_dict(),
        "step":  step,
        "rng":   {"torch": torch.get_rng_state()},
        "config": vars(args),
    }, rolling_ckpt_path)

# SIGTERM handler (for spot preemption)
import signal
def save_and_exit(*a):
    torch.save(..., emergency_ckpt_path)
    sys.exit(0)
signal.signal(signal.SIGTERM, save_and_exit)
```

Vast.ai and RunPod both send SIGTERM before preemption on spot
instances with a few-minute grace window.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Teacher download fails or is slow | Start delay | Pre-download to provider disk via their CLI; verify hash |
| GPUs get preempted mid-run (spot only) | Rework from last checkpoint | Periodic checkpointing (above) |
| DeepSpeed ZeRO-3 OOMs at 30B unexpectedly | Wasted $ + time | Test at 7B on same config first |
| Paper's hyperparameters don't transfer to our Qwen target | Run produces bad PPL | Use Samsung's exact LR / schedule / loss config; don't diverge |
| Slow I/O on spot marketplace node | 2× training time | Sanity-check with 100-step smoke before committing |

## Decision framework

Before spending cloud $:

1. Does our optimized stack match run-2's PPL 54.8 on 0.5B? → if no,
   fix the stack first.
2. Does 1.5B / 3B / 7B locally beat their FP-SVD floors with the
   optimized stack? → if no, the method isn't scaling well and 30B
   cloud spend is high-risk.
3. If both (1) and (2), pick a scenario and run it once.

Total commitment for one successful 30B reproduction: **~$200 cloud +
one weekend of active monitoring** if everything goes right on the
first try. Budget 1.5-2× that for uncertainty: **~$300-400 and ~one
week calendar time**.

## Sources

Pricing data as of April 2026:
- [RunPod A100 PCIe pricing](https://www.runpod.io/gpu-models/a100-pcie)
- [Vast.ai vs RunPod 2026 comparison](https://medium.com/@velinxs/vast-ai-vs-runpod-pricing-in-2026-which-gpu-cloud-is-cheaper-bd4104aa591b)
- [H100 rental price comparison 2026](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [Lambda Labs pricing page](https://lambda.ai/pricing)
- [GPU rental market trends](https://www.thundercompute.com/blog/ai-gpu-rental-market-trends)
