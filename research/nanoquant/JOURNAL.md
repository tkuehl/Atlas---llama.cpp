# NanoQuant Replication — Journal

Running log of decisions, findings, and course corrections. Append-only; don't rewrite past entries — supersede with new entries instead.

---

## 2026-04-23 — Vein opened

Started from the arxiv 2602.06694v1 HTML. Paper is v1, Feb 2026, no code release. Authors: Chong, Kim, Kim, Choi.

### Scope decisions

- Faithful reimplementation, not improvement.
- Quality claims only (Tables 2, 3, 5, 6). Inference speedup (Table 4) deferred — it requires custom binary GEMV/GEMM kernels the paper doesn't release, and the quality claims are independently reproducible without them (STE training uses `torch.sign(U) @ torch.sign(V).T` in FP, numerics identical to a packed XNOR-popcount path).
- 70B off the table: 64 GB system RAM cannot hold Llama2/3-70B FP16 (~140 GB) even with block-wise CPU offload.
- Plain PyTorch + HuggingFace Transformers. No C++ / llama.cpp integration until replication succeeds.
- Ambiguities resolved case-by-case in this journal; related-work precedent (DBF, LittleBit, HBLLM) as first reference. Author contact deferred.

### Primary targets

- Qwen3-8B — Atlas daily driver, in paper's eval set.
- Llama2-7B — paper's strongest advertised gap: 10.34 PPL vs BiLLM 19.87.

### Stretch targets

- Gemma3-12B, Llama2-13B.

### Red flags carried forward from the reading

- **Paper loses at scale.** Llama3-70B @ 1-bit: NanoQuant 11.32 PPL vs HBLLM 8.88 PPL. The headline hides this. We can't test at 70B; will note it but not spend on it.
- **No code release; many "configurable" knobs.** K-FAC preconditioning and ADMM are both tricky numerics. High chance of subtle implementation divergence even when we match the headline PPL. Keep the worst-case expectation calibrated.
- **FLOPs-bound consumer decode** (prior finding, 2026-04-20): NanoQuant's compression is a bytes story; their speedup claim requires reduced effective FLOPs via XNOR/popcount. Phase 4 is where that tension gets tested — and it's deferred until quality replicates.

### Hardware

- RTX 5080 Laptop, 16 GB VRAM, Blackwell cc 12.0 — stronger than the paper's RTX 3050 8GB baseline, fine for 7–13B block-wise.
- 64 GB system RAM.

### Next decision point

Phase 0. Install pinned stack, fetch Qwen3-8B and Llama2-7B FP16, run WikiText-2 PPL baselines, reproduce one public BiLLM number to calibrate the eval harness.

---

## 2026-04-23 — Phase 0, step 1: environment bring-up

Clean venv at `research/nanoquant/.venv` on system Python 3.14.0. Pinned stack:

- `torch==2.11.0+cu130` (from the CUDA 13.0 index)
- `transformers==5.6.1`, `accelerate==1.13.0`, `datasets==4.8.4`
- `lm-eval==0.4.11` (transitively brings scipy 1.17.1, scikit-learn 1.8.0, nltk, rouge-score, sacrebleu — useful for later phases)
- `numpy==2.4.4`, `safetensors==0.7.0`, `sentencepiece==0.2.1`

### Blackwell gotcha — don't use cu126 wheels

First attempt installed torch from the `cu126` index. Import worked, `torch.cuda.is_available()` returned True, `get_device_properties` reported `sm_120` correctly — but any actual kernel launch failed with `CUDA error: no kernel image is available for execution on the device`. The warning spelled it out:

> The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.

cu126 wheels are compiled for Hopper and older — no Blackwell. Fix was swapping to `cu130` (CUDA 13.0 index), which does include `sm_120`. cu128 also works per PyTorch docs; cu130 was chosen for future-proofing.

**Rule of thumb for this machine:** when installing torch, use the cu128 or cu130 index. A successful import is not enough — verify with an actual `a @ b` on device.

### Verification

FP16 2048² matmul works, ~3.6 TFLOPS in a cold Python loop (not a real benchmark — no tensor-core warmup, no CUDA graphs, Python-per-call overhead). bf16 works. Sign-tensor matmul (the Phase 1 core op `sign(U) @ sign(V).T` at r=4) produces the expected `{-4, -2, 0, 2, 4}` unique values. Environment is fit for purpose.

### Next

Build the deterministic WikiText-2 calibration loader (128 samples × 2048 tokens) and run FP16 PPL baselines on Qwen3-8B and Llama2-7B.

---

## 2026-04-23 — Phase 0, step 2: eval harness + results log

### Switch core target: Qwen3-8B → Qwen3-4B

Swapped the primary proving-ground model to Qwen3-4B. Same family, smaller
and faster — a full WikiText-2 eval sweep is a few minutes instead of 20+,
which matters because we'll run hundreds of these (every method × ablation ×
hyperparameter point over the replication). Qwen3-8B is still a valid final
target but the inner loop wants the cheaper model. Llama2-7B remains the
second core target (it's the paper's strongest advertised gap).

The paper does not report a Qwen3-4B number, so this run establishes our own
FP16 reference — NanoQuant quality at 1 bit on Qwen3-4B will be judged
against this reference, not against a paper number.

### Layout

Four flat files under `research/nanoquant/`:

- `data.py` — WikiText-2 calibration (128 × 2048 random windows, seeded) +
  eval stream (full test split, concatenated with `\n\n`, tokenized once).
  Uses `wikitext-2-raw-v1`.
- `ppl.py` — non-overlapping sliding-window PPL matching the GPTQ/BiLLM
  convention (stride = seq_len, trailing partial dropped, PPL = exp(mean NLL
  over all windows)). Returns window count, token count, and mean NLL so
  every entry is fully characterized.
- `results.py` — append-only log writer for `results.json`. Each entry
  records git commit + dirty flag + branch, model HF id + revision + dtype,
  method name + params dict, eval metadata, and hardware (GPU name + torch
  version). Atomic temp-file-then-replace write so a crashed run can't
  corrupt the log.
- `run_baseline.py` — CLI entrypoint. `--model Qwen/Qwen3-4B --dtype float16`
  etc. Will be the pattern for Phase 1+ runners too (same results.json).

### Global results.json schema (v1)

One entry per run. Fields are set up so a Phase 1 run writing `method.name =
"phase1-ste-r2"` with `method.params = {"r": 2, "K_ste": 500, ...}` slots
into the same log as the baselines. `git.commit` + `git.dirty` means we can
always reproduce — and know when we can't, because the commit was dirty.

### Smoke tests

Tokenizer + data loader verified: Qwen3-4B tokenizer loads, WikiText-2 test
split tokenizes to 298,938 tokens → 145 non-overlapping windows at
seq_len=2048. Calibration sampling deterministic under `seed=0`. The fast
tokenizer emits the usual "sequence longer than max" warning on the
full-stream tokenize — harmless, because we window before feeding the model.

### Qwen3-4B FP16 reference — 13.6436 PPL

First entry in `results.json`:

- Model: `Qwen/Qwen3-4B`, rev `1cfa9a72…`, float16
- Eval: WikiText-2-raw-v1 test, seq_len=2048, stride=2048
- 145 windows × 2048 tokens = 296,960 scored tokens (trailing 1,978 tokens
  dropped — expected, matches the GPTQ/BiLLM convention)
- **PPL = 13.6436**, mean NLL = 2.6133
- Wall: ~1m47s on the 5080 (1.35 it/s, stable — no tensor-core warmup drift)
- Hardware entry: RTX 5080 Laptop, torch 2.11.0+cu130

Sanity-check: this sits in the right ballpark for a 4B Qwen3 on WikiText-2
(the paper quotes Qwen3-8B FP16 ~10–11; a 4B member of the same family
landing at ~13.6 is consistent with the usual scaling gap). Good enough to
serve as the reference point NanoQuant quality will be judged against on
this model.

Commit was dirty at run time — flagged in the entry's `git.dirty: true`.
Next action is to commit the harness before the Llama2-7B run so that
result lands on a clean SHA.

### Next

- Llama2-7B FP16 baseline (gated; needs `HF_TOKEN`). Once clean, this
  populates the second reference point and lets us reproduce the paper's
  advertised BiLLM-beating gap as a sanity check on the eval harness.
- Commit the Phase 0 harness (data/ppl/results/run_baseline) and the first
  Qwen3-4B entry in one commit so the baseline result has a clean SHA.

---

## 2026-04-23 — Phase 1, step 1: STE skeleton up, two bugs preempted (one didn't)

### Layout

Four more files under `research/nanoquant/`:

- `quant.py` — `BinaryFactoredLinear` (SVD init + clipped-STE forward) and
  `replace_linears_with_quant` / `quant_params` helpers. STE is clipped
  identity (grad = 1 for `|x| < 1`, else 0), applied via a custom
  `torch.autograd.Function`. Forward never materializes the effective
  weight — cost is `O((d_in + d_out) · r)` per batch-element.
- `activations.py` — builds a per-boundary teacher activation cache on
  disk. Captures aux kwargs (rotary `position_embeddings`, attention mask)
  via a `with_kwargs=True` forward-pre-hook on layer 0 so block-wise
  training can reconstruct the forward without threading them manually.
  Cache stored as fp16 regardless of model dtype (training casts on load).
- `phase1.py` — block-by-block quantize → freeze-non-quant-params →
  AdamW (lr=1e-4, wd=0, betas=(0.9, 0.95)) → per-step MSE against cached
  teacher output. Writes a status file after every block so a crash is
  recoverable.
- `run_phase1.py` — CLI, wires the pieces together, appends to
  `results.json` as `method.name = "phase1-ste-svd-init"`.

### Two bugs I preempted from cross-layer-svd's 2026-04-22 lessons

The new memory notes cross-layer-svd's Stage 4 gotchas and I took them
seriously when writing the code, so two of those four bugs never got to
ship:

1. **Only train quant params.** `phase1.py` explicitly freezes the whole
   block, then unfreezes only what `quant_params()` yields. RMSNorm
   weights are not touched. No silent normalization drift.
2. **AdamW `weight_decay=0.0`.** Scale vectors `s1, s2` stay free to grow.
3. (Line-buffered stdout and per-block status files are both in
   `run_phase1.py` from the start.)

### One bug I didn't preempt — fp16 overflow in the binary forward

Smoke at `n_calib=4, steps=10` produced `final_mse=nan` on every block
and `init_mse=inf` on blocks 6 and 16. Cause: Qwen3-4B ships as fp16
(range ±65504), but the rank-2 sign-approx of some linear weights
produces activation outliers that overflow that range inside the
attention/MLP forward — specifically on blocks 6 and 16, before any
training step. Other blocks overflowed only after a few AdamW steps.

**Fix:** default `--dtype` is now `bfloat16` (fp32-like range, fine
precision for this purpose). Also cast MSE computation to fp32 on both
train and eval paths for safety, and added a `not torch.isfinite(loss)`
guard in the STE loop that skips bad steps and counts them. bf16 alone
was enough to fix it on this smoke, but the guards stay as cheap
insurance.

This is worth remembering: the fp32 guidance in LittleBit's notes
("keep factored-binary latent params in fp32 because `tanh(tau·x)`
precision near zero matters for SmoothSign") is one axis; the separate
axis is that the **forward** through a signed rank-r approximation can
blow fp16's range on some layers, regardless of STE flavor. On Qwen3-4B
r=2 this hits at least two of 36 blocks at SVD init.

### Smoke result — pipeline works, PPL catastrophic (expected)

`n_calib=4, steps=20/block, r=2, bf16` on Qwen3-4B:

- Runtime: cache 15s, phase1 157s (4.4s/block), eval 85s. Total ~5 min.
- No NaN, no inf, no bad steps on any block.
- Per-block MSE trajectory climbs with depth (residual stream grows):
  block 0 = 0.026, block 5 = 0.062, block 17 = 0.11, block 30 = 3.5,
  block 35 = 151 → 144 after 20 STE steps.
- Two pathological blocks: 6 (init 13.75) and 16 (init 9.53) —
  ~100× worse than neighbors. SVD rank-2 sign-approx is catastrophic
  for some specific linear(s) in those blocks. Not diagnosed further;
  Phase 2's Dual-SVID / LB-ADMM init may or may not fix it.
- **PPL = 143,754,976** (≈1.4 × 10⁸). The research brief projected 10⁴
  to 10⁶ for Phase 1; we're above that, because at only 20 steps with
  `n_calib=4` STE barely moves the init (block 35: 151 → 144, ~5%).
- `results.json` entry skipped (`--no-log`) so the smoke doesn't pollute
  the reference log.

### Next

1. Commit the Phase 1 harness on a clean SHA so the real run's entry is
   tagged cleanly.
2. Full Phase 1 run: `steps=500`, `n_calib=32`, `r=2`, bf16. Expected
   runtime ~25–30 min.
3. Log the result, journal the trajectory. Accept that the number will
   be big — it's the Phase 1 floor.

---

## 2026-04-23 — Phase 1, step 2: floor is 29,668 PPL

### Headline

Qwen3-4B, bf16, `r=2`, `n_calib=32`, `steps=500`/block, plain-SVD init,
clipped-identity STE, AdamW(lr=1e-4, wd=0, betas=(0.9, 0.95)), pure
teacher input per block. One linear's output is a single `diag(s1) ·
sign(U) · sign(V)^T · diag(s2)` product, `U, V ∈ ℝ^{d × 2}`.

- **Qwen3-4B Phase 1 WikiText-2 PPL = 29,668.6** (nll_mean 10.30)
- vs FP16 baseline 13.64 → **2174× worse**
- Mean block init MSE 6.50 → final 3.05 (53% reduction averaged across
  36 blocks). Per-block behavior highly non-uniform.
- Wall: cache 56s, phase1 813s (22.6s/block), eval 85s. End-to-end ~16 min.
- Entry id `2026-04-23T14:37:02Z-qwen-qwen3-4b-phase1-ste-svd-init`
  tagged on `f51681384` (clean harness commit; results append made it
  dirty, a later journal commit re-cleans).

### Per-block trajectories — the diagnosis

Grouping the 36 blocks by behavior at 500 STE steps:

1. **Modest STE wins, small starting MSE** (~0 .02–0.2 init; final within
   20–50% of init). Blocks 0–5, 7–15, 17–23. STE is effective but the
   residual-stream context just doesn't require much correction —
   rank-2 sign-approx is already "close enough" for these blocks.
2. **Catastrophic init, STE nearly useless** at 500 steps. Block 6: init
   13.81 → final 13.70 (0.8% reduction). Block 16: init 9.66 → final
   9.14 (5.4% reduction). 10–100× worse than neighbors and STE can't
   climb out. These are the clear failure cases.
3. **Large init, but STE does meaningful work.** Late blocks where
   residual stream grows. Block 24: 0.85 → 0.65 (24%). Block 30:
   3.54 → 2.83 (20%). Block 34: 29.93 → 20.80 (31%). Block 35:
   151.57 → **41.02 (73% reduction)**. The depth-bias reduces but
   doesn't disappear.

Blocks 6 and 16 are the interesting failures. Same story as the
smoke (though smoke had only 20 steps of STE, so the question was
whether more training helps — it doesn't). Some linear in those
blocks has a singular-value spectrum where rank-2 captures essentially
none of the energy, and signing the factors destroys what little is
there. Flagging as a Phase 2 target — if LB-ADMM fixes blocks 6/16
cleanly, that alone validates the paper's algorithmic choice.

### Calibration against the research brief

The 2026-04-23 brief projected "post-init PPL 10⁴–10⁶, STE-refined low
thousands." We landed at 10⁴ — specifically, above the low-thousands
band but the right order of magnitude. Sources of pessimism vs the
brief:

- `r=2` not `r=4` (minimum chosen deliberately per DESIGN §5 primary).
- `n_calib=32` not 128 (tradeoff for iteration speed; we can scale up
  without changing code).
- Plain SVD, not Dual-SVID (Phase 1's point — measure this floor).
- No error propagation, no global scale opt.

The brief also projected a 10× init→refined improvement as the success
bar for "scaffold works." We got: mean block final MSE is 47% of init
on average (53% reduction), and final PPL 30K is consistent with STE
having moved something real per-block but not enough in aggregate
because the residual stream compounds — per-block 53% → end-to-end
catastrophe. This is the expected Phase 1 signature.

### Infrastructure lessons

1. **bf16 is the right default** for the forward. fp16 overflows on
   blocks 6/16 at SVD init (we saw `init_mse=inf` directly in the
   earlier smoke). Committed harness hardcodes `--dtype bfloat16`.
2. **Cache-build chunk_size=2 fits fine on the 5080** at n=32 seq=2048
   (hidden-states peak ~8 GB VRAM).
3. **Activation cache is ~13 GB on disk** at n=32 (37 boundaries × 335 MB).
   Fine on NVMe. Scaling to n=128 would be ~52 GB — order of magnitude
   for a full reproduction. Budget for it.
4. **22.6s/block is dominated by the 500 STE steps at ~25 it/s** (~20s)
   plus ~2s for SVD init of 7 linears and MSE eval. SVD at r=2 on a
   9728×2560 matrix is quick.
5. **Training loss display**: log every `steps // 20` (so 25 times per
   block at 500 steps) feels right — enough resolution to spot
   divergence, not so much the display is all tqdm overhead.

### Next

1. Commit the results.json entry + this journal on a new SHA.
2. Sanity-check whether the blocks-6/16 pathology is specific to one
   linear type (q, k, v, o, gate, up, or down). Cheap diagnostic —
   re-run SVD init on just those blocks and measure per-linear
   reconstruction error.
3. Phase 2 vein: implement LB-ADMM init. That's where K-FAC + ADMM +
   SVID come in — the paper's actual contribution. The PPL target is
   roughly "close the gap between Phase 1 (30K) and Phase 3 with full
   pipeline (paper says 25 at 8B, so ~30–50 at 4B if it scales)."
4. (Optional, cheap) Re-run Phase 1 at `r=4` to confirm the brief's
   intuition that rank matters here — should drop PPL substantially
   just from the extra rank, independent of init quality.

---

## 2026-04-23 — Phase 1, step 3: per-linear diagnostic reframes the story

Ran `diag_block_linears.py` across blocks 5–7 and 15–17 (the suspect
pair and their neighbors) to test the hypothesis that **one specific
linear** in blocks 6 and 16 was responsible for the ~10× init-MSE gap
seen in Phase 1.

**Hypothesis falsified.** The per-linear table is essentially flat:

- Rank-2 signed-SVD reconstruction error is 98.8%–99.9% on **every
  linear in every block**. The numbers look the same in block 6 as
  in blocks 5 and 7.
- Top-2 singular energy fraction is 0.003–0.046 everywhere — all
  Qwen3-4B linears have roughly flat singular spectra past the top
  handful of components. Rank-2 captures almost nothing of the
  Frobenius energy; this is uniform across blocks.
- Signing cost (err_signed − err_trunc) is 0.0005–0.008 — tiny and
  also uniform. Signing the rank-2 factors adds <1% relative error
  on top of truncation.
- Block 6's worst linear (`k_proj`, signE 0.9906) vs block 5's
  counterpart (signE 0.9881) differs by 0.3%, nowhere near enough to
  explain a 100× block-level MSE gap.

So: **the Phase 1 block-6/16 pathology is not a weight-space
phenomenon.** It's activation-space.

### Reframe

Rank-2 SVD preserves the top-2 singular directions of `W`. What makes
the block fail isn't how much of `W` is reconstructed — it's how much
of the **input activation energy** falls on those preserved directions.
`block_output_error ≈ (W − Ŵ) · x`, so a block with inputs concentrated
on singular directions 3+ will suffer catastrophically even if `W − Ŵ`
looks identical in Frobenius terms to a neighbor's.

Blocks 6 and 16 are presumably the layers where Qwen3-4B's activations
align with singular directions not preserved by rank-2 truncation.
This is the same phenomenon the cross-layer-svd §13.2 activation-
weighted experiment documented: **Frobenius-optimal low-rank ≠
activation-optimal low-rank**.

### What this implies for Phase 2

The paper's Phase 2 is K-FAC-preconditioned LB-ADMM. Re-expressing
that: it solves `min ||X·(W − Ŵ)^T||_F²` — the activation-weighted
Frobenius objective — which is precisely the thing the above
diagnostic says we need. K-FAC shrinks the empirical `E[xx^T]` with
Ledoit-Wolf; ADMM handles the non-convex `{±1}` constraint.

If LB-ADMM closes the block-6/16 gap specifically (bringing their
init MSE within the same band as neighbors), that's the single
cleanest Phase-2-value-prop story.

### Artifact

[`diag_block_linears.py`](diag_block_linears.py) — the diagnostic
script. `err_trunc` / `err_signed` / `signing_cost` per-linear, plus
`top_r_energy` and `cond_r` for spectrum characterization.
Output saved to `diag_blocks_6_16.log`.

### Next

Phase 2. Start with the K-FAC activation Gramian
(`E[xx^T]` per linear) collection — that's the prerequisite for every
other piece of the LB-ADMM machinery.

---

## 2026-04-23 — Phase 2, step 1: paper-faithful LB-ADMM scaffold (WIP, open bug)

Built the paper's LB-ADMM init path end-to-end. Four new files
(`preconditioner.py`, `admm.py`, plus extensions to `quant.py` and
`phase1.py`, wired together through `run_phase2.py`). Stopped in a WIP
state with a reproducible bug and clean diagnosis below.

### Paper source of truth

Re-fetched the paper via WebFetch and pdftotext. Appendix C pinned
several implementation details that the arXiv HTML abstract omits.
**Key numbers** that now live in DESIGN defaults or `run_phase2.py`:

- K = 400 ADMM steps per matrix, with a **linear ρ scheduler**
  (values not specified by the paper; chose 0.1→10 as a starting guess).
- `γ = 0.2` Ledoit-Wolf for Llama/Qwen, `0.6` for Gemma/Rnj.
- TuneLatentSTE: lr=1e-5, bs=1, 8 epochs, cosine.
- TuneFP (Step 1 Error-Prop Mitigation) and TuneScalesKD (Phase 3
  global KL) are **deferred** — Phase 2 tests ADMM init alone.

Also clarified that **SVID is from OneBit (Xu et al. 2024)**, not
Pouransari 2020 (the paper cites both, but the algorithmic content
belongs to OneBit). OneBit's Prop 1 is the definition:

    SVID(M) := sign(M) ⊙ (a b^T)  where  |M| ≈ a b^T

The `a b^T` is a rank-1 SVD of `|M|` — so the full m×r sign pattern is
preserved while the magnitudes collapse to a rank-1 outer product.

### Algorithm implemented

Paper Eq. 5 / 6 / 24 with scaled dual `Λ = Y/ρ`:

    # U update
    (V^T V + (ρ+λ)I) · U^T = V^T W̃^T + ρ (Z_U - Λ_U)^T       (Cholesky)
    # V update: symmetric
    Z_U = SVID(U + Λ_U)                                       (paper Eq. 26)
    Λ_U = Λ_U + (U - Z_U)

then magnitude balancing (paper Eq. 7-9) to extract `(s1, s2, U_latent,
V_latent)` in our stored format. Verified on a random 256×512 weight
that ADMM at D=I matches SVD init Frobenius quality (0.9924 vs 0.9928);
with non-trivial `(D_in, D_out)` Frobenius error correctly goes up
(0.9951) because we're optimizing the activation-weighted norm, not
plain Frobenius.

### Process fix: use the paper, not local related code

First web-search agent fell back to `research/cross-layer-svd/`'s
LittleBit work when its WebFetch was denied, and proposed we "port" the
existing Dual-SVID + ALS code. User pushed back: *"we need to be
faithful to the nanoquant paper."* The second web-fetch run succeeded
(WebFetch on the abstract + pdftotext on the downloaded PDF) and gave
Algorithm 1, Appendix B (ADMM statement + convergence proof), and
Appendix C (hyperparameters) verbatim. Memorialized as a `feedback`
memory so future sessions default to the paper over nearby local
reimplementations.

### Bugs found during build (root-caused)

1. **Windows stdout γ → UnicodeEncodeError** — replaced γ with `gamma`.
2. **Hook `.cpu()` per-fire — not actually the bottleneck.** Hypothesis
   was wrong: swapping to on-device fp32 accumulators saved ~5% not 10×.
   Actual bottleneck is Qwen3-4B backward through seq=2048 with HF
   gradient checkpointing: ~2 min/sample on the 5080. One-time ~1 hour
   cost at n=32 calib; cached to disk, so subsequent runs reuse.
3. **My own over-normalization** — I had `U_latent /= s1[:, None]` at
   the end of `lb_admm_init` (copied the svd_init pattern). Paper Eq. 9
   says `U := η · Û`, no such divide. With SVID's rank-1 magnitude
   structure (every row of Z_U proportional to `(a[i], b[k])`), normalizing
   by per-row mean-abs pushes every latent entry to |x| ≈ 1, landing
   exactly on the clipped-STE threshold and killing the gradient. Fixed.
4. **bf16 latent params kill AdamW updates when init magnitudes are
   small.** After fixing (3), `|U_latent|` sits near 0.04. bf16's
   relative precision 2^-7 = 0.008 gives absolute precision ~3e-4 at
   that magnitude, which is *above* an AdamW step of magnitude 1e-4
   (lr) × 1 (normalized moment) = 1e-4. Updates round to zero in bf16
   storage. Fix: store factor params as fp32, cast to the input's dtype
   in forward. LittleBit's notes warned about this specifically.

### Open bug (WIP, documented so a future session can pick up)

After fix (4), a standalone unit test with a single linear + fake
target confirms AdamW on fp32 latents updates params (loss 1.0071 →
1.0045, `|U_latent|` changes by 0.002 over 20 steps).

But the **end-to-end Phase 2 smoke still reports `final_mse == init_mse`
to 4 decimals on every block**, and PPL = inf. So something is
different between the unit test and the real block-by-block training
loop. Hypotheses to test next session (in order):

1. bytecode cache — already confirmed no stale `__pycache__` was
   surviving, but worth a hard re-check.
2. `quantize_model_phase1` — is `quant_params(block)` actually yielding
   the fp32 params after `replace_linears_with_quant` with the init_fn
   path? Add an explicit `assert U_latent.dtype == torch.float32` at
   the top of `train_block`.
3. Silent skip: if the MSE computation returns a value that happens to
   be reported as finite but the backward pass produces NaN grads that
   AdamW handles by leaving params unchanged? The `not torch.isfinite(loss)`
   guard would have to count it; add a `bad=N` display check —
   probably already fine since `bad=0` is shown, but verify.
4. The `block.to(device)` after `replace_linears_with_quant` may be
   implicitly casting fp32 factor params to the block's bf16 dtype.
   Check with `print([p.dtype for p in quant_params(block)])` after
   `.to(device)`. If bf16, skip `.to(device)` or use `.to(device,
   dtype=None)` with per-param overrides.
5. The training loop's `x.to(dtype=target_dtype)` where `target_dtype`
   is taken from `next(p for p in block.parameters()).dtype` — that
   parameter is RMSNorm's bf16 weight, so the forward is bf16, which is
   fine. But if during `_eval_block_mse` we also cast the block inputs
   to bf16 while the factor params are fp32, the mixed-type backward
   may drop gradients. Check explicitly.

### Files landed (WIP)

- [preconditioner.py](preconditioner.py) — D_in / D_out collection via
  forward + backward hooks, percentile-clipped, Ledoit-Wolf shrunk.
- [admm.py](admm.py) — SVID operator, LB-ADMM K-iteration loop,
  magnitude-balanced scale extraction.
- [quant.py](quant.py) — added `from_factors` + fp32-latent storage,
  `replace_linears_with_quant(init_fn=…)` callback.
- [phase1.py](phase1.py) — `init_fn_factory` parameter for
  `quantize_model_phase1` so Phase 2 reuses the same STE loop.
- [run_phase2.py](run_phase2.py) — end-to-end CLI.
- [diag_block_linears.py](diag_block_linears.py) — already landed in
  the prior session's diagnostic (kept as-is).

Preconditioner cache at n=4 is persisted under
`cache/qwen-qwen3-4b/n4_L2048_seed0/preconditioners.pt`. Don't rebuild
on smoke; it's a 520 s cost.

### Next

1. Fix the STE-no-move bug (item 4 in the WIP list is my leading
   hypothesis: `block.to(device)` silently downcasting fp32 factor
   params). Add the assert, run the smoke, confirm.
2. Once movement shows up at n=4 (even if it produces a bad PPL because
   preconditioner is too noisy), scale up to n=32. Full run: ~60 min
   preconditioner + ~15 min phase2 + eval.
3. Measure against Phase 1's floor of PPL 29,668 and see if the
   activation-weighted init closes the block-6/16 gap the Phase 1
   diagnostic flagged.

---

## 2026-04-23 — Phase 2, step 2: root-caused (D scaling) — smoke produces real gradients

Earlier WIP hypotheses 1–5 were all wrong. The actual bug came from a
scale the paper never writes down:

**`D_in` and `D_out` need to be normalized to mean=1 before ADMM, or
the λ ridge regularizer dominates the optimum and drives U, V to zero.**

### Trace

Wrote `diag_phase2_grad.py` — a focused single-block diagnostic that
replicates the full Phase 2 pipeline (model load, cache load, LB-ADMM
replace, forward+backward) with `retain_grad()` on every intermediate
inside `BinaryFactoredLinear.forward`. Output pinned the failure:

- Forward graph intact (`loss.grad_fn = MseLossBackward0`, `y.requires_grad = True`).
- `loss = 0.025` (nonzero).
- `loss.backward()` runs without error.
- **Every param.grad and every retained intermediate grad = exactly 0.**

That pattern only happens if the loss's *numerical* dependence on the
params is below the precision floor of the arithmetic path. Two more
prints identified what was small:

    q_proj D_in: mean=0.023
    q_proj D_out: mean=1.05e-4        ← 200× smaller than D_in
    W_eff = s1·(sign·sign)·s2: mean=2.2e-31  ← essentially zero

Compare to a healthy, unquantized layer's W: mean 0.017.

### Why

Paper Eq. 15 uses `W̃ = D_out · W · D_in`. My `preconditioner.py` built
`D_out` from RMS of per-output-channel cross-entropy gradient — and on
a mean-normalized CE loss over 4 × 2048 tokens, those gradients are
~10⁻⁴. `D_in` is activation RMS, ~10⁻². Both are fine as *relative*
importance signals, but their absolute magnitudes differ by 200× and
the product `D_out · D_in ≈ 2×10⁻⁶` scales W down by six orders of
magnitude.

The ADMM objective is

    min ½‖W̃ − UV^T‖² + (λ/2)(‖U‖² + ‖V‖²)

At the optimum (gradient = 0):

    U ≈ W̃·V / (V^T V + λ·I)

When `‖W̃‖` is tiny and `V^T V` is also tiny (because `V` shrank to
match `W̃`), the `λ·I` term (λ = 1e-3) dominates the denominator and
drives `U` to ≈ `W̃·V / λ` ≈ `10⁻⁸ · 10⁻⁴ / 10⁻³` = `10⁻⁹`. The
factored-binary effective weight is then `s1 · s2 · sign-product`
with s1, s2 ≈ `10⁻⁸`, giving W_eff ≈ `10⁻³¹` — a weight matrix
numerically indistinguishable from zero.

With W_eff ≈ 0, q_proj output is zero, attention softmax is uniform,
gradient through a uniform softmax shrinks by ~`1/seq_len²` = `1/4M`
per layer, and the chain underflows to zero at fp32 precision for
every intermediate.

### Fix

`admm.py` now normalizes both preconditioners to mean=1 before ADMM:

    D_in  = D_in  / D_in.mean()
    D_out = D_out / D_out.mean()

This preserves the relative channel-importance ratios (the actual
content of K-FAC preconditioning) while keeping `W̃` at the same
scale as `W` so `λ=1e-3` can't dominate. Only 3 lines; the fix is
structural, not a workaround.

### Instrumentation

Also added `--verbose-blocks N` to `run_phase2.py` (and a `verbose`
flag to `train_block`), which logs for each of the first N blocks:

- Number of trainables, their dtypes, first-param shape and mean-abs.
- Step-0 loss and grad norms for the first 6 trainables.
- Post-training `max_abs_delta` and how many of the trainables moved.

Plus `diag_phase2_grad.py` stays in the tree as a reusable
gradient-flow probe for future debugging.

### Smoke result (n=4, steps=20, K=50, lr=1e-4, fixed)

- `|U_latent| mean = 0.020` (was 1.6×10⁻⁸ before fix).
- Grad norms per param: 10⁻⁵ to 10⁻³ — healthy.
- **All 28 trainables per block move** (`max_abs_delta ≈ 2×10⁻³`,
  `bad_steps = 0`).
- Block 0: 0.0257 → 0.0247. Block 1: 0.0184 → 0.0162. Block 35:
  137.5 → 117.9. STE actually reduces MSE now.
- Blocks 6 and 16 still pathological (init 13.77 and 9.76) but STE
  moves them ~0.05%. Preconditioner at n=4 doesn't have enough signal
  to fix them — deferred to n=32+.
- **PPL = 1,376,899** (1.4×10⁶), nll_mean 14.14.

For reference: Phase 1 smoke at the same `(n=4, steps=20)` was PPL
143,754,976 (1.4×10⁸). **Phase 2 smoke is 100× better on the same
budget**, just from LB-ADMM init + correctly-flowing STE.

### Next

1. Commit the fix + instrumentation on a clean SHA.
2. Full Phase 2 run: `n_calib=32`, `K=400` ADMM, `lr=1e-5`, `steps=500`
   (paper defaults). Expected wall: preconditioner ~60 min (backward
   through 4B at seq=2048 with gradient checkpointing) + Phase 2 ADMM
   + STE ~30 min + eval 2 min. Total ~90–100 min.
3. Compare against Phase 1's floor of PPL 29,668. The Phase 1
   diagnostic flagged block 6/16 as activation-space problems; at
   n=32 LB-ADMM should finally have enough preconditioner signal to
   close that gap.

---

## 2026-04-23 — Phase 2, step 3: full run — LB-ADMM alone LOSES to SVD + STE

### Headline

- **Phase 2 (LB-ADMM + STE): PPL = 277,154** (nll 12.53)
- Phase 1 (SVD + STE):    PPL = 29,669 (nll 10.30)
- **Phase 2 is 9.3× worse than Phase 1** on the same budget.

Results entry: `2026-04-23T18:42:58Z-qwen-qwen3-4b-phase2-lbadmm-ste`,
tagged on `d1e5e2f7b`. Hyperparameters: `r=2, n_calib=32, K=400,
ρ: 0.1→10 linear, λ=1e-3, γ=0.2, lr=1e-5, 500 steps/block, bf16 compute
+ fp32 latents`. Wall: cache cached, precond cached (from earlier run),
Phase 2 ADMM+STE 2386s, eval 90s. Effectively 40 min once caches were
warm (1 hr cold with precond rebuild).

### Per-block behaviour — bimodal

LB-ADMM init is catastrophically varied at the Frobenius block-MSE
level:

- **Early-mid blocks 0–15**: inits range 0.02–13 (vs Phase 1's 0.02–0.14).
  Some start worse than Phase 1, some better. STE closes most of the
  gap: block 1 went init 12.51 → final 0.089 (140×), block 20 went
  106.12 → 0.18 (588×), block 30 went 674.24 → 7.37 (91×).
- **Blocks 6, 16 (the Phase 1 pathological ones)**: init_mse 13.79 and
  10.04; final_mse 13.76 and 9.37. *Barely moved* — same failure
  pattern as Phase 1 SVD init, just landing at slightly different bad
  values. The activation-weighted objective at our hyperparameters
  did **not** close the gap the Phase 1 diagnostic flagged.
- **Late blocks 30–35**: Phase 2 finals 7.4, 3.2, 4.4, 9.9, 15.4,
  231.8. Phase 1 finals 2.8, 3.2, 4.6, 5.8, 20.8, 41.0. Phase 2 is
  roughly equal in the middle of this range but **5–6× worse on
  block 35** (the last block, feeding directly into the LM head).
- **Mean block final_mse: 8.51 vs Phase 1's 3.05** — 2.8× worse on
  average.

### Why LB-ADMM loses here

The Phase 1 diagnostic established that block 6/16 are activation-space
problems: input activations concentrate on singular directions rank-2
SVD doesn't preserve. The expectation was that LB-ADMM's K-FAC-weighted
objective (Eq. 2) would preferentially preserve the activation-aligned
directions and close the gap. It did **not** on these two blocks,
within our budget.

Specific reasons this run under-performs Phase 1:

1. **LB-ADMM optimizes activation-weighted Frobenius, not
   plain Frobenius.** Init block-MSE (our Phase 1 metric) is *not*
   what LB-ADMM is targeting. Its higher block-init-MSE (mean 268 vs
   Phase 1's 6.5) is partly expected — the algorithm is preserving
   what matters for activation reconstruction, not what matters for
   a Frobenius diff on cached teacher outputs.

2. **We ran only Step 2-2 + Step 2-3 of Algorithm 1** — no
   **TuneFP** (Step 1, error propagation mitigation) and no
   **TuneScalesKD** (Phase 3, global logit-KL fine-tune). The paper's
   Table 5 ablation ("LB-ADMM > Dual-SVID init") is measured after
   the *full* pipeline. Isolated init comparison isn't what the
   paper claims to win. **This replication tests what we built; it
   doesn't disprove the paper's headline.**

3. **lr=1e-5 × 500 steps** (paper's TuneLatentSTE values, scaled for
   a 500-step budget) produces less parameter movement than Phase 1's
   lr=1e-4 × 500 steps. At bf16 forward + fp32 latent storage the
   signal IS making it through (confirmed in step-2 diagnostic), but
   the total update budget is ~10× smaller. If the paper's *true*
   recipe is 8 epochs × 128 samples = ~1024 × 8 = 8192 steps, we're
   at 500 (6% of their budget).

### What LB-ADMM *did* help

Several non-pathological blocks got dramatic STE reductions from their
high inits (see 140–588× numbers above). The algorithm is doing
meaningful work — just not enough of it, and not on the blocks that
most need help. On clean blocks 7–14 Phase 2 final MSEs are nearly
identical to Phase 1's.

### Implementation-fix bugs caught and resolved

Four bugs discovered and fixed before producing this number:

1. **D_in / D_out unnormalized** → λ-ridge dominated ADMM, U,V collapse
   to ~1e-8, W_eff ~1e-31, backward underflows to zero. Fixed by
   normalizing each to mean=1 inside `lb_admm_init`. (Previous entry.)
2. **fp32 latent params + bf16 compute dtype inferred from first
   block param** — after factor params moved to fp32, the
   training-loop `target_dtype = next(p for p in block.parameters()).dtype`
   was picking U_latent's fp32, forcing the *entire block forward*
   (attention, MLP, norm) into fp32. That disabled Qwen3's SDPA
   fast-path and made block training **~100× slower** (11 s/step
   vs 0.06 s/step). Extrapolated full run would have been ~20
   hours. Fixed in `phase1.py`: pick compute dtype from the first
   NON-quant-param (RMSNorm bf16 weight), not the first param.
3. **Unicode γ, Δ** — second round on Windows stdout. Replaced with
   ASCII.
4. **Preconditioner takes ~2 min/sample** — this is just physical
   cost of Qwen3-4B backward at seq=2048 with gradient checkpointing.
   n=32 takes ~60 min. One-time, cached to disk.

### What this means for the replication

Phase 2 as *just the paper's Algorithm 1 Step 2-2 + Step 2-3* is not
a win on Qwen3-4B r=2 at our budget. The natural next steps:

1. **Add TuneFP (Step 1).** The paper explicitly adjusts the FP
   weights of the current block *before* quantization, to absorb
   quantization error accumulated in preceding blocks. This might
   specifically address the late-block explosion we see (block 35
   Phase 2 final 231 vs Phase 1 41).

2. **Increase LR or step budget.** Paper lr=1e-5 × 8 epochs × 128
   samples is 8192 steps; we did 500. Try lr=1e-4 + 500 steps (match
   Phase 1 total movement), or lr=1e-5 + 2000 steps. The latter is
   closer to the paper's recipe.

3. **Ablate the four Phase 2 hyperparameters we guessed at.**
   Specifically: ρ schedule (paper doesn't specify start/end), λ (not
   specified), K=400 (confirmed from Appendix C but on H100 — maybe
   too many on our smaller budget). Narrow sweep on a single block
   with instrumentation.

4. **Implement TuneScalesKD (Phase 3).** Freeze binary signs, fit
   s1/s2 end-to-end against FP16 logits via KL. Paper says this is
   a late-stage optimizer that closes any remaining gap from
   fixed-sign approximation.

Path forward recommended: **TuneFP next**, because (a) it directly
addresses the late-block compounding error signature we see in the
data, and (b) it's the only Step-1 contribution we haven't
implemented, so it isolates a clean algorithmic variable.

### Memory to save

Added `feedback` memory earlier about paper-replication discipline;
this entry reinforces it with a datapoint. LB-ADMM's claimed benefit
over Dual-SVID in Table 5 is measured with the full pipeline; do
not assume any single paper step is a win in isolation.
