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

---

## 2026-04-23 — Phase 2, step 4: LR ablation + blocks 6/16 sweep — hyperparameters have hit diminishing returns

Two experiments, both cheap, designed to pin down what the 9× PPL gap
between Phase 2 (LB-ADMM+STE) and Phase 1 (SVD+STE) actually comes from.

### Experiment 1: LR ablation (`lr=1e-5` → `lr=1e-4`)

Same Phase 2 config otherwise. Results entry
`2026-04-23T19:21:45Z-qwen-qwen3-4b-phase2-lbadmm-ste` on `2b771f545`:

| metric | lr=1e-5 | lr=1e-4 | Phase 1 SVD |
|---|---:|---:|---:|
| PPL | 277,154 | **249,378** | 29,669 |
| mean block final MSE | 8.51 | ~3.24 | 3.05 |
| block 6 final | 13.76 (stuck) | **9.90** | 13.69 (stuck) |
| block 16 final | 9.37 (stuck) | **7.23** | 9.14 (stuck) |
| block 35 final | 231.8 | 64.5 | 41.0 |

**Block-level MSE at `lr=1e-4` now matches Phase 1 on most blocks** —
the previously-stuck blocks 6 and 16 moved 28% and 21% respectively,
and blocks 30–34 converged to Phase 1's range. But **PPL only dropped
10%** (277K → 249K), still 8× above Phase 1.

The per-block average converging to Phase 1's while PPL stays 8× worse
means the PPL gap **is not an aggregate block-MSE problem** — it's
dominated by specific blocks whose error compounds through the
residual stream. Specifically block 35 (last block, feeds LM head
after final norm): 64 in Phase 2@1e-4 vs 41 in Phase 1 (1.57× worse
block-MSE, but massive PPL leverage).

Conclusion: **the LR ceiling for closing the Phase 2 gap is very low.**
Further LR tuning wouldn't meaningfully help. The remaining gap is
structural.

### Experiment 2: `(K, λ)` sweep on blocks 6 and 16

Nine combos each — `K ∈ {100, 400, 1000} × λ ∈ {1e-4, 1e-3, 1e-2}` —
with instrumented `lb_admm_init` recording primal/dual residuals,
latent magnitudes, Frobenius rel-err, and activation-weighted rel-err.
Script: `diag_blocks_sweep.py`, output `runs/blocks_sweep.json`.

**Block 6 best init_mse: 13.26 at K=1000, λ=1e-3** (our default was
15.93 at K=400 λ=1e-3; Phase 1 SVD was 13.79). LB-ADMM *can* beat SVD
on this block's init but by only 4%.

**Block 16 best init_mse: 9.97 at K=100, any λ** (our default 10.01
at K=400 λ=1e-3; Phase 1 SVD was 9.14). **Phase 1 SVD actually wins
on block 16 init.** LB-ADMM's activation-weighted objective doesn't
translate to better Frobenius block-MSE here.

Key patterns across the 18 combos:

1. **K=100 beats K=400 beats K=1000** on activation-weighted rel-err
   for most combos. ADMM overshoots with more iterations — primal
   residuals on V are already 0.1–0.5 at K=100 and don't tighten.
2. **K=1000 + small λ diverges** on block 16 (init_mse = 863 at
   K=1000, λ=1e-4). Confirms instability at high iterations.
3. **All 18 combos have `frob_rel ≥ 1.0`** — the quantized weight is
   farther from the original than zero is, in Frobenius. Rank-2 signed
   binary is **architecturally insufficient** to represent these two
   specific weight matrices regardless of LB-ADMM hyperparameters.
4. **Per-block optima diverge**: block 6 wants K=1000, block 16 wants
   K=100. No single (K, λ) is universally optimal.
5. **Activation-weighted rel-err IS meaningfully lower than Frobenius
   rel-err** on the best combos (act_rel 0.80 vs frob_rel 0.996 on
   block 6 at K=100 λ=1e-4). The algorithm is doing its job on its own
   objective; the Frobenius metric just doesn't reward that.

### Combined conclusion

Hyperparameter tuning on LB-ADMM has hit diminishing returns:

- **LR** cap: `lr=1e-4` brings per-block MSE to parity with Phase 1
  but leaves an 8× PPL gap. Further LR won't help.
- **K, λ**: per-block best init gains are 4–5% over defaults. Maybe
  ~5% PPL improvement achievable with per-block adaptive settings.
  Not the main lever.
- **rank r**: out of scope here but the larger lever (r=4 vs r=2 is
  the obvious direction).

**The dominant remaining lever is the paper's Step 1, TuneFP** (error
propagation mitigation): adjust the FP weights of the current block
*before* quantization, to absorb error accumulated from preceding
quantized blocks. This directly targets the late-block compounding
error signature we see (block 35 is the PPL killer, and its input is
the residual-stream output of 34 preceding blocks of accumulated
quantization noise).

### Next

1. Build TuneFP (Algorithm 1 Step 1). Inputs per block: the block's
   original FP weights + cached teacher input + **actual current
   student input** (output of preceding already-quantized blocks). FP
   weights get fine-tuned (~50–100 AdamW steps at lr=1e-4, per paper's
   TuneFP defaults) to make the FP block produce the teacher's output
   *given* the student input. This absorbs upstream quantization error
   into the FP weights *before* we then quantize them with LB-ADMM.
2. Re-run Phase 2 with TuneFP + LB-ADMM + STE. Expected dominant
   effect on block 35 (currently 64× vs Phase 1's 41×, where the
   compounding error concentrates).
3. Only then revisit per-block (K, λ) adaptation — it's a second-order
   optimization that only matters if TuneFP closes the big gap first.

### Artifacts

- `diag_blocks_sweep.py` — reusable per-block `(K, λ, ρ)` sweep with
  instrumented ADMM residuals.
- `runs/blocks_sweep.json` — the 18-row data dump from this sweep.
- Phase 2 @ lr=1e-4 entry in `results.json` (id
  `2026-04-23T19:21:45Z-qwen-qwen3-4b-phase2-lbadmm-ste`).

---

## 2026-04-23 — Phase 2, step 5: TuneFP implemented — surprisingly makes PPL 8× WORSE

Built TuneFP (Algorithm 1 Step 1) + student-input chaining for the STE
step, per paper spec. Clean implementation, unit tests pass, full run
finished. Result:

### Headline

- **Phase 2 + TuneFP (n=32, K=400, lr=1e-4, tune_fp=200 steps): PPL 1,950,548**
- Phase 2 no TuneFP (same lr=1e-4):                              PPL   249,378
- Phase 2 no TuneFP (paper's lr=1e-5):                           PPL   277,154
- Phase 1 (SVD + STE, lr=1e-4):                                  PPL    29,669
- FP16 baseline:                                                 PPL        14

**TuneFP made PPL 7.8× WORSE than no-TuneFP, 66× worse than Phase 1.**
Results entry `2026-04-23T21:31:01Z-qwen-qwen3-4b-phase2-tunefp-lbadmm-ste`
on `bc68acf9f`. Wall 93 min (TuneFP is expensive — 200 step full-block
backward ×36 blocks).

### The paradox

Block 35 **final MSE = 9.64** — best we've ever measured (Phase 1: 41,
Phase 2 no-TuneFP: 64). The LAST block is dramatically better. But PPL
is 8× worse.

Why? Because the *mean* block final MSE is ≈ 18 (vs 3.2 for Phase 2
no-TuneFP). The middle blocks are the issue.

### Error floor phenomenon

- Blocks 7–15 all land at STE final MSE ≈ **11.9** (± 0.1).
- Blocks 17–29 all land at STE final MSE ≈ **23–35**.
- Block 35 drops to 9.6 because the LM head is the final target, and
  STE can always fit the final layer well.

The floors are NOT random — they're the **irreducible distance**
between what the teacher block *would* produce on the student's drifted
input and the cached teacher-on-teacher target. No per-block training
can drive this below zero.

### Why TuneFP hurt at our hyperparameters

Two mismatched objectives:

1. **Phase 1 / Phase 2 no-TuneFP** use cached `(X_teacher, Y_teacher)`
   pairs: train each block to reproduce teacher behavior on clean
   input. Easy objective → each block's individual quality is high
   (final MSE 0.1–2). At deploy time, errors compound but each block
   is individually well-conditioned.

2. **Phase 2 with TuneFP** uses `(X_student, Y_teacher)`: train each
   block to produce teacher output when fed *realistic drifted input*.
   Hard objective → each block's final MSE reflects the upstream
   noise floor (11–35). At deploy time, no compositional advantage
   because the "training distribution" already baked in noise the
   block can't correct.

The irreducible-floor interpretation: for training signal
`‖block(X_student) − Y_teacher‖²`, the optimum is capped at
`‖teacher_block(X_student) − teacher_block(X_teacher)‖²`. No
per-block learning can get below this. TuneFP doesn't change the
floor — it just moves the FP weights around within the floor.

### What WORKED in TuneFP

To be clear, TuneFP itself does the work the paper claims. Looking at
the per-block TuneFP loss reductions:

- Block 2: 231 → 0.05 (4600× absorption).
- Block 9: 192 → 11.9 (16× absorption).
- Block 17: 271 → 22.5 (12×).
- Block 30: 4164 → 39.8 (104×).
- Block 35: 9959 → 9.6 (**1040× absorption**, best measured).

TuneFP is genuinely compensating for upstream quantization error. The
issue is downstream: LB-ADMM quantization of the TuneFP-tuned FP
weights reintroduces error, and the subsequent blocks' STE (on the
noisy student input path) can't push below the floor.

### Hypotheses for why the paper's Table 2 claims this works

1. **Paper uses lr=1e-5 for TuneLatentSTE × 8 epochs = ~1024 steps**
   per block at n=128, ~32× more signal than our 500 steps. That
   might be enough to drive STE below the measured floors.
2. **Paper has `weighted MSE` in block reconstruction** (Appendix C
   mentions "weighted MSE function, utilized in previous quantization
   works"). We use plain MSE. The weighting may explicitly
   down-weight the compounded-noise directions.
3. **Paper's TuneFP batch size = 4** (ours matches) but with longer
   training. More absorption, less LB-ADMM reintroduction.
4. **Global TuneScalesKD (Phase 3)** is the final compensator that
   fixes the residual errors after all the block-wise work. We
   haven't implemented this. Possibly the main fix.

### Where to go next (ordered by likely impact)

1. **Implement TuneScalesKD (Phase 3)** — the paper's end-stage global
   KL fine-tune. Freeze binary signs, tune only scales s1 / s2 across
   the whole model against FP logits via KL. This is the only
   component of Algorithm 1 we haven't built, and it directly targets
   compositional error rather than per-block. Expected biggest
   single-step win.
2. **If TuneScalesKD doesn't close the gap**, try paper's lr=1e-5
   with more steps (1000–2000) for both TuneFP and STE — matches
   paper's effective budget more closely.
3. **Add weighted MSE** per Appendix C mention. Implementation
   unclear from paper; probably Hessian-diagonal-weighted from the
   same K-FAC preconditioners.

### Concrete honest takeaways

- TuneFP as **implemented** (paper-faithful to my reading of
  Algorithm 1 + Appendix C) underperforms the simpler Phase 2
  pipeline at our constrained budget. Not a critique of the paper —
  at their 1024-step-per-block, n=128, 8-epoch-cosine-LR budget on
  an H100, it may well work. At our budget it doesn't.
- The "student-input chain" training approach creates a noise floor
  that's absent in pure teacher-forcing. Any comparison of
  LB-ADMM+STE variants must hold this constant.
- Per-block MSE is MISLEADING as a proxy for PPL when measured on
  different input distributions. Phase 2 no-TuneFP has low MSEs but
  noisy deployment; Phase 2 with TuneFP has high MSEs (reflecting
  realistic input) but still worse deployment. PPL is the only
  reliable final metric.

### Files landed this step

- `tune_fp.py` — TuneFP training loop, fully-trainable block params,
  cosine LR schedule, batch-sampled steps.
- `phase2.py` — `quantize_model_with_tunefp` orchestrator chaining
  TuneFP → LB-ADMM init → STE → forward update per block.
- `run_phase2.py` — new flags: `--tune-fp-steps`, `--tune-fp-lr`,
  `--tune-fp-batch`. Calls `phase2.quantize_model_with_tunefp` when
  `tune_fp_steps > 0`, else falls back to `phase1.quantize_model_phase1`.
- Results entry id `2026-04-23T21:31:01Z-qwen-qwen3-4b-phase2-tunefp-lbadmm-ste`.

### Next

Implement TuneScalesKD (Phase 3) per paper spec.

---

## 2026-04-24 — Paper vs implementation audit (mid-Fisher+late-block run)

Best result so far: Fisher-weighted MSE full pipeline, Qwen3-0.6B r=512 at Phase 3 step 500 → **PPL 135.20**. Paper claims 27.56 at the same bpw — ~5× gap. Triggered a structured audit comparing arXiv 2602.06694v1 PDF against our code. Findings ranked by likely contribution to the gap:

### Deviations to fix (actionable)

1. **Wrong target for TuneFP and STE (HIGH).** Paper Algorithm 1 lines 8–9 computes `Y ← B*_b(X)` fresh each block, where X is the student-side partially-quantized input. We use the cached teacher→teacher boundary at `phase2.py:171` (`Y_b = cache.load_boundary(b + 1)`). The comment there claiming "loss would be trivially zero" is wrong — the block is still FP at TuneFP start, so `block(X_student)` ≠ `B*_b(X_student)` exactly because X_student carries upstream drift. This breaks Step 1's error-propagation-mitigation objective and Step 3 STE's Eq. 10 simultaneously.
   **Fix:** recompute `Y_b = FP_block(X_student)` (detached snapshot) before TuneFP starts each block, and reuse that snapshot for STE too.

2. **TuneFP runs 4× too long (HIGH).** Appendix C says 8 epochs, batch=4, n=128 → 256 steps. We run `--tune-fp-steps 1024` = 32 epochs. Overtraining likely drifts FP weights away from the point LB-ADMM can cleanly sign.
   **Fix:** default `--tune-fp-steps` to n_calib * 8 / batch = 256 (derive from other args).

3. **TuneFP trains RMSNorm weights (MEDIUM-HIGH).** `tune_fp.py:134-136` unfreezes every block parameter. Paper's "full-precision weights" in standard PTQ lineage means Linear weights only — RMSNorm/LayerNorm shouldn't move. Our own `feedback_ste_training_gotchas` memory already flagged this exact failure mode in a prior sprint.
   **Fix:** filter to `nn.Linear` weights only, like `quant_params` does for the STE step.

4. **Fisher ≠ paper's weighted MSE (MEDIUM).** Paper's weighted MSE cites DBF (Boža & Macko 2025) which uses `‖D_out · (Y_t − Y_s)‖_F²` — the same D_out we already compute in `preconditioner.py`. Our `fisher_diag.py` computes `E[(∂L_CE/∂h_b)²]` at the residual-stream level, on a different axis (d_model vs d_out of each Linear). One-line reuse replaces a full backward pass.
   **Fix:** retire `fisher_diag.py`; in TuneFP/STE loss, weight each Linear's output-contribution by its `D_out`.

5. **ρ schedule endpoints unverified (MEDIUM).** Paper only says "linear schedule, K=400" — no values. Our `rho_start=0.1 rho_end=10.0` in `admm.py` are guesses. Thm 3 in the paper requires ρ > L_f, and L_f scales with `τ_max²·‖W‖₂` (Corollary 2) — `ρ_start=0.1` is probably below the threshold early on.
   **Fix:** sweep ρ endpoints, or compute a data-driven lower bound from the first-iteration residual.

6. **RobustDiag uses single-shot percentile (LOW-MEDIUM).** Paper (Appx B.2 Lemma 1) specifies `τ_t = max(τ_{t-1}, q_t)` — cumulative-max across calibration forward. We use one-shot `torch.quantile` at `preconditioner.py:203-206`. Our comment acknowledges the simplification. Likely minor.
   **Fix:** optional, low priority.

### Non-deviations (confirmed matching)

- KL direction: `F.kl_div(log_s, log_t, log_target=True)` = `KL(teacher ‖ student)` = paper's Eq. 11.
- STE clipped-identity `|x|<1`: not specified in paper, matches cited baselines (LittleBit, OneBit, DBF).
- Rank uniform across Linears: matches paper's Figure 2 / Table convention.
- Quantization scope: Linears under `.layers` only (embed + lm_head skipped) matches paper Section 3.
- Calibration: 128 × 2048 WikiText-2 train seed=0 matches Appx C.
- Cosine LR schedule with unspecified `eta_min` — we use `lr*0.1`; paper unspecified, inconsequential.

### Instrumentation to add: block-level KL

**Why:** Our Phase 2 block-MSE data shows late blocks accumulate error at ~25% per-block TuneFP absorption. But MSE isn't the thing we care about — it's the *effect on the teacher-vs-student logit KL*. We can run a per-block ablation: with the full quantized student, restore block `b`'s FP weights (or zero out its quantization delta) and measure the KL change. This tells us which blocks are actually load-bearing for global quality.

**How:**
- Script: `diag_block_kl.py` (new file).
- For each block 0..N-1: save current quantized state → restore FP weights for block b only → forward on 32 calibration samples → KL against teacher → restore quantized → next.
- Output: a tensor `dkl_per_block[b]` = (KL with block b FP) − (KL with everything quantized). Negative = that block was hurting KL; the more negative, the more hurt it caused.
- Cost: N+1 forwards × 32 samples × seq=2048 ≈ 5 min on 0.6B.

**Uses:**
- Cross-reference with Phase 2 block-MSE: do blocks with high MSE actually dominate KL, or is the error surprisingly distributed?
- Decide if late-block lever is targeting the right blocks.
- Decide if mid-block dynamic rank (higher rank for load-bearing blocks) is worth implementing.

### Priorities

The audit suggests the gap is mostly fixable. Order of attack once the current Fisher+late-block run completes:

1. Fix TuneFP/STE target (#1) — cheap, probably biggest single win.
2. Cut TuneFP steps to 256 (#2) — cheap, frees ~75% of Phase 2 time for more runs.
3. Freeze RMSNorm in TuneFP (#3) — one-liner.
4. Add block-level KL diagnostic — informs whether late-block lever is worthwhile at all.
5. Swap Fisher for D_out-weighted MSE (#4) — retires a bespoke file for one-line reuse.
6. ρ-endpoint sweep (#5) — lower priority.

If the current run lands near Fisher-only's 135.20 — as expected given #1/#2 are unchanged — it confirms the audit reading: our current lever additions can't beat the target-mismatch ceiling. Fixing #1 should be the next run.

---

## 2026-04-24 (later) — Three post-audit experiments. All killed. Audit vindicated.

Three back-to-back runs under the same unfixed pipeline (target-mismatch + 4× overtrained TuneFP still present). Goal was A/B testing the levers we'd already shipped (Fisher, late-block, lr adjustment, save-best-KL). Result: **none of them beat Fisher-only's PPL 135.20, and the signals strongly suggest no lever built on top of the current pipeline can**.

### Save-best-KL landed first

- `tune_scales_kd.py` now tracks a 50-step windowed mean of per-step KL, saves to `*.phase3_best.pt` on new min. Throttled to `log_every` cadence so I/O stays bounded.
- `TuneScalesKDStats` gained `best_windowed_kl`, `best_step`.
- Trace JSON and results-log `method.params` reflect both.

### Experiment 1: Phase-3-only lr=3e-7 resume (killed at step ~100)

Motivated by my earlier (wrong) story that the baseline and Fisher runs were overshooting their best KL. Resumed from Fisher's `post_phase2.pt`, dropped lr 1e-6 → 3e-7, turned on save-best. At step 91:

- Best windowed-mean KL: **7276.59** — significantly *worse* than prior Fisher at matched step 100 (windowed mean 6814).
- Per-step KL range 6055-8368 across last 20 steps — noise envelope. No trend.

Then I went back and **recomputed the prior Fisher run's windowed means** from its saved `step_kls`:

- First-50-step mean: 7364.93
- First-100-step mean: 6814.59
- **Best windowed mean: 4027.78 at step 495**
- **Final windowed mean: 4065.11 at step 500** (only 38 units above best)

The "overshoot" I'd attributed to the baseline and Fisher runs was an illusion. I'd been reading single-sample KL values (last≈4218 vs min≈2722 for Fisher) and calling the 35% gap overshoot — but windowed mean shows near-monotonic convergence, 1% separation between best and final. **Lowering LR doesn't fix a problem that didn't exist**; it just trains slower.

Killed Experiment 1. Takeaway: save-best-KL is a small-payoff defensive measure, not a lever. Useful if we later actually do see overshoot (e.g., after fixing the target, with sharper loss landscape) but meaningless right now.

### Experiment 2: Fisher + late-block 2× on last 3 blocks (killed at step 21 of Phase 3)

The theory: late-block 2× TuneFP/STE budget should do extra work on blocks 25-27, where Fisher weighting is already 30-50× concentrated. Expectation was ~15-20% PPL improvement from a direct attack on the error-accumulation tail.

Full pipeline ran. Phase 2 results on the late blocks:

| Block | Fisher-only STE final | Fisher+late2× STE final | Gain |
|-------|----------------------|-------------------------|------|
| 25    | 43.98                | 43.27                   | +1.6% |
| 26    | 49.09                | 47.73                   | +3%   |
| 27    | **25.11**            | **22.33**               | **+11%** |

Block 27 **improved 11%** on its own MSE metric. Encouraging at first glance.

Then Phase 3 init KL arrived (measured on fixed sample 0, so deterministic and cross-run comparable):

| Run               | Phase 3 init KL |
|-------------------|-----------------|
| Baseline          | 7,604.67        |
| **Fisher-only**   | **6,141.55**    |
| Fisher+late2×     | 6,391.23        |

**Fisher+late2× is 4% *worse* than Fisher-only at end-of-Phase-2 KL, despite local block-27 MSE being 11% better.**

This is the MSE-vs-KL disconnect the Phase 1 rank-ladder analysis already hinted at — block MSE varies 3.4× while PPL varies 112×. Here the late-block 2× budget extracted local MSE wins by drifting weights off the residual-stream manifold in ways the lm_head sees but block-MSE doesn't.

Also worth noting: this is exactly deviation #1's predicted failure mode. When you train against the wrong target (cached teacher→teacher rather than `B*_b(X_student)` recomputed each block), giving more gradient steps against the wrong target produces better fit *on that wrong target*. Global KL doesn't reward it.

Killed Experiment 2. Takeaway: **local block-MSE is no longer a decision-useful metric for levers in this pipeline.** Until the target is fixed, MSE-guided interventions are as likely to hurt as help global quality.

### The cross-run metric we can trust: Phase 3 init KL

What emerged as actually useful across these three runs is **Phase 3 init KL measured on fixed sample 0** (`tune_scales_kd.py` line near the `init KL = ...` print). It's:

- **Deterministic** given the same post-Phase-2 state — confirmed by the lr=3e-7 resume matching Fisher-only's 6141.55 to 6 sig figs after loading the same checkpoint.
- **Cross-run comparable** — same sample, same teacher, same init procedure.
- **A direct measure of end-of-Phase-2 quality** — what happens before any Phase 3 scale tuning runs.
- **Decoupled from Phase 3 LR or budget choices** — lets us isolate Phase 2 gains from Phase 3 gains cleanly.

Should be the primary metric for all Phase 2 interventions going forward, over block-MSE.

### Monitor script worth keeping

During Experiment 2 I used `Monitor` with a poll-loop that grepped the log for phase transitions, step progress, ckpt mtimes, and error signatures, emitting state-change notifications every 30 min. Filter pattern covered Traceback / CUDA error / OOM / FAILED / Killed so silence wasn't success. Worked well and didn't over-fire; ~1 event per 20-30 min of real state change. Keep the pattern for long runs.

### Bottom line

Three experiments × ~5 hours of GPU each, and **best PPL still sits at Fisher-only's 135.20**. The audit was right: the current pipeline has a ceiling that no tuning of its existing knobs can break. The next run must fix deviations #1 (target), #2 (TuneFP budget), #3 (RMSNorm freeze). Those are the path to closing 135 → 27; everything else is noise at this layer.

### Next

1. Fix #1/#2/#3 together (estimated 1 hour of code work, likely one commit).
2. Add `diag_block_kl.py` — per-block KL ablation — so the next run gets a real global-KL readout per block, not just block-MSE.
3. Rerun full pipeline. If init KL drops substantially (target fix is predicted to be the biggest Phase-2 improvement), we'll see it immediately at Phase 3 step 0.

---

## 2026-04-24 (later still) — Audit fixes implemented. Predictions empirically wrong.

Implemented the three audit fixes and clarified fix #1's scope via a follow-up agent read of Algorithm 1:

- **Fix #1 (revised):** STE target changed to `B(X_student)` where `B` is the post-TuneFP tuned FP block (paper Eq. 10). Recomputed fresh per block just before LB-ADMM binarizes. TuneFP target kept as cached teacher-teacher boundary — paper Algorithm 1 line 9 literally specifies `Y* = B*_b(X_student)` as a shared target, but at block level this yields zero initial TuneFP loss. The paper's Algorithm 1 is ambiguous here; a faithful reading implies per-layer interleaving within LB-ADMM, which we don't implement. Block-level pragmatic target preserved TuneFP signal.
- **Fix #2:** `--tune-fp-epochs` CLI flag added to `run_phase2.py`. When set, derives steps as `n_calib * epochs / batch`. Paper's 8 epochs at n=128 batch=4 = 256 steps (vs our previous 1024 = 32 epochs).
- **Fix #3:** `tune_fp.py` now unfreezes only `nn.Linear` weights/biases; RMSNorm stays frozen. Matches QuaRot/GPTQ lineage and avoids the residual-stream scale drift flagged in feedback memory.

Smoke test passed end-to-end. Launched full Fisher pipeline with all three fixes at paper-faithful budget. Monitor caught the first meaningful pipeline event, the derived budget:

```
[tunefp-budget] epochs=8 n_calib=128 batch=4 -> 256 steps
```

### Phase 2 wall time halved

Pre-fix Fisher: 137 min (8142s). Post-fix: **71 min (4263s)**. The 4× TuneFP budget cut accounts for most of the savings — each block now does 256 TuneFP steps instead of 1024. Confirms deviation #2 was real bloat, not load-bearing.

### Per-block trajectory: TuneFP init ~25% higher throughout

Post-fix block-by-block TuneFP init MSEs are ~1.2–2.9× the pre-fix values through blocks 3–26. Absorption ratio drops from pre-fix's 20–25% to post-fix's 10–17% (less budget → less absorption, expected). Block 27 anomaly persists: tunefp 109.15→13.13 (88% absorption) vs pre-fix 103.70→7.82 (92%). Ratio tightens with depth — by block 26 only 1.21× higher init.

STE finals are not directly comparable across runs (target changed). They're all small (0.05–0.15 middle blocks) because Eq. 10's target `tuned_FP(X_student)` is easy to match — LB-ADMM init already gets close.

### The prediction that broke

Audit predicted: correcting the STE target would substantially improve Phase 3 init KL and final PPL. Prior Fisher ran with "wrong" STE target + overtrained TuneFP + trainable RMSNorm → PPL 135.20. We expected post-fix to beat it.

Measured instead:

| Config | Post-Phase-2 PPL | Phase 3 init KL | Phase 2 wall |
|--------|-----------------:|----------------:|-------------:|
| **Pre-fix Fisher** | **1156.59** | 6141.55 | 137 min |
| Post-fix Fisher | 1449.05 | 7249.85 | 71 min |

**Post-fix is 25% WORSE at post-Phase-2 PPL and 18% worse at init KL.** Killed the run early rather than burn the ~8h Phase 3 to confirm the projection; direct post-Phase-2 PPL eval is sufficient.

### Why the audit's prediction failed

Interpretation — the pre-fix pipeline was *accidentally* acting as a global-KL regularizer via three interacting "bugs":

1. **STE against cached teacher-teacher boundary** asks the binary block to reproduce the teacher's *clean-input output* while receiving drifted input. Literally impossible to minimize this loss to zero, but the unsatisfied gradient pressure pushes the binary block's behavior to *compensate* for upstream drift — a form of implicit error-correction not present in Eq. 10's local objective.
2. **4× overtrained TuneFP** amplifies (1) because each block's FP weights have more opportunity to absorb drift before binarization. Eq. 10's paper-faithful STE target sees this fully-tuned FP block as its target — which is "cleaner" but discards the drift-compensation pressure.
3. **Trainable RMSNorm** adds another dimension of freedom for TuneFP to compensate with.

Fix #1 individually removes the most important of these (STE target → Eq. 10), which the paper describes as the theoretically correct objective. In isolation, it *is* the cleaner local objective — matching what the FP block produces on drifted input. What it loses is the implicit global regularization that came from forcing the binary block to fight the drift, not just reproduce the FP block's already-drifted response.

This is a concrete example of the "paper-replication discipline" feedback memory: a bug that works as well or better than the paper-faithful version, for reasons the paper's theoretical framing doesn't surface. The audit correctly identified the deviation; it was wrong that the deviation was load-bearing for quality in our direction.

### Concrete honest takeaways

- **The audit was right about WHAT the paper says. It was wrong that fixing made things better for this model.** The three pragmatic deviations compound into an implicit regularization that our paper-faithful replacement doesn't have.
- **Don't trust audit severity rankings without an actual measurement.** We assumed deviation #1 was the biggest single win; it was actually a regression.
- **Per-block MSE continues to be misleading.** Phase 2's per-block trajectories in post-fix look fine (consistent, small, monotonic-ish) — but global KL says the blocks are worse aligned.
- **Post-Phase-2 PPL is now our cross-run quality metric**, alongside init KL. Lets us A/B without waiting 8h for Phase 3 to finish. Should add `--eval-post-phase2` as a built-in pipeline flag for future runs.

### What's preserved

- `post_phase2_prefix.pt` (pre-fix Fisher Phase 2 state) — still on disk for future diag_block_kl.py experiments.
- `post_phase2.pt` (post-fix Fisher Phase 2 state) — preserved too.
- Three fix patches are in `tune_fp.py`, `phase2.py`, `run_phase2.py`; `diag_block_kl.py` exists as a new file. None reverted yet — they're available under CLI flags / default-off where applicable.

### Artifacts by run

**Pre-fix Fisher run** (the earlier run that reached PPL 135.20 at Phase 3 step 500):
- Full-pipeline log: [fullpipeline_fisher.log](fullpipeline_fisher.log)
- Diagnostic log (Fisher concentration stats): [diag_fisher.log](diag_fisher.log)
- Phase 3 lr=3e-7 retry log (killed at step ~100): [fullpipeline_fisher_lr3e-7.log](fullpipeline_fisher_lr3e-7.log)
- Post-Phase-2 PPL eval log: [eval_prefix_post_phase2.log](eval_prefix_post_phase2.log)
- Phase 3 step-500 PPL eval log: [eval_fisher_step500.log](eval_fisher_step500.log)
- Post-Phase-2 checkpoint: [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.post_phase2_prefix.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.post_phase2_prefix.pt)
- Phase 3 ckpt at step 500: [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.phase3_progress_lr1e-6_step500.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.phase3_progress_lr1e-6_step500.pt)
- Phase 3 lr=3e-7 best-KL ckpt (step 91): [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.phase3_best_lr3e-7_step91.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.phase3_best_lr3e-7_step91.pt)
- Per-block status JSON: [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.status_prefix.json](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.status_prefix.json)

**Fisher + late-block 2× run** (killed at Phase 3 step 21, confirmed MSE-vs-KL disconnect):
- Full-pipeline log: [fullpipeline_fisher_late.log](fullpipeline_fisher_late.log)
- Post-Phase-2 checkpoint: [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher_late3x2.post_phase2.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher_late3x2.post_phase2.pt)
- Status JSON: [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher_late3x2.status.json](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher_late3x2.status.json)

**Post-fix Fisher run** (this entry — audit fixes #1/#2/#3 applied):
- Full-pipeline log: [fullpipeline_postfix.log](fullpipeline_postfix.log)
- Smoke test log (validated pipeline before full run): [smoke_post_fix.log](smoke_post_fix.log)
- Post-Phase-2 PPL eval log: [eval_postfix_post_phase2.log](eval_postfix_post_phase2.log)
- Post-Phase-2 checkpoint: [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.post_phase2.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.post_phase2.pt)

**Baseline (no Fisher, no fixes, paper Appendix C budget)** — the 389.59 PPL reference:
- Resume log (the live run that produced 389.59): [fullpipeline_resume.log](fullpipeline_resume.log)
- Full-pipeline log (earlier Phase 2 that produced the post-Phase-2 ckpt later reused by the resume): [fullpipeline_full.log](fullpipeline_full.log)
- Post-Phase-2 checkpoint: [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0.post_phase2.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0.post_phase2.pt)
- Eval at Phase 3 step 200 (PPL 389.59): [eval_step200.log](eval_step200.log)

**Supporting artifacts (cross-run):**
- Results DB: [results.json](results.json)
- K-FAC preconditioners (cached, reused across runs): [cache/qwen-qwen3-0-6b/n128_L2048_seed0/preconditioners.pt](cache/qwen-qwen3-0-6b/n128_L2048_seed0/preconditioners.pt)
- Fisher diagonal (cached): [cache/qwen-qwen3-0-6b/n128_L2048_seed0/fisher_diag.pt](cache/qwen-qwen3-0-6b/n128_L2048_seed0/fisher_diag.pt)

### Next

Three options, ranked by leverage vs time:

1. **Ablate each fix individually** (3× Phase-2-only runs at ~75 min each with `--tune-scales-steps 0`). Gives clean attribution — which of #1/#2/#3 actually hurt, and by how much. Useful if we ever want to revisit a "cleaner" pipeline from a different angle.
2. **Revert all three, accept pre-fix config as baseline PPL 135.20, invest elsewhere.** Either Phase 3 improvements (LR schedule, best-KL + early stopping), or new Phase 2 knobs (per-block rank, different loss weighting than Fisher/D_out, etc.).
3. **Partial revert guess.** Keep whichever fix we suspect is neutral/helpful, revert the others. Risky without ablation data.

Going with (1) to get definitive attribution before committing direction. After the three ablations land, we'll have a proper understanding of which knob was load-bearing and can design the next lever from data.

---

## 2026-04-24 (later still 2) — Drift review + full paper deviation audit

Context: post-fix Fisher run showed drift accumulating badly in the second half of blocks. Reviewed per-block data from both status JSONs and discovered:

- Per-block drift-amplification ratios are **the same** pre-fix and post-fix (~1.35-1.41× per block in the second half). Structural to the residual stream.
- Post-fix enters the second half from a **2.2× higher drift floor** at block 14 (init_loss 1.96 vs pre-fix 0.90).
- Two compounding causes: (A) TuneFP absorption halved by the 4× budget cut — each block passes ~90% of upstream drift instead of ~80%, compounding to a 20× difference across 26 blocks; (B) Eq. 10 STE target has ~2-unit gap (easy) vs pre-fix ~10-unit gap (impossible), so post-fix STE no longer applies the implicit drift-correction gradient pressure that pre-fix accidentally benefited from.

With that diagnosis in hand, re-read the NanoQuant paper (arXiv:2602.06694v1) end-to-end via agent and cross-referenced against every file in the pipeline. Findings organized by whether they're likely contributing to the second-half drift.

### Deviations — HIGH (plausibly explain the drift)

**D1. STE target recomputed AFTER TuneFP, not shared with TuneFP ([phase2.py:197-199](phase2.py)).** Paper Algorithm 1 lines 7-9 compute `Y* ← B*_b(X)` **once** (teacher block on student input), then line 11 TuneFP and line 18 TuneLatentSTE **share** that Y*. Our code recomputes `Y_b_ste = compute_teacher_on_student(block, X_student, ...)` AFTER TuneFP runs — so STE sees `tuned_FP(X_student)` as its target, not `teacher(X_student)`. Consequence: STE fits a post-TuneFP block that has already absorbed drift, and STE itself gets no drift-correction gradient signal. This is the direct mechanism for the second-half drift: fix #1 accidentally removed a regularization that the paper's shared-Y* protocol was supposed to preserve.

**D2. TuneFP target is cached teacher-to-teacher boundary, not `B*_b(X_student)` ([phase2.py:169](phase2.py)).** Already flagged, partially reverted. The comment's "zero initial loss" argument is literally correct at block level — which points to an unresolved paper ambiguity: Algorithm 1 Step 1's prose ("errors introduced by ... previously factorized layers in the current block") suggests per-layer TuneFP/LB-ADMM interleaving, not block-level. Our block-level implementation substitutes a target to keep gradient signal.

**D3. Weighted MSE = Fisher on residual stream, not `D_out` on per-Linear output channels ([fisher_diag.py](fisher_diag.py) vs [preconditioner.py:196](preconditioner.py)).** Paper cites DBF (Boža & Macko 2025): weighted MSE is `‖D_out·(Y_t − Y_s)‖²` per-Linear. Ours weights the `d_model` residual-stream dimension once per block. Different axis, different statistic. Audit #4, not fixed.

### Deviations — MEDIUM (structural but not obviously drift-related)

**D4. K-FAC preconditioner normalized to mean=1 in LB-ADMM ([admm.py:97-106](admm.py)).** Paper Eq. 15: raw `W̃ = D_out · W · D_in`. Our code divides `D_in` and `D_out` by their means before forming `W̃`. Documented as a fix for gradient vanishing when `D_out` magnitudes were ≈1e-4. Structurally deviates from paper; possibly masks an upstream estimator mis-scale.

**D5. RobustDiag uses single-shot percentile, not cumulative-max ([preconditioner.py:203-206](preconditioner.py)).** Paper Appx B.2 Lemma 1 specifies `τ_t = max(τ_{t-1}, q_t)`. Ours uses `torch.quantile` one-shot. Audit #6, low priority.

**D6. ρ schedule endpoints unverified ([admm.py:74](admm.py)).** Paper: "linear, K=400", no endpoints. Ours: `rho_start=0.1, rho_end=10.0`. Paper's Thm 3 requires `ρ > L_f` per step. `ρ_start=0.1` may not satisfy that early on. Audit #5.

### Verified matching (non-deviations)

K=400 LB-ADMM iterations ✓; TuneFP 256 steps / lr=1e-4 / batch=4 ✓; STE lr=1e-5 ✓; STE trains (U, V, s1, s2) ✓; Phase 3 lr=1e-6 / scales-only ✓; Phase 3 KL direction = forward KL from teacher ✓; calibration 128×2048 seed=0 ✓; RMSNorm frozen ✓; Ledoit-Wolf shrinkage on D̃ ✓; SVID ✓; magnitude balancing Eq. 7-9 ✓.

### Paper ambiguities (unresolvable from text)

STE backward form (plain vs clipped-identity `|x|<1`) — ours clipped; block-level vs per-layer structure of Step 1+2; ρ endpoints; "weighted MSE" exact form (paper punts to DBF / Kim 2025); ADMM λ value.

### Priority

D1 is the new finding that directly maps to the drift mechanism. Implementing first. Change is ~10 lines in [phase2.py](phase2.py): compute `Y_b = compute_teacher_on_student(block, X_student, ...)` once BEFORE TuneFP, pass the same `Y_b` to both `tune_fp` and `train_block`. This supersedes the current ablation plan — if D1 recovers the drift-correction pressure via the paper-faithful mechanism, ablations become less urgent.

Open item: at block level, paper-faithful D1 gives TuneFP zero initial loss (block still equals teacher, target = teacher-on-student = block(student)). TuneFP would be a no-op. That's acceptable — STE is the stage we want to fix. If we then want TuneFP to do real work, we'd need to address D2 by either restructuring to per-layer interleaving or substituting a different non-teacher target.

### Next

1. Log audit (this entry).
2. Implement D1 in `phase2.py`.
3. Run post-D1 Phase 2 + post-Phase-2 PPL eval. Compare against pre-fix PPL 1156 and current post-fix 1449.
4. Decide D2 direction based on D1 result.

---

## 2026-04-24 (later still 3) — D1 empirically regresses 500×; paper-faithful audit branched off master

Implemented D1 and ran full Qwen3-0.6B Fisher pipeline. D1 regresses catastrophically. Code preserved on branch `research/nanoquant-d1-shared-y-star` (tip commit `0fd4e208b`); master reverted to pre-audit-fixes state (`1ac0d3c89`).

### D1 implementation

- [phase2.py](phase2.py): `Y_b = compute_teacher_on_student(block, X_student, ...)` computed ONCE on the pristine FP (teacher) block BEFORE TuneFP. Same `Y_b` passed to both `tune_fp` and `train_block`. Cached teacher-to-teacher boundary no longer used in Phase 2.
- [tune_fp.py](tune_fp.py): docstring updated to note the shared-`Y*` protocol and the block-level zero-gradient caveat.
- Smoke test on Qwen3-0.6B n=4 rank=32 steps=20 confirmed wiring: all 28 blocks reported `tunefp 0.0000->0.0000` (target == block output at init), STE init_mse in "impossible target" regime (block 27 init_mse = 215.7 vs post-fix 33.3).

### Full run results

Full pipeline Qwen3-0.6B with Fisher: rank=512, K=400, 1024 STE steps/block, 8 TuneFP epochs (256 steps), Phase 3 500 steps lr=1e-6. Phase 2 completed in 70.7 min (matches post-fix 71 min). Phase 3 killed at step 146 of 500 after trajectory confirmed a plateau well above baselines.

| config | post-Phase-2 PPL | Phase 3 init KL | best Phase 3 PPL | notes |
|--------|------------------:|-----------------:|-----------------:|-------|
| pre-fix Fisher | 1,156 | 6,142 | 135.20 @ step 500 | baseline |
| post-fix Fisher (audit #1/#2/#3) | 1,449 | 7,250 | — (killed) | +25% post-P2 |
| **D1 Fisher** | **588,507** | **22,531** | **19,767 @ step 146** | **509× post-P2, 146× vs pre-fix final** |

### The prediction that broke (the other way this time)

I predicted D1 would recover drift-correction pressure that fix #1 accidentally removed. The smoke test data looked correct: STE was visibly doing work (70% reductions vs pre-fix's 5%), per-block MSE growth looked similar across configs. Pre-fix's "STE barely moves" I interpreted as unsatisfied gradient pressure — the paper-faithful shared-`Y*` target would let STE fully converge while still pulling binary toward teacher's distribution.

Measured: the opposite. STE's strong convergence on the paper-faithful target **preserves** drift rather than fighting it.

### Why D1 preserves drift — the drift-amplification mechanism

Consider block b with student input `X_s[b]` (drifted from teacher input `X_t[b]` by accumulated upstream quantization error). Call the drift `δ[b] = X_s[b] − X_t[b]`. Each block in each config optimizes:

- **Pre-fix:** `min‖binary_block(X_s[b]) − teacher_block(X_t[b])‖²` — target is pristine teacher output on TEACHER input. Unreachable (binary can only process `X_s[b]`, can't reverse upstream drift on arbitrary input). STE residual gradient pushes binary's output partway toward the teacher manifold, effectively subtracting a component of `δ`. Downstream, `X_s[b+1]` is a mix of "teacher behavior on X_s[b]" and "pulled toward clean teacher output". Per-block drift growth: bounded below teacher-Jacobian rate.
- **D1:** `min‖binary_block(X_s[b]) − teacher_block(X_s[b])‖²` — target is teacher's response to DRIFTED input. Reachable in principle (binary ≈ teacher is the whole quantization goal). STE converges strongly (~70% reduction). Downstream, `X_s[b+1] ≈ teacher_block(X_s[b])`. Linearizing teacher around `X_t[b]`: `teacher_block(X_t[b] + δ[b]) ≈ teacher_block(X_t[b]) + J_b · δ[b]`. So `δ[b+1] ≈ J_b · δ[b]` — drift propagates through the teacher's Jacobian at FULL amplification.
- **Post-fix:** similar to D1 but with TuneFP having absorbed some drift into the FP weights first, so the target is slightly different from pristine teacher-on-student. Drift propagates at near-full Jacobian rate with a constant-factor attenuation.

Transformer Jacobian norms at typical input scales are `||J_b|| ≥ 1` (residual connections contribute identity; MLP + attention contribute magnitude > 1 on the directions that carry signal). Over 28 blocks, `Π ||J_b||` compounds geometrically. Pre-fix's drift-cancellation subtracted a component each block, bringing effective per-block amplification below 1 on average. D1 does not — drift grows at full teacher-Jacobian rate every block.

### Empirical evidence for the mechanism

D1 per-block STE init_mse (pre-STE LB-ADMM error measured against `teacher(X_s[b])`):
| block | D1 | pre-fix | post-fix |
|------|-----|---------|----------|
| 14 | 0.87 | 0.93 | 0.15 |
| 20 | 5.35 | 10.86 | 2.85 |
| 27 | 119.42 | 42.66 | 33.29 |

Counter-intuitive: D1 init_mse at block 14 is comparable to pre-fix, but at block 27 it's **2.8× larger**. Pre-fix's drift-cancellation was kicking in at middle-to-late blocks — by block 27, pre-fix's `||X_s||` is smaller than D1's because each block subtracted a drift component. D1's `||X_s||` grew geometrically via the teacher Jacobian, so the same LB-ADMM relative factorization error (`ε_LB-ADMM · ||W||₂ · ||x||`) produced much larger absolute errors.

Per-block STE init_mse growth ratio (B14→B26 average):
- pre-fix: 1.41× per block
- post-fix: 1.35× per block
- D1: **1.46× per block** (highest)

D1's ratio is highest. If `||J||_avg ≈ 1.4` in the latter half, D1 tracks that exactly. Pre-fix undershoots by ~0.05 per block; over 12 blocks that's `1.05^12 ≈ 1.8×` less drift at block 26. That matches the observed pre-fix init_mse 42.66 vs D1 119.42 at block 27 (ratio 2.8, close to the predicted factor once you include LB-ADMM scaling with `||x||`).

### Why local fidelity ≠ global fidelity

Paper's Eq. 10 is the locally correct objective: each block should reproduce teacher's mapping. For a composition `f = f_28 ∘ f_27 ∘ … ∘ f_1`, local fidelity `f̂_b ≈ f_b` should in theory compose to `f̂ ≈ f`. This fails for two reasons at binary precision:

1. **Input domain shift.** `f̂_b` is optimized over the student input distribution `X_s[b]`, which differs from `X_t[b]` by the accumulated drift. But `f̂_b` only gets trained on a 128-sample calibration — it's not a global function approximation, it's a local fit. The "approximation" is valid only in a neighborhood of the training inputs. At block 27, if `X_s[27]` has drifted far from `X_t[27]`, the training distribution is also drifted, so `f̂_27` converges to approximate `f_27` **on the drifted distribution** — which is precisely what we asked for, but the test distribution at inference time is whatever `X_s[27]` happens to be, and if that distribution is fatter-tailed than calibration, generalization breaks.

2. **Error composition under bounded STE capacity.** Even if `||f̂_b − f_b|| ≤ ε_b` uniformly, composition yields error bounds `||f̂ − f|| ≤ Σ Lip(f) · ε_b` — growing linearly at best, geometrically if Lipschitz constants exceed 1. Pre-fix's "impossible target" biased STE toward a different point in its capacity ball — specifically the point closest to the teacher-on-clean-input manifold, which happens to compose multiplicatively BETTER even though per-block it's strictly worse.

### Lessons

1. **Paper-faithful is not always quality-faithful.** The audit correctly identified 3 deviations from the paper. All three, when fixed, regressed quality. D1 is the cleanest example — literally implementing what Algorithm 1 line 9 says produced 500× worse PPL than the pre-fix "bug". The paper's protocol likely works at their scale (H100, larger calibration, possibly different regularization pulled in via "weighted MSE") but breaks at ours. Replication discipline requires distinguishing "paper-faithful" from "paper-intent" — sometimes our bugs are the paper's intent by accident.

2. **Per-block MSE is misleading three ways.** The journal has noted this twice before but D1 produced the sharpest example yet. D1's per-block STE reductions (70%) looked like the pipeline was working **better**. It was working better at the local objective; the local objective is decoupled from global KL; and the composed network was the only thing that mattered for PPL.

3. **Implicit regularization can be load-bearing.** Pre-fix had two "accidental regularizers" working together: (a) impossible STE target → drift-cancellation gradient, (b) overtrained TuneFP → extra FP weight adaptation absorbing drift. Fix #1 removed (a); fix #2 removed most of (b). D1 doubled down on removing (a). All three operate on the same drift-control axis the paper's explicit mechanisms apparently control differently — possibly via the undefined "weighted MSE", possibly via budget.

4. **Post-Phase-2 PPL eval is the right cross-run metric.** It predicted D1's failure at 90 minutes instead of 8 hours. Every future Phase 2 lever should be A/B'd at post-Phase-2 PPL first. Add `--eval-post-phase2` as a built-in flag next time we revisit this pipeline.

### Artifacts preserved

- **Branch:** `research/nanoquant-d1-shared-y-star` @ `0fd4e208b` — paper-faithful audit fixes #1/#2/#3 + D1 bundled.
- **D1 Phase 2 ckpt:** [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.post_phase2.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.post_phase2.pt)
- **D1 Phase 3 best-KL ckpt (step 146, KL 12938):** [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.phase3_best.pt](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.phase3_best.pt)
- **D1 full pipeline log:** [fullpipeline_d1.log](fullpipeline_d1.log)
- **D1 Phase 2 per-block status:** [runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.status.json](runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0_fisher.status.json)
- **D1 PPL eval logs:** [eval_d1_post_phase2.log](eval_d1_post_phase2.log), [eval_d1_phase3_best.log](eval_d1_phase3_best.log)
- **Post-fix Fisher artifacts (preserved via rename):** `.post_phase2_postfix.pt`, `.status_postfix.json`
- **Pre-fix Fisher artifacts (earlier preservation):** `.post_phase2_prefix.pt`, `.status_prefix.json`

### Master state

Master code reverted to `1ac0d3c89`. Audit fixes #1/#2/#3 and D1 all live on the branch. To reproduce the paper-faithful pipeline (regressive): `git checkout research/nanoquant-d1-shared-y-star`. To reproduce pre-fix 135.20 baseline: stay on master.

### Next

With D1 ruled out and the paper-faithful direction empirically wrong at our scale, the research question shifts: **what direction is actually paying?**

Options to consider for the next session:

1. **Bank pre-fix 135.20, invest in Phase 3.** LR schedule, save-best + early stopping (already built), validation-window-based stopping, temperature-scaled KL. Phase 3 pre-fix ended at step 500 with windowed mean 4028, only 1% above the min at step 495 — likely close to its own limit, but worth verifying.
2. **Different Phase 2 loss.** Replace Fisher with the paper's actual D_out-weighted MSE (audit D3). This is the one unattacked axis that's paper-faithful but hasn't been tried.
3. **Per-block rank allocation.** Fisher diagnostic showed 10× concentration in late blocks. Rank-allocation that gives more rank to late blocks may directly reduce per-block factorization error, slowing drift propagation even under the current STE regime. This is an orthogonal lever to target/budget choices.
4. **Larger model.** Qwen3-0.6B has 28 blocks; the drift compounding is measurable here precisely because depth is modest. Qwen3-4B has 36 blocks and possibly better-conditioned Jacobians. If the drift mechanism is scale-sensitive, D1 might stop regressing at 4B — would bring replication closer to paper scale.

Option 3 is the highest-leverage low-risk move. Option 2 closes the last paper-audit deviation without re-introducing D1's mechanism. Option 4 is informative but takes real GPU time.
