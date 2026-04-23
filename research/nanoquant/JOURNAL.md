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
