# Experiment 01 — Hadamard Rotation Alone

**Question:** Does a single Hadamard residual-stream rotation, applied
before quantization, meaningfully reduce Qwen3-8B's 2-bit perplexity
gap vs FP16?

**Quant choice:** Q2_K (non-imatrix k-quant, ~2.6 bpw). The original
plan was Q2_K, but llama.cpp's i-quants require an imatrix, which
would confound the rotation signal (we'd be testing rotation + imatrix
jointly). Q2_K needs no imatrix, so rotation is the only variable.
Q2_K is also more aggressive than Q2_K, which amplifies any
rotation benefit rather than masking it.

**Scope:** Only the R1 (residual stream) rotation from QuaRot — no R2
attention-head rotation, no learned rotations, no SpinQuant. This is
the cheapest, most composable version. If R1 alone closes the gap,
the rest of the stack's burden drops sharply.

## Configurations compared

| ID  | Description                          | GGUF                                  |
|-----|--------------------------------------|---------------------------------------|
| A   | Baseline FP16                        | `gguf/qwen3-8b-f16.gguf`              |
| B   | Unrotated Q2_K (default quant)    | `gguf/qwen3-8b-q2k.gguf`           |
| C   | Hadamard-rotated Q2_K             | `gguf/qwen3-8b-rotated-q2k.gguf`   |

Metric: perplexity on `wikitext-2-raw-v1` test split (standard
llama.cpp perplexity eval, 512-token context windows).

Result contract:
- **A** is the reference — quantization cannot beat it, only approach it.
- **B − A** = the gap we're trying to close.
- **C − A** = the residual gap after Hadamard rotation.
- **B − C** = the Hadamard contribution.

Secondary comparison: Q2_K GGUF already on disk
(`Atlas/models/Qwen3-8B-Q2_K.gguf`) — sanity reference point at 2-bit.

## Pipeline

```
HF checkpoint (models-hf/Qwen3-8B)
  ├─► convert → f16.gguf ──► perplexity (A)
  │                      └─► quantize Q2_K → q2k.gguf → perplexity (B)
  └─► rotate.py ──► rotated-hf/ ──► convert → rotated-f16.gguf
                                       └─► quantize Q2_K → rotated-q2k.gguf → perplexity (C)
```

## How to run

From this directory, with the venv activated:

```bash
VENV="../../cross-layer-svd/venv-research/Scripts/python.exe"
BIN="../../../bin-upstream"
HF_IN="../../models-hf/Qwen3-8B"

# 0. Fetch wikitext test set (one-time)
$VENV fetch_wikitext.py

# 1. Baseline FP16 conversion
$VENV ../../../convert_hf_to_gguf.py $HF_IN --outfile gguf/qwen3-8b-f16.gguf --outtype f16

# 2. Baseline perplexity A
$BIN/llama-perplexity.exe -m gguf/qwen3-8b-f16.gguf -f data/wiki.test.raw \
  -ngl 18 --ctx-size 512 2>&1 | tee results/A_fp16_ppl.log

# 3. Unrotated Q2_K + perplexity B
$BIN/llama-quantize.exe gguf/qwen3-8b-f16.gguf gguf/qwen3-8b-q2k.gguf Q2_K
$BIN/llama-perplexity.exe -m gguf/qwen3-8b-q2k.gguf -f data/wiki.test.raw \
  -ngl 99 --ctx-size 512 2>&1 | tee results/B_q2k_ppl.log

# 4. Apply Hadamard rotation
$VENV rotate.py --in $HF_IN --out rotated-hf/

# 5. Sanity check — must show <1e-3 logit delta
$VENV sanity_check.py --original $HF_IN --rotated rotated-hf/

# 6. Convert rotated, quantize, perplexity C
$VENV ../../../convert_hf_to_gguf.py rotated-hf/ --outfile gguf/qwen3-8b-rotated-f16.gguf --outtype f16
$BIN/llama-quantize.exe gguf/qwen3-8b-rotated-f16.gguf gguf/qwen3-8b-rotated-q2k.gguf Q2_K
$BIN/llama-perplexity.exe -m gguf/qwen3-8b-rotated-q2k.gguf -f data/wiki.test.raw \
  -ngl 99 --ctx-size 512 2>&1 | tee results/C_rotated_q2k_ppl.log
```

## Notes on hardware

RTX 5080 Laptop has 12 GB VRAM.
- F16 8B = ~16 GB → partial offload (`-ngl 18` fits ~18/36 layers).
  F16 perplexity is the slow step (~30-60 min).
- Q2_K 8B = ~3.5 GB → fully on GPU (`-ngl 99`). Fast (~10 min).
- Rotation itself runs on CPU in bf16 (~8 GB peak RAM).

## Why only R1

QuaRot defines four rotations (R1-R4). R1 is the residual-stream
rotation and is:
- Mathematically free (doesn't interact with RoPE)
- The only one applicable via pure weight manipulation with no runtime changes
- Per QuaRot ablations, the dominant contributor for Llama-class models

R2/R3 require either online Hadamard ops at inference or more
architectural surgery. R4 touches RoPE. Starting with R1 gives the
cleanest signal.
