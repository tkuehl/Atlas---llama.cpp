"""Evaluate WikiText-2 PPL on a saved post-Phase-2 / Phase-3 checkpoint.

Loads the base FP model, replaces every linear under `model.model.layers`
with a shell BinaryFactoredLinear (so state_dict keys match), loads the
checkpoint, and runs sliding-window PPL. No training.

Usage:
    python eval_checkpoint.py --model Qwen/Qwen3-0.6B --rank 512 \
        --ckpt runs/phase2_qwen-qwen3-0-6b_r512_K400_steps1024_n128_L2048_seed0.phase3_progress.pt
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import eval_token_stream
from ppl import sliding_window_ppl
from quant import replace_linears_with_quant


sys.stdout.reconfigure(line_buffering=True)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = DTYPE_MAP[args.dtype]

    print(f"[eval] loading {args.model} ({args.dtype})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device
    )
    model.eval()

    print(f"[eval] replacing linears with BinaryFactoredLinear shells (r={args.rank})", flush=True)
    t0 = time.time()
    for b in range(len(model.model.layers)):
        block = model.model.layers[b]
        replace_linears_with_quant(block, r=args.rank)
        block.to(args.device)
    print(f"[eval] shell replacement done in {time.time() - t0:.1f}s", flush=True)

    print(f"[eval] loading state from {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
        extra = {k: v for k, v in ckpt.items() if k != "model_state"}
        if "step" in extra:
            print(f"[eval] ckpt is at training step {extra['step']}", flush=True)
        if "step_kls" in extra and len(extra["step_kls"]) > 0:
            kls = extra["step_kls"]
            print(
                f"[eval] KL trajectory: first={kls[0]:.2f} last={kls[-1]:.2f} "
                f"min={min(kls):.2f} n_steps={len(kls)}",
                flush=True,
            )
    else:
        state = ckpt
    model.load_state_dict(state)
    print(f"[eval] state loaded", flush=True)

    print(f"[eval] tokenizing WikiText-2 test split", flush=True)
    stream = eval_token_stream(tok)

    print(f"[eval] sliding-window PPL (seq_len={args.seq_len})", flush=True)
    result = sliding_window_ppl(model, stream, seq_len=args.seq_len, device=args.device)
    print(
        f"[eval] PPL={result.ppl:.4f} nll_mean={result.nll_mean:.4f} "
        f"windows={result.num_windows} tokens={result.num_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
