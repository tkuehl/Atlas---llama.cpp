"""Minimal in-memory debug: load Qwen3-8B, measure baseline logits,
apply rotation, measure post-rotation logits, compare. Skips save/load.

Also lets us test with Q=Identity (pure gamma fusion) vs Q=Hadamard
to isolate which step introduces error.
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import rotate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../../models-hf/Qwen3-8B")
    ap.add_argument("--q", choices=["identity", "hadamard"], default="hadamard")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[load] {args.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    print(f"  {time.time()-t0:.1f}s")

    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits_before = model(ids).logits.float().clone()
    print(f"[pre ] logits shape {tuple(logits_before.shape)}  argmax[-1]={logits_before[0, -1].argmax().item()}")

    hidden = model.config.hidden_size
    if args.q == "identity":
        Q = torch.eye(hidden, dtype=torch.float64)
        print(f"[q   ] using Q = I (isolates gamma fusion)")
    else:
        Q = rotate.hadamard_matrix(hidden, dtype=torch.float64)
        print(f"[q   ] using Q = Hadamard")

    rotate.rotate_qwen3(model, Q)

    with torch.no_grad():
        logits_after = model(ids).logits.float().clone()
    print(f"[post] argmax[-1]={logits_after[0, -1].argmax().item()}")

    diff = (logits_before - logits_after).abs()
    print(f"\n[compare] max_abs={diff.max().item():.4e}   mean_abs={diff.mean().item():.4e}")
    argmax_before = logits_before.argmax(-1)
    argmax_after = logits_after.argmax(-1)
    disagree = (argmax_before != argmax_after).float().mean().item()
    print(f"[compare] argmax disagree fraction: {disagree:.3f}")


if __name__ == "__main__":
    main()
