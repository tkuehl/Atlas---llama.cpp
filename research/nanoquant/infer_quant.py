"""Run a few inference prompts on a quantized checkpoint vs FP16 teacher.

Qualitative quality read: side-by-side completions on factual, reasoning,
code, and prose prompts. Not a benchmark — just a human-readable sanity
check complementary to PPL.

Usage:
    python infer_quant.py --model Qwen/Qwen3-0.6B --rank 512 \
        --ckpt runs/....phase3_progress_lr1e-6_step500.pt
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant import replace_linears_with_quant


sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROMPTS = [
    ("factual", "The capital of France is"),
    ("arithmetic", "Q: What is 17 + 25? A:"),
    ("code", "def fibonacci(n):\n    "),
    ("prose", "The old lighthouse keeper watched the storm"),
    ("list", "Three primary colors are:\n1."),
]


def load_quantized(model_id: str, rank: int, ckpt_path: str, dtype, device):
    print(f"[infer] loading base {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    model.eval()
    print(f"[infer] replacing linears (rank={rank})", flush=True)
    for b in range(len(model.model.layers)):
        block = model.model.layers[b]
        replace_linears_with_quant(block, r=rank)
        block.to(device)
    print(f"[infer] loading state dict from {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if "step" in ckpt:
        print(f"[infer] ckpt at training step {ckpt['step']}", flush=True)
    model.load_state_dict(state)
    return model


def load_fp(model_id: str, dtype, device):
    print(f"[infer] loading FP {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    model.eval()
    return model


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new: int = 40, temperature: float = 0.0):
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-5),
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    quant_model = load_quantized(args.model, args.rank, args.ckpt, dtype, args.device)

    print(f"\n=== QUANTIZED ({args.ckpt}) ===\n", flush=True)
    for label, prompt in PROMPTS:
        completion = generate(quant_model, tok, prompt, args.max_new, args.temperature)
        print(f"[{label}] {prompt!r}\n  -> {completion!r}\n", flush=True)

    del quant_model
    torch.cuda.empty_cache()

    fp_model = load_fp(args.model, dtype, args.device)
    print(f"\n=== FP TEACHER ({args.model}) ===\n", flush=True)
    for label, prompt in PROMPTS:
        completion = generate(fp_model, tok, prompt, args.max_new, args.temperature)
        print(f"[{label}] {prompt!r}\n  -> {completion!r}\n", flush=True)


if __name__ == "__main__":
    main()
