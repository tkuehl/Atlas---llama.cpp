"""Collect per-linear input Gramians for scale-refinement init.

For each nn.Linear inside the transformer blocks, accumulates
H_l = E[X_l^T X_l] via a forward pre-hook over calibration data.
These Gramians feed into activation-weighted scale refinement (ALS)
in `littlebit_init_refined.py`.

Output format: a single .pt file containing a dict keyed by the
fully-qualified module name (e.g. `model.layers.0.self_attn.q_proj`)
mapping to the fp32 Gramian.  Linear layers OUTSIDE the transformer
stack (embeddings, lm_head) are skipped — they stay FP16 in LittleBit.

Usage:
    python littlebit_gramians.py --model Qwen/Qwen2.5-0.5B \\
        --samples 128 --seq-len 2048 \\
        --out qwen05b_gramians.pt
"""

from __future__ import annotations

import os as _os
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":
    import subprocess as _subprocess
    _env = dict(_os.environ)
    _env["PYTHONUTF8"] = "1"
    _sys.exit(_subprocess.run(
        [_sys.executable] + _sys.argv, env=_env
    ).returncode)

try:
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import argparse
import time
from pathlib import Path

import torch
from torch import nn


def collect_all_gramians(
    model_id: str,
    num_samples: int,
    seq_len: int,
    min_chars: int = 400,
    device: str = "cuda",
) -> dict:
    """Forward FP teacher over calibration data; accumulate per-linear
    input Gramians via forward pre-hooks on every nn.Linear inside the
    transformer blocks."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[gram] loading {model_id} (bfloat16)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[gram]   loaded in {time.time() - t0:.1f}s")

    blocks = model.model.layers
    num_blocks = len(blocks)

    # Discover all nn.Linear modules inside blocks.
    linear_names = []
    linear_modules = {}
    for b_idx, block in enumerate(blocks):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"model.layers.{b_idx}.{name}"
                linear_names.append(full_name)
                linear_modules[full_name] = module
    print(f"[gram] {len(linear_names)} linear layers across {num_blocks} blocks")

    # Init Gramian accumulators on CPU fp32 (big matrices; fp32 is
    # plenty for collection; fp64 only matters for the ALS solve).
    gramians = {}
    for name in linear_names:
        d_in = linear_modules[name].in_features
        gramians[name] = torch.zeros(d_in, d_in, dtype=torch.float32)

    token_counts = {name: 0 for name in linear_names}

    def make_hook(name):
        def hook(_mod, inputs):
            x = inputs[0]
            flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            # Accumulate on CPU to keep GPU memory free for teacher forward.
            gramians[name].add_((flat.T @ flat).cpu())
            token_counts[name] += flat.shape[0]
        return hook

    handles = []
    for name in linear_names:
        handles.append(
            linear_modules[name].register_forward_pre_hook(make_hook(name))
        )

    # Load calibration data.
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print(f"[gram] calibrating: {num_samples} samples, seq_len={seq_len}")
    t0 = time.time()
    count = 0
    total_tokens = 0
    with torch.inference_mode():
        for row in ds:
            text = row["text"].strip()
            if len(text) < min_chars:
                continue
            enc = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=seq_len,
            )
            ids = enc.input_ids.to(device)
            if ids.shape[1] < 8:
                continue
            _ = model(ids, use_cache=False)
            count += 1
            total_tokens += ids.shape[1]
            if count % 16 == 0:
                rate = count / max(time.time() - t0, 1e-6)
                print(f"[gram]   {count}/{num_samples}  "
                      f"({rate:.1f} samp/s, {total_tokens} tokens)")
            if count >= num_samples:
                break

    for h in handles:
        h.remove()

    elapsed = time.time() - t0
    print(f"[gram] {count} samples / {total_tokens} tokens in {elapsed:.1f}s")

    # Normalize each Gramian by token count (gives E[X^T X]).
    # Note: q_proj, k_proj, v_proj share the same input in each block, so
    # their token counts will match.  But o_proj sees attention-output
    # tokens (same count).  Normalization is per-linear so any per-path
    # differences are fine.
    result = {}
    for name in linear_names:
        tc = token_counts[name]
        if tc > 0:
            result[name] = gramians[name] / tc
        else:
            print(f"[gram] WARNING: {name} had 0 tokens")
            result[name] = gramians[name]

    return {
        "gramians": result,
        "token_counts": token_counts,
        "model_id": model_id,
        "sample_count": count,
        "seq_len": seq_len,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--out", default="qwen05b_gramians.pt")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[gram] WARNING: {out_path} already exists — will overwrite")

    data = collect_all_gramians(
        model_id=args.model,
        num_samples=args.samples,
        seq_len=args.seq_len,
        min_chars=args.min_chars,
        device=args.device,
    )

    # Quick sanity: print Gramian diagonal trace per unique shape.
    shapes_seen = {}
    for name, H in data["gramians"].items():
        s = tuple(H.shape)
        if s not in shapes_seen:
            shapes_seen[s] = (name, float(H.diagonal().sum()))
    print(f"[gram] sanity: Gramian shapes present:")
    for s, (n, t) in shapes_seen.items():
        print(f"[gram]   {s}  e.g. {n}  trace={t:.2e}")

    torch.save(data, out_path)
    print(f"[gram] saved {out_path}  ({len(data['gramians'])} Gramians)")


if __name__ == "__main__":
    main()
