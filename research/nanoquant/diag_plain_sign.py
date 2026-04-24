"""Plain 1-bit sign quantization — lower-bound sanity check.

For every `nn.Linear` under `model.model.layers`, replace

    W  →  sign(W) * mean(|W|, axis=1)[:, None]

i.e. one scalar scale per output row, multiplying a pure ±1 sign
matrix. This is the simplest possible 1-bit quantization — no ranks,
no STE, no LB-ADMM, no training — and serves as a hard lower bound
for what binary quantization can possibly achieve on a given model.

If this baseline produces a sensible PPL on Qwen3-0.6B and our
binary-factored pipeline at r=512 doesn't, the binary-factored
pipeline has a framework-level bug. If this baseline is also
catastrophic, then 1-bit on 0.6B is just inherently hopeless and the
pipeline can't be blamed.

Does NOT use the results.json logger (this is a pure diagnostic).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import eval_token_stream
from ppl import sliding_window_ppl


sys.stdout.reconfigure(line_buffering=True)


class PlainSignLinear(nn.Module):
    """W ≈ diag(s) · sign(W_orig),  s = per-row mean(|W_orig|)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        W = lin.weight.data
        # scale[i] = mean over j of |W[i, j]|
        scale = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-12)
        sign = torch.sign(W)
        # Store as bf16/fp16 matching original to keep memory modest.
        self.sign = nn.Parameter(sign.to(W.dtype), requires_grad=False)
        self.scale = nn.Parameter(scale.to(W.dtype).squeeze(1), requires_grad=False)
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.in_features = lin.in_features
        self.out_features = lin.out_features

    def forward(self, x):
        cdtype = x.dtype
        # (…, d_in) @ (d_in, d_out) where effective W = scale * sign
        # Compute as: (x @ sign.T) * scale[None, :]
        y = x @ self.sign.to(cdtype).T
        y = y * self.scale.to(cdtype)
        if self.bias is not None:
            y = y + self.bias.to(cdtype)
        return y


def replace_linears_with_plain_sign(module: nn.Module) -> list[str]:
    replaced: list[str] = []

    def walk(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and "lm_head" not in path:
                setattr(parent, name, PlainSignLinear(child))
                replaced.append(path)
            else:
                walk(child, path)

    walk(module, "")
    return replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"[diag] loading {args.model} ({args.dtype})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device
    )
    model.eval()

    # Replace every Linear under model.model.layers with a PlainSignLinear.
    # Leave lm_head alone.
    print("[diag] replacing linears with plain sign quantization (1-bit)", flush=True)
    t0 = time.time()
    replaced = []
    for b, block in enumerate(model.model.layers):
        r = replace_linears_with_plain_sign(block)
        replaced.extend([f"model.layers.{b}.{p}" for p in r])
    print(
        f"[diag] replaced {len(replaced)} linears in {time.time() - t0:.1f}s",
        flush=True,
    )

    print("[diag] tokenizing WikiText-2 test split", flush=True)
    stream = eval_token_stream(tok)

    print(f"[diag] sliding-window PPL (seq_len={args.seq_len})", flush=True)
    result = sliding_window_ppl(
        model,
        stream,
        seq_len=args.seq_len,
        device=args.device,
    )
    print(
        f"[diag] plain-sign quantization PPL={result.ppl:.4f} "
        f"nll_mean={result.nll_mean:.4f} "
        f"windows={result.num_windows} tokens={result.num_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
