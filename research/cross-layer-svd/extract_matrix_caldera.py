"""Extract (W, XTX) for one linear in an HF model, saved as a pickle
compatible with caldera_validate.py / balanced_snapshot.pkl.

CALDERA only needs the input-activation gramian (XTX), so we do a
forward-only calibration in bf16 — no backward pass, no GGT, much lower
VRAM and faster than balanced_test.py's fp32 + backward path.

Usage:
  python extract_matrix_caldera.py --model Qwen/Qwen2.5-3B \\
         --role mlp.gate_proj --layer 12 --samples 32 --seq-len 2048 \\
         --out snapshot_3b_gate12.pkl

For models that don't fit fully in VRAM (7B on 12 GB), pass
`--device-map auto` and HF Accelerate spreads layers across GPU+CPU.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--role", default="mlp.gate_proj",
                   help="submodule path under model.layers.N")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default=None,
                   help='HF device_map; e.g. "auto" for >VRAM models')
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype}, "
          f"device_map={args.device_map or 'cuda'})")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=args.device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype,
        ).to("cuda")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"  loaded in {time.time() - t0:.1f}s")

    target_name = f"model.layers.{args.layer}.{args.role}"
    named = dict(model.named_modules())
    if target_name not in named:
        available = [n for n in named if ".layers.0." in n]
        raise SystemExit(f"{target_name} not found. Available L0 linears: "
                         + ", ".join(available[:20]))
    target_mod = named[target_name]
    d_in = target_mod.in_features
    d_out = target_mod.out_features
    # fp32 XTX accumulator on CPU (stable across many samples, small size)
    xtx = torch.zeros(d_in, d_in, dtype=torch.float32)

    def fwd_hook(_mod, inputs):
        x = inputs[0].detach()
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        xtx.add_((flat.T @ flat).cpu())

    h = target_mod.register_forward_pre_hook(fwd_hook)

    print(f"target     : {target_name}  [{d_out} x {d_in}]")
    print(f"loading wikitext-2 train…")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    print(f"calibrating: samples={args.samples} seq_len={args.seq_len}")
    t0 = time.time()
    count = 0
    with torch.inference_mode():
        for row in ds:
            t = row["text"].strip()
            if len(t) < args.min_chars:
                continue
            enc = tokenizer(t, return_tensors="pt",
                            truncation=True, max_length=args.seq_len)
            ids = enc.input_ids
            if ids.shape[1] < 8:
                continue
            # Input always on first device; HF handles device_map routing
            device = next(model.parameters()).device
            ids = ids.to(device)
            _ = model(ids)
            count += 1
            if count % 8 == 0:
                print(f"  {count}/{args.samples}  "
                      f"({(time.time()-t0)/count:.1f}s/sample)")
            if count >= args.samples:
                break
    h.remove()

    print(f"calibration done: {count} samples in {time.time()-t0:.1f}s")
    W = target_mod.weight.data.detach().cpu().clone().to(torch.float32)

    snap = {
        "model_id": args.model,
        "role": args.role,
        "layer_idx": args.layer,
        "W": W,
        "XTX": xtx,
        "calib_samples": count,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
    }
    Path(args.out).write_bytes(pickle.dumps(snap))
    print(f"wrote {args.out}: W={tuple(W.shape)} "
          f"XTX diag mean={xtx.diag().mean().item():.4f}")


if __name__ == "__main__":
    main()
