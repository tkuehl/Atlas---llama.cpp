"""Activation-weighted QAT for a single LittleBit layer.

Closer to the paper's actual objective than plain Frobenius.  The
paper minimises KL + intermediate-MSE through the whole model; both
gradients ultimately flow through each linear as "weighted by what
activations actually look like."  Without running the full model,
the single-layer analogue is:

    L = || X @ W.T  -  X @ W_hat.T ||_F^2
      = tr( (W - W_hat) @ XTX @ (W - W_hat).T )

where XTX is the d_in x d_in input Gramian collected by running the
model over a calibration corpus (same code path as CALDERA's
extract_matrix_caldera.py).  We don't need to store raw activations —
XTX is enough and reusable across runs.

If this plateaus at a similar rel Frobenius err to the plain
Frobenius run (~0.75), then the LittleBit format at r=512 is
capacity-limited and full-model QAT will not recover it either.
If it drops materially (say below 0.5), then activation weighting
is where the paper's recovery comes from and full-model QAT is
worth the compute.

Usage:
  python littlebit_qat_activation.py --model Qwen/Qwen2.5-0.5B \\
         --role mlp.gate_proj --layer 12 --rank 512 \\
         --calib-samples 32 --seq-len 2048 \\
         --steps 5000 --lr 1e-3 \\
         --out littlebit_qat_activation.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Reuse SmoothSign + LittleBitLinear.
from littlebit_qat_single import SmoothSign, smooth_sign, LittleBitLinear


def collect_xtx(model_id: str, role: str, layer: int,
                samples: int, seq_len: int,
                min_chars: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the teacher model over wikitext, collect the input Gramian
    XTX and the target layer's weight matrix W.  XTX accumulated on CPU
    in fp32 for numerical stability."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {model_id} (bfloat16)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"  loaded in {time.time() - t0:.1f}s")

    target = f"model.layers.{layer}.{role}"
    target_mod = dict(model.named_modules())[target]
    d_in = target_mod.in_features
    xtx = torch.zeros(d_in, d_in, dtype=torch.float32)

    def fwd_hook(_mod, inputs):
        x = inputs[0].detach()
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        xtx.add_((flat.T @ flat).cpu())

    h = target_mod.register_forward_pre_hook(fwd_hook)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print(f"calibrating: {samples} samples, seq_len={seq_len}")
    t0 = time.time()
    count = 0
    with torch.inference_mode():
        for row in ds:
            t = row["text"].strip()
            if len(t) < min_chars:
                continue
            enc = tokenizer(t, return_tensors="pt",
                            truncation=True, max_length=seq_len)
            ids = enc.input_ids.to("cuda")
            if ids.shape[1] < 8:
                continue
            _ = model(ids)
            count += 1
            if count >= samples:
                break
    h.remove()
    print(f"  {count} samples in {time.time() - t0:.1f}s")

    W = target_mod.weight.data.detach().clone().to(torch.float32).cpu()
    return W, xtx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--role", default="mlp.gate_proj")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default="littlebit_qat_activation.json")
    p.add_argument("--cache", default="qwen05b_gate12_xtx.pkl",
                   help="Cache XTX + W here; skip collection if exists")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = Path(args.cache)
    if cache.exists():
        import pickle
        print(f"loading cached W + XTX from {cache}")
        data = pickle.loads(cache.read_bytes())
        W_cpu, xtx = data["W"], data["XTX"]
    else:
        W_cpu, xtx = collect_xtx(
            args.model, args.role, args.layer,
            args.calib_samples, args.seq_len,
        )
        import pickle
        cache.write_bytes(pickle.dumps({"W": W_cpu, "XTX": xtx,
                                        "samples": args.calib_samples,
                                        "seq_len": args.seq_len}))
        print(f"cached to {cache}")

    W = W_cpu.to(device)
    XTX = xtx.to(device)
    W_norm = torch.linalg.norm(W).item()
    # Baseline: ||X @ W.T||_F^2 = tr(W @ XTX @ W.T)
    baseline_act_energy = torch.einsum("ij,jk,ik->", W, XTX, W).item()
    print(f"||W||_F={W_norm:.4f}  "
          f"activation energy ||XW.T||_F^2={baseline_act_energy:.1f}")

    layer = LittleBitLinear(W, r=args.rank, tau=args.tau).to(device)

    @torch.no_grad()
    def rel_act_err(layer_):
        Wh = layer_.reconstruct()
        D = W - Wh
        # ||X D.T||_F^2 = tr(D @ XTX @ D.T)
        num = torch.einsum("ij,jk,ik->", D, XTX, D)
        return float((num / baseline_act_energy).sqrt().item()), \
               float((torch.linalg.norm(D) / W_norm).item())

    init_act, init_frob = rel_act_err(layer)
    print(f"Dual-SVID init:   activation-rel={init_act:.4f}  "
          f"Frobenius-rel={init_frob:.4f}")

    opt = torch.optim.AdamW(layer.parameters(), lr=args.lr)
    history = [{"step": 0, "loss": None,
                "rel_act_err": init_act, "rel_frob_err": init_frob}]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        W_hat = layer.reconstruct()
        D = W - W_hat
        # Activation-weighted loss
        loss = torch.einsum("ij,jk,ik->", D, XTX, D)
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            act, frob = rel_act_err(layer)
            history.append({
                "step": step, "loss": float(loss.item()),
                "rel_act_err": act, "rel_frob_err": frob,
            })
            print(f"  step {step:5d}  loss={loss.item():.1f}  "
                  f"act-rel={act:.4f}  frob-rel={frob:.4f}  "
                  f"elapsed={time.time() - t0:.1f}s")

    best = min(history, key=lambda h: h["rel_act_err"])
    print(f"\nfinal: init act-rel={init_act:.4f}  "
          f"best act-rel={best['rel_act_err']:.4f} (step {best['step']})  "
          f"improvement={init_act - best['rel_act_err']:+.4f}")
    print(f"       init frob-rel={init_frob:.4f}  "
          f"best frob-rel={best['rel_frob_err']:.4f}")

    out = {
        "model": args.model, "role": args.role, "layer": args.layer,
        "W_shape": list(W.shape), "W_frob": W_norm,
        "baseline_act_energy": baseline_act_energy,
        "rank": args.rank, "steps": args.steps, "lr": args.lr,
        "tau": args.tau, "calib_samples": args.calib_samples,
        "init_rel_act_err": init_act, "init_rel_frob_err": init_frob,
        "final_rel_act_err": history[-1]["rel_act_err"],
        "final_rel_frob_err": history[-1]["rel_frob_err"],
        "best_rel_act_err": best["rel_act_err"],
        "best_rel_frob_err": best["rel_frob_err"],
        "best_step": best["step"],
        "history": history,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
