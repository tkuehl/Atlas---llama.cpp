"""Per-block Fisher diagonal collection for BRECQ-style LittleBit QAT.

Collects `f_b[i] = E[(dL_ce / dz_b[i])^2]` at each transformer block's
output, where L_ce is the causal-LM cross-entropy loss against
ground-truth next-token and z_b is block b's output hidden state.

This is the activation-weighted importance score used in the Stage 4
objective (see `stage_4_brecq_plan.md` §3):

    L_b = E_n [ || f_b^{1/2} * (Z_b^student - Z_b^teacher) ||^2 ]

The Fisher diagonal approximates the diagonal of the Hessian of the
final CE loss w.r.t. that block's output — it tells us which output
coordinates matter most for final prediction and should be preserved
most tightly by the quantized block.

Output: a single .pt file with shape (num_blocks, d_model) plus
metadata.  Reusable across all subsequent Stage 4 experiments and
across every model initialization we try (Dual-SVID, activation-
weighted, matching pursuit, etc.) — the Fisher depends only on the
teacher, not on the student.

Usage:
    python littlebit_fisher.py --model Qwen/Qwen2.5-0.5B \\
        --samples 128 --seq-len 2048 \\
        --out qwen05b_fisher.pt

Run cost: ~3-5 min on Qwen 2.5 0.5B / RTX 5080 Laptop with 128 seqs.
"""

from __future__ import annotations

# Windows / torch.compile bootstrap — same pattern as littlebit_qat_model.py.
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


def collect_fisher(
    model_id: str,
    num_samples: int,
    seq_len: int,
    min_chars: int = 400,
    device: str = "cuda",
) -> dict:
    """Run FP teacher forward+backward on calibration data; accumulate
    per-block Fisher diagonal at block outputs."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[fisher] loading {model_id} (bfloat16)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    # Freeze all parameters — we don't want weight gradients, only the
    # gradient at block outputs (captured via backward hooks below).
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[fisher]   loaded in {time.time() - t0:.1f}s")

    # Qwen / Llama family: transformer blocks live at model.model.layers.
    blocks = model.model.layers
    num_blocks = len(blocks)
    d_model = model.config.hidden_size
    print(f"[fisher] {num_blocks} blocks, d_model={d_model}")

    # Accumulator: float64 on CPU, per-block per-dim sum of squared grads.
    # Kept on CPU to avoid contending with forward/backward VRAM.
    fisher = torch.zeros(num_blocks, d_model, dtype=torch.float64)
    token_count = 0

    # Register full_backward_hook on each block. grad_output[0] is the
    # gradient flowing back through the block's output tensor, shape
    # (batch, seq, d_model). We sum squared grads over batch and seq dims
    # to accumulate the Fisher diagonal.
    def make_hook(idx: int):
        def hook(_module, _grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return
            # Square, sum over (batch, seq), stream to CPU fp64.
            sq_sum = (g.float() ** 2).sum(dim=(0, 1))
            fisher[idx].add_(sq_sum.detach().cpu().double())
        return hook

    handles = []
    for i, block in enumerate(blocks):
        handles.append(block.register_full_backward_hook(make_hook(i)))

    # Autograd requires at least one leaf tensor with requires_grad=True for
    # loss.backward() to run. Weights are all frozen, so we inject the
    # requires_grad on inputs_embeds (a non-leaf constructed from a leaf
    # that requires grad is the standard trick).
    embed_layer = model.get_input_embeddings()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print(f"[fisher] calibrating: {num_samples} samples, seq_len={seq_len}")
    t0 = time.time()
    count = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    for row in ds:
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        enc = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=seq_len,
        )
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 8:
            continue

        # Build an inputs_embeds leaf that requires grad. The embedding
        # weight is frozen; we just need a differentiable path so that
        # autograd has somewhere to terminate.
        with torch.no_grad():
            embeds = embed_layer(input_ids).detach()
        embeds.requires_grad_(True)

        # Forward. use_cache=False suppresses KV caching.
        outputs = model(inputs_embeds=embeds, use_cache=False)
        logits = outputs.logits  # (1, seq, vocab)

        # Standard causal-LM shift: predict token t+1 from position t.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        # Sum reduction: accumulated Fisher is sum over tokens; we
        # normalize by token_count at the end.
        loss.backward()

        # Clean up the inputs_embeds graph to free memory between samples.
        del outputs, logits, loss, embeds
        token_count += int(input_ids.shape[1] - 1)  # shifted count
        count += 1

        if count % 16 == 0:
            elapsed = time.time() - t0
            rate = count / max(elapsed, 1e-6)
            print(f"[fisher]   {count}/{num_samples} samples "
                  f"({rate:.2f} samp/s, {token_count} tokens)")

        if count >= num_samples:
            break

    for h in handles:
        h.remove()

    elapsed = time.time() - t0
    print(f"[fisher] {count} samples / {token_count} tokens in {elapsed:.1f}s")

    # Normalize: fisher_accum was sum over tokens of squared grads.
    # Empirical expectation = sum / token_count.
    fisher_norm = fisher / max(token_count, 1)

    # Summary stats per block
    print(f"[fisher] per-block Fisher stats (mean, max, sum):")
    for i in range(num_blocks):
        f = fisher_norm[i]
        print(f"  block {i:2d}: mean={f.mean().item():.3e}  "
              f"max={f.max().item():.3e}  sum={f.sum().item():.3e}")

    return {
        "fisher": fisher_norm.float(),  # (num_blocks, d_model) fp32
        "fisher_fp64": fisher_norm,     # retain fp64 for numerical fidelity
        "token_count": token_count,
        "sample_count": count,
        "model_id": model_id,
        "num_blocks": num_blocks,
        "d_model": d_model,
        "seq_len": seq_len,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                   help="HF model id")
    p.add_argument("--samples", type=int, default=128,
                   help="Number of calibration sequences")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Max tokens per sequence")
    p.add_argument("--min-chars", type=int, default=400,
                   help="Skip samples shorter than this many characters")
    p.add_argument("--out", default="qwen05b_fisher.pt",
                   help="Output .pt file for the Fisher diagonals")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[fisher] WARNING: {out_path} already exists — "
              f"will overwrite after collection")

    data = collect_fisher(
        model_id=args.model,
        num_samples=args.samples,
        seq_len=args.seq_len,
        min_chars=args.min_chars,
        device=args.device,
    )

    torch.save(data, out_path)
    print(f"[fisher] saved {out_path}  "
          f"(fisher.shape={tuple(data['fisher'].shape)}, "
          f"tokens={data['token_count']})")


if __name__ == "__main__":
    main()
