"""Measure top-k probability coverage and KL approximation error
across multiple k values on Qwen2.5's 152k vocab.

For a random batch of wikitext-2 text, run the teacher forward and
for each k in {32, 128, 256, 512, 1024, 2048}:
  - cumulative probability mass in top-k
  - relative error of kl_topk_loss vs full-vocab KL against a
    pseudo-student.

Goal: find the smallest k that gets the top-k KL approximation
under ~3% of full-vocab KL.  That's the smallest cache-storage
footprint that'll reproduce live-teacher training quality.
"""

from __future__ import annotations

import os as _os
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":
    import subprocess as _subprocess
    _env = dict(_os.environ)
    _env["PYTHONUTF8"] = "1"
    _sys.exit(_subprocess.run([_sys.executable] + _sys.argv, env=_env).returncode)

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from teacher_cache import kl_topk_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    from littlebit_qat_model import prepare_train_stream, iter_batches
    tokens = prepare_train_stream(tokenizer)
    gen = torch.Generator()
    gen.manual_seed(0)
    it = iter_batches(tokens, gen, args.seq_len, 1)

    k_values = [32, 128, 256, 512, 1024, 2048, 4096]

    cov_per_k = {k: [] for k in k_values}
    kl_err_per_k = {k: [] for k in k_values}

    torch.manual_seed(42)
    for b in range(args.batches):
        batch = next(it).to(device)
        with torch.no_grad():
            t_logits = teacher(batch).logits
        # Pseudo-student for gradient-signal comparison
        s_logits = t_logits.float() + 3.0 * torch.randn_like(t_logits.float())

        # Full-vocab KL (truth)
        t_log_probs_full = F.log_softmax(t_logits.float(), dim=-1)
        t_probs_full = t_log_probs_full.exp()
        s_log_probs_full = F.log_softmax(s_logits, dim=-1)
        kl_full = (t_probs_full * (t_log_probs_full - s_log_probs_full)).sum(-1).mean().item()

        for k in k_values:
            topk_vals, topk_idx = torch.topk(t_logits, k=k, dim=-1)
            # Coverage: cumulative probability mass in top-k
            cum = t_probs_full.gather(-1, topk_idx).sum(-1).mean().item()
            cov_per_k[k].append(cum)
            # KL approximation error
            kl_approx = kl_topk_loss(s_logits, topk_vals, topk_idx).item()
            kl_err_per_k[k].append(abs(kl_approx - kl_full) / kl_full)

    print(f"\n{'k':>6} {'p_top_k (mean)':>16} {'|kl_topk - kl_full|/kl_full':>30}")
    for k in k_values:
        avg_cov = sum(cov_per_k[k]) / len(cov_per_k[k])
        avg_err = sum(kl_err_per_k[k]) / len(kl_err_per_k[k])
        print(f"{k:>6d} {avg_cov:>16.4f} {avg_err:>30.4f}")


if __name__ == "__main__":
    main()
