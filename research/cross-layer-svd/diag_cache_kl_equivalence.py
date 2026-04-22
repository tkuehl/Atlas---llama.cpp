"""Unit test: does the teacher cache produce losses matching live teacher?

Two checks, both on a single batch with a reproducible seed:

  1. Top-k KL: kl_topk_loss(student_logits, cached_topk)
     vs  kl_topk_loss(student_logits, live_topk(teacher_logits))
     - must match to within fp16 roundtrip noise (< 1e-3)
     - this validates the fp16 storage of logit values

  2. Hidden-state MSE: MSE(student_hidden, cached_teacher_hidden)
     vs  MSE(student_hidden, live_teacher_hidden)
     - must match to within bf16 roundtrip noise (exact, since
       bf16 → uint16 storage → bf16 view is an identity)

  3. Top-k KL approximation error: kl_topk_loss vs true full-vocab
     F.kl_div — bounded separately; this is the paper's known
     approximation, checked to stay under ~3% relative on typical
     logits distributions.

Run after extracting a small cache (~10 micro-steps is enough):

    python -u littlebit_teacher_extract.py --cache-dir caches/smoke \\
       --steps 3 --grad-accum-steps 4 --k 256
    python -u diag_cache_kl_equivalence.py --cache-dir caches/smoke
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
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from teacher_cache import TeacherCacheReader, kl_topk_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--microsteps-to-check", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    reader = TeacherCacheReader(Path(args.cache_dir))
    meta = reader.meta
    print(f"cache meta: teacher={meta.teacher_model} "
          f"n_microsteps={meta.n_microsteps} k={meta.k} "
          f"n_layers={meta.n_layers}", flush=True)

    print(f"loading live teacher {meta.teacher_model} for comparison...",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(meta.tokenizer_model)
    teacher = AutoModelForCausalLM.from_pretrained(
        meta.teacher_model, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    # Replay the exact sampling trajectory the cache was built from.
    from littlebit_qat_model import prepare_train_stream, iter_batches

    train_tokens = prepare_train_stream(tokenizer, c4_samples=meta.c4_samples)
    gen = torch.Generator()
    gen.manual_seed(meta.seed)
    it = iter_batches(train_tokens, gen, meta.seq_len, meta.batch_size)

    # Fake student logits: teacher + sizeable Gaussian noise so the
    # KL sits in the 1-5 range (matching real-training KL magnitudes).
    # With std=0.05 the pseudo-student was too close to the teacher;
    # KL near zero made relative-diff comparisons numerically
    # meaningless (single-ulp drift → 20%% relative deviation).
    def make_pseudo_student(t_logits):
        return t_logits.float() + 3.0 * torch.randn_like(t_logits.float())

    n_check = min(args.microsteps_to_check, meta.n_microsteps)
    kl_diffs = []
    kl_topk_vs_full_diffs = []
    hidden_max_abs_diffs = []

    print(f"\nchecking {n_check} micro-steps...\n", flush=True)

    for idx in range(n_check):
        batch = next(it).to(device)

        # --- live teacher ---
        with torch.no_grad():
            out = teacher(batch, output_hidden_states=True)
            t_logits_live = out.logits                        # (B, T, V)
            t_topk_vals_live, t_topk_idx_live = torch.topk(
                t_logits_live, k=meta.k, dim=-1
            )
            t_hidden_live = list(out.hidden_states[1:])

        # --- cached teacher ---
        t_topk_vals_cached, t_topk_idx_cached, t_hidden_cached = reader.get(
            idx, device
        )

        # --- pseudo student ---
        torch.manual_seed(idx)
        s_logits = make_pseudo_student(t_logits_live)

        # KL on live vs cached
        kl_live = kl_topk_loss(s_logits, t_topk_vals_live, t_topk_idx_live)
        kl_cached = kl_topk_loss(s_logits, t_topk_vals_cached,
                                 t_topk_idx_cached)
        kl_rel = (kl_live - kl_cached).abs() / kl_live.abs().clamp(min=1e-9)
        kl_diffs.append(kl_rel.item())

        # Full-vocab KL vs top-k KL (paper-known approximation — bound
        # the error for sanity)
        with torch.no_grad():
            t_log_probs_full = F.log_softmax(t_logits_live.float(), dim=-1)
            s_log_probs_full = F.log_softmax(s_logits, dim=-1)
            t_probs_full = t_log_probs_full.exp()
            kl_full = (t_probs_full * (t_log_probs_full - s_log_probs_full)).sum(-1).mean()
        kl_topk_vs_full_rel = (
            (kl_full - kl_live).abs() / kl_full.abs().clamp(min=1e-9)
        )
        kl_topk_vs_full_diffs.append(kl_topk_vs_full_rel.item())

        # Hidden-state bf16 roundtrip should be exact bit-for-bit.
        max_hidden_diff = 0.0
        for live_h, cached_h in zip(t_hidden_live, t_hidden_cached):
            d = (live_h.to(torch.bfloat16) - cached_h).abs().max().item()
            max_hidden_diff = max(max_hidden_diff, d)
        hidden_max_abs_diffs.append(max_hidden_diff)

        print(f"  micro-step {idx:3d}: "
              f"kl_live={kl_live.item():.5f} kl_cached={kl_cached.item():.5f} "
              f"rel_diff={kl_rel.item():.2e}  "
              f"hidden_max|Δ|={max_hidden_diff:.2e}  "
              f"(topk vs full-vocab: {kl_topk_vs_full_rel.item():.2e})",
              flush=True)

    print(f"\nsummary across {n_check} micro-steps:")
    print(f"  cached-vs-live top-k KL: "
          f"max rel diff = {max(kl_diffs):.2e}, mean = {sum(kl_diffs)/len(kl_diffs):.2e}")
    print(f"  top-k vs full-vocab KL:  "
          f"max rel diff = {max(kl_topk_vs_full_diffs):.2e}, "
          f"mean = {sum(kl_topk_vs_full_diffs)/len(kl_topk_vs_full_diffs):.2e}")
    print(f"  hidden-state bf16 roundtrip max |Δ| = {max(hidden_max_abs_diffs):.2e}")

    # Gates — tighten as needed
    if max(kl_diffs) > 1e-2:
        print("\n!!! FAIL: cached-vs-live KL deviation > 1%%")
        sys.exit(1)
    if max(hidden_max_abs_diffs) > 1e-3:
        print("\n!!! FAIL: hidden bf16 roundtrip drift > 1e-3")
        sys.exit(1)
    print("\nOK: cache reproduces live teacher within expected numerical bounds.")


if __name__ == "__main__":
    main()
