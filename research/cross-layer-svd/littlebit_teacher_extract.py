"""Offline teacher extraction for the Sprint 3 teacher cache.

Runs the teacher forward over a deterministic sampling trajectory
(same seed/config as a planned training run) and writes top-k
logits + all-layer hidden states to a mmap'd binary cache that
replaces the teacher forward at training time.

Usage example (matches tonight's final smoke):

    python -u littlebit_teacher_extract.py \\
      --model Qwen/Qwen2.5-0.5B \\
      --steps 100 \\
      --seq-len 512 \\
      --grad-accum-steps 4 \\
      --batch-size 1 \\
      --k 256 \\
      --cache-dir caches/qwen05b_s0_n100

The cache produced is positional and keyed to exactly this
(seed, seq_len, batch_size, grad_accum_steps, c4_samples,
teacher_model) tuple.  Training reads from it via
TeacherCacheReader, which validates the config on load.
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
import time
from pathlib import Path

import numpy as np
import torch

from teacher_cache import (
    CacheMetadata, TeacherCacheWriter, compute_corpus_hash,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--cache-dir", required=True,
                   help="Directory to write cache files into.  Will be "
                        "created if missing; overwrites existing files.")
    p.add_argument("--seed", type=int, default=0,
                   help="Matches --resume/--sampler seed in training; "
                        "cache is trajectory-keyed, so train and extract "
                        "must share this.")
    p.add_argument("--steps", type=int, default=100,
                   help="Number of OPT-steps to cache.  Actual cached "
                        "micro-steps = steps * grad_accum_steps.")
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--c4-samples", type=int, default=0,
                   help="Must match training.  0 = wikitext-2 only.")
    p.add_argument("--k", type=int, default=256,
                   help="Top-k logits to cache per position.  "
                        "k=256 is essentially lossless vs full-vocab KL.")
    p.add_argument("--tf32", action="store_true", default=True,
                   help="Enable TF32 for teacher extraction.  Same "
                        "numerical regime as training.")
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--log-every", type=int, default=25,
                   help="Progress log every N micro-steps.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    if device.type == "cuda" and args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("tf32: on (matmul), cudnn.benchmark: on", flush=True)

    # Import training helpers AFTER torch setup so the bootstrap
    # doesn't re-exec through them.
    from littlebit_qat_model import prepare_train_stream, iter_batches
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading teacher {args.model} (bfloat16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    # Dimensions
    config = teacher.config
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    # Number of layers whose hidden states we cache == number of
    # entries in output_hidden_states[1:], which is n_decoder_layers
    # (post-each-layer except the last, whose slot is the
    # post-final-norm tensor).
    n_layers = config.num_hidden_layers
    print(f"  vocab={vocab_size}, hidden={hidden_size}, layers={n_layers}",
          flush=True)

    # Token stream identical to training.
    train_tokens = prepare_train_stream(tokenizer, c4_samples=args.c4_samples)
    n_tokens = train_tokens.shape[0]
    corpus_hash = compute_corpus_hash(train_tokens)
    print(f"  train stream: {n_tokens:,} tokens  "
          f"(corpus_hash={corpus_hash})", flush=True)

    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(args.seed)
    it = iter_batches(train_tokens, sampler_gen, args.seq_len, args.batch_size)

    n_microsteps = args.steps * args.grad_accum_steps

    # Disk budget sanity check
    bytes_topk = n_microsteps * args.batch_size * args.seq_len * args.k * (2 + 4)
    bytes_hidden = (n_microsteps * args.batch_size * args.seq_len
                    * hidden_size * 2 * n_layers)
    total_gb = (bytes_topk + bytes_hidden) / (1024 ** 3)
    print(f"  disk budget: top-k {bytes_topk / 1024**3:.2f} GB, "
          f"hidden {bytes_hidden / 1024**3:.2f} GB "
          f"(total {total_gb:.2f} GB)", flush=True)

    # Teacher PPL on wikitext-2 test — cached so training can display
    # it without reloading the teacher.  Uses the exact same eval
    # harness as training for consistency.
    from littlebit_qat_model import wikitext_ppl
    print("  measuring teacher PPL (for metadata)...", flush=True)
    teacher_ppl = float(wikitext_ppl(
        teacher, tokenizer, seq_len=args.seq_len,
        max_tokens=50_000, device=device,
    ))
    print(f"  teacher PPL = {teacher_ppl:.3f}", flush=True)

    meta = CacheMetadata(
        teacher_model=args.model,
        tokenizer_model=args.model,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        k=args.k,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        n_microsteps=n_microsteps,
        opt_steps=args.steps,
        c4_samples=args.c4_samples,
        seed=args.seed,
        hidden_layer_indices=list(range(n_layers)),
        corpus_hash=corpus_hash,
        teacher_ppl=teacher_ppl,
    )

    cache_dir = Path(args.cache_dir)
    print(f"  cache dir: {cache_dir}", flush=True)
    writer = TeacherCacheWriter(cache_dir, meta)

    print(f"\nextracting {n_microsteps} micro-steps "
          f"({args.steps} opt-steps × {args.grad_accum_steps} accum)...",
          flush=True)
    t0 = time.time()
    for idx in range(n_microsteps):
        batch = next(it).to(device)
        with torch.no_grad():
            out = teacher(batch, output_hidden_states=True)
            # Top-k over vocab dim
            topk_vals, topk_idx = torch.topk(out.logits, k=args.k, dim=-1)
            # hidden_states has N+1 entries: [embed, post-layer-0, ...,
            # post-final-norm].  [1:] gives N entries matching our
            # training-time target (and post-hook-fix HiddenCapture).
            hiddens = list(out.hidden_states[1:])
        writer.write_microstep(idx, topk_vals, topk_idx, hiddens)
        if (idx + 1) % args.log_every == 0:
            rate = (idx + 1) / (time.time() - t0)
            eta = (n_microsteps - idx - 1) / max(rate, 1e-6)
            print(f"  micro-step {idx + 1:5d}/{n_microsteps}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s",
                  flush=True)

    writer.finalize()
    elapsed = time.time() - t0
    print(f"\ndone.  {n_microsteps} micro-steps in {elapsed:.0f}s "
          f"({n_microsteps / max(elapsed, 1e-6):.1f} micro-steps/s)",
          flush=True)
    print(f"cache written to {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
