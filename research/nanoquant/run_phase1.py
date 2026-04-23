"""Phase 1 entrypoint.

Pipeline:
    1. Load FP model + tokenizer.
    2. Build (or reuse) the teacher activation cache.
    3. Block-by-block STE refinement with SVD init.
    4. Evaluate the quantized model with the same sliding-window PPL
       harness used for baselines.
    5. Append one entry to results.json.

Usage:
    python run_phase1.py --model Qwen/Qwen3-4B --rank 2 --steps 500
    python run_phase1.py --model Qwen/Qwen3-4B --rank 2 --steps 100 --n-calib 8
        # smoke-test configuration

Cache dir defaults to research/nanoquant/cache/<model-slug>/<config-slug>/ so
repeat runs at the same (model, n_calib, seq_len) skip the cache build.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activations import Cache, build_cache
from data import eval_token_stream
from phase1 import quantize_model_phase1
from ppl import sliding_window_ppl
from results import append_entry


HERE = Path(__file__).parent
CACHE_ROOT = HERE / "cache"
STATUS_ROOT = HERE / "runs"

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500, help="STE steps per block")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=2, help="cache build batch")
    # bf16 is the default: the binary-factored forward can produce outliers
    # that overflow fp16's ±65504 range (observed on Qwen3-4B blocks 6, 16
    # at rank=2 under plain SVD init). bf16 keeps fp32-like range.
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    ap.add_argument("--eval-seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--notes", default=None)
    ap.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="force rebuild even if cache dir exists",
    )
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="skip appending to results.json (for smoke tests)",
    )
    args = ap.parse_args()

    torch_dtype = DTYPE_MAP[args.dtype]

    cache_slug = f"n{args.n_calib}_L{args.seq_len}_seed{args.seed}"
    cache_dir = CACHE_ROOT / _slug(args.model) / cache_slug
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    status_file = STATUS_ROOT / (
        f"phase1_{_slug(args.model)}_r{args.rank}_steps{args.steps}_"
        f"{cache_slug}.status.json"
    )

    print(f"[load] tokenizer: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f"[load] model: {args.model} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()

    need_build = args.rebuild_cache or not (cache_dir / "meta.json").exists()
    if need_build:
        print(
            f"[cache] building teacher activations -> {cache_dir} "
            f"(n={args.n_calib}, L={args.seq_len})",
            flush=True,
        )
        t0 = time.time()
        cache = build_cache(
            model,
            tok,
            cache_dir=cache_dir,
            n_samples=args.n_calib,
            seq_len=args.seq_len,
            seed=args.seed,
            chunk_size=args.chunk_size,
            device=args.device,
        )
        print(f"[cache] built in {time.time() - t0:.1f}s", flush=True)
    else:
        print(f"[cache] reusing {cache_dir}", flush=True)
        cache = Cache.load(cache_dir)

    print(
        f"[phase1] rank={args.rank} steps={args.steps} lr={args.lr} "
        f"blocks={cache.n_blocks}",
        flush=True,
    )
    t0 = time.time()
    block_stats = quantize_model_phase1(
        model,
        cache=cache,
        r=args.rank,
        steps_per_block=args.steps,
        lr=args.lr,
        device=args.device,
        status_file=status_file,
    )
    phase1_secs = time.time() - t0
    print(
        f"[phase1] done in {phase1_secs:.1f}s "
        f"({phase1_secs / max(len(block_stats), 1):.1f}s/block)",
        flush=True,
    )

    print("[eval] tokenizing WikiText-2 test split", flush=True)
    stream = eval_token_stream(tok)
    print(
        f"[eval] sliding-window PPL (seq_len={args.eval_seq_len})",
        flush=True,
    )
    result = sliding_window_ppl(
        model,
        stream,
        seq_len=args.eval_seq_len,
        device=args.device,
    )
    print(
        f"[eval] ppl={result.ppl:.4f} nll_mean={result.nll_mean:.4f} "
        f"windows={result.num_windows} tokens={result.num_tokens}",
        flush=True,
    )

    mean_init = sum(s.init_mse for s in block_stats) / max(len(block_stats), 1)
    mean_final = sum(s.final_mse for s in block_stats) / max(len(block_stats), 1)
    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu"
    )
    revision = getattr(model.config, "_commit_hash", None)

    if args.no_log:
        print("[log] --no-log set, skipping results.json append", flush=True)
        return

    entry = append_entry(
        model_hf_id=args.model,
        model_revision=revision,
        model_dtype=args.dtype,
        method_name="phase1-ste-svd-init",
        method_params={
            "r": args.rank,
            "steps_per_block": args.steps,
            "lr": args.lr,
            "init": "svd",
            "ste": "clipped-identity",
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "n_calib": args.n_calib,
            "calib_seq_len": args.seq_len,
            "calib_seed": args.seed,
            "mean_block_init_mse": mean_init,
            "mean_block_final_mse": mean_final,
            "phase1_wall_seconds": phase1_secs,
        },
        eval_info={
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
            "seq_len": result.seq_len,
            "stride": result.stride,
            "num_tokens": result.num_tokens,
            "num_windows": result.num_windows,
            "metric": "ppl",
            "value": result.ppl,
            "nll_mean": result.nll_mean,
        },
        hardware={"gpu": gpu_name, "torch": torch.__version__},
        notes=args.notes,
    )
    print(f"[log] appended entry id={entry['id']}", flush=True)

    # Also drop a final block-stats snapshot alongside the status file.
    final_status_file = status_file.with_name(status_file.stem + ".final.json")
    with open(final_status_file, "w") as f:
        json.dump(
            {
                "entry_id": entry["id"],
                "ppl": result.ppl,
                "block_stats": [vars(s) for s in block_stats],
            },
            f,
            indent=2,
        )
    print(f"[log] block stats -> {final_status_file}", flush=True)


if __name__ == "__main__":
    main()
