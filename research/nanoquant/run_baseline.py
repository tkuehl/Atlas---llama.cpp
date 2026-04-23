"""FP16 baseline: load an HF model, compute WikiText-2 PPL, append to results.json.

Usage:
    python run_baseline.py --model Qwen/Qwen3-4B
    python run_baseline.py --model Qwen/Qwen3-4B --seq-len 2048 --dtype float16
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import eval_token_stream
from ppl import sliding_window_ppl
from results import append_entry


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="float16", choices=list(DTYPE_MAP))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    torch_dtype = DTYPE_MAP[args.dtype]

    print(f"[load] tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f"[load] model: {args.model} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()

    print("[data] tokenizing WikiText-2 test split")
    stream = eval_token_stream(tok)
    print(f"[data] stream length: {stream.numel()} tokens")

    print(f"[eval] sliding-window PPL (seq_len={args.seq_len})")
    result = sliding_window_ppl(
        model,
        stream,
        seq_len=args.seq_len,
        stride=args.stride,
        device=args.device,
    )
    print(
        f"[eval] ppl={result.ppl:.4f}  nll_mean={result.nll_mean:.4f}  "
        f"windows={result.num_windows}  tokens={result.num_tokens}"
    )

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu"
    )
    revision = getattr(model.config, "_commit_hash", None)

    entry = append_entry(
        model_hf_id=args.model,
        model_revision=revision,
        model_dtype=args.dtype,
        method_name="fp16-baseline",
        method_params={},
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
    print(f"[log] appended entry id={entry['id']}")


if __name__ == "__main__":
    main()
