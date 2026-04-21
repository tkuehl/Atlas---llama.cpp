"""
prune_layers.py — Drop specified transformer layers from an HF checkpoint,
save to a new directory. No retraining, no calibration — purely structural
surgery on the ModuleList.

Usage:
    python prune_layers.py \
        --input /path/to/Qwen3-8B \
        --output /path/to/Qwen3-8B-pruned \
        --drop "24,25,26,27,28,29,30,31,32"

Drop list is 0-indexed layer indices (0 = first transformer block).
For A1 per the literature: target middle-to-late layers (Gromov et al.).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop", required=True, help="Comma-separated 0-based layer indices to drop")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    drop = sorted(set(int(x) for x in args.drop.split(",")))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[prune] loading {in_path}")
    model = AutoModelForCausalLM.from_pretrained(in_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    orig_n = model.config.num_hidden_layers
    print(f"[prune] original layers: {orig_n}")
    print(f"[prune] dropping indices: {drop}")

    if any(d >= orig_n or d < 0 for d in drop):
        print(f"[prune] ERROR: drop list contains out-of-range index (model has {orig_n} layers)", file=sys.stderr)
        sys.exit(2)

    # Remove in reverse so earlier indices stay valid
    for idx in sorted(drop, reverse=True):
        del model.model.layers[idx]

    new_n = len(model.model.layers)
    model.config.num_hidden_layers = new_n
    # Keep layer_types in sync with num_hidden_layers; newer transformers
    # versions validate these must match. Same treatment for max_window_layers.
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = [
            t for i, t in enumerate(model.config.layer_types) if i not in drop
        ]
    if getattr(model.config, "max_window_layers", None):
        model.config.max_window_layers = min(model.config.max_window_layers, new_n)
    print(f"[prune] new layers: {new_n}")

    # Renumber remaining layers — transformers indexes blocks by position in the
    # ModuleList, not by a stored integer, so the delete is enough.
    # But we want the GGUF converter to emit sequential blk.N names, which it
    # does automatically based on ModuleList order.

    print(f"[prune] saving to {out_path}")
    model.save_pretrained(out_path, safe_serialization=True)

    # Tokenizer is unchanged — copy over
    tok = AutoTokenizer.from_pretrained(in_path)
    tok.save_pretrained(out_path)

    # Sanity-check: dump new config
    with open(out_path / "config.json") as f:
        cfg = json.load(f)
    print(f"[prune] output config.num_hidden_layers = {cfg['num_hidden_layers']}")
    print(f"[prune] done")


if __name__ == "__main__":
    main()
