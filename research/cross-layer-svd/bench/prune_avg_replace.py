"""
prune_avg_replace.py — Drop a contiguous span of N transformer layers
and insert a single replacement layer whose weights are the element-wise
mean of the dropped layers. No gradients, no calibration.

Simplification of LLM-Streamline's linear-LSQ replacement. The full LSQ
version needs calibration-set hidden states to fit, plus an encoding of
the resulting linear transformation as a transformer block (non-trivial
given SwiGLU + RMSNorm). Mean-of-span is the smallest first step that
still inserts *some* learned-by-target-data transformation where the
dropped span used to be.

Usage:
    python prune_avg_replace.py \
        --input /path/to/Qwen3-8B \
        --output /path/to/Qwen3-8B-avgspan \
        --drop-start 20 \
        --drop-count 10

Produces an HF checkpoint with (original_layers - drop_count + 1) layers.
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop-start", type=int, required=True, help="First layer index (0-based) of the span to replace")
    ap.add_argument("--drop-count", type=int, required=True, help="Number of contiguous layers to replace with one averaged layer")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[avgspan] loading {in_path}")
    model = AutoModelForCausalLM.from_pretrained(in_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    orig_n = model.config.num_hidden_layers
    start = args.drop_start
    count = args.drop_count
    end = start + count  # exclusive

    if start < 0 or end > orig_n or count < 2:
        print(f"[avgspan] ERROR: invalid span [{start},{end}) for a {orig_n}-layer model; count must be ≥ 2", file=sys.stderr)
        sys.exit(2)

    print(f"[avgspan] original layers: {orig_n}")
    print(f"[avgspan] replacing span [{start}..{end-1}] ({count} layers) with 1 averaged layer")

    # Take the span and element-wise average every parameter. Deep-copy one of
    # the span layers as the scaffold so module structure / names are valid,
    # then overwrite each parameter tensor with the mean across span indices.
    span = [model.model.layers[i] for i in range(start, end)]
    avg_layer = deepcopy(span[0])
    with torch.no_grad():
        state_dicts = [lyr.state_dict() for lyr in span]
        avg_sd = {}
        for k in state_dicts[0]:
            stack = torch.stack([sd[k].to(torch.float32) for sd in state_dicts], dim=0)
            avg_sd[k] = stack.mean(dim=0).to(dtype)
        missing, unexpected = avg_layer.load_state_dict(avg_sd, strict=True)
        assert not missing and not unexpected, f"averaged state_dict mismatch: {missing} {unexpected}"

    # Build new ModuleList: keep prefix (0..start-1), add averaged layer, keep suffix (end..).
    new_layers = torch.nn.ModuleList()
    for i in range(start):
        new_layers.append(model.model.layers[i])
    new_layers.append(avg_layer)
    for i in range(end, orig_n):
        new_layers.append(model.model.layers[i])
    model.model.layers = new_layers
    new_n = len(new_layers)
    print(f"[avgspan] new layers: {new_n}")

    model.config.num_hidden_layers = new_n
    if getattr(model.config, "layer_types", None):
        # Prefix + single "full_attention" (averaged layer inherits the span's first type) + suffix
        lt = model.config.layer_types
        new_lt = lt[:start] + [lt[start]] + lt[end:]
        assert len(new_lt) == new_n, f"{len(new_lt)} != {new_n}"
        model.config.layer_types = new_lt
    if getattr(model.config, "max_window_layers", None):
        model.config.max_window_layers = min(model.config.max_window_layers, new_n)

    print(f"[avgspan] saving to {out_path}")
    model.save_pretrained(out_path, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(in_path)
    tok.save_pretrained(out_path)
    print(f"[avgspan] done")


if __name__ == "__main__":
    main()
