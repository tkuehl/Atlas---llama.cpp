"""Offline Dual-SVID init cache builder.

Run once per (model, rank) pair.  Loads the HF model, wraps every
Linear (except --skip) with LittleBitLinearHF via Dual-SVID, saves
the state_dict to --out.  Does not touch the GPU and does not load
a teacher — pure CPU SVDs.

Example (Qwen2.5-1.5B at r=768):

    python -u build_init_cache.py \\
      --model Qwen/Qwen2.5-1.5B \\
      --rank 768 \\
      --out littlebit_qat_init_qwen15b_r768.pt

The resulting file is consumed by littlebit_qat_model.py's
`--init-cache` flag, which skips the expensive Dual-SVID pass on
re-runs at the same (model, rank).
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

import torch
from transformers import AutoModelForCausalLM

from littlebit_qat_model import wrap_model_littlebit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. Qwen/Qwen2.5-1.5B")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--out", required=True,
                   help="Output path for the init cache .pt")
    p.add_argument("--shadow-dtype", default="fp32",
                   choices=("fp32", "bf16"),
                   help="Save U_fp/V_fp at this precision.  The "
                        "training script will up/down-cast at load "
                        "time if its --shadow-dtype differs, so fp32 "
                        "here is the most re-usable default.")
    p.add_argument("--skip", default="lm_head",
                   help="Comma-separated substrings of module names "
                        "to skip wrapping (default: lm_head).")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(
            f"output path already exists: {out_path}\n"
            "refusing to overwrite — rm first if that's intentional"
        )

    shadow_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.shadow_dtype]
    skip_tuple = tuple(s.strip() for s in args.skip.split(",") if s.strip())

    print(f"loading {args.model} (fp32 on CPU)...", flush=True)
    t0 = time.time()
    # fp32 so SVD precision is good; Dual-SVID's internal math uses
    # fp64 anyway but the source weights shouldn't be the bottleneck.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    )
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    # Sanity on architecture
    cfg = model.config
    print(f"  architecture: layers={getattr(cfg, 'num_hidden_layers', '?')}, "
          f"hidden={getattr(cfg, 'hidden_size', '?')}, "
          f"vocab={getattr(cfg, 'vocab_size', '?')}", flush=True)

    print(f"wrapping with Dual-SVID at r={args.rank}, skip={skip_tuple}...",
          flush=True)
    t0 = time.time()
    wrapped = wrap_model_littlebit(model, r=args.rank, tau=args.tau,
                                   skip=skip_tuple, log_every=10,
                                   shadow_dtype=shadow_dtype)
    print(f"  wrapped {wrapped} Linear layers "
          f"in {time.time() - t0:.1f}s "
          f"(shadow_dtype={args.shadow_dtype})", flush=True)

    print(f"saving init cache to {out_path}...", flush=True)
    t0 = time.time()
    torch.save(model.state_dict(), out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.0f} MB in {time.time() - t0:.1f}s", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
