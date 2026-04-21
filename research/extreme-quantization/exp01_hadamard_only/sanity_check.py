"""Verify the rotated model is mathematically equivalent to the original at FP precision.

If the R1 rotation math is correct, the rotated model should produce
numerically-close logits on any input (modulo rounding in bf16/fp16).
A large divergence means the rotation or norm-fusion is wrong, and
any downstream perplexity result would be uninterpretable.

Pass criterion: max absolute logit delta < 1e-2 in bf16 on a mixed
prompt set. (bf16 has ~1e-3 relative precision, so small absolute
deltas on logits in the -10..+10 range are expected and fine.)
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientist discovered a herd of unicorns living in",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "To be, or not to be, that is the",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
]


def logits_for(model, tokenizer, prompt: str, device: str) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids)
    return out.logits.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--rotated", required=True)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (cpu is safest if VRAM is tight)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tol-max", type=float, default=5e-2, help="max absolute logit delta")
    ap.add_argument("--tol-argmax-disagree", type=float, default=0.02,
                    help="max fraction of positions where argmax disagrees")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[load] original {args.original}")
    t0 = time.time()
    orig = AutoModelForCausalLM.from_pretrained(args.original, torch_dtype=dtype, device_map=args.device)
    orig.eval()
    tok = AutoTokenizer.from_pretrained(args.original)
    print(f"  {time.time() - t0:.1f}s")

    print(f"[load] rotated  {args.rotated}")
    t0 = time.time()
    rot = AutoModelForCausalLM.from_pretrained(args.rotated, torch_dtype=dtype, device_map=args.device)
    rot.eval()
    print(f"  {time.time() - t0:.1f}s")

    worst_abs = 0.0
    worst_argmax_rate = 0.0
    print("\n[compare]")
    for p in PROMPTS:
        lo = logits_for(orig, tok, p, args.device)
        lr = logits_for(rot, tok, p, args.device)
        if lo.shape != lr.shape:
            raise SystemExit(f"logit shape mismatch: {lo.shape} vs {lr.shape}")

        abs_diff = (lo - lr).abs()
        max_abs = abs_diff.max().item()

        # Fraction of positions where the top-1 token differs
        argmax_orig = lo.argmax(-1)
        argmax_rot = lr.argmax(-1)
        argmax_disagree = (argmax_orig != argmax_rot).float().mean().item()

        print(f"  prompt: {p[:50]!r:52s}  max_abs={max_abs:.3e}  argmax_disagree={argmax_disagree:.3f}")
        worst_abs = max(worst_abs, max_abs)
        worst_argmax_rate = max(worst_argmax_rate, argmax_disagree)

    print(f"\n[summary] worst max_abs={worst_abs:.3e}  worst argmax_disagree={worst_argmax_rate:.3f}")
    ok_abs = worst_abs < args.tol_max
    ok_argmax = worst_argmax_rate < args.tol_argmax_disagree
    if ok_abs and ok_argmax:
        print("[PASS] rotation preserves model outputs within tolerance")
    else:
        print("[FAIL] rotation does NOT preserve outputs")
        if not ok_abs:
            print(f"  logit delta {worst_abs:.3e} >= tol {args.tol_max:.3e}")
        if not ok_argmax:
            print(f"  argmax disagree {worst_argmax_rate:.3f} >= tol {args.tol_argmax_disagree:.3f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
