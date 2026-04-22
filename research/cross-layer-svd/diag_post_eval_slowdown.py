"""Isolate the post-eval compile slowdown.

Reproduces the pattern:
    for _ in range(N):   train_forward_backward      # warmup compile
    eval_forward()                                   # trigger transition
    for _ in range(N):   train_forward_backward      # measure post-eval

Prints per-step timings, dynamo counters, and the compile cache state
before/after eval so we can see exactly what dynamo does at the
transition.

Usage (with PYTHONUTF8 fix):
    python -u diag_post_eval_slowdown.py [--eval-stance {default,force_eager,disable}] [--eval-kwargs {same,default}]
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
import torch
from transformers import AutoModelForCausalLM


def _counts():
    """Snapshot a handful of dynamo counters worth diffing."""
    from torch._dynamo.utils import counters
    interesting = {}
    for bucket_name, bucket in counters.items():
        for k, v in bucket.items():
            if k in ("frames", "captures", "cache_size", "recompiles"):
                interesting[f"{bucket_name}.{k}"] = v
            if "recompile" in k or "cache" in k:
                interesting[f"{bucket_name}.{k}"] = v
    return interesting


def _diff_counters(before, after):
    keys = set(before) | set(after)
    out = {}
    for k in sorted(keys):
        b = before.get(k, 0)
        a = after.get(k, 0)
        if a != b:
            out[k] = f"{b} -> {a}"
    return out


def time_block(name, fn, iters):
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        fn(i)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  {name}: {elapsed:.2f}s for {iters} iters "
          f"-> {elapsed/iters*1000:.1f} ms/iter", flush=True)
    return elapsed / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-stance", default="force_eager",
                   choices=("default", "force_eager", "disable"))
    p.add_argument("--eval-kwargs", default="same",
                   choices=("same", "default"),
                   help="same: call eval with the same kwargs as training. "
                        "default: mimic wikitext_ppl's bare model(x) call.")
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--measure-steps", type=int, default=30)
    p.add_argument("--eval-iters", type=int, default=10,
                   help="Number of eval forward passes")
    p.add_argument("--mode-switch", action="store_true", default=True,
                   help="Call model.eval()/model.train() around eval (matches "
                        "wikitext_ppl's behaviour).")
    p.add_argument("--no-mode-switch", dest="mode_switch",
                   action="store_false")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    seq_len = 512
    batch = 1

    print("loading Qwen2.5-0.5B student (fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
    ).to(device)

    # Match training-time stack
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2(
            rope=True, rms_norm=True, swiglu=False,
            cross_entropy=False, fused_linear_cross_entropy=False,
            model=model,
        )
        print("liger applied")
    except Exception as e:
        print(f"liger skipped: {e}")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = torch.compile(model, mode="default", fullgraph=False,
                          dynamic=False)

    # Simple training step: forward with same kwargs the real training
    # loop uses when use_hook_mse=False (output_hidden_states=True).
    opt = torch.optim.SGD(model.parameters(), lr=1e-6)

    def train_step(i):
        opt.zero_grad(set_to_none=True)
        batch_ids = torch.randint(0, 1000, (batch, seq_len), device=device)
        out = model(batch_ids, output_hidden_states=True)
        # sum of logits + hidden states = trivial loss that exercises the
        # same output tensors training does
        loss = out.logits.sum() + sum(h.sum() for h in out.hidden_states[1:])
        loss.backward()
        opt.step()

    def eval_pass(i):
        ids = torch.randint(0, 1000, (batch, seq_len), device=device)
        with torch.no_grad():
            if args.eval_kwargs == "same":
                _ = model(ids, output_hidden_states=False).logits
            else:
                _ = model(ids).logits

    print(f"\n=== config: eval_stance={args.eval_stance} "
          f"eval_kwargs={args.eval_kwargs} "
          f"mode_switch={args.mode_switch} ===")

    print("\n[1] training warmup (compile cold start)")
    c0 = _counts()
    t_warm = time_block("warmup", train_step, args.warmup_steps)
    c1 = _counts()
    print(f"  counter delta: {_diff_counters(c0, c1)}")

    print("\n[2] training steady-state (pre-eval)")
    c2 = _counts()
    t_pre = time_block("pre-eval", train_step, args.measure_steps)
    c3 = _counts()
    print(f"  counter delta: {_diff_counters(c2, c3)}")

    print(f"\n[3] eval ({args.eval_iters} iters, "
          f"stance={args.eval_stance}, "
          f"mode_switch={args.mode_switch})")
    if args.mode_switch:
        model.eval()
    c4 = _counts()
    if args.eval_stance == "force_eager":
        with torch.compiler.set_stance("force_eager"):
            time_block("eval", eval_pass, args.eval_iters)
    elif args.eval_stance == "disable":
        with torch.compiler.disable():
            time_block("eval", eval_pass, args.eval_iters)
    else:
        time_block("eval", eval_pass, args.eval_iters)
    c5 = _counts()
    print(f"  counter delta: {_diff_counters(c4, c5)}")
    if args.mode_switch:
        model.train()

    print("\n[4] training post-eval")
    c6 = _counts()
    t_post = time_block("post-eval", train_step, args.measure_steps)
    c7 = _counts()
    print(f"  counter delta: {_diff_counters(c6, c7)}")

    print(f"\n=== SUMMARY ===")
    print(f"  pre-eval  ms/iter: {t_pre*1000:.1f}")
    print(f"  post-eval ms/iter: {t_post*1000:.1f}")
    ratio = t_post / t_pre if t_pre > 0 else float("inf")
    print(f"  slowdown ratio:    {ratio:.2f}x")

    # Final counter dump
    print("\n=== final dynamo counters (non-zero only) ===")
    from torch._dynamo.utils import counters
    for bucket_name, bucket in sorted(counters.items()):
        nz = {k: v for k, v in bucket.items() if v}
        if nz:
            print(f"  {bucket_name}: {nz}")


if __name__ == "__main__":
    main()
