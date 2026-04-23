"""Full-model build with multi-rank-magnitude LittleBit and eval PPL.

Construct a Qwen-flavored LittleBit student where every linear in
the transformer stack uses LittleBitLinearMultiRankHF with rank r
and magnitude rank K.  Evaluate WikiText-2 PPL at init (no training).

Compare to bare Dual-SVID's PPL (~380k from prior runs).  If K=2
multi-rank drops init PPL by 5× or more, this is a genuine init lever;
if it barely moves, something about the single-matrix → full-model
composition is breaking it.

Usage:
    python littlebit_init_multirank_full.py --model Qwen/Qwen2.5-0.5B \\
        --rank 512 --K 2 --out s6_0_mr_K2.student.pt --eval

Debug mode (--debug) prints per-layer reconstruction rel-err at init,
per-block timing, full parameter breakdown, and PPL trajectory.
"""

from __future__ import annotations

import os as _os
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":
    import subprocess as _subprocess
    _env = dict(_os.environ)
    _env["PYTHONUTF8"] = "1"
    _sys.exit(_subprocess.run(
        [_sys.executable] + _sys.argv, env=_env
    ).returncode)

try:
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from littlebit_multirank import (
    LittleBitLinearMultiRankHF,
    convert_block_to_multirank,
    count_params,
)
from littlebit_qat_brecq import get_or_eval_teacher_ppl, write_status
from littlebit_qat_brecq_full import eval_ppl


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.float()
    b32 = b.float()
    return float((torch.linalg.norm(a32 - b32) /
                  (torch.linalg.norm(b32) + 1e-12)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--K", type=int, default=2,
                   help="Magnitude approximation rank (K=1 = original Dual-SVID)")
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eval-max-tokens", type=int, default=25000)
    p.add_argument("--out", default="s6_0_mr_K2.student.pt")
    p.add_argument("--eval", action="store_true",
                   help="Evaluate student PPL after build")
    p.add_argument("--debug", action="store_true",
                   help="Verbose per-layer debug output")
    p.add_argument("--probe-block", type=int, default=12,
                   help="Report reconstruction quality on this block's linears")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load teacher + clone for student -----
    print(f"[mr-build] loading teacher {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)
    print(f"[mr-build]   teacher loaded in {time.time()-t0:.1f}s")

    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    student.eval()
    for pr in student.parameters():
        pr.requires_grad_(False)

    num_blocks = len(student.model.layers)
    print(f"[mr-build] num_blocks={num_blocks} rank={args.rank} K={args.K}")

    # ----- Build status path -----
    status_path = str(Path(args.out).with_suffix(".status.json"))
    write_status(status_path, {
        "phase": "building",
        "blocks_total": num_blocks,
        "blocks_done": 0,
        "K": args.K,
        "rank": args.rank,
    })

    # ----- Probe: report single-matrix reconstruction quality on one block
    # BEFORE replacing, so we have a reference.  Also runs with --debug on. -----
    if args.debug and 0 <= args.probe_block < num_blocks:
        print(f"[mr-build] pre-conversion probe on block {args.probe_block}:")
        probe_block = teacher.model.layers[args.probe_block]
        for name, module in probe_block.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.detach().to(torch.float64).cpu().numpy()
                import numpy as np
                W_norm = np.linalg.norm(W)
                print(f"[mr-build]   {name}: shape={tuple(W.shape)}  "
                      f"||W||_F={W_norm:.4f}")

    # ----- Per-block conversion with timing + verification -----
    t_build = time.time()
    per_block_times = []
    total_params = {"total": 0}

    for b in range(num_blocks):
        t_b = time.time()
        if args.debug:
            print(f"[mr-build] === block {b}/{num_blocks-1} ===")

        # Capture teacher block weights so we can verify reconstruction.
        if args.debug and b == args.probe_block:
            pre_weights = {}
            for name, module in teacher.model.layers[b].named_modules():
                if isinstance(module, nn.Linear):
                    pre_weights[name] = module.weight.data.clone()

        student.model.layers[b] = convert_block_to_multirank(
            student.model.layers[b],
            rank=args.rank, K=args.K, tau=args.tau,
            debug=args.debug and b == args.probe_block,
        ).to(device)

        # Post-conversion: verify reconstruction on this block's linears.
        if args.debug and b == args.probe_block:
            print(f"[mr-build] post-conversion reconstruction check (block {b}):")
            for name, module in student.model.layers[b].named_modules():
                if isinstance(module, LittleBitLinearMultiRankHF):
                    with torch.no_grad():
                        W_hat = module.reconstruct_W()
                    W_orig = pre_weights.get(name)
                    if W_orig is None:
                        continue
                    err = rel_err(W_hat, W_orig.to(device))
                    print(f"[mr-build]   {name}  Frob rel-err={err:.4f}")

        blk_time = time.time() - t_b
        per_block_times.append(blk_time)
        if not args.debug:
            # Non-debug concise per-block line
            print(f"[mr-build]   block {b}/{num_blocks-1} done in {blk_time:.1f}s")
        write_status(status_path, {
            "phase": "building",
            "blocks_total": num_blocks,
            "blocks_done": b + 1,
            "elapsed_s": time.time() - t_build,
        })

    total_build = time.time() - t_build
    avg = sum(per_block_times) / len(per_block_times)
    print(f"[mr-build] total build: {total_build:.1f}s  "
          f"(avg {avg:.1f}s/block)")

    # ----- Parameter budget debug -----
    print(f"[mr-build] parameter budget breakdown (across all blocks):")
    pb = count_params(student.model)
    total = pb["total"]
    for role, n in pb.items():
        if role == "total":
            continue
        print(f"[mr-build]   {role:>10s}: {n:>12d}  ({100*n/total:5.2f}%)")
    print(f"[mr-build]   {'total':>10s}: {total:>12d}")

    # ----- Save checkpoint -----
    ckpt_path = Path(args.out)
    print(f"[mr-build] saving state_dict -> {ckpt_path}")
    torch.save(student.state_dict(), ckpt_path)

    # ----- Optional eval -----
    if args.eval:
        teacher_ppl = get_or_eval_teacher_ppl(
            teacher, tokenizer, device,
            model_id=args.model,
            seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
        )
        print(f"[mr-build] eval student PPL at init (K={args.K})...")
        t_e = time.time()
        student_ppl, n_tok = eval_ppl(
            student, tokenizer, device,
            seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
        )
        ratio = student_ppl / teacher_ppl
        print(f"[mr-build]   student PPL = {student_ppl:.3f} "
              f"({ratio:.2f}x teacher, {time.time()-t_e:.1f}s)")

        result = {
            "config": {
                "model": args.model, "rank": args.rank, "K": args.K,
                "tau": args.tau, "seq_len": args.seq_len,
            },
            "teacher_ppl": teacher_ppl,
            "student_ppl": student_ppl,
            "ppl_ratio": ratio,
            "build_seconds": total_build,
            "avg_block_seconds": avg,
            "param_breakdown": pb,
        }
        summary_path = ckpt_path.with_suffix(".json")
        summary_path.write_text(json.dumps(result, indent=2))
        print(f"[mr-build] saved {summary_path}")


if __name__ == "__main__":
    main()
