"""Diagnostic for S4.1: measure end-to-end hidden-state drift layer-by-layer.

Loads the teacher and the BRECQ-trained student checkpoint, feeds the same
calibration samples through both in full propagation mode, and reports
per-block:

  1. Block-output rel-err at pure-teacher input (what BRECQ trained against)
  2. Block-output rel-err at student-propagated input (what inference sees)
  3. Delta-contribution rel-err (block's f(X) contribution only, residual removed)

The hypothesis from s4_1_full_500spb.json (PPL 1397 vs teacher 12.26):
the residual stream dominates block-output rel-err, so 0.03-0.09 per-block
numbers hid catastrophic delta-contribution errors that compounded.

If the diagnostic shows:
  - Pure-teacher rel-err matches the saved training numbers (0.03-0.3):
    confirms our training-time measurement was honest per se.
  - Propagated rel-err much higher (>0.3 across most blocks):
    confirms composition failure; residual stream diverges at inference.
  - Delta-contribution rel-err >0.5 across most blocks:
    confirms residual-dominance hid the actual block-contribution quality.

Usage:
    python littlebit_brecq_diag.py --model Qwen/Qwen2.5-0.5B \\
        --rank 512 --ckpt s4_1_full_500spb.student.pt \\
        --samples 8 --seq-len 2048 \\
        --out s4_1_diag.json
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

from littlebit_qat_brecq import convert_block_to_littlebit


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.float()
    b32 = b.float()
    num = torch.linalg.norm(a32 - b32)
    den = torch.linalg.norm(b32)
    return float((num / (den + 1e-12)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--ckpt", default="s4_1_full_500spb.student.pt")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--out", default="s4_1_diag.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load teacher -----
    print(f"[diag] loading teacher {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)
    num_blocks = len(teacher.model.layers)
    print(f"[diag]   {num_blocks} blocks")

    # ----- Rebuild student architecture + load checkpoint -----
    print(f"[diag] rebuilding student architecture (LittleBit blocks)")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    student.eval()
    for pr in student.parameters():
        pr.requires_grad_(False)

    for b in range(num_blocks):
        student.model.layers[b] = convert_block_to_littlebit(
            student.model.layers[b], rank=args.rank,
        ).to(device)
    print(f"[diag]   loading checkpoint {args.ckpt}")
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(sd, strict=True)
    print(f"[diag]   loaded in {time.time() - t0:.1f}s")

    # ----- Load calibration -----
    print(f"[diag] preparing calibration pool ({args.samples} samples)")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_ids = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < args.min_chars:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=args.seq_len)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        calib_ids.append(ids)
        if len(calib_ids) >= args.samples:
            break
    print(f"[diag]   {len(calib_ids)} sequences")

    # ----- Hook: capture every block's input and output on both models -----
    teacher_block_io = [None] * num_blocks
    student_block_io = [None] * num_blocks

    def make_teacher_hook(idx):
        def pre(_m, args, kwargs):
            teacher_block_io[idx] = {"X": args[0].detach()}
        def post(_m, _i, output):
            z = output[0] if isinstance(output, tuple) else output
            teacher_block_io[idx]["Z"] = z.detach()
        return pre, post

    def make_student_hook(idx):
        def pre(_m, args, kwargs):
            student_block_io[idx] = {"X": args[0].detach()}
        def post(_m, _i, output):
            z = output[0] if isinstance(output, tuple) else output
            student_block_io[idx]["Z"] = z.detach()
        return pre, post

    t_handles = []
    for i, blk in enumerate(teacher.model.layers):
        pre, post = make_teacher_hook(i)
        t_handles.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
        t_handles.append(blk.register_forward_hook(post))
    s_handles = []
    for i, blk in enumerate(student.model.layers):
        pre, post = make_student_hook(i)
        s_handles.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
        s_handles.append(blk.register_forward_hook(post))

    # ----- Also need to compare block f(X) contribution (= Z - X) -----
    # Delta = Z_b - X_b is the block's actual contribution to the residual stream.
    # rel_err(Delta_student, Delta_teacher) tells us how well the block's f(X)
    # matches, with residual removed.

    # ----- Run diagnostics -----
    print(f"[diag] running propagated forward on both models")
    # Accumulators: per-block averages across samples
    metric_sums = {
        "teacher_in_student_out_rel":   [0.0] * num_blocks,  # pure-teacher input → student block
        "student_in_student_out_rel":   [0.0] * num_blocks,  # propagated student input → student block
        "delta_rel":                    [0.0] * num_blocks,  # (Z_s - X_s) vs (Z_t - X_t)
        "propagation_drift_X":          [0.0] * num_blocks,  # X_s vs X_t at each block's input
        "propagation_drift_Z":          [0.0] * num_blocks,  # Z_s vs Z_t at each block's output
    }
    n_ok = 0

    with torch.inference_mode():
        for sample_idx, ids in enumerate(calib_ids):
            # (a) Full teacher forward — captures teacher X_b and Z_b for all b.
            _ = teacher(ids, use_cache=False)
            t_cap = [{k: v.clone() for k, v in io.items()}
                     for io in teacher_block_io]

            # (b) Full student forward — captures student X_b and Z_b for all b.
            _ = student(ids, use_cache=False)
            s_cap = [{k: v.clone() for k, v in io.items()}
                     for io in student_block_io]

            # (c) Per-block, also run student block with TEACHER's X_b.
            # This isolates "how well does student block b match teacher block b
            # at the calibration-time input distribution."
            for b in range(num_blocks):
                X_t = t_cap[b]["X"]
                Z_t = t_cap[b]["Z"]
                X_s = s_cap[b]["X"]
                Z_s = s_cap[b]["Z"]

                # Drift at block input (how far has propagation drifted by block b's input)
                metric_sums["propagation_drift_X"][b] += rel_err(X_s, X_t)
                # Drift at block output (propagated path)
                metric_sums["propagation_drift_Z"][b] += rel_err(Z_s, Z_t)
                # (b) = (propagated path at output) = same number
                metric_sums["student_in_student_out_rel"][b] += rel_err(Z_s, Z_t)

                # Delta contribution: f(X) = Z - X (residual removed)
                Df_t = Z_t - X_t
                Df_s = Z_s - X_s
                metric_sums["delta_rel"][b] += rel_err(Df_s, Df_t)

                # Student block applied to teacher's input (one-shot forward, not propagated)
                # To get a fresh forward with student block b on teacher's X_b, we
                # need to call the block directly. We don't have kwargs captured,
                # so skip this unless the block is called directly.

            # For (a) — pure-teacher-input → student-block-output — we need to
            # call student's block on teacher's X_b. We need the kwargs teacher
            # saw (position_embeddings etc). Skip for now; the other three
            # metrics diagnose the core question.
            n_ok += 1
            if (sample_idx + 1) % 2 == 0:
                print(f"[diag]   sample {sample_idx + 1}/{len(calib_ids)}")

    # Average
    for k in metric_sums:
        metric_sums[k] = [v / max(n_ok, 1) for v in metric_sums[k]]

    # ----- Report -----
    print(f"\n[diag] ===== Per-block diagnostic (avg over {n_ok} samples) =====")
    print(f"{'b':>3s}  {'drift_X':>8s}  {'drift_Z':>8s}  {'delta':>8s}")
    for b in range(num_blocks):
        print(f"{b:>3d}  "
              f"{metric_sums['propagation_drift_X'][b]:>8.4f}  "
              f"{metric_sums['propagation_drift_Z'][b]:>8.4f}  "
              f"{metric_sums['delta_rel'][b]:>8.4f}")

    # ----- Cleanup and save -----
    for h in t_handles + s_handles:
        h.remove()

    result = {
        "config": {
            "model": args.model,
            "rank": args.rank,
            "ckpt": args.ckpt,
            "samples": n_ok,
            "seq_len": args.seq_len,
        },
        "per_block": {
            b: {
                k: metric_sums[k][b]
                for k in ["propagation_drift_X", "propagation_drift_Z", "delta_rel",
                          "student_in_student_out_rel"]
            }
            for b in range(num_blocks)
        },
        "summary": {
            "mean_propagation_drift_X": sum(metric_sums["propagation_drift_X"]) / num_blocks,
            "mean_propagation_drift_Z": sum(metric_sums["propagation_drift_Z"]) / num_blocks,
            "mean_delta_rel":           sum(metric_sums["delta_rel"])           / num_blocks,
            "max_propagation_drift_Z":  max(metric_sums["propagation_drift_Z"]),
            "final_propagation_drift_Z": metric_sums["propagation_drift_Z"][-1],
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[diag] saved {args.out}")
    print(f"[diag] SUMMARY:")
    print(f"  mean drift_X (propagation)   = "
          f"{result['summary']['mean_propagation_drift_X']:.4f}")
    print(f"  mean drift_Z (propagation)   = "
          f"{result['summary']['mean_propagation_drift_Z']:.4f}")
    print(f"  final block drift_Z          = "
          f"{result['summary']['final_propagation_drift_Z']:.4f}")
    print(f"  mean delta-contribution err  = "
          f"{result['summary']['mean_delta_rel']:.4f}")


if __name__ == "__main__":
    main()
