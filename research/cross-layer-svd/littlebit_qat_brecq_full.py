"""Full-model BRECQ-style per-block QAT for LittleBit — S4.1.

Extension of `littlebit_qat_brecq.py` (S4.0) to the full 24-block
pipeline.  For each transformer block b in order:

  1. Capture (X_b^teacher, Z_b^teacher, kwargs) from pure-teacher forward.
  2. Construct student_block_b = LittleBit-factored copy of teacher_block_b
     via Dual-SVID init.
  3. Train student_block_b for N steps against Fisher-weighted MSE.
  4. Replace student_model.model.layers[b] with student_block_b.

At the end, student_model has all blocks as LittleBit.  Evaluate WikiText-2
PPL against the frozen teacher.

The two-model pattern is load-bearing: teacher is never modified, so
X_b^teacher for block b is always the true FP32 teacher activation (pure-
teacher input per Stage 4 design, not propagated from student).

Usage:
    python littlebit_qat_brecq_full.py --model Qwen/Qwen2.5-0.5B \\
        --rank 512 --steps-per-block 500 \\
        --fisher qwen05b_fisher.pt \\
        --calib-samples 32 --seq-len 2048 \\
        --out s4_1_full.json
"""

from __future__ import annotations

# Windows UTF-8 bootstrap.
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
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from littlebit_qat_brecq import (
    convert_block_to_littlebit,
    capture_block_io,
    rel_err,
    enable_littlebit_grads,
    fisher_weighted_mse,
    write_status,
    get_or_eval_teacher_ppl,
)


def load_calib(tokenizer, seq_len, samples, min_chars, device):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=seq_len)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        out.append(ids)
        if len(out) >= samples:
            break
    return out


def eval_ppl(model, tokenizer, device, seq_len=2048, max_tokens=25000):
    """Standard wikitext-2 test PPL, capped at max_tokens."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([row["text"] for row in ds if row["text"].strip()])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids.to(device)
    n_tok = min(ids.shape[1], max_tokens)
    ids = ids[:, :n_tok]

    nlls = []
    stride = seq_len
    prev_end = 0
    with torch.inference_mode():
        for begin in range(0, ids.shape[1], stride):
            end = min(begin + seq_len, ids.shape[1])
            target_len = end - prev_end
            window = ids[:, begin:end]
            if window.shape[1] < 2:
                break
            outputs = model(window, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = window[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            nll = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(nll.item())
            prev_end = end
            if end >= ids.shape[1]:
                break
    total_nll = sum(nlls)
    total_tokens = ids.shape[1] - 1
    ppl = math.exp(total_nll / max(total_tokens, 1))
    return ppl, total_tokens


def train_one_block(
    teacher,
    student,
    block_idx,
    rank,
    steps,
    lr,
    tau,
    f_b,
    calib_ids,
    log_every,
    device,
):
    """Train student.model.layers[block_idx] via BRECQ objective.

    Mutates `student` in-place: replaces its block `block_idx` with a
    LittleBit-factored block trained against teacher's block output.
    Returns a history list.
    """
    # Source teacher block (never modified).
    teacher_block = teacher.model.layers[block_idx]

    # Build student block via deepcopy + convert (seeds from current student
    # block's weights, which for block 0 == teacher, and for later blocks
    # may have been altered by earlier student trainings IF we propagated,
    # which we don't — but deepcopy from teacher_block keeps pure-teacher).
    student_block = copy.deepcopy(teacher_block)
    student_block = convert_block_to_littlebit(
        student_block, rank=rank, tau=tau,
    ).to(device)
    # Only LittleBit params get grads; RMSNorms stay frozen at teacher values.
    enable_littlebit_grads(student_block)

    # Teacher I/O capture via hooks on teacher's block.
    captured, handles = capture_block_io(teacher, block_idx)

    def run_student(X_b, args, kwargs):
        out = student_block(X_b, *args, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    f_b_gpu = f_b.to(device)

    # Init rel-err (avg over 4 samples).
    with torch.no_grad():
        init_errs = []
        for ids in calib_ids[: min(4, len(calib_ids))]:
            _ = teacher(ids, use_cache=False)
            X_b = captured["X_b"]
            Z_t = captured["Z_b"]
            Z_s = run_student(X_b, captured["args"], captured["kwargs"])
            init_errs.append(rel_err(Z_s, Z_t))
    init_rel = sum(init_errs) / len(init_errs)

    # weight_decay=0 to avoid shrinking scale vectors (h, g, ell).
    opt = torch.optim.AdamW(
        [pr for pr in student_block.parameters() if pr.requires_grad],
        lr=lr, weight_decay=0.0,
    )
    history = [{"step": 0, "block_rel_err": init_rel}]

    for step in range(1, steps + 1):
        ids = calib_ids[(step - 1) % len(calib_ids)]
        with torch.no_grad():
            _ = teacher(ids, use_cache=False)
        X_b = captured["X_b"]
        Z_t = captured["Z_b"]
        cap_args = captured["args"]
        cap_kwargs = captured["kwargs"]

        Z_s = run_student(X_b, cap_args, cap_kwargs)
        loss = fisher_weighted_mse(Z_s, Z_t, f_b_gpu)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                vals = []
                for vids in calib_ids[-4:]:
                    _ = teacher(vids, use_cache=False)
                    Xv = captured["X_b"]
                    Zvt = captured["Z_b"]
                    Zvs = run_student(Xv, captured["args"],
                                      captured["kwargs"])
                    vals.append(rel_err(Zvs, Zvt))
                val_rel = sum(vals) / len(vals)
            history.append({
                "step": step,
                "loss": float(loss.item()),
                "block_rel_err": val_rel,
            })

    for h in handles:
        h.remove()

    # Commit: replace student's block.
    student.model.layers[block_idx] = student_block

    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--steps-per-block", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--fisher", default="qwen05b_fisher.pt")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-max-tokens", type=int, default=25000)
    p.add_argument("--out", default="s4_1_full.json")
    p.add_argument("--blocks",
                   default=None,
                   help="Block range 'start:end', default all blocks")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load teacher (frozen) + student (copy) -----
    print(f"[brecq-full] loading teacher {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)

    # Student = separate model on same weights; blocks replaced as we go.
    # Loading twice is cleaner than deepcopying the HF model.
    print(f"[brecq-full] loading student copy")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    student.eval()
    for pr in student.parameters():
        pr.requires_grad_(False)
    print(f"[brecq-full]   loaded in {time.time() - t0:.1f}s")

    num_blocks = len(teacher.model.layers)
    if args.blocks:
        start, end = [int(x) for x in args.blocks.split(":")]
    else:
        start, end = 0, num_blocks
    print(f"[brecq-full] training blocks [{start}:{end}] "
          f"({end - start} of {num_blocks})")

    # ----- Fisher -----
    print(f"[brecq-full] loading Fisher from {args.fisher}")
    fisher_data = torch.load(args.fisher, weights_only=False)
    fisher = fisher_data["fisher"]  # (num_blocks, d_model) CPU fp32

    # ----- Calibration pool -----
    print(f"[brecq-full] preparing calibration pool")
    calib_ids = load_calib(
        tokenizer, args.seq_len, args.calib_samples, args.min_chars, device,
    )
    print(f"[brecq-full]   {len(calib_ids)} calibration sequences")

    # ----- Pre-training PPL (teacher reference; cached) -----
    teacher_ppl = get_or_eval_teacher_ppl(
        teacher, tokenizer, device,
        model_id=args.model,
        seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
    )

    # Status file for live progress (readable while training runs).
    status_path = str(Path(args.out).with_suffix(".status.json"))

    # ----- Per-block training -----
    all_histories = {}
    t_total = time.time()
    write_status(status_path, {
        "phase": "training",
        "blocks_total": end - start,
        "blocks_done": 0,
        "elapsed_s": 0.0,
        "latest": None,
    })

    for b in range(start, end):
        f_b = fisher[b]
        print(f"[brecq-full] === block {b}/{num_blocks - 1} === "
              f"(Fisher sum={f_b.sum().item():.3f})")
        t_blk = time.time()
        hist = train_one_block(
            teacher=teacher,
            student=student,
            block_idx=b,
            rank=args.rank,
            steps=args.steps_per_block,
            lr=args.lr,
            tau=args.tau,
            f_b=f_b,
            calib_ids=calib_ids,
            log_every=args.log_every,
            device=device,
        )
        all_histories[str(b)] = hist
        blk_time = time.time() - t_blk
        last = hist[-1]
        first = hist[0]
        print(f"[brecq-full]   block {b}: rel-err "
              f"{first['block_rel_err']:.4f} -> "
              f"{last['block_rel_err']:.4f} in {blk_time:.1f}s")
        write_status(status_path, {
            "phase": "training",
            "blocks_total": end - start,
            "blocks_done": b - start + 1,
            "elapsed_s": time.time() - t_total,
            "latest": {
                "block": b,
                "init_rel_err": first["block_rel_err"],
                "final_rel_err": last["block_rel_err"],
                "block_seconds": blk_time,
            },
        })

    total_train = time.time() - t_total
    print(f"[brecq-full] total per-block training: {total_train:.1f}s")

    # ----- Save student state dict BEFORE eval, so a crash here
    # doesn't lose the trained weights. -----
    ckpt_path = Path(args.out).with_suffix(".student.pt")
    print(f"[brecq-full] saving student state_dict -> {ckpt_path}")
    torch.save(student.state_dict(), ckpt_path)

    # ----- Final eval: student PPL -----
    print(f"[brecq-full] eval student PPL...")
    t_eval = time.time()
    student_ppl, _ = eval_ppl(
        student, tokenizer, device,
        seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
    )
    print(f"[brecq-full]   student PPL = {student_ppl:.3f} "
          f"({time.time() - t_eval:.1f}s)")
    print(f"[brecq-full]   teacher - student gap: "
          f"{teacher_ppl:.2f} -> {student_ppl:.2f} "
          f"({student_ppl / teacher_ppl:.2f}x)")

    # ----- Save -----
    result = {
        "config": {
            "model": args.model,
            "rank": args.rank,
            "steps_per_block": args.steps_per_block,
            "lr": args.lr,
            "tau": args.tau,
            "calib_samples": args.calib_samples,
            "seq_len": args.seq_len,
            "blocks_trained": [start, end],
        },
        "teacher_ppl": teacher_ppl,
        "student_ppl": student_ppl,
        "ppl_ratio": student_ppl / teacher_ppl,
        "total_train_seconds": total_train,
        "block_histories": all_histories,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[brecq-full] saved {args.out}")


if __name__ == "__main__":
    main()
