"""BRECQ-propagated: sequential per-block QAT with inference-time inputs.

Fix for the residual-dominance + distribution-shift failure diagnosed in
`s4_1_diag.json`.  Each block b is calibrated on its ACTUAL inference-time
input distribution (student's propagated X_b through already-quantized
blocks 0..b-1), against the target `teacher_block_b(X_s_b)` — i.e., what
the teacher block would produce given the same drifted input.

This is the GPTQ-style sequential variant of BRECQ, which published work
uses for scalar quantization.  For LittleBit's factored form with
residual connections, the pure-teacher-input BRECQ (S4.1) failed
catastrophically: block-output rel-err looked clean (0.03-0.09) but
delta-contribution rel-err was 0.83, and assembled PPL was 114x teacher
(1397 vs 12.26).  The residual stream hid the actual block-contribution
error; propagated inference then exposed it.

Algorithm:

    for b = 0..B-1:
        1. Build student_block_b = LittleBit(teacher_block_b) via Dual-SVID.
        2. For step in 1..N:
             run student forward on ids  -> captures X_s_b at block b's input
                (student has LittleBit blocks for 0..b-1, unchanged teacher
                blocks for b..B-1)
             run teacher forward on ids  -> captures teacher's args/kwargs
                for block b (position_embeddings, attention_mask, etc.)
             compute target = teacher_block_b(X_s_b, *args, **kwargs)
             compute Z_s = student_block_b(X_s_b, *args, **kwargs)
             loss = fisher-weighted MSE(Z_s, target)
             backward, step
        3. Commit: student.model.layers[b] = student_block_b

Teacher is never modified, so teacher.model.layers[b] is always the
pristine FP block.  Student accumulates LittleBit blocks sequentially.

Usage:
    python littlebit_qat_brecq_prop.py --model Qwen/Qwen2.5-0.5B \\
        --rank 512 --steps-per-block 500 \\
        --fisher qwen05b_fisher.pt \\
        --calib-samples 32 --seq-len 2048 \\
        --out s4_2_propagated.json
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
import copy
import json
import time
from pathlib import Path

import torch

from littlebit_qat_brecq import (
    convert_block_to_littlebit,
    rel_err,
    enable_littlebit_grads,
    fisher_weighted_mse,
    write_status,
    get_or_eval_teacher_ppl,
)
from littlebit_qat_brecq_full import load_calib, eval_ppl


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
    p.add_argument("--out", default="s4_2_propagated.json")
    p.add_argument("--blocks", default=None,
                   help="Block range 'start:end', default all blocks")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load teacher + student (separate instances) -----
    print(f"[brecq-prop] loading teacher")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)

    print(f"[brecq-prop] loading student copy")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    student.eval()
    for pr in student.parameters():
        pr.requires_grad_(False)
    print(f"[brecq-prop]   loaded in {time.time() - t0:.1f}s")

    num_blocks = len(teacher.model.layers)
    if args.blocks:
        start, end = [int(x) for x in args.blocks.split(":")]
    else:
        start, end = 0, num_blocks
    print(f"[brecq-prop] training blocks [{start}:{end}]")

    # ----- Fisher -----
    print(f"[brecq-prop] loading Fisher from {args.fisher}")
    fisher_data = torch.load(args.fisher, weights_only=False)
    fisher = fisher_data["fisher"]

    # ----- Calibration -----
    print(f"[brecq-prop] preparing calibration pool")
    calib_ids = load_calib(
        tokenizer, args.seq_len, args.calib_samples, args.min_chars, device,
    )
    print(f"[brecq-prop]   {len(calib_ids)} sequences")

    # ----- Teacher reference PPL (cached) -----
    teacher_ppl = get_or_eval_teacher_ppl(
        teacher, tokenizer, device,
        model_id=args.model,
        seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
    )

    # ===== Helper: capture X and kwargs at block idx in a model =====
    def capture_at_block(model, block_idx, input_ids):
        """Run model forward; return (X, args_after_X, kwargs) seen by
        model.model.layers[block_idx].
        """
        cap = {}
        blk = model.model.layers[block_idx]

        def hook(_m, args, kwargs):
            cap["X"] = args[0].detach()
            cap["args"] = tuple(
                a.detach() if torch.is_tensor(a) else a
                for a in args[1:]
            )
            # kwargs can contain tensors, tuples of tensors, or bool flags.
            cap_kwargs = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    cap_kwargs[k] = v.detach()
                elif isinstance(v, tuple):
                    cap_kwargs[k] = tuple(
                        t.detach() if torch.is_tensor(t) else t for t in v
                    )
                else:
                    cap_kwargs[k] = v
            cap["kwargs"] = cap_kwargs

        h = blk.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                _ = model(input_ids, use_cache=False)
        finally:
            h.remove()
        return cap["X"], cap["args"], cap["kwargs"]

    # Status file for live progress.
    status_path = str(Path(args.out).with_suffix(".status.json"))

    # ===== Per-block training loop =====
    t_total = time.time()
    block_histories = {}
    write_status(status_path, {
        "phase": "training",
        "blocks_total": end - start,
        "blocks_done": 0,
        "elapsed_s": 0.0,
        "latest": None,
    })

    for b in range(start, end):
        f_b = fisher[b].to(device)
        teacher_block_b = teacher.model.layers[b]  # frozen, never modified

        # Seed the new student block from TEACHER's block (pristine weights).
        new_block = copy.deepcopy(teacher_block_b)
        new_block = convert_block_to_littlebit(
            new_block, rank=args.rank, tau=args.tau,
        ).to(device)
        # Only LittleBit params get grads; RMSNorms stay frozen.
        enable_littlebit_grads(new_block)

        # Measure init block-output rel-err on student's propagated input.
        # (Probing; not used as loss.)
        X_s, pos_args, pos_kwargs = capture_at_block(
            student, b, calib_ids[0],
        )
        with torch.no_grad():
            tgt_out = teacher_block_b(X_s, *pos_args, **pos_kwargs)
            Z_target = tgt_out[0] if isinstance(tgt_out, tuple) else tgt_out
            stu_out = new_block(X_s, *pos_args, **pos_kwargs)
            Z_student = stu_out[0] if isinstance(stu_out, tuple) else stu_out
        init_rel = rel_err(Z_student, Z_target)

        print(f"[brecq-prop] === block {b}/{num_blocks - 1} === "
              f"(Fisher sum={f_b.sum().item():.3f}, init rel-err={init_rel:.4f})")

        # weight_decay=0 to avoid shrinking scale vectors.
        opt = torch.optim.AdamW(
            [pr for pr in new_block.parameters() if pr.requires_grad],
            lr=args.lr, weight_decay=0.0,
        )
        hist = [{"step": 0, "block_rel_err": init_rel}]
        t_blk = time.time()

        for step in range(1, args.steps_per_block + 1):
            ids = calib_ids[(step - 1) % len(calib_ids)]

            # Capture student's propagated X at block b.
            X_s, pos_args, pos_kwargs = capture_at_block(student, b, ids)

            # Target: teacher block b applied to the same drifted input.
            with torch.no_grad():
                tgt_out = teacher_block_b(X_s, *pos_args, **pos_kwargs)
                Z_target = tgt_out[0] if isinstance(tgt_out, tuple) else tgt_out

            # Student block forward (with grad).
            stu_out = new_block(X_s, *pos_args, **pos_kwargs)
            Z_student = stu_out[0] if isinstance(stu_out, tuple) else stu_out

            loss = fisher_weighted_mse(Z_student, Z_target, f_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % args.log_every == 0 or step == args.steps_per_block:
                # Validation: 4 held-out samples, same metric.
                with torch.no_grad():
                    vals = []
                    for vids in calib_ids[-4:]:
                        Xv, va, vk = capture_at_block(student, b, vids)
                        tout = teacher_block_b(Xv, *va, **vk)
                        Ztv = tout[0] if isinstance(tout, tuple) else tout
                        sout = new_block(Xv, *va, **vk)
                        Zsv = sout[0] if isinstance(sout, tuple) else sout
                        vals.append(rel_err(Zsv, Ztv))
                    val_rel = sum(vals) / len(vals)
                hist.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "block_rel_err": val_rel,
                })
                print(f"[brecq-prop]   step {step:4d}  "
                      f"loss={loss.item():.6f}  rel-err={val_rel:.4f}  "
                      f"t={time.time() - t_blk:.1f}s")

        # Commit: swap new block into student BEFORE moving to block b+1.
        student.model.layers[b] = new_block
        for pr in student.model.layers[b].parameters():
            pr.requires_grad_(False)
        block_histories[str(b)] = hist
        print(f"[brecq-prop]   block {b} done in {time.time() - t_blk:.1f}s, "
              f"rel-err {init_rel:.4f} -> {hist[-1]['block_rel_err']:.4f}")

    total_train = time.time() - t_total
    print(f"[brecq-prop] total: {total_train:.1f}s "
          f"({total_train / max(end - start, 1):.1f}s/block)")

    # ----- Save student checkpoint BEFORE eval -----
    ckpt_path = Path(args.out).with_suffix(".student.pt")
    print(f"[brecq-prop] saving student state_dict -> {ckpt_path}")
    torch.save(student.state_dict(), ckpt_path)

    # ----- Final eval -----
    print(f"[brecq-prop] eval student PPL")
    t_eval = time.time()
    student_ppl, _ = eval_ppl(
        student, tokenizer, device,
        seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
    )
    print(f"[brecq-prop]   student PPL = {student_ppl:.3f} "
          f"({time.time() - t_eval:.1f}s)")
    print(f"[brecq-prop]   teacher -> student: {teacher_ppl:.2f} -> "
          f"{student_ppl:.2f} ({student_ppl / teacher_ppl:.2f}x)")

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
            "variant": "brecq-propagated",
        },
        "teacher_ppl": teacher_ppl,
        "student_ppl": student_ppl,
        "ppl_ratio": student_ppl / teacher_ppl,
        "total_train_seconds": total_train,
        "block_histories": block_histories,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[brecq-prop] saved {args.out}")


if __name__ == "__main__":
    main()
