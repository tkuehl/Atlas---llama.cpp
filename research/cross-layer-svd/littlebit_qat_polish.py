"""End-to-end polish on top of BRECQ-propagated block-local init.

Stage 4 on its own (both pure-teacher and propagated variants) cannot
close the composition gap: block-local calibration has no visibility
into how a block's output affects downstream blocks.  KL-only end-to-end
QAT (§15) closed the loop but required 8000 steps from a broken
Dual-SVID init.

This polish run starts from the S4.2 checkpoint (propagated block-local
init, PPL 775, drift_Z mean 0.24) and runs a short end-to-end KL +
intermediate-MSE fine-tune to let blocks co-adapt.  The hypothesis:
with a well-initialized starting point, a small number of end-to-end
steps should collapse PPL materially because blocks are already close
enough that the cross-block gradient signal points in a productive
direction.

Loss (matches LittleBit paper recipe):
    L = KL(student_logits || teacher_logits) + λ · MSE(hiddens)
    λ = 10  (paper default)

Usage:
    python littlebit_qat_polish.py --model Qwen/Qwen2.5-0.5B \\
        --ckpt s4_2_propagated.student.pt \\
        --steps 500 --lr 1e-4 --lambda-mse 10 \\
        --out s4_3_polish.json
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
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from littlebit_qat_brecq import (
    convert_block_to_littlebit,
    write_status,
    get_or_eval_teacher_ppl,
)
from littlebit_qat_brecq_full import load_calib, eval_ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--ckpt", default="s4_2_propagated.student.pt",
                   help="Path to BRECQ checkpoint to load. Pass empty string "
                        "('') to skip loading and polish from bare Dual-SVID init.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-mse", type=float, default=10.0)
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=100,
                   help="Evaluate PPL every N steps (0 = only at end)")
    p.add_argument("--eval-max-tokens", type=int, default=25000)
    p.add_argument("--out", default="s4_3_polish.json")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear warmup from 0 to --lr over this many steps. "
                        "0 disables warmup (flat LR from step 1).")
    p.add_argument("--cosine-decay", action="store_true",
                   help="After warmup, cosine-decay LR from --lr to --min-lr "
                        "over remaining steps. Reduces post-descent oscillation.")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="Minimum LR at end of cosine decay.")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch size).")
    p.add_argument("--multirank-K", type=int, default=0,
                   help="If >0, rebuild student as multi-rank-magnitude "
                        "LittleBit (K specifies magnitude rank). Required "
                        "when loading a multi-rank checkpoint.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load teacher (frozen) -----
    print(f"[polish] loading teacher {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)

    # ----- Rebuild student and load checkpoint -----
    is_multirank = args.multirank_K > 0
    print(f"[polish] rebuilding student architecture  "
          f"({'multi-rank K=' + str(args.multirank_K) if is_multirank else 'standard LittleBit'})")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    for pr in student.parameters():
        pr.requires_grad_(False)
    num_blocks = len(student.model.layers)
    if is_multirank:
        from littlebit_multirank import convert_block_to_multirank
        for b in range(num_blocks):
            student.model.layers[b] = convert_block_to_multirank(
                student.model.layers[b],
                rank=args.rank, K=args.multirank_K,
            ).to(device)
    else:
        for b in range(num_blocks):
            student.model.layers[b] = convert_block_to_littlebit(
                student.model.layers[b], rank=args.rank,
            ).to(device)
    if args.ckpt:
        print(f"[polish] loading checkpoint {args.ckpt}")
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)
        student.load_state_dict(sd, strict=True)
    else:
        print(f"[polish] NO checkpoint — polishing from bare Dual-SVID init "
              f"(head-to-head vs KL QAT from scratch)")

    # ----- Enable gradients on LittleBit params (not norms/embed/lm_head) -----
    # Include both classic LittleBit (U_fp, V_fp, h, g, ell, bias) and
    # multi-rank variant (U_mag, V_mag_u, V_mag_g, V_mag_lv).  "layers."
    # prefix restricts to transformer blocks — no embedding / lm_head.
    LB_SUFFIXES = (
        "U_fp", "V_fp", ".h", ".g", ".ell", ".bias",  # original Dual-SVID
        "U_mag", "V_mag_u", "V_mag_g", "V_mag_lv",    # multi-rank magnitudes
    )
    trainable = []
    trainable_breakdown = {}
    for name, pr in student.named_parameters():
        matched_suffix = None
        for s in LB_SUFFIXES:
            if name.endswith(s):
                matched_suffix = s
                break
        is_lb = matched_suffix is not None and "layers." in name
        if is_lb:
            pr.requires_grad_(True)
            trainable.append(pr)
            trainable_breakdown[matched_suffix] = (
                trainable_breakdown.get(matched_suffix, 0) + pr.numel()
            )
        else:
            pr.requires_grad_(False)
    n_trainable = sum(pr.numel() for pr in trainable)
    print(f"[polish] trainable params: {n_trainable} across "
          f"{len(trainable)} tensors")
    for s, n in sorted(trainable_breakdown.items(), key=lambda x: -x[1]):
        print(f"[polish]   {s:>10s}: {n:>12d} ({100*n/n_trainable:5.2f}%)")
    print(f"[polish]   loaded in {time.time() - t0:.1f}s")

    # ----- Calibration -----
    print(f"[polish] preparing calibration pool")
    calib_ids = load_calib(
        tokenizer, args.seq_len, args.calib_samples, args.min_chars, device,
    )
    print(f"[polish]   {len(calib_ids)} sequences")

    # ----- Reference PPL (teacher + student-at-init) -----
    teacher_ppl = get_or_eval_teacher_ppl(
        teacher, tokenizer, device,
        model_id=args.model,
        seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
    )

    print(f"[polish] eval student PPL at init (pre-polish)")
    init_ppl, _ = eval_ppl(
        student, tokenizer, device,
        seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
    )
    print(f"[polish]   student PPL at init = {init_ppl:.3f} "
          f"({init_ppl / teacher_ppl:.2f}x)")

    # ----- Hooks to capture hidden states at each block output -----
    # For intermediate-MSE, we need per-block hidden states on both teacher
    # and student.  Use output_hidden_states=True — Qwen supports it.

    # ----- Training loop -----
    # weight_decay=0 to leave scale vectors and shadow weights undecayed.
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # LR schedule: linear warmup then optional cosine decay.  Addresses the
    # oscillation we saw in s4_3_polish (PPL dipped to 387 at step 300,
    # bounced back to 465 at step 400) — flat LR at 1e-4 was too aggressive
    # past the initial descent.
    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        if args.cosine_decay:
            progress = (step - args.warmup_steps) / max(
                args.steps - args.warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            cos_min = args.min_lr / max(args.lr, 1e-12)
            return cos_min + (1.0 - cos_min) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = [{
        "step": 0,
        "loss_total": None,
        "loss_kl": None,
        "loss_mse": None,
        "student_ppl": init_ppl,
    }]

    def kl_loss(student_logits, teacher_logits):
        """Forward KL: student towards teacher."""
        # Flatten (batch, seq, vocab) -> (batch*seq, vocab)
        s = student_logits.float().reshape(-1, student_logits.size(-1))
        t = teacher_logits.float().reshape(-1, teacher_logits.size(-1))
        log_s = F.log_softmax(s, dim=-1)
        t_probs = F.softmax(t, dim=-1)
        # KL(t || s) = sum t * (log t - log s)
        log_t = F.log_softmax(t, dim=-1)
        return (t_probs * (log_t - log_s)).sum(dim=-1).mean()

    print(f"[polish] training: {args.steps} steps, lr={args.lr}, "
          f"λ_mse={args.lambda_mse}")
    t0 = time.time()
    status_path = str(Path(args.out).with_suffix(".status.json"))
    write_status(status_path, {
        "phase": "training",
        "steps_total": args.steps,
        "steps_done": 0,
        "elapsed_s": 0.0,
        "init_ppl": init_ppl,
        "teacher_ppl": teacher_ppl,
        "latest_ppl": init_ppl,
    })

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)

        # Gradient accumulation: accumulate grads over `grad_accum` micro-steps
        # before stepping.  Effective batch size = grad_accum × 1 sequence.
        for micro in range(args.grad_accum):
            micro_idx = (step - 1) * args.grad_accum + micro
            ids = calib_ids[micro_idx % len(calib_ids)]

            # Teacher forward (no grad) with hidden states for intermediate MSE
            with torch.no_grad():
                t_out = teacher(ids, use_cache=False,
                                output_hidden_states=True)
                t_logits = t_out.logits
                t_hiddens = t_out.hidden_states

            # Student forward (with grad)
            s_out = student(ids, use_cache=False,
                            output_hidden_states=True)
            s_logits = s_out.logits
            s_hiddens = s_out.hidden_states

            # KL on logits
            loss_kl_micro = kl_loss(s_logits, t_logits)

            # MSE across hidden states (averaged over depth)
            loss_mse_micro = 0.0
            for t_h, s_h in zip(t_hiddens, s_hiddens):
                loss_mse_micro = loss_mse_micro + (
                    s_h.float() - t_h.float()
                ).pow(2).mean()
            loss_mse_micro = loss_mse_micro / len(t_hiddens)

            loss_total_micro = loss_kl_micro + args.lambda_mse * loss_mse_micro
            (loss_total_micro / args.grad_accum).backward()

            # Record the last micro-step's losses for logging.
            loss_kl = loss_kl_micro.detach()
            loss_mse = loss_mse_micro.detach() if torch.is_tensor(
                loss_mse_micro) else torch.tensor(float(loss_mse_micro))
            loss_total = loss_total_micro.detach()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        scheduler.step()

        if step % args.log_every == 0 or step == args.steps:
            cur_lr = opt.param_groups[0]["lr"]
            print(f"[polish]   step {step:4d}  "
                  f"loss={loss_total.item():.4f} "
                  f"(kl={loss_kl.item():.4f} "
                  f"mse={loss_mse.item():.4f})  "
                  f"lr={cur_lr:.2e}  "
                  f"t={time.time() - t0:.1f}s")

        entry = {"step": step,
                 "loss_total": float(loss_total.item()),
                 "loss_kl": float(loss_kl.item()),
                 "loss_mse": float(loss_mse.item())}

        if args.eval_every and step % args.eval_every == 0:
            ppl, _ = eval_ppl(
                student, tokenizer, device,
                seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
            )
            entry["student_ppl"] = ppl
            print(f"[polish]     PPL at step {step} = {ppl:.3f} "
                  f"({ppl / teacher_ppl:.2f}x)")

        history.append(entry)

        # Status file — overwritten every step; cheap and atomic.
        write_status(status_path, {
            "phase": "training",
            "steps_total": args.steps,
            "steps_done": step,
            "elapsed_s": time.time() - t0,
            "init_ppl": init_ppl,
            "teacher_ppl": teacher_ppl,
            "latest_ppl": entry.get("student_ppl", history[-2].get(
                "student_ppl", init_ppl,
            )) if step > 1 else init_ppl,
            "latest_loss": float(loss_total.item()),
        })

    total_train = time.time() - t0
    print(f"[polish] training done in {total_train:.1f}s")

    # ----- Save + final eval -----
    ckpt_path = Path(args.out).with_suffix(".student.pt")
    print(f"[polish] saving -> {ckpt_path}")
    torch.save(student.state_dict(), ckpt_path)

    # Free optimizer state (Adam moments hold ~2x trainable-param VRAM)
    # before final eval; large-vocab CE loss at seq=2048 on Qwen 1.5B
    # (vocab=151936) allocates >1 GB for softmax tensors, which at 98%
    # VRAM pressure causes CUDA allocator stalls.
    opt = None
    scheduler = None
    for pr in student.parameters():
        pr.requires_grad_(False)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[polish] freed optimizer state, final eval")
    final_ppl, _ = eval_ppl(
        student, tokenizer, device,
        seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
    )
    print(f"[polish] final student PPL = {final_ppl:.3f} "
          f"({final_ppl / teacher_ppl:.2f}x)")
    print(f"[polish] trajectory: "
          f"init {init_ppl:.2f} -> final {final_ppl:.2f} "
          f"(reduction {init_ppl / final_ppl:.2f}x)")

    result = {
        "config": {
            "model": args.model,
            "rank": args.rank,
            "ckpt": args.ckpt,
            "steps": args.steps,
            "lr": args.lr,
            "lambda_mse": args.lambda_mse,
            "calib_samples": args.calib_samples,
            "seq_len": args.seq_len,
        },
        "teacher_ppl": teacher_ppl,
        "student_ppl_init": init_ppl,
        "student_ppl_final": final_ppl,
        "ppl_ratio_init": init_ppl / teacher_ppl,
        "ppl_ratio_final": final_ppl / teacher_ppl,
        "total_train_seconds": total_train,
        "history": history,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[polish] saved {args.out}")


if __name__ == "__main__":
    main()
