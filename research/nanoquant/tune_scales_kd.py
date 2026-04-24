"""TuneScalesKD — paper Algorithm 1 Phase 3 (global scale reconstruction).

After all blocks are quantized via the block-reconstruction loop, freeze
the binary sign matrices (sign(U_latent), sign(V_latent)) everywhere and
fine-tune **only the per-linear scale vectors** `s1, s2` end-to-end
against the teacher's logits via KL divergence.

Paper Appendix C: lr=1e-6, batch_size=1, 8 epochs, cosine LR scheduler.

This directly targets compositional / end-to-end error rather than any
per-block reconstruction objective. It's the paper's only component
that operates on the whole model's forward simultaneously.

Memory note: needs both the quantized student model AND a fresh FP
teacher model loaded simultaneously during KL training. Quant student
is small after factorization (~1–2 GB factor params for Qwen3-4B at
r=2 fp32); teacher is the usual ~8 GB bf16. Total ~12 GB peak, fits
on a 16 GB GPU. Gradient checkpointing is not needed because only
scale-vector gradients are retained (tiny).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import calibration_samples
from quant import BinaryFactoredLinear


@dataclass
class TuneScalesKDStats:
    init_loss: float
    final_loss: float
    bad_steps: int
    n_scale_params: int
    steps_taken: int
    step_kls: list = None
    step_lrs: list = None
    temperature: float = 1.0

    def __post_init__(self):
        if self.step_kls is None:
            self.step_kls = []
        if self.step_lrs is None:
            self.step_lrs = []


def _collect_scale_params(student_model) -> list[torch.nn.Parameter]:
    """All s1 / s2 parameters of BinaryFactoredLinear submodules."""
    scales: list[torch.nn.Parameter] = []
    for m in student_model.modules():
        if isinstance(m, BinaryFactoredLinear):
            scales.append(m.s1)
            scales.append(m.s2)
    return scales


@torch.no_grad()
def _precompute_teacher_log_probs(
    teacher_model, input_ids, device, chunk_size=1, dtype=torch.float16,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Run the teacher once on every calibration sample; return stacked log-probs on CPU.

    With `temperature != 1.0`, log-probs are computed as
    `log_softmax(logits / T)`. The STUDENT side must use the same
    temperature at train time to keep the KL consistent. Temperature > 1
    flattens the distribution (Hinton distillation trick) which
    dramatically reduces the magnitude of the KL when teacher has a few
    near-one probabilities — makes optimization easier at low LR.

    Returns a tensor of shape (n_samples, seq_len, vocab_size) in `dtype`.
    For Qwen3-4B (vocab≈152k, seq=2048, n=32) this is ~20 GB on CPU
    (host has 64 GB — fine).
    """
    teacher_model.eval()
    n = input_ids.shape[0]
    outs: list[torch.Tensor] = []
    for start in tqdm(range(0, n, chunk_size), desc="teacher logprobs"):
        ids = input_ids[start : start + chunk_size].to(device)
        logits = teacher_model(ids, use_cache=False).logits
        if temperature != 1.0:
            logits = logits / temperature
        lp = F.log_softmax(logits.float(), dim=-1).to(dtype=dtype, device="cpu")
        outs.append(lp)
        del logits
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _eval_kd_loss_from_cache(student_model, teacher_log_probs_cpu, input_ids, device) -> float:
    student_model.eval()
    ids = input_ids[:1].to(device)
    s_logits = student_model(ids, use_cache=False).logits
    s_lp = F.log_softmax(s_logits.float(), dim=-1)
    t_lp = teacher_log_probs_cpu[:1].to(device=device, dtype=torch.float32)
    return F.kl_div(s_lp, t_lp, reduction="batchmean", log_target=True).item()


def tune_scales_kd(
    student_model,
    teacher_model,
    tokenizer,
    n_samples: int,
    seq_len: int,
    seed: int,
    steps: int,
    lr: float,
    device,
    batch_size: int = 1,
    cosine_schedule: bool = True,
    verbose: bool = False,
    free_teacher_after_precompute: bool = True,
    temperature: float = 1.0,
    ckpt_path: "str | Path | None" = None,
    ckpt_every: int = 200,
    resume_from_ckpt: bool = True,
) -> TuneScalesKDStats:
    """Freeze all student params except `s1, s2`, then train via KL against teacher.

    Assumes `student_model` is already block-wise quantized (all Linears
    replaced with BinaryFactoredLinear). `teacher_model` is a fresh FP
    copy of the original model in eval mode (no quantization).
    """
    # Calibration data: same seed-deterministic sampling used in phase2.
    calib = calibration_samples(tokenizer, n=n_samples, seq_len=seq_len, seed=seed)
    input_ids = calib.input_ids  # (n_samples, seq_len)

    # --- Freeze everything, then unfreeze only the scale vectors ---
    for p in student_model.parameters():
        p.requires_grad_(False)
    scales = _collect_scale_params(student_model)
    for p in scales:
        p.requires_grad_(True)
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    if not scales:
        raise RuntimeError("no BinaryFactoredLinear.s1/s2 parameters found — was the model quantized?")

    print(
        f"  [tune_scales_kd] {len(scales)} scale params trainable, "
        f"lr={lr:g} steps={steps} batch={batch_size}",
        flush=True,
    )

    opt = torch.optim.AdamW(scales, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
        if cosine_schedule
        else None
    )

    # Precompute teacher log-probs once over all calibration samples so the
    # teacher model can be freed before student backward starts. This
    # removes 8 GB of teacher weights from VRAM during the student's
    # gradient path and gets us out of OOM territory.
    if temperature != 1.0:
        print(f"  [tune_scales_kd] temperature scaling tau={temperature}", flush=True)
    print(f"  [tune_scales_kd] precomputing teacher log-probs for {n_samples} samples", flush=True)
    teacher_log_probs = _precompute_teacher_log_probs(
        teacher_model, input_ids, device, chunk_size=1, dtype=torch.float16,
        temperature=temperature,
    )
    print(
        f"  [tune_scales_kd] teacher log-probs cached: shape={tuple(teacher_log_probs.shape)} "
        f"dtype={teacher_log_probs.dtype} on {teacher_log_probs.device}",
        flush=True,
    )
    if free_teacher_after_precompute:
        teacher_model.to("cpu")
        torch.cuda.empty_cache()
        print("  [tune_scales_kd] teacher moved to CPU, VRAM freed", flush=True)

    # Student stays in eval mode for stability (no dropout/BN in Qwen3).
    # Gradient checkpointing makes student backward at seq=2048 fit in
    # the remaining VRAM budget.
    student_model.eval()
    try:
        student_model.gradient_checkpointing_enable()
        student_model.enable_input_require_grads()
    except Exception as e:
        print(f"  [tune_scales_kd] could not enable gradient checkpointing: {e}", flush=True)

    # Initial KL snapshot (cache-based, avoids live teacher forward).
    init_loss = _eval_kd_loss_from_cache(student_model, teacher_log_probs, input_ids, device)
    print(f"  [tune_scales_kd] init KL = {init_loss:.6f}", flush=True)

    # Resume from prior checkpoint if available.
    start_step = 0
    step_kls: list[float] = []
    step_lrs: list[float] = []
    if ckpt_path is not None and resume_from_ckpt and Path(ckpt_path).exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            student_model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["opt_state"])
            if sched is not None and "sched_state" in ckpt:
                sched.load_state_dict(ckpt["sched_state"])
            start_step = int(ckpt.get("step", 0))
            step_kls = list(ckpt.get("step_kls", []))
            step_lrs = list(ckpt.get("step_lrs", []))
            print(
                f"  [tune_scales_kd] resumed from {ckpt_path} at step {start_step}",
                flush=True,
            )
        except Exception as e:
            print(
                f"  [tune_scales_kd] could not resume from {ckpt_path}: {e} — starting fresh",
                flush=True,
            )

    bad_steps = 0
    log_every = max(1, steps // 20)
    pbar = tqdm(range(steps), desc="TuneScalesKD", leave=False)

    steps_taken = 0
    # If resuming, skip ahead on the pbar count.
    if start_step > 0:
        pbar.update(start_step)
    for step in pbar:
        if step < start_step:
            continue
        idx = torch.randperm(n_samples)[:batch_size]
        ids = input_ids[idx].to(device)
        t_lp = teacher_log_probs[idx].to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.enable_grad():
            s_logits = student_model(ids, use_cache=False).logits
            if temperature != 1.0:
                s_logits = s_logits / temperature
            s_lp = F.log_softmax(s_logits.float(), dim=-1)
            loss = F.kl_div(s_lp, t_lp, reduction="batchmean", log_target=True)
            # Compensate for temperature (gradient scales as 1/T²)
            if temperature != 1.0:
                loss = loss * (temperature ** 2)

        if not torch.isfinite(loss):
            bad_steps += 1
            opt.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()
            step_kls.append(float("nan"))
            step_lrs.append(opt.param_groups[0]["lr"])
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        steps_taken += 1
        step_kls.append(loss.item())
        step_lrs.append(opt.param_groups[0]["lr"])

        # Explicit cleanup so intermediate logit tensors don't accumulate.
        del s_logits, s_lp, t_lp

        if step % log_every == 0 or step == steps - 1:
            pbar.set_postfix(kl=f"{loss.item():.5f}", bad=bad_steps)
            if verbose:
                print(
                    f"  [tune_scales_kd] step {step}: kl={loss.item():.5f} bad={bad_steps}",
                    flush=True,
                )

        # Periodic checkpoint.
        if (
            ckpt_path is not None
            and ckpt_every > 0
            and (step + 1) % ckpt_every == 0
            and step + 1 < steps
        ):
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save(
                {
                    "model_state": student_model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "sched_state": sched.state_dict() if sched is not None else None,
                    "step": step + 1,
                    "step_kls": step_kls,
                    "step_lrs": step_lrs,
                    "bad_steps": bad_steps,
                },
                tmp,
            )
            import os as _os
            _os.replace(tmp, ckpt_path)
            if verbose:
                print(f"  [tune_scales_kd] ckpt saved @ step {step + 1}", flush=True)

    # Final loss snapshot.
    final_loss = _eval_kd_loss_from_cache(student_model, teacher_log_probs, input_ids, device)
    print(
        f"  [tune_scales_kd] final KL = {final_loss:.6f}  (delta = {init_loss - final_loss:+.6f})",
        flush=True,
    )

    # Re-freeze scales — keeps model state predictable for the caller.
    for p in scales:
        p.requires_grad_(False)

    # Disable gradient checkpointing again so eval forward runs at normal speed.
    try:
        student_model.gradient_checkpointing_disable()
    except Exception:
        pass

    return TuneScalesKDStats(
        init_loss=init_loss,
        final_loss=final_loss,
        bad_steps=bad_steps,
        n_scale_params=len(scales),
        steps_taken=steps_taken,
        step_kls=step_kls,
        step_lrs=step_lrs,
        temperature=temperature,
    )
