"""TuneFP — Algorithm 1 Step 1, error propagation mitigation.

For each block b during the block-reconstruction loop, *before* we
factorize it into binary form, we briefly tune the block's full-precision
weights so that

    block_tuned(X_student)  ≈  block_original(X_student)

where `X_student` is the actual (partially-quantized) hidden state
flowing into block b at this point in the pipeline — not the
teacher-to-teacher cached boundary. This lets the block absorb the
quantization error accumulated in preceding blocks *before* we then
quantize it via LB-ADMM, so that when its own quantization error piles
on, the input it receives is pre-compensated.

The target `Y_teacher` is computed fresh per block as the *original*
FP block's output on the current student input — not the cached
teacher-to-teacher output.

Paper Appendix C hyperparameters:
  lr = 1e-4,  batch_size = 4,  8 epochs,  cosine LR schedule.

This implementation exposes those as arguments so we can sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class TuneFPStats:
    init_loss: float
    final_loss: float
    bad_steps: int
    n_params_trained: int


def _block_forward(block: nn.Module, x: torch.Tensor, aux_kwargs: dict) -> torch.Tensor:
    out = block(x, **aux_kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.inference_mode()
def compute_teacher_on_student(
    block: nn.Module,
    X_student: torch.Tensor,
    aux_kwargs: dict,
    device,
    target_dtype: torch.dtype,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Run the (still-FP) block on the student input chunk-by-chunk.

    Returns an fp16 CPU tensor matching `X_student`'s shape. This is the
    target Y_b that both TuneFP and the subsequent STE step fit against.
    """
    block.eval()
    n = X_student.shape[0]
    outs = []
    for start in range(0, n, chunk_size):
        x = X_student[start : start + chunk_size].to(device=device, dtype=target_dtype)
        y = _block_forward(block, x, aux_kwargs)
        outs.append(y.to(torch.float16).cpu())
    return torch.cat(outs, dim=0)


@torch.inference_mode()
def forward_block_chunked(
    block: nn.Module,
    X: torch.Tensor,
    aux_kwargs: dict,
    device,
    target_dtype: torch.dtype,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Apply `block` to every sample in `X`, returning fp16 on CPU.

    Used after TuneFP + quantization to compute the student input to the
    next block.
    """
    return compute_teacher_on_student(block, X, aux_kwargs, device, target_dtype, chunk_size)


def tune_fp(
    block: nn.Module,
    X_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    aux_kwargs: dict,
    steps: int,
    lr: float,
    device,
    target_dtype: torch.dtype,
    batch_size: int = 4,
    weight_decay: float = 0.0,
    cosine_schedule: bool = True,
    verbose: bool = False,
    block_idx: int = -1,
) -> TuneFPStats:
    """Fine-tune every parameter of `block` so `block(X_student) ≈ Y_teacher`.

    All block parameters are unfrozen for this call (including RMSNorm
    weights — the paper says "full-precision weights of the current
    block", and these are what will be frozen again before STE turns on
    only the quant params).
    """
    # Unfreeze all block parameters.
    for p in block.parameters():
        p.requires_grad_(True)
    trainable = [p for p in block.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"block {block_idx} has no parameters to TuneFP")

    opt = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
        if cosine_schedule
        else None
    )

    n_samples = X_student.shape[0]
    block.train()

    # Initial loss snapshot
    with torch.inference_mode():
        block.eval()
        y_init = _block_forward(
            block,
            X_student[:batch_size].to(device=device, dtype=target_dtype),
            aux_kwargs,
        )
        z_init = Y_teacher[:batch_size].to(device=device, dtype=target_dtype)
        init_loss = F.mse_loss(y_init.float(), z_init.float()).item()
    block.train()

    bad_steps = 0
    log_every = max(1, steps // 10)

    pbar = tqdm(range(steps), desc=f"TuneFP b{block_idx:2d}", leave=False)
    for step in pbar:
        idx = torch.randperm(n_samples)[:batch_size]
        x = X_student[idx].to(device=device, dtype=target_dtype, non_blocking=True)
        z = Y_teacher[idx].to(device=device, dtype=target_dtype, non_blocking=True)

        y = _block_forward(block, x, aux_kwargs)
        loss = F.mse_loss(y.float(), z.float())

        if not torch.isfinite(loss):
            bad_steps += 1
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        if step % log_every == 0 or step == steps - 1:
            pbar.set_postfix(mse=f"{loss.item():.4f}", bad=bad_steps)

    # Final loss on the same initial batch (for apples-to-apples delta).
    with torch.inference_mode():
        block.eval()
        y_final = _block_forward(
            block,
            X_student[:batch_size].to(device=device, dtype=target_dtype),
            aux_kwargs,
        )
        z_final = Y_teacher[:batch_size].to(device=device, dtype=target_dtype)
        final_loss = F.mse_loss(y_final.float(), z_final.float()).item()

    # Re-freeze — caller will unfreeze just quant params for STE.
    for p in block.parameters():
        p.requires_grad_(False)

    if verbose:
        print(
            f"  [tune_fp b{block_idx}] {len(trainable)} params, "
            f"{init_loss:.4f} -> {final_loss:.4f} (bad={bad_steps})",
            flush=True,
        )

    return TuneFPStats(
        init_loss=init_loss,
        final_loss=final_loss,
        bad_steps=bad_steps,
        n_params_trained=len(trainable),
    )
