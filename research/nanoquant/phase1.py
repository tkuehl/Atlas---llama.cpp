"""Phase 1: block-by-block STE refinement with SVD init.

For each transformer block b in order 0..N-1:

  1. Replace every nn.Linear in the block with a BinaryFactoredLinear
     initialized from the FP weight via rank-r SVD.
  2. Load cached teacher (X_b, Z_b) — the FP hidden states at boundaries
     b and b+1 — and train the block via AdamW + MSE against Z_b.
  3. Status is written to disk after every block completes so a crash
     doesn't lose hours of training.

Freezes every non-quantization parameter inside the block (RMSNorm
weights etc.) — training those drifts the teacher's normalization and
was a documented bug in prior block-local work in this repo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from activations import Cache
from quant import quant_params, replace_linears_with_quant


@dataclass
class BlockStats:
    idx: int
    init_mse: float
    final_mse: float
    n_linears_replaced: int


def _block_forward(block: nn.Module, x: torch.Tensor, aux_kwargs: dict) -> torch.Tensor:
    out = block(x, **aux_kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.inference_mode()
def _eval_block_mse(
    block: nn.Module,
    X: torch.Tensor,
    Z: torch.Tensor,
    aux_kwargs: dict,
    device,
    target_dtype: torch.dtype,
    n: int = 4,
) -> float:
    block.eval()
    idx = torch.arange(min(n, X.shape[0]))
    x = X[idx].to(device=device, dtype=target_dtype)
    z = Z[idx].to(device=device, dtype=target_dtype)
    y = _block_forward(block, x, aux_kwargs)
    return F.mse_loss(y.float(), z.float()).item()


def train_block(
    block: nn.Module,
    X: torch.Tensor,
    Z: torch.Tensor,
    aux_kwargs: dict,
    steps: int,
    lr: float,
    device,
    block_idx: int,
    verbose: bool = False,
) -> BlockStats:
    # Freeze everything, then unfreeze only the quant parameters.
    for p in block.parameters():
        p.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for p in quant_params(block):
        p.requires_grad_(True)
        trainable.append(p)
    if not trainable:
        raise RuntimeError(f"block {block_idx} has no quantized parameters")

    if verbose:
        dtypes = {str(p.dtype) for p in trainable}
        first = trainable[0]
        print(
            f"  [dbg b{block_idx}] {len(trainable)} trainables, dtypes={dtypes}, "
            f"first.shape={tuple(first.shape)} |first|mean={first.detach().abs().mean().item():.4g}",
            flush=True,
        )

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    # Cache is stored as fp16 for disk space; the block's compute dtype
    # is the dtype of NON-quant params (RMSNorm weights, bias). We must
    # not use quant-param dtype (fp32 since Phase 2's latent fix), because
    # casting the block's input to fp32 makes attention fall off the
    # flash-attention / SDPA fast path — Phase 2 full run was ~90× slower
    # per step than it should be before this. Take the first non-quant
    # parameter's dtype instead.
    from quant import BinaryFactoredLinear
    quant_param_ids = {id(p) for p in quant_params(block)}
    compute_dtype = next(
        (p.dtype for p in block.parameters() if id(p) not in quant_param_ids),
        torch.float32,
    )
    target_dtype = compute_dtype

    init_mse = _eval_block_mse(block, X, Z, aux_kwargs, device, target_dtype)
    block.train()
    n_samples = X.shape[0]

    # Snapshot a fixed subset of trainables so we can report max|delta| at end.
    if verbose:
        snap = [p.detach().clone() for p in trainable]

    pbar = tqdm(range(steps), desc=f"block {block_idx:2d}", leave=False)
    bad_steps = 0
    log_every = max(1, steps // 20)
    for step in pbar:
        idx = torch.randint(0, n_samples, (1,)).item()
        x = X[idx : idx + 1].to(device=device, dtype=target_dtype, non_blocking=True)
        z = Z[idx : idx + 1].to(device=device, dtype=target_dtype, non_blocking=True)

        y = _block_forward(block, x, aux_kwargs)
        # MSE in fp32 for numerical safety; y may be bf16 / fp16.
        loss = F.mse_loss(y.float(), z.float())

        if not torch.isfinite(loss):
            bad_steps += 1
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if verbose and step == 0:
            grad_norms = [
                (p.grad.norm().item() if p.grad is not None else None)
                for p in trainable[:6]
            ]
            grad_dtypes = [
                (str(p.grad.dtype) if p.grad is not None else None)
                for p in trainable[:6]
            ]
            print(
                f"  [dbg b{block_idx}] step 0: loss={loss.item():.6f} "
                f"grad_norms(first 6)={grad_norms} dtypes={grad_dtypes}",
                flush=True,
            )

        opt.step()

        if step % log_every == 0 or step == steps - 1:
            pbar.set_postfix(mse=f"{loss.item():.4f}", bad=bad_steps)

    if verbose:
        deltas = [
            (p.detach() - s).abs().max().item() for p, s in zip(trainable, snap)
        ]
        max_d = max(deltas)
        any_moved = sum(1 for d in deltas if d > 0)
        print(
            f"  [dbg b{block_idx}] post-train: max_abs_delta={max_d:.6g} "
            f"({any_moved}/{len(deltas)} params moved, bad_steps={bad_steps})",
            flush=True,
        )

    final_mse = _eval_block_mse(block, X, Z, aux_kwargs, device, target_dtype)
    return BlockStats(
        idx=block_idx,
        init_mse=init_mse,
        final_mse=final_mse,
        n_linears_replaced=sum(
            1 for m in block.modules() if type(m).__name__ == "BinaryFactoredLinear"
        ),
    )


def quantize_model_phase1(
    model,
    cache: Cache,
    r: int,
    steps_per_block: int,
    lr: float,
    device,
    status_file: Path | None = None,
    init_fn_factory=None,
    verbose_first_blocks: int = 0,
) -> list[BlockStats]:
    """Block-by-block quantize + STE refine.

    `init_fn_factory`, if given, is called as
        init_fn_factory(block_idx, block) -> init_fn
    where init_fn follows the `replace_linears_with_quant` contract
    (path, linear) -> (U, V, s1, s2). Default is None → plain SVD init
    (Phase 1 behaviour).
    """
    n_blocks = len(model.model.layers)
    if n_blocks != cache.n_blocks:
        raise ValueError(
            f"model has {n_blocks} blocks but cache has {cache.n_blocks}"
        )
    aux_kwargs = cache.load_aux_kwargs(device)

    stats: list[BlockStats] = []
    for b in range(n_blocks):
        block = model.model.layers[b]
        init_fn = None if init_fn_factory is None else init_fn_factory(b, block)
        replaced = replace_linears_with_quant(block, r=r, init_fn=init_fn)
        block.to(device)
        if not replaced:
            print(f"[block {b:2d}] no Linears found — skipping")
            continue

        X = cache.load_boundary(b)
        Z = cache.load_boundary(b + 1)

        s = train_block(
            block,
            X,
            Z,
            aux_kwargs,
            steps=steps_per_block,
            lr=lr,
            device=device,
            block_idx=b,
            verbose=(b < verbose_first_blocks),
        )
        stats.append(s)
        print(
            f"[block {b:2d}] init_mse={s.init_mse:.4f} final_mse={s.final_mse:.4f} "
            f"linears={s.n_linears_replaced}",
            flush=True,
        )

        if status_file is not None:
            with open(status_file, "w") as f:
                json.dump(
                    {
                        "completed_blocks": b + 1,
                        "total_blocks": n_blocks,
                        "stats": [vars(x) for x in stats],
                    },
                    f,
                    indent=2,
                )

        del X, Z
        torch.cuda.empty_cache()

    return stats
