"""Phase 2 full pipeline with TuneFP — paper Algorithm 1 Step 1 + 2 + 3.

Per block `b` in order 0..N-1:

    1. `Y_b = block_b_original(X_student)`  — target for this block (teacher
       on student input, *not* cached teacher-to-teacher).
    2. `TuneFP`: briefly fine-tune block_b's FP weights so it maps
       `X_student` onto `Y_b`. Absorbs the quantization error that has
       accumulated in the preceding (now-quantized) blocks.
    3. LB-ADMM init on the tuned FP weights → replace linears with
       `BinaryFactoredLinear`.
    4. STE refine against the same `(X_student, Y_b)` pair.
    5. Forward the now-quantized tuned block on `X_student` to produce
       `X_student` for block b+1.

The big break from Phase 1 / no-TuneFP Phase 2: the targets `(X, Y)` for
both TuneFP and STE use the **student's actual current hidden state**,
not the cached teacher-to-teacher boundaries. Cached `boundary_0` is
still used as the initial `X_student` (embedding output); cached
boundaries for b > 0 are unused here.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from activations import Cache
from phase1 import BlockStats, train_block
from quant import BinaryFactoredLinear, quant_params, replace_linears_with_quant
from tune_fp import (
    TuneFPStats,
    compute_teacher_on_student,
    forward_block_chunked,
    tune_fp,
)


def _capture_admm_init_stats(block: nn.Module, orig_linears: dict[str, nn.Linear]) -> dict:
    """Per-linear diagnostics immediately after LB-ADMM init, before STE.

    Records weight-space relative Frobenius error and per-param magnitudes.
    Compares against the pre-TuneFP original Linear weights stored in
    `orig_linears` — so the error includes both TuneFP drift and the
    signing cost of LB-ADMM.
    """
    out: dict = {}
    for name, bf in block.named_modules():
        if not isinstance(bf, BinaryFactoredLinear):
            continue
        orig = orig_linears.get(name)
        if orig is None:
            continue
        with torch.no_grad():
            W = orig.weight.data.float()
            s1 = bf.s1.float().unsqueeze(1)
            s2 = bf.s2.float().unsqueeze(0)
            U_sign = torch.sign(bf.U_latent.float())
            V_sign = torch.sign(bf.V_latent.float())
            W_eff = (s1 * U_sign) @ (V_sign.T * s2)
            diff = (W - W_eff).norm().item()
            denom = max(W.norm().item(), 1e-12)
        out[name] = {
            "shape": tuple(W.shape),
            "frob_rel_err": diff / denom,
            "U_abs_mean": bf.U_latent.abs().mean().item(),
            "U_abs_max": bf.U_latent.abs().max().item(),
            "V_abs_mean": bf.V_latent.abs().mean().item(),
            "V_abs_max": bf.V_latent.abs().max().item(),
            "s1_abs_mean": bf.s1.abs().mean().item(),
            "s2_abs_mean": bf.s2.abs().mean().item(),
            "frac_U_under_1_ste_active": (bf.U_latent.abs() < 1).float().mean().item(),
        }
    return out


@dataclass
class Phase2BlockStats:
    idx: int
    tune_fp: TuneFPStats | None
    ste: BlockStats
    wall_seconds: float = 0.0
    admm_init: dict = None  # per-linear: frob_rel, U_abs_mean, s1_abs_mean, etc.

    def __post_init__(self):
        if self.admm_init is None:
            self.admm_init = {}


def _pick_compute_dtype(block: nn.Module) -> torch.dtype:
    """Compute dtype = first non-quant-param dtype (RMSNorm weight in Qwen).

    Avoids forcing fp32 forward when the quant params are fp32 but the
    surrounding block is bf16 (would disable SDPA fast-path).
    """
    quant_ids = {id(p) for p in quant_params(block)}
    for p in block.parameters():
        if id(p) not in quant_ids:
            return p.dtype
    return torch.float32


def quantize_model_with_tunefp(
    model,
    cache: Cache,
    r: int,
    steps_per_block: int,
    lr: float,
    device,
    init_fn_factory,
    tune_fp_steps: int,
    tune_fp_lr: float,
    tune_fp_batch: int = 4,
    status_file: Path | None = None,
    verbose_first_blocks: int = 0,
) -> list[Phase2BlockStats]:
    n_blocks = len(model.model.layers)
    if n_blocks != cache.n_blocks:
        raise ValueError(
            f"model has {n_blocks} blocks but cache has {cache.n_blocks}"
        )
    aux_kwargs = cache.load_aux_kwargs(device)

    # X_student starts as the teacher's input to block 0 — the embedding
    # output, which we already cached as boundary_0.
    X_student = cache.load_boundary(0)

    stats: list[Phase2BlockStats] = []
    for b in range(n_blocks):
        t_block_start = time.time()
        block = model.model.layers[b]
        verbose = b < verbose_first_blocks

        # Initial compute dtype (block is still all-FP here; use its
        # first param's dtype — for a pristine Qwen3 block that's bf16).
        compute_dtype = next(p for p in block.parameters()).dtype

        # Snapshot the original FP Linear modules for admm-init diagnostic.
        # We need them to compute frob_rel_err after ADMM replaces them.
        orig_linears: dict[str, nn.Linear] = {}
        for name, m in block.named_modules():
            if isinstance(m, nn.Linear):
                orig_linears[name] = copy.deepcopy(m)

        # Target Y_b: cached teacher block output on teacher input
        # (= boundary_{b+1} in our cache). The paper's Algorithm 1 line 9
        # `Y ← B_b(X)` uses the *calibration* X, not the running student
        # X — otherwise Y would equal the block's current output on the
        # student input and TuneFP loss would be trivially zero. We
        # verified this empirically on a smoke run (loss=0 across all
        # blocks when Y was recomputed as block_FP(X_student)).
        Y_b = cache.load_boundary(b + 1)

        # Step 1: TuneFP — fine-tune block FP weights to absorb upstream
        # quantization error before factorizing.
        tunefp_stats: TuneFPStats | None = None
        if tune_fp_steps > 0:
            tunefp_stats = tune_fp(
                block,
                X_student,
                Y_b,
                aux_kwargs,
                steps=tune_fp_steps,
                lr=tune_fp_lr,
                device=device,
                target_dtype=compute_dtype,
                batch_size=tune_fp_batch,
                verbose=verbose,
                block_idx=b,
            )

        # Step 2: LB-ADMM init using the (possibly tuned) FP weights.
        init_fn = None if init_fn_factory is None else init_fn_factory(b, block)
        replaced = replace_linears_with_quant(block, r=r, init_fn=init_fn)
        block.to(device)
        if not replaced:
            print(f"[block {b:2d}] no Linears found — skipping", flush=True)
            continue

        # Diagnostic snapshot AFTER ADMM init, BEFORE STE.
        admm_init_stats = _capture_admm_init_stats(block, orig_linears)

        # Step 3: STE refinement on (X_student, Y_b) — not the cached
        # teacher-to-teacher pair.
        ste_stats = train_block(
            block,
            X_student,
            Y_b,
            aux_kwargs,
            steps=steps_per_block,
            lr=lr,
            device=device,
            block_idx=b,
            verbose=verbose,
        )
        block_wall = time.time() - t_block_start
        stats.append(
            Phase2BlockStats(
                idx=b,
                tune_fp=tunefp_stats,
                ste=ste_stats,
                wall_seconds=block_wall,
                admm_init=admm_init_stats,
            )
        )

        # Free the deepcopy'd originals.
        del orig_linears

        tfp_line = (
            f" tunefp {tunefp_stats.init_loss:.4f}->{tunefp_stats.final_loss:.4f}"
            if tunefp_stats is not None
            else ""
        )
        print(
            f"[block {b:2d}]{tfp_line}  init_mse={ste_stats.init_mse:.4f} "
            f"final_mse={ste_stats.final_mse:.4f}",
            flush=True,
        )

        if status_file is not None:
            with open(status_file, "w") as f:
                json.dump(
                    {
                        "completed_blocks": b + 1,
                        "total_blocks": n_blocks,
                        "stats": [
                            {
                                "idx": s.idx,
                                "tune_fp": (
                                    {
                                        "init_loss": s.tune_fp.init_loss,
                                        "final_loss": s.tune_fp.final_loss,
                                        "bad_steps": s.tune_fp.bad_steps,
                                        "n_params_trained": s.tune_fp.n_params_trained,
                                    }
                                    if s.tune_fp is not None
                                    else None
                                ),
                                "ste": {
                                    "init_mse": s.ste.init_mse,
                                    "final_mse": s.ste.final_mse,
                                    "n_linears_replaced": s.ste.n_linears_replaced,
                                },
                            }
                            for s in stats
                        ],
                    },
                    f,
                    indent=2,
                )

        # Step 4: propagate X_student through the now-quantized block.
        X_student_next = forward_block_chunked(
            block, X_student, aux_kwargs, device, compute_dtype, chunk_size=4
        )
        del X_student, Y_b
        X_student = X_student_next
        torch.cuda.empty_cache()

    return stats
