"""LB-ADMM hyperparameter sweep on specific blocks.

Measures per-combo:

- **Block init MSE**: teacher-vs-quantized block output MSE on cached
  calibration activations (the metric Phase 1's diagnostic flagged
  blocks 6, 16 as pathological on).
- **Per-linear Frobenius rel error**: `‖W − W_eff‖_F / ‖W‖_F`. The
  pure-weight fidelity the paper does NOT directly optimize.
- **Per-linear activation-weighted rel error**: `‖X(W−W_eff)^T‖_F / ‖X·W^T‖_F`
  where `X` is the cached pre-linear activation. This IS LB-ADMM's
  actual objective; if Frobenius is bad but activation-weighted is
  good, the algorithm is doing its job.
- **|U_latent| and |s1| stats** — diagnose magnitude regime.
- **ADMM convergence residuals** — primal `‖U − Z_U‖_F` and dual
  change `‖Z_U^{(K)} − Z_U^{(K-1)}‖_F` at the final iteration, to
  see if ADMM actually converged.

Outputs a table plus a JSON dump for downstream analysis.

Usage:
    python diag_blocks_sweep.py --blocks 6 16 \
        --K-values 100 400 1000 --lam-values 1e-4 1e-3 1e-2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from activations import Cache
from admm import svid, _rho_schedule
from preconditioner import load_preconditioners
from quant import BinaryFactoredLinear


sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "runs"


LINEAR_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


@torch.no_grad()
def lb_admm_init_instrumented(
    W: torch.Tensor,
    D_in: torch.Tensor,
    D_out: torch.Tensor,
    r: int,
    K: int,
    rho_start: float,
    rho_end: float,
    lam: float,
    eps_diag: float = 1e-8,
    target_dtype: torch.dtype | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """LB-ADMM init with internal diagnostics returned as a stats dict.

    Same math as admm.lb_admm_init; instrumentation-only extension for
    the sweep script. Tracks convergence residuals across iterations.
    """
    device = W.device
    if target_dtype is None:
        target_dtype = W.dtype

    W_f = W.detach().to(torch.float32)
    d_out, d_in = W_f.shape

    D_in = D_in.to(device=device, dtype=torch.float32).clamp(min=eps_diag)
    D_out = D_out.to(device=device, dtype=torch.float32).clamp(min=eps_diag)
    D_in = D_in / D_in.mean().clamp(min=eps_diag)
    D_out = D_out / D_out.mean().clamp(min=eps_diag)

    W_target = D_out.unsqueeze(1) * W_f * D_in.unsqueeze(0)

    g = torch.Generator(device=device).manual_seed(seed)
    U = torch.randn(d_out, r, generator=g, device=device, dtype=torch.float32) / (r ** 0.5)
    V = torch.randn(d_in, r, generator=g, device=device, dtype=torch.float32) / (r ** 0.5)
    Z_U = torch.zeros_like(U)
    Z_V = torch.zeros_like(V)
    Lam_U = torch.zeros_like(U)
    Lam_V = torch.zeros_like(V)

    eye_r = torch.eye(r, device=device, dtype=torch.float32)
    schedule = _rho_schedule(K, rho_start, rho_end)

    # Track convergence
    primal_residual_U_history = []
    primal_residual_V_history = []
    Z_U_prev = torch.zeros_like(Z_U)
    Z_V_prev = torch.zeros_like(Z_V)
    dual_residual_U_history = []
    dual_residual_V_history = []

    for k in range(K):
        rho = schedule[k]

        VtV = V.T @ V
        M = VtV + (rho + lam) * eye_r
        L = torch.linalg.cholesky(M)
        rhs = W_target @ V + rho * (Z_U - Lam_U)
        U = torch.cholesky_solve(rhs.T, L).T

        UtU = U.T @ U
        M = UtU + (rho + lam) * eye_r
        L = torch.linalg.cholesky(M)
        rhs = W_target.T @ U + rho * (Z_V - Lam_V)
        V = torch.cholesky_solve(rhs.T, L).T

        Z_U_prev.copy_(Z_U)
        Z_V_prev.copy_(Z_V)
        Z_U = svid(U + Lam_U)
        Z_V = svid(V + Lam_V)
        Lam_U = Lam_U + (U - Z_U)
        Lam_V = Lam_V + (V - Z_V)

        # Residuals
        primal_residual_U_history.append((U - Z_U).norm().item() / max(U.norm().item(), 1e-12))
        primal_residual_V_history.append((V - Z_V).norm().item() / max(V.norm().item(), 1e-12))
        dual_residual_U_history.append((Z_U - Z_U_prev).norm().item() / max(Z_U.norm().item(), 1e-12))
        dual_residual_V_history.append((Z_V - Z_V_prev).norm().item() / max(Z_V.norm().item(), 1e-12))

    U_hat = U / D_out.unsqueeze(1)
    V_hat = V / D_in.unsqueeze(1)
    U_hat_norm = U_hat.norm()
    V_hat_norm = V_hat.norm()
    eta = (V_hat_norm.clamp(min=1e-12) / U_hat_norm.clamp(min=1e-12)).sqrt()

    U_latent = eta * U_hat
    V_latent = V_hat / eta

    s1 = U_latent.abs().mean(dim=1).clamp(min=eps_diag)
    s2 = V_latent.abs().mean(dim=1).clamp(min=eps_diag)

    stats = {
        "primal_res_U_final": primal_residual_U_history[-1],
        "primal_res_V_final": primal_residual_V_history[-1],
        "dual_res_U_final":   dual_residual_U_history[-1],
        "dual_res_V_final":   dual_residual_V_history[-1],
        "primal_res_U_min":   min(primal_residual_U_history),
        "primal_res_V_min":   min(primal_residual_V_history),
        "U_latent_abs_mean":  U_latent.abs().mean().item(),
        "U_latent_abs_max":   U_latent.abs().max().item(),
        "V_latent_abs_mean":  V_latent.abs().mean().item(),
        "s1_abs_mean":        s1.abs().mean().item(),
        "s2_abs_mean":        s2.abs().mean().item(),
        "eta":                eta.item(),
        "frac_U_under_1":     (U_latent.abs() < 1).float().mean().item(),
    }
    return (
        U_latent.to(target_dtype),
        V_latent.to(target_dtype),
        s1.to(target_dtype),
        s2.to(target_dtype),
        stats,
    )


def _get_sub(module, dotted: str):
    m = module
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


def _set_sub(parent, dotted: str, new_mod):
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


@torch.inference_mode()
def eval_block_init_mse(block, X, Z, aux_kwargs, device, target_dtype):
    block.eval()
    x = X.to(device=device, dtype=target_dtype)
    z = Z.to(device=device, dtype=target_dtype)
    y = block(x, **aux_kwargs)
    if isinstance(y, tuple):
        y = y[0]
    return F.mse_loss(y.float(), z.float()).item()


@torch.inference_mode()
def per_linear_stats(lin_orig, lin_quant, x_sample):
    """Frobenius + (optional) activation-weighted reconstruction error.

    Activation-weighted is only meaningful for linears whose input dim
    matches `x_sample`'s last dim (q/k/v/gate/up which take the block's
    normed hidden state). For o_proj (d_in = num_heads * head_dim) and
    down_proj (d_in = d_ff), x_sample has the wrong shape — return
    Frobenius only.
    """
    W = lin_orig.weight.data.float()
    W_eff = (
        lin_quant.s1.unsqueeze(1).float() * torch.sign(lin_quant.U_latent.float())
    ) @ (
        torch.sign(lin_quant.V_latent.float()).T * lin_quant.s2.unsqueeze(0).float()
    )
    W_diff = W - W_eff
    frob_rel = W_diff.norm().item() / max(W.norm().item(), 1e-12)

    act_rel = None
    if x_sample.shape[-1] == W.shape[1]:  # d_in matches
        x_flat = x_sample.reshape(-1, x_sample.shape[-1]).float()
        xW = x_flat @ W.T
        xW_eff = x_flat @ W_eff.T
        act_rel = (xW - xW_eff).norm().item() / max(xW.norm().item(), 1e-12)
    return frob_rel, act_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--blocks", type=int, nargs="+", default=[6, 16])
    ap.add_argument("--K-values", type=int, nargs="+", default=[100, 400, 1000])
    ap.add_argument("--lam-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--rho-start", type=float, default=0.1)
    ap.add_argument("--rho-end", type=float, default=10.0)
    ap.add_argument("--cache-dir", default=str(HERE / "cache" / "qwen-qwen3-4b" / "n32_L2048_seed0"))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out",
        default=str(RESULTS_DIR / "blocks_sweep.json"),
    )
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"[diag] loading model ({args.dtype})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device)
    model.eval()

    print(f"[diag] loading cache + preconditioners from {args.cache_dir}", flush=True)
    cache = Cache.load(args.cache_dir)
    precond = load_preconditioners(args.cache_dir)
    aux_kwargs = cache.load_aux_kwargs(args.device)
    print(
        f"[diag] cache: n_samples={cache.n_samples} seq_len={cache.seq_len} n_blocks={cache.n_blocks}",
        flush=True,
    )
    print(
        f"[diag] sweeping {len(args.blocks)} blocks x {len(args.K_values)} K x {len(args.lam_values)} lam "
        f"= {len(args.blocks) * len(args.K_values) * len(args.lam_values)} total",
        flush=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Capture the pre-block input activations *as the linears see them*.
    # For the 7 Qwen linears within a block: q/k/v are fed from input_layernorm(x);
    # o_proj from attention's attn_output; gate/up from post_attention_layernorm(x+attn);
    # down_proj from gate_out * up_out. We bucket by role and measure with the cached
    # block input for a cheap approximation: use one calib-sample through the block's
    # norm + attention chain. Activation-weighted errors are approximate but consistent
    # across runs.

    header = (
        f"{'block':>5}  {'K':>5}  {'lam':>8}  {'init_mse':>10}  "
        f"{'|U|':>10}  {'|s1|':>10}  {'frac<1':>7}  "
        f"{'primU':>8}  {'primV':>8}  {'eta':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for block_idx in args.blocks:
        # Snapshot original linears once so we can restore after each combo.
        orig_block = model.model.layers[block_idx]
        fp_linears = {
            name: copy.deepcopy(_get_sub(orig_block, name))
            for name in LINEAR_NAMES
            if isinstance(_get_sub(orig_block, name), nn.Linear)
        }

        X = cache.load_boundary(block_idx)
        Z = cache.load_boundary(block_idx + 1)
        # Take a small sample for activation-weighted reconstruction check.
        x_sample = X[:1].to(device=args.device, dtype=dtype)

        for K in args.K_values:
            for lam in args.lam_values:
                t0 = time.time()

                # Replace each linear with LB-ADMM init at these params.
                combo_stats = {}
                for name, lin in fp_linears.items():
                    full_path = f"model.layers.{block_idx}.{name}"
                    D_in, D_out = precond[full_path]
                    U, V, s1, s2, admm_stats = lb_admm_init_instrumented(
                        lin.weight.data,
                        D_in=D_in.to(args.device),
                        D_out=D_out.to(args.device),
                        r=args.rank,
                        K=K,
                        rho_start=args.rho_start,
                        rho_end=args.rho_end,
                        lam=lam,
                        target_dtype=lin.weight.dtype,
                        seed=args.rank + block_idx * 31 + K * 7 + int(lam * 1e6),
                    )
                    new = BinaryFactoredLinear.from_factors(
                        lin, args.rank, U, V, s1, s2
                    )
                    _set_sub(orig_block, name, new)
                    # Per-linear reconstruction stats
                    fr, ar = per_linear_stats(lin, new, x_sample)
                    combo_stats[name] = {
                        "frob_rel": fr,
                        "act_rel": ar,
                        **admm_stats,
                    }

                # Block init MSE
                target_dtype = dtype
                init_mse = eval_block_init_mse(
                    orig_block, X, Z, aux_kwargs, args.device, target_dtype
                )

                wall = time.time() - t0

                # Aggregate combo-level metrics
                avg_U = sum(s["U_latent_abs_mean"] for s in combo_stats.values()) / len(combo_stats)
                avg_s1 = sum(s["s1_abs_mean"] for s in combo_stats.values()) / len(combo_stats)
                avg_frac = sum(s["frac_U_under_1"] for s in combo_stats.values()) / len(combo_stats)
                avg_primU = sum(s["primal_res_U_final"] for s in combo_stats.values()) / len(combo_stats)
                avg_primV = sum(s["primal_res_V_final"] for s in combo_stats.values()) / len(combo_stats)
                avg_eta = sum(s["eta"] for s in combo_stats.values()) / len(combo_stats)
                avg_frob = sum(s["frob_rel"] for s in combo_stats.values()) / len(combo_stats)
                act_rels = [s["act_rel"] for s in combo_stats.values() if s["act_rel"] is not None]
                avg_act = (sum(act_rels) / len(act_rels)) if act_rels else float("nan")

                print(
                    f"{block_idx:>5d}  {K:>5d}  {lam:>8.4g}  {init_mse:>10.4f}  "
                    f"{avg_U:>10.4g}  {avg_s1:>10.4g}  {avg_frac:>7.3f}  "
                    f"{avg_primU:>8.4f}  {avg_primV:>8.4f}  {avg_eta:>7.3f}"
                    f"   (frob_rel={avg_frob:.3f} act_rel={avg_act:.3f}, {wall:.1f}s)",
                    flush=True,
                )

                rows.append({
                    "block": block_idx,
                    "K": K,
                    "lam": lam,
                    "rho_start": args.rho_start,
                    "rho_end": args.rho_end,
                    "init_mse": init_mse,
                    "avg_U_latent_abs_mean": avg_U,
                    "avg_s1_abs_mean": avg_s1,
                    "avg_frac_U_under_1": avg_frac,
                    "avg_primal_res_U_final": avg_primU,
                    "avg_primal_res_V_final": avg_primV,
                    "avg_eta": avg_eta,
                    "avg_frob_rel": avg_frob,
                    "avg_act_rel": avg_act,
                    "wall_seconds": wall,
                    "per_linear": combo_stats,
                })

                # Restore FP linears for next combo
                for name, fp_lin in fp_linears.items():
                    _set_sub(orig_block, name, copy.deepcopy(fp_lin))

        del fp_linears
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "rank": args.rank,
                "rho_start": args.rho_start,
                "rho_end": args.rho_end,
                "blocks": args.blocks,
                "K_values": args.K_values,
                "lam_values": args.lam_values,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\n[diag] wrote {len(rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
