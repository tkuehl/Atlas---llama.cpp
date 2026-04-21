"""Numerical sanity checks for LittleBit (arXiv 2506.13771).

Implements the three checks from littlebit_math.md §10.1:

  A. Bit-for-bit verify Proposition 1 (Eq. 5) on a toy matrix.
  B. Dual-SVID (Eq. 6-8) initial-point quality on a real trained matrix:
     measure ||W - W_pri_0||_F / ||W||_F across a rank sweep.
  C. Rank-1 separability of |U'| - the Dual-SVID assumption that
     |U'|_{ik} is approximately h_i * l_u_k.

Target for B and C: Qwen 2.5 0.5B mlp.gate_proj layer 12 (same
matrix the archived CALDERA validation ran on, so the findings
compare directly to the archived PPL 86 at r=512 post-training SVD
floor). Auto-downloads if not cached.

Usage:
  python littlebit_sanity.py --model Qwen/Qwen2.5-0.5B \\
         --role mlp.gate_proj --layer 12 \\
         --ranks 4,8,16,32,64,128,256,512 \\
         --out littlebit_sanity.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def verify_proposition_1(seed: int = 0, d_out: int = 128, d_in: int = 96,
                         r: int = 16, batch: int = 4) -> dict:
    """Part A. Bit-for-bit verify that

        Y_naive = X @ (diag(h) @ sign(U) @ diag(l) @ sign(V).T @ diag(g)).T

    equals

        Y_eff = ((((X * g) @ sign(V)) * l) @ sign(U).T) * h

    on random inputs.  Modulo floating-point rounding only.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((d_out, r)).astype(np.float64)
    V = rng.standard_normal((d_in, r)).astype(np.float64)
    U_sign = np.sign(U)
    V_sign = np.sign(V)
    h = rng.standard_normal(d_out).astype(np.float64)
    g = rng.standard_normal(d_in).astype(np.float64)
    ell = rng.standard_normal(r).astype(np.float64)
    X = rng.standard_normal((batch, d_in)).astype(np.float64)

    # Naive: materialize W_pri, then Y = X @ W_pri.T
    W_pri = np.diag(h) @ U_sign @ np.diag(ell) @ V_sign.T @ np.diag(g)
    Y_naive = X @ W_pri.T

    # Efficient form (Eq. 5)
    # Step by step to keep the broadcast semantics explicit.
    Y_eff = X * g                          # (B, d_in)
    Y_eff = Y_eff @ V_sign                 # (B, r)
    Y_eff = Y_eff * ell                    # (B, r)
    Y_eff = Y_eff @ U_sign.T               # (B, d_out)
    Y_eff = Y_eff * h                      # (B, d_out)

    diff = np.abs(Y_naive - Y_eff)
    return {
        "d_out": d_out, "d_in": d_in, "r": r, "batch": batch,
        "max_abs_diff": float(diff.max()),
        "rel_diff": float(diff.max() /
                          max(1e-30, np.abs(Y_naive).max())),
        "equal_within_1e-10": bool(diff.max() < 1e-10),
    }


def load_matrix(model_id: str, role: str, layer: int,
                load_dtype: str = "bfloat16") -> torch.Tensor:
    """Load a single weight matrix from an HF model; returns float32 CPU.

    Model is loaded at `load_dtype` (default bf16 to keep RAM usage
    reasonable on 7B+ models); only the extracted target matrix is
    upcast to float32 for the SVD step.
    """
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, load_dtype)
    print(f"  loading {model_id} ({load_dtype}) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    target = f"model.layers.{layer}.{role}"
    named = dict(model.named_modules())
    if target not in named:
        avail = [n for n in named if f".layers.{layer}." in n and
                 hasattr(named[n], "weight")]
        raise SystemExit(f"{target} not found. Available L{layer}: {avail}")
    W = named[target].weight.data.detach().clone().to(torch.float32)
    print(f"  extracted {target} [{W.shape[0]} x {W.shape[1]}]")
    return W


def dual_svid(W: torch.Tensor, r: int) -> dict:
    """Part B. Run Dual-SVID at rank r on matrix W.

    Returns dict with:
      - frob_err_rel:    ||W - W_pri_0||_F / ||W||_F after Dual-SVID
      - frob_err_fp_svd: ||W - fp_svd_r||_F / ||W||_F  (the FP16-SVD
                         floor our archived sweep measured)
      - sep_u, sep_v:    rank-1 separability of |U'|, |V'|
                         (sigma_1^2 / sum sigma_i^2)
    """
    W_np = W.numpy().astype(np.float64)
    d_out, d_in = W_np.shape

    # Truncated rank-r SVD, symmetric fold: U' = Uk * sqrt(Sk), V' = Vk * sqrt(Sk).
    U_full, S_full, VT_full = np.linalg.svd(W_np, full_matrices=False)
    Uk = U_full[:, :r]
    Sk = S_full[:r]
    Vk = VT_full[:r, :].T  # d_in x r
    sqrt_S = np.sqrt(Sk)
    Up = Uk * sqrt_S[None, :]   # d_out x r
    Vp = Vk * sqrt_S[None, :]   # d_in  x r

    # Baseline FP-SVD reconstruction (what the archived SVD sweep used).
    W_fp = Up @ Vp.T
    fp_svd_err_rel = (np.linalg.norm(W_np - W_fp) /
                      max(1e-30, np.linalg.norm(W_np)))

    # Eq. 6: sign factors.
    U_sign = np.sign(Up)
    V_sign = np.sign(Vp)

    # Eq. 7: rank-1 SVD of the magnitude matrices.  Full SVD then take
    # the leading singular triple.
    U_abs = np.abs(Up)
    V_abs = np.abs(Vp)

    uU, sU, vtU = np.linalg.svd(U_abs, full_matrices=False)
    uV, sV, vtV = np.linalg.svd(V_abs, full_matrices=False)

    # Rank-1 factors, absorb the singular value symmetrically.
    h0       = uU[:, 0] * np.sqrt(sU[0])           # d_out
    l_u0     = vtU[0, :] * np.sqrt(sU[0])          # r
    g0       = uV[:, 0] * np.sqrt(sV[0])           # d_in
    l_v0     = vtV[0, :] * np.sqrt(sV[0])          # r

    # The paper doesn't fix the overall sign convention; the rank-1
    # SVD of a nonneg matrix admits either (u,v) or (-u,-v).  Flip if
    # needed so h0 and l_u0 are majority positive (matching |U'| >= 0).
    if h0.sum() < 0:
        h0 = -h0; l_u0 = -l_u0
    if g0.sum() < 0:
        g0 = -g0; l_v0 = -l_v0

    # Separability diagnostics.
    sep_u = float(sU[0] ** 2 / np.sum(sU ** 2))
    sep_v = float(sV[0] ** 2 / np.sum(sV ** 2))

    # Eq. 8.
    ell0 = l_u0 * l_v0

    # Eq. 4.
    W_pri = np.diag(h0) @ U_sign @ np.diag(ell0) @ V_sign.T @ np.diag(g0)

    frob_err_rel = (np.linalg.norm(W_np - W_pri) /
                    max(1e-30, np.linalg.norm(W_np)))

    return {
        "r": r,
        "d_out": d_out,
        "d_in": d_in,
        "bpw_primary": (r * (1.0 / d_in + 1.0 / d_out) +
                        16.0 * (1.0 / d_in + 1.0 / d_out +
                                r / (d_out * d_in))),
        "frob_err_rel": float(frob_err_rel),
        "frob_err_fp_svd_rel": float(fp_svd_err_rel),
        "sep_u_rank1": sep_u,
        "sep_v_rank1": sep_v,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--role", default="mlp.gate_proj")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--ranks", default="4,8,16,32,64,128,256,512")
    p.add_argument("--out", default="littlebit_sanity.json")
    args = p.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]

    out = {"part_a_prop1": [], "part_b_dual_svid": []}

    print("=" * 60)
    print("Part A - Proposition 1 numerical equivalence")
    print("=" * 60)
    for r in [4, 16, 64, 256]:
        res = verify_proposition_1(r=r)
        out["part_a_prop1"].append(res)
        print(f"  r={r:4d}  max|diff|={res['max_abs_diff']:.2e}  "
              f"rel={res['rel_diff']:.2e}  "
              f"equal_within_1e-10={res['equal_within_1e-10']}")

    print()
    print("=" * 60)
    print(f"Part B - Dual-SVID on {args.model} "
          f"layers.{args.layer}.{args.role}")
    print("=" * 60)
    W = load_matrix(args.model, args.role, args.layer)
    W_norm = torch.linalg.norm(W).item()
    print(f"  ||W||_F = {W_norm:.4f}")
    print()
    print(f"  {'rank':>5}  {'BPW':>7}  {'fp-SVD err':>10}  "
          f"{'DualSVID err':>12}  {'sep |U|':>8}  {'sep |V|':>8}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}")
    for r in ranks:
        if r > min(W.shape):
            print(f"  r={r} > min(shape)={min(W.shape)}, skipping")
            continue
        res = dual_svid(W, r)
        out["part_b_dual_svid"].append(res)
        print(f"  {r:>5d}  {res['bpw_primary']:>7.3f}  "
              f"{res['frob_err_fp_svd_rel']:>10.4f}  "
              f"{res['frob_err_rel']:>12.4f}  "
              f"{res['sep_u_rank1']:>8.3f}  "
              f"{res['sep_v_rank1']:>8.3f}")

    out["model"] = args.model
    out["role"] = args.role
    out["layer"] = args.layer
    out["W_shape"] = list(W.shape)
    out["W_frob"] = W_norm
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
