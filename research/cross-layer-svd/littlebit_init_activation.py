"""Stage 2 probe: activation-weighted Dual-SVID init, single-matrix.

Compares two init schemes on one matrix:
  (a) Frobenius Dual-SVID (current baseline) — minimizes ||W - W_hat||_F^2
  (b) Activation-weighted Dual-SVID — minimizes ||X·W^T - X·W_hat^T||_F^2
       where the activation Gramian H = E[X^T X] replaces the identity.

Math: for quadratic form ||A||_H^2 := tr(A H A^T), the best rank-r
approximation comes from SVD of A·H^(1/2) (whitening the row space),
then unwhitening the right factor.  Concretely:

    W @ H^(1/2) = U Σ V^T      (standard SVD of the whitened weight)
    Up_aw  = U_r · sqrt(Σ_r)
    Vp_aw  = H^(-1/2) @ V_r · sqrt(Σ_r)

Then Dual-SVID's magnitude rank-1 split (|Up| ≈ h·l_u, |Vp| ≈ g·l_v)
proceeds identically.  The sign matrix signs(Up_aw), sign(Vp_aw) are
the activation-weighted versions — they preserve the subspace that
actually carries input energy, rather than the Frobenius-dominant
directions that may carry none.

[littlebit_math.md §13.2] showed that gradient-trained activation-
weighted QAT recovers ~90% activation energy on a single matrix vs
Frobenius QAT's ~25%.  This script asks: how much of that recovery
comes for free from picking the right signs at init?

Reuses `collect_xtx` from `littlebit_qat_activation.py` for Gramian
collection and `LittleBitLinearHF.from_linear` for Dual-SVID
comparison.

Usage:
    python littlebit_init_activation.py --model Qwen/Qwen2.5-0.5B \\
        --role mlp.gate_proj --layer 12 --rank 512 \\
        --calib-samples 32 --seq-len 2048 \\
        --out activation_init_gate12.json
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
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from littlebit_qat_activation import collect_xtx
from littlebit_qat_model import LittleBitLinearHF


def dual_svid_from_W(W: np.ndarray, r: int):
    """Standard (Frobenius) Dual-SVID decomposition.
    Returns (Up, Vp, h, g, ell) as numpy arrays.
    Matches LittleBitLinearHF.from_linear's math in fp64 for stability.
    """
    d_out, d_in = W.shape
    r_eff = min(r, d_out, d_in)
    U_full, S_full, VT_full = np.linalg.svd(W, full_matrices=False)
    Uk = U_full[:, :r_eff]
    Sk = S_full[:r_eff]
    Vk = VT_full[:r_eff, :].T
    sqrt_S = np.sqrt(Sk)
    Up = Uk * sqrt_S[None, :]
    Vp = Vk * sqrt_S[None, :]

    U_abs = np.abs(Up)
    V_abs = np.abs(Vp)
    uU, sU, vtU = np.linalg.svd(U_abs, full_matrices=False)
    uV, sV, vtV = np.linalg.svd(V_abs, full_matrices=False)
    h0 = uU[:, 0] * np.sqrt(sU[0])
    l_u0 = vtU[0, :] * np.sqrt(sU[0])
    g0 = uV[:, 0] * np.sqrt(sV[0])
    l_v0 = vtV[0, :] * np.sqrt(sV[0])
    if h0.sum() < 0:
        h0 = -h0; l_u0 = -l_u0
    if g0.sum() < 0:
        g0 = -g0; l_v0 = -l_v0
    ell0 = l_u0 * l_v0
    return Up, Vp, h0, g0, ell0


def activation_weighted_dual_svid(W: np.ndarray, H: np.ndarray, r: int,
                                   reg: float = 1e-6):
    """Activation-weighted Dual-SVID.

    H is the d_in x d_in input Gramian E[X^T X].  Regularized with `reg * I`
    before inversion to handle near-singular cases.

    Returns (Up, Vp, h, g, ell) with the same shapes as dual_svid_from_W,
    but the sign patterns are chosen to minimize the activation-weighted
    approximation error.
    """
    d_out, d_in = W.shape
    r_eff = min(r, d_out, d_in)

    # Regularize H, then compute H^(1/2) and H^(-1/2) via eigendecomp.
    # H is symmetric PSD.  We use its eigenvalues; for near-zero
    # eigenvalues the reg floor prevents H^(-1/2) from blowing up.
    H_reg = H + reg * np.eye(d_in) * (np.trace(H) / d_in)
    w, V_H = np.linalg.eigh(H_reg)
    w = np.clip(w, a_min=reg * (np.trace(H) / d_in), a_max=None)
    H_sqrt = V_H @ np.diag(np.sqrt(w)) @ V_H.T
    H_invsqrt = V_H @ np.diag(1.0 / np.sqrt(w)) @ V_H.T

    # SVD of whitened weight.
    Wp = W @ H_sqrt  # (d_out, d_in)
    U_full, S_full, VT_full = np.linalg.svd(Wp, full_matrices=False)
    Uk = U_full[:, :r_eff]
    Sk = S_full[:r_eff]
    Vk_white = VT_full[:r_eff, :].T  # (d_in, r) in whitened basis
    sqrt_S = np.sqrt(Sk)

    Up = Uk * sqrt_S[None, :]  # (d_out, r) — unchanged (row space whitened)
    # Unwhiten right factor back to original input basis.
    Vp = (H_invsqrt @ Vk_white) * sqrt_S[None, :]  # (d_in, r)

    # Dual-SVID magnitude rank-1 split (identical to Frobenius version).
    U_abs = np.abs(Up)
    V_abs = np.abs(Vp)
    uU, sU, vtU = np.linalg.svd(U_abs, full_matrices=False)
    uV, sV, vtV = np.linalg.svd(V_abs, full_matrices=False)
    h0 = uU[:, 0] * np.sqrt(sU[0])
    l_u0 = vtU[0, :] * np.sqrt(sU[0])
    g0 = uV[:, 0] * np.sqrt(sV[0])
    l_v0 = vtV[0, :] * np.sqrt(sV[0])
    if h0.sum() < 0:
        h0 = -h0; l_u0 = -l_u0
    if g0.sum() < 0:
        g0 = -g0; l_v0 = -l_v0
    ell0 = l_u0 * l_v0
    return Up, Vp, h0, g0, ell0


def reconstruct_from_svid(Up, Vp, h, g, ell) -> np.ndarray:
    """Reconstruct the approximated W from Dual-SVID params (as the
    LittleBit forward would compute it, minus the x term)."""
    U_sign = np.sign(Up)
    V_sign = np.sign(Vp)
    # W_hat = diag(h) · sign(U) · diag(ell) · sign(V^T) · diag(g)
    #       (d_out x d_out)(d_out x r)(r x r)(r x d_in)(d_in x d_in)
    return (h[:, None] * U_sign) @ np.diag(ell) @ V_sign.T @ np.diag(g)


def reconstruct_from_signs(sU, sV, h, g, ell) -> np.ndarray:
    """Same reconstruction but take pre-signed matrices."""
    return (h[:, None] * sU) @ np.diag(ell) @ sV.T @ np.diag(g)


def scale_refinement_aw(
    W: np.ndarray,
    H: np.ndarray,
    sU: np.ndarray,
    sV: np.ndarray,
    h: np.ndarray,
    g: np.ndarray,
    ell: np.ndarray,
    iters: int = 20,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple:
    """Activation-weighted scale refinement via alternating least squares.

    Given fixed sign matrices sU, sV and an initial (h, g, ell), refine
    (h, g, ell) to minimize tr((W - W_hat) H (W - W_hat)^T) where
    W_hat = diag(h) · sU · diag(ell) · sV^T · diag(g).

    Each of h, g, ell enters W_hat linearly given the others, so each
    ALS substep is a normal-equation solve (closed form).

    Cost: O(iters · (d_out + d_in + r) · d²) per iteration, dominated
    by the intermediate M · H matmuls.
    """
    d_out, d_in = W.shape
    r = sU.shape[1]
    assert sU.shape == (d_out, r) and sV.shape == (d_in, r)

    h = h.astype(np.float64).copy()
    g = g.astype(np.float64).copy()
    ell = ell.astype(np.float64).copy()

    def cur_err():
        W_hat = reconstruct_from_signs(sU, sV, h, g, ell)
        D = W - W_hat
        return float(np.sqrt(np.trace(D @ H @ D.T)))

    prev_err = cur_err()
    if verbose:
        print(f"  [als] init act-err: {prev_err:.4f}")

    for it in range(iters):
        # --- Solve for ell holding h, g fixed ---
        # W_hat[i, j] = sum_k h[i] * sU[i,k] * ell[k] * sV[j,k] * g[j]
        # Let B[i, j, k] = h[i] * sU[i,k] * sV[j,k] * g[j]
        # W_hat = sum_k ell[k] * B[:,:,k]  (linear in ell)
        # Normal eq in H-norm: A_ll ell = b_ll where
        #   A_ll[k1, k2] = <B[:,:,k1], B[:,:,k2]>_H = tr(B_k1 H B_k2^T)
        #   b_ll[k] = <W, B[:,:,k]>_H = tr(W H B_k^T)
        hU = h[:, None] * sU  # (d_out, r)
        gV = g[:, None] * sV  # (d_in, r)
        # tr(B_k1 H B_k2^T) = sum_{i,j1,j2} hU[i,k1]gV[j1,k1]H[j1,j2]hU[i,k2]gV[j2,k2]
        #                   = (sum_i hU[i,k1]hU[i,k2]) · (gV^T H gV)[k1,k2]
        hUhU = hU.T @ hU            # (r, r)
        gVHgV = gV.T @ H @ gV       # (r, r)
        A_ll = hUhU * gVHgV         # elementwise; shape (r, r)
        # b_ll[k] = tr(W H B_k^T) = sum_{i, j1, j2} W[i,j1] H[j1,j2] hU[i,k] gV[j2,k]
        #        = (hU^T @ W) @ (H @ gV) ... let's do shapes carefully
        # B_k = hU[:, k:k+1] @ gV[:, k:k+1].T    (d_out, d_in)
        # tr(W H B_k^T) = tr(W H gV[:,k] hU[:,k]^T) = hU[:,k]^T W H gV[:,k]
        WHgV = W @ H @ gV           # (d_out, r)
        b_ll = np.sum(hU * WHgV, axis=0)  # (r,)
        # Regularize to handle any rank deficiency
        A_ll_reg = A_ll + 1e-10 * np.eye(r) * np.trace(A_ll) / r
        ell = np.linalg.solve(A_ll_reg, b_ll)

        # --- Solve for h holding g, ell fixed ---
        # W_hat[i, j] = h[i] * (sum_k sU[i,k] * ell[k] * sV[j,k]) * g[j]
        # Let C[i, j] = (sum_k sU[i,k] ell[k] sV[j,k]) g[j]
        #             = (sU · diag(ell) · sV^T · diag(g))[i, j]
        # Then W_hat = diag(h) · C, so W_hat[i, :] = h[i] * C[i, :]
        # In H-norm, ||W - diag(h) C||_H^2 = sum_i (W[i,:] - h[i] C[i,:]) H (W[i,:]- h[i] C[i,:])^T
        # Each row is independent. For row i:
        #   min_{h[i]} (W[i] - h[i] C[i]) H (W[i] - h[i] C[i])^T
        #   Optimal: h[i] = (W[i] H C[i]^T) / (C[i] H C[i]^T)
        C = sU @ np.diag(ell) @ sV.T @ np.diag(g)  # (d_out, d_in)
        CH = C @ H
        num = np.sum(W * CH, axis=1)           # (d_out,) — equivalent to diag(W @ H @ C^T) wait
        # Actually: W[i] H C[i]^T = row i of (W @ H @ C^T) — i.e. diagonal element
        # But since W and C have same row dim, diag(W @ H @ C^T) = sum_j (W H)[:,j] * C[:,j]
        # Simpler: (W @ H) elementwise (C).sum(axis=1) = diag(W @ H @ C^T)
        num = (W @ H * C).sum(axis=1)          # (d_out,)
        den = (C @ H * C).sum(axis=1)          # (d_out,)
        h = num / np.maximum(den, 1e-20)

        # --- Solve for g holding h, ell fixed ---
        # W_hat[i, j] = (sum_k h[i] sU[i,k] ell[k] sV[j,k]) * g[j]
        # Let D[i, j] = h[i] * (sum_k sU[i,k] ell[k] sV[j,k])
        # Then W_hat = D · diag(g), so W_hat[:, j] = g[j] * D[:, j]
        # In H-norm: ||W - D diag(g)||_H^2 = tr((W - D diag(g)) H (W - D diag(g))^T)
        # Vectorize: Minimize over each g[j] — but g[j] affects column j which appears
        #   in H-weighted sum across ALL columns. Not independent per column.
        # Normal equation: for each j,
        #   d(L)/d(g[j]) = -2 sum_i D[i,j] · ((W - D diag(g)) H)[i, j] = 0
        # Defining R = W - D diag(g): sum_i D[i, j] · (R H)[i, j] = 0 for all j
        # -> D^T (R H) has zero diagonal
        # Let Q = W - D diag(g), then Q H is d_out × d_in; elementwise product with D sums to 0 per column
        # Equivalently: diag(D^T W H) = diag(D^T D diag(g) H)
        #   D^T D diag(g) H = M diag(g) H where M = D^T D (d_in × d_in)
        #   diag(M diag(g) H)[j] = sum_k M[j, k] g[k] H[k, j] = (M ⊙ H^T)[j, :] @ g  (scaled by vector)
        # So normal eq: (M ⊙ H) g = diag(D^T W H)  (using H symmetric so H^T = H)
        D_mat = (h[:, None] * sU) @ np.diag(ell) @ sV.T  # (d_out, d_in)
        M = D_mat.T @ D_mat   # (d_in, d_in)
        A_g = M * H           # elementwise
        b_g = np.sum(D_mat * (W @ H), axis=0)  # diag(D^T W H) = col-sum of elementwise product
        A_g_reg = A_g + 1e-10 * np.eye(d_in) * np.trace(A_g) / d_in
        g = np.linalg.solve(A_g_reg, b_g)

        err = cur_err()
        if verbose:
            print(f"  [als] iter {it + 1}: act-err={err:.4f}  "
                  f"Δ={prev_err - err:.4f}")
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return h, g, ell


def rel_err(approx: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(approx - target) /
                 (np.linalg.norm(target) + 1e-12))


def activation_rel_err(W_hat: np.ndarray, W: np.ndarray,
                        H: np.ndarray) -> float:
    """||X(W - W_hat)^T||_F / ||XW^T||_F computed via the Gramian:
        ||XW^T||_F^2 = tr(W H W^T)
        ||X(W-W_hat)^T||_F^2 = tr((W-W_hat) H (W-W_hat)^T)
    """
    D = W - W_hat
    num = float(np.trace(D @ H @ D.T))
    den = float(np.trace(W @ H @ W.T))
    return float(np.sqrt(num / (den + 1e-12)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--role", default="mlp.gate_proj")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--cache", default="qwen05b_gate12_xtx.pkl",
                   help="Reuse existing XTX cache from littlebit_qat_activation.py")
    p.add_argument("--out", default="activation_init_gate12.json")
    args = p.parse_args()

    cache = Path(args.cache)
    if cache.exists():
        print(f"[aw-init] loading cached W + XTX from {cache}")
        data = pickle.loads(cache.read_bytes())
        W_cpu, xtx = data["W"], data["XTX"]
    else:
        print(f"[aw-init] collecting W + XTX (no cache at {cache})")
        W_cpu, xtx = collect_xtx(
            args.model, args.role, args.layer,
            args.calib_samples, args.seq_len,
        )
        cache.write_bytes(pickle.dumps({"W": W_cpu, "XTX": xtx,
                                        "samples": args.calib_samples,
                                        "seq_len": args.seq_len}))
        print(f"[aw-init] cached to {cache}")

    W = W_cpu.to(torch.float64).numpy()
    H = xtx.to(torch.float64).numpy()
    d_out, d_in = W.shape
    print(f"[aw-init] W shape {W.shape}, ||W||_F={np.linalg.norm(W):.4f}, "
          f"||XW^T||_F={np.sqrt(np.trace(W @ H @ W.T)):.4f}")

    # ---------- Frobenius Dual-SVID ----------
    print(f"[aw-init] Frobenius Dual-SVID at r={args.rank}...")
    t0 = time.time()
    Up_f, Vp_f, h_f, g_f, ell_f = dual_svid_from_W(W, args.rank)
    W_hat_f = reconstruct_from_svid(Up_f, Vp_f, h_f, g_f, ell_f)
    frob_rel_err_f = rel_err(W_hat_f, W)
    act_rel_err_f = activation_rel_err(W_hat_f, W, H)
    print(f"[aw-init]   time {time.time() - t0:.1f}s")
    print(f"[aw-init]   Frobenius rel-err:   {frob_rel_err_f:.4f}")
    print(f"[aw-init]   Activation rel-err:  {act_rel_err_f:.4f}")

    # ---------- Activation-weighted Dual-SVID ----------
    print(f"[aw-init] Activation-weighted Dual-SVID at r={args.rank}...")
    t0 = time.time()
    Up_a, Vp_a, h_a, g_a, ell_a = activation_weighted_dual_svid(
        W, H, args.rank,
    )
    W_hat_a = reconstruct_from_svid(Up_a, Vp_a, h_a, g_a, ell_a)
    frob_rel_err_a = rel_err(W_hat_a, W)
    act_rel_err_a = activation_rel_err(W_hat_a, W, H)
    print(f"[aw-init]   time {time.time() - t0:.1f}s")
    print(f"[aw-init]   Frobenius rel-err:   {frob_rel_err_a:.4f}  "
          f"(expect higher than Frobenius init)")
    print(f"[aw-init]   Activation rel-err:  {act_rel_err_a:.4f}  "
          f"(expect lower than Frobenius init)")

    # ---------- Sign agreement ----------
    sign_U_agree = float((np.sign(Up_f) == np.sign(Up_a)).mean())
    sign_V_agree = float((np.sign(Vp_f) == np.sign(Vp_a)).mean())
    print(f"[aw-init] sign agreement Frobenius vs activation-weighted:")
    print(f"[aw-init]   sign(U): {sign_U_agree:.4f}")
    print(f"[aw-init]   sign(V): {sign_V_agree:.4f}")

    # ---------- Scale refinement (ALS on h, g, ell with frozen Frobenius signs) ----------
    print(f"[aw-init] ALS scale refinement on Frobenius signs...")
    t0 = time.time()
    sU_f = np.sign(Up_f)
    sV_f = np.sign(Vp_f)
    h_ref, g_ref, ell_ref = scale_refinement_aw(
        W, H, sU_f, sV_f, h_f, g_f, ell_f, iters=30, verbose=True,
    )
    W_hat_ref = reconstruct_from_signs(sU_f, sV_f, h_ref, g_ref, ell_ref)
    frob_rel_err_ref = rel_err(W_hat_ref, W)
    act_rel_err_ref = activation_rel_err(W_hat_ref, W, H)
    print(f"[aw-init]   time {time.time() - t0:.1f}s")
    print(f"[aw-init]   Frobenius rel-err:   {frob_rel_err_ref:.4f}  "
          f"(vs Frobenius init: {frob_rel_err_f:.4f})")
    print(f"[aw-init]   Activation rel-err:  {act_rel_err_ref:.4f}  "
          f"(vs Frobenius init: {act_rel_err_f:.4f})")

    # ---------- Report ----------
    improvement_aw = act_rel_err_f / max(act_rel_err_a, 1e-12)
    improvement_ref = act_rel_err_f / max(act_rel_err_ref, 1e-12)
    print(f"[aw-init] =====")
    print(f"[aw-init] Activation rel-err summary:")
    print(f"[aw-init]   Frobenius Dual-SVID:        {act_rel_err_f:.4f}  (baseline)")
    print(f"[aw-init]   Activation-weighted signs:  {act_rel_err_a:.4f}  ({improvement_aw:.2f}x)")
    print(f"[aw-init]   Frob signs + ALS scales:    {act_rel_err_ref:.4f}  ({improvement_ref:.2f}x)")

    result = {
        "config": {
            "model": args.model, "role": args.role, "layer": args.layer,
            "rank": args.rank, "calib_samples": args.calib_samples,
            "seq_len": args.seq_len,
        },
        "frobenius_dual_svid": {
            "frob_rel_err": frob_rel_err_f,
            "act_rel_err": act_rel_err_f,
        },
        "activation_weighted_dual_svid": {
            "frob_rel_err": frob_rel_err_a,
            "act_rel_err": act_rel_err_a,
        },
        "frobenius_signs_als_scales": {
            "frob_rel_err": frob_rel_err_ref,
            "act_rel_err": act_rel_err_ref,
        },
        "sign_agreement_U": sign_U_agree,
        "sign_agreement_V": sign_V_agree,
        "aw_signs_improvement_ratio": improvement_aw,
        "als_scales_improvement_ratio": improvement_ref,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[aw-init] saved {args.out}")


if __name__ == "__main__":
    main()
