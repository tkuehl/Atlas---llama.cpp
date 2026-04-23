"""Multi-rank magnitude approximation test for Dual-SVID.

Current Dual-SVID (rank-1 magnitudes):
  |Up| ≈ h · l_u^T       (rank-1 outer product)
  |Vp| ≈ g · l_v^T

This is the format's main information loss after sign truncation — the
rank-1 magnitude assumption discards any magnitude structure that isn't
representable as a single outer product.

This script measures whether higher-rank magnitude approximations
(K=1, 2, 4, 8) materially improve reconstruction quality.  If K=2
beats K=1 by >10% on activation rel-err, rank-1 is the bottleneck
and it's worth considering a format extension.  If K=2 barely moves
anything, magnitude structure is adequately captured at rank 1.

We don't worry about inference cost yet — a K^2 increase in forward
compute would be a separate engineering question.  Here we just test
whether the format has enough capacity given unlimited magnitude rank.

Usage:
    python littlebit_init_multirank.py --cache qwen05b_gate12_xtx.pkl \\
        --rank 512 --mag-ranks 1,2,4,8,16
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


def dual_svid_multirank(W: np.ndarray, r: int, K: int):
    """Dual-SVID with rank-K magnitude approximation.

    Rank-1 (K=1) matches the paper's original Dual-SVID.  Higher K keeps
    more SVD components of |Up| and |Vp| for the magnitude split.

    Returns (S_u, S_v, U_mag, V_mag_u, V_mag_g, V_mag_l_v) where
      S_u: (d_out, r) binary sign
      S_v: (d_in, r) binary sign
      U_mag: (d_out, K) — rows of |Up| approx factor
      V_mag_u: (K, r) — cols of |Up| approx factor (scaled by sqrt sing val)
      V_mag_g: (d_in, K)
      V_mag_l_v: (K, r)

    Reconstruction:
      |Up|_hat = U_mag @ V_mag_u
      |Vp|_hat = V_mag_g @ V_mag_l_v
      W_hat = (S_u * |Up|_hat) @ (S_v * |Vp|_hat)^T
    """
    d_out, d_in = W.shape
    r_eff = min(r, d_out, d_in)
    K_eff = min(K, r_eff)

    U_full, S_full, VT_full = np.linalg.svd(W, full_matrices=False)
    Uk = U_full[:, :r_eff]
    Sk = S_full[:r_eff]
    Vk = VT_full[:r_eff, :].T
    sqrt_S = np.sqrt(Sk)
    Up = Uk * sqrt_S[None, :]
    Vp = Vk * sqrt_S[None, :]

    S_u = np.sign(Up)
    S_v = np.sign(Vp)
    U_abs = np.abs(Up)
    V_abs = np.abs(Vp)

    # Rank-K SVD of |Up| and |Vp|.
    uU, sU, vtU = np.linalg.svd(U_abs, full_matrices=False)
    uV, sV, vtV = np.linalg.svd(V_abs, full_matrices=False)

    # Keep top-K components with sqrt scaling (same as rank-1 case but K-dim)
    U_mag = uU[:, :K_eff] * np.sqrt(sU[:K_eff])[None, :]  # (d_out, K)
    V_mag_u = vtU[:K_eff, :] * np.sqrt(sU[:K_eff])[:, None]  # (K, r)
    V_mag_g = uV[:, :K_eff] * np.sqrt(sV[:K_eff])[None, :]  # (d_in, K)
    V_mag_l_v = vtV[:K_eff, :] * np.sqrt(sV[:K_eff])[:, None]  # (K, r)

    # Sign fix for consistency (pick positive-sum convention, K=1-compatible)
    # This is mostly cosmetic; SVD sign ambiguity doesn't affect |U_mag @ V_mag|
    # since it's the product that matters.  Skip for simplicity.

    return S_u, S_v, U_mag, V_mag_u, V_mag_g, V_mag_l_v


def reconstruct_multirank(S_u, S_v, U_mag, V_mag_u, V_mag_g, V_mag_l_v):
    """Reconstruct W from multi-rank Dual-SVID factors."""
    Up_abs_approx = U_mag @ V_mag_u             # (d_out, r)
    Vp_abs_approx = V_mag_g @ V_mag_l_v         # (d_in, r)
    Up_approx = S_u * Up_abs_approx
    Vp_approx = S_v * Vp_abs_approx
    return Up_approx @ Vp_approx.T


def rel_err(approx, target):
    return float(np.linalg.norm(approx - target) /
                 (np.linalg.norm(target) + 1e-12))


def activation_rel_err(W_hat, W, H):
    D = W - W_hat
    num = float(np.trace(D @ H @ D.T))
    den = float(np.trace(W @ H @ W.T))
    return float(np.sqrt(num / (den + 1e-12)))


def extra_bits_per_weight(d_out, d_in, r, K):
    """Incremental FP32 storage for magnitudes at rank K vs K=1.
    Total weight count in the linear = d_out * d_in.
    Extra bits = (extra magnitude storage at K) * 32 / (d_out * d_in).
    """
    # Rank-1: (d_out + r) + (d_in + r) FP32 values for magnitudes
    # Rank-K: K * (d_out + r) + K * (d_in + r) FP32 values for magnitudes
    # Extra over K=1: (K-1) * (d_out + d_in + 2r) FP32 values
    extra_floats = max(K - 1, 0) * (d_out + d_in + 2 * r)
    return extra_floats * 32 / (d_out * d_in)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="qwen05b_gate12_xtx.pkl")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--mag-ranks", default="1,2,4,8,16",
                   help="Comma-separated magnitude rank Ks to test")
    p.add_argument("--out", default="multirank_probe.json")
    args = p.parse_args()

    cache = Path(args.cache)
    assert cache.exists(), f"missing cache: {cache}"
    data = pickle.loads(cache.read_bytes())
    W_cpu, xtx = data["W"], data["XTX"]
    W = W_cpu.to(torch.float64).numpy()
    H = xtx.to(torch.float64).numpy()
    d_out, d_in = W.shape
    print(f"[mr] W shape {W.shape}, rank r={args.rank}, "
          f"||W||_F={np.linalg.norm(W):.4f}, "
          f"||XW^T||_F={np.sqrt(np.trace(W @ H @ W.T)):.4f}")

    # Reference: upper bound — full rank magnitude approximation
    # (K = r_eff gives exact |Up|, |Vp| so only sign truncation loss).
    Ks = [int(k) for k in args.mag_ranks.split(",")]
    r_eff = min(args.rank, d_out, d_in)
    Ks_with_upper = Ks + [r_eff]

    results = []
    for K in Ks_with_upper:
        t0 = time.time()
        S_u, S_v, U_mag, V_mag_u, V_mag_g, V_mag_l_v = dual_svid_multirank(
            W, args.rank, K,
        )
        W_hat = reconstruct_multirank(S_u, S_v, U_mag, V_mag_u, V_mag_g,
                                       V_mag_l_v)
        frob = rel_err(W_hat, W)
        act = activation_rel_err(W_hat, W, H)
        bpw = extra_bits_per_weight(d_out, d_in, args.rank, K)
        tag = "UPPER_BOUND" if K == r_eff and K not in Ks else f"K={K}"
        print(f"[mr] mag-rank K={K:4d}  "
              f"frob-rel={frob:.4f}  "
              f"act-rel={act:.4f}  "
              f"+bits/wt={bpw:.3f}  "
              f"({tag}, {time.time()-t0:.1f}s)")
        results.append({
            "K": K,
            "frob_rel_err": frob,
            "act_rel_err": act,
            "extra_bits_per_weight_vs_K1": bpw,
            "tag": tag,
        })

    # Report the most interesting comparisons
    k1 = next(r for r in results if r["K"] == 1)
    print(f"\n[mr] ===== Relative to K=1 (current Dual-SVID) =====")
    print(f"{'K':>4s}  {'frob':>8s}  {'act':>8s}  {'act_gain':>8s}  {'+BPW':>8s}")
    for r in results:
        act_gain = k1["act_rel_err"] / max(r["act_rel_err"], 1e-12)
        print(f"{r['K']:>4d}  "
              f"{r['frob_rel_err']:>8.4f}  "
              f"{r['act_rel_err']:>8.4f}  "
              f"{act_gain:>8.2f}x  "
              f"{r['extra_bits_per_weight_vs_K1']:>8.3f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[mr] saved {args.out}")


if __name__ == "__main__":
    main()
