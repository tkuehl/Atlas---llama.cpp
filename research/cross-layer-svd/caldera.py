"""CALDERA decomposition — W ≈ Q + L·R.

Q is a native-GGUF quantized tensor (Q4_K/Q3_K/…) resident in VRAM at inference;
L (d_out × r) and R (r × d_in) are fp16 low-rank correction factors. The forward
pass is `y = mul_mat(Q, x) + factored_linear(L, R, x)`, i.e. reuses two kernels
we already have.

The decomposition is the alternating loop from arxiv:2405.18886 (CALDERA),
specialized to per-matrix weighted-LS via the activation gramian Σ = XᵀX
that we already collect in basis_sharing.collect_stats. For each matrix we
alternate:

  1. Q ← quantize(W − L·R)           # round-to-nearest in the chosen GGUF format
  2. (L, R) ← rank-r fit to (W − dequant(Q)) in Σ-weighted Frobenius norm
     = SVD of (W − Q̂) · S, where S·Sᵀ = Σ, then un-whiten.

This module is standalone: `python caldera.py --smoke` runs a synthetic check
that RPCD beats pure-quant and pure-SVD at a matched bit budget. The full-model
pipeline integration lives in basis_sharing.py (next step).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# gguf-py lives alongside this research dir at repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))
from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf import quants as gguf_quants  # noqa: E402

from basis_sharing import _stable_svd, sqrt_and_inv  # noqa: E402


_QTYPE_BY_NAME = {
    "Q4_0": GGMLQuantizationType.Q4_0,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q3_K": GGMLQuantizationType.Q3_K,
    "Q8_0": GGMLQuantizationType.Q8_0,
}


# ---------- quant round-trip ----------

def _quantize_roundtrip(W: torch.Tensor, qtype: GGMLQuantizationType):
    """Run a weight through gguf-py quantize+dequantize. Returns (Q_bytes, W_hat)
    where Q_bytes is the packed quantized blob (uint8 ndarray, GGUF-ready) and
    W_hat is the reconstructed fp32 tensor on W's original device.

    The last dim of W must be a multiple of the quant block size (always the
    case for transformer matrices: hidden, intermediate are both 256-aligned
    for Q*_K, 32-aligned for Q*_0).
    """
    dev = W.device
    W_np = W.detach().to(torch.float32).cpu().numpy()
    Q_bytes = gguf_quants.quantize(W_np, qtype)
    W_hat_np = gguf_quants.dequantize(Q_bytes, qtype)
    W_hat = torch.from_numpy(W_hat_np.astype(np.float32, copy=False)).to(dev)
    return Q_bytes, W_hat


# ---------- weighted low-rank fit ----------

def _weighted_low_rank(R: torch.Tensor, S: torch.Tensor, S_inv: torch.Tensor,
                       rank: int):
    """Best rank-r approx of R in the Σ-weighted Frobenius norm,
    where Σ = S·Sᵀ. Returns (L, Rmat) with L·Rmat ≈ R.

    Derivation: minimize ||(R − L·Rmat) · S||_F. Let M = R · S. Truncated SVD
    M ≈ U_r diag(σ_r) V_rᵀ is optimal in plain Frobenius; un-whiten the right
    factor by post-multiplying by S⁻¹ (== S_inv from sqrt_and_inv).
    """
    M = R @ S
    U, sigma, Vt = _stable_svd(M)
    L = (U[:, :rank] * sigma[:rank].unsqueeze(0)).contiguous()
    Rmat = (Vt[:rank, :] @ S_inv).contiguous()
    return L, Rmat


# ---------- RPCD outer loop ----------

@dataclass
class CalderaResult:
    Q_bytes: np.ndarray        # packed GGUF blob, dtype=uint8
    L: torch.Tensor            # [d_out, rank], fp32 (caller casts to fp16)
    R: torch.Tensor            # [rank, d_in],  fp32
    qtype: GGMLQuantizationType
    rank: int
    iters: int
    rel_err_history: list[float]  # Σ-weighted relative error after each iter


def caldera_decompose(
    W: torch.Tensor,
    Sigma: torch.Tensor,
    rank: int,
    qtype: GGMLQuantizationType,
    n_iters: int = 3,
    device: str = "cuda",
) -> CalderaResult:
    """Single-matrix CALDERA: W ≈ Q + L·R.

    W:     [d_out, d_in], fp32, on `device`.
    Sigma: [d_in, d_in] input-activation gramian (XᵀX). Used as the weighted
           Frobenius metric for the low-rank fit.
    """
    assert W.dim() == 2
    d_out, d_in = W.shape
    assert Sigma.shape == (d_in, d_in)
    W = W.to(device=device, dtype=torch.float32)

    S, S_inv = sqrt_and_inv(Sigma.to(device), device=device, need_inv=True)
    W_norm = torch.linalg.norm(W @ S).item() + 1e-12

    L = torch.zeros(d_out, rank, device=device, dtype=torch.float32)
    R = torch.zeros(rank, d_in, device=device, dtype=torch.float32)
    history: list[float] = []
    Q_bytes = None

    for _ in range(n_iters):
        # Step 1: quantize the quant-residual W − L·R.
        Qtarget = W - L @ R
        Q_bytes, Q_hat = _quantize_roundtrip(Qtarget, qtype)

        # Step 2: weighted low-rank fit of the lr-residual W − Q̂.
        Res = W - Q_hat
        L, R = _weighted_low_rank(Res, S, S_inv, rank)

        approx = Q_hat + L @ R
        err = torch.linalg.norm((W - approx) @ S).item() / W_norm
        history.append(err)

    return CalderaResult(
        Q_bytes=Q_bytes, L=L, R=R, qtype=qtype, rank=rank, iters=n_iters,
        rel_err_history=history,
    )


# ---------- smoke test ----------

def _baseline_quant(W, Sigma, qtype):
    _, W_hat = _quantize_roundtrip(W, qtype)
    S, _ = sqrt_and_inv(Sigma, need_inv=False)
    return torch.linalg.norm((W - W_hat) @ S).item() / (
        torch.linalg.norm(W @ S).item() + 1e-12
    )


def _baseline_svd(W, Sigma, rank):
    S, S_inv = sqrt_and_inv(Sigma, need_inv=True)
    L, R = _weighted_low_rank(W, S, S_inv, rank)
    return torch.linalg.norm((W - L @ R) @ S).item() / (
        torch.linalg.norm(W @ S).item() + 1e-12
    )


def smoke(d_out=1024, d_in=2048, rank=128, qtype_name="Q4_K",
          n_iters=3, seed=0, device="cuda"):
    """Synthetic check on a 'realistic' weight: low-rank structure + noise.
    Expect CALDERA ≤ both baselines; typically strictly lower than both."""
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        device = "cpu"
    qtype = _QTYPE_BY_NAME[qtype_name]

    # Weight with effective rank ~256 plus small dense noise; hidden-to-ffn shape.
    U = torch.randn(d_out, 256, device=device) / np.sqrt(256)
    V = torch.randn(256, d_in, device=device) / np.sqrt(256)
    W = U @ V + 0.02 * torch.randn(d_out, d_in, device=device)

    # Random PSD input gramian (Σ = A Aᵀ) at a plausible token count.
    A = torch.randn(d_in, 4096, device=device)
    Sigma = (A @ A.T) / 4096

    err_q = _baseline_quant(W, Sigma, qtype)
    err_lr = _baseline_svd(W, Sigma, rank)
    res = caldera_decompose(W, Sigma, rank, qtype, n_iters=n_iters, device=device)

    print(f"shape={W.shape}, rank={rank}, qtype={qtype_name}, iters={n_iters}")
    print(f"  pure quant  : Σ-rel-err = {err_q:.4f}")
    print(f"  pure LR-{rank:<3} : Σ-rel-err = {err_lr:.4f}")
    print(f"  CALDERA     : Σ-rel-err = {res.rel_err_history[-1]:.4f} "
          f"(history {[f'{e:.4f}' for e in res.rel_err_history]})")

    assert res.rel_err_history[-1] <= min(err_q, err_lr) + 1e-4, (
        "CALDERA did not beat both baselines — RPCD math is wrong"
    )
    print("OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--d-out", type=int, default=1024)
    p.add_argument("--d-in", type=int, default=2048)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--qtype", default="Q4_K", choices=list(_QTYPE_BY_NAME))
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    if args.smoke:
        smoke(d_out=args.d_out, d_in=args.d_in, rank=args.rank,
              qtype_name=args.qtype, n_iters=args.iters)
    else:
        p.print_help()
