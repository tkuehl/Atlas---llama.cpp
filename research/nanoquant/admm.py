"""LB-ADMM initialization for the factored-binary format (NanoQuant Phase 2, Step 2).

Solves the activation-weighted rank-r factorization

    min_{U, V, Z_U, Z_V}  ½‖W̃ − U V^T‖_F²  +  (λ/2)(‖U‖_F² + ‖V‖_F²)
    s.t.  U = Z_U,  V = Z_V,  Z_U, Z_V ∈ S  (image of the SVID operator)

where the preconditioned target is

    W̃ = D_out · W_fp · D_in

(both D's diagonal, collected in `preconditioner.py`). The SVID operator
projects onto the set of matrices expressible as a full sign matrix
elementwise-multiplied by a rank-1 outer product of magnitudes
(Xu et al. 2024 / OneBit):

    SVID(M):
        sign_pattern = sign(M)
        |M| ≈ a b^T      (rank-1 SVD of |M|)
        return sign_pattern ⊙ (a b^T)

The ADMM updates (paper Eq. 5, 6, 24) using scaled dual Λ = Y/ρ:

    U ← solve (V^T V + (ρ+λ) I)  ·  U^T = V^T W̃^T + ρ (Z_U − Λ_U)^T
    V ← symmetric
    Z_U ← SVID(U + Λ_U)
    Λ_U ← Λ_U + (U − Z_U)

After K iterations, magnitude balancing (Eq. 7–9) splits the continuous
proxies into (s1, s2, U_latent, V_latent) in our stored format.

Returns the four tensors in the model's target dtype (bf16 / fp16).
"""

from __future__ import annotations

import torch


def svid(M: torch.Tensor) -> torch.Tensor:
    """Rank-1 sign-value decomposition (Xu et al. 2024, Prop. 1).

    For M ∈ ℝ^{m × r}:
        sign_pattern = sign(M)
        |M| ≈ a b^T   via rank-1 SVD of |M|
    Returns sign_pattern ⊙ (a b^T).
    """
    sign_pat = torch.sign(M)
    abs_M = M.abs()
    # torch.linalg.svd on abs_M; take first singular component.
    # For an (m × r) matrix this is cheap even at m=9728, r=2.
    U_, S_, Vh_ = torch.linalg.svd(abs_M, full_matrices=False)
    a = U_[:, 0] * S_[0].sqrt()  # (m,)
    b = Vh_[0, :] * S_[0].sqrt()  # (r,)
    return sign_pat * (a.unsqueeze(1) * b.unsqueeze(0))


def _rho_schedule(K: int, rho_start: float, rho_end: float) -> list[float]:
    """Linear ρ schedule (paper Appendix C: 'linear ADMM penalty scheduler')."""
    if K == 1:
        return [rho_end]
    return [rho_start + (rho_end - rho_start) * k / (K - 1) for k in range(K)]


@torch.no_grad()
def lb_admm_init(
    W: torch.Tensor,
    D_in: torch.Tensor,
    D_out: torch.Tensor,
    r: int,
    K: int = 400,
    rho_start: float = 0.1,
    rho_end: float = 10.0,
    lam: float = 1e-3,
    eps_diag: float = 1e-8,
    target_dtype: torch.dtype | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper-faithful LB-ADMM init of a single linear weight.

    Returns (U_latent, V_latent, s1, s2) in `target_dtype` (defaults to
    W.dtype), shaped to slot straight into `BinaryFactoredLinear`.

    All internal computation is fp32 on the input device for numerical
    stability — ADMM on r=2 matrices is tiny.
    """
    device = W.device
    if target_dtype is None:
        target_dtype = W.dtype

    W_f = W.detach().to(torch.float32)
    d_out, d_in = W_f.shape

    D_in = D_in.to(device=device, dtype=torch.float32).clamp(min=eps_diag)
    D_out = D_out.to(device=device, dtype=torch.float32).clamp(min=eps_diag)

    # Normalize D_in and D_out to mean=1 so that only their RELATIVE
    # channel-importance matters. Without this, D_out with raw gradient-RMS
    # magnitudes (can be 1e-4 on a standard CE-normalized loss) shrinks
    # W̃ so far below λ that the ridge term dominates ADMM's optimum and
    # drives U, V to near-zero — which then gives an effective binary
    # weight of magnitude 10^-30 and zero gradient downstream. The
    # normalization preserves the preconditioning structure (channel
    # importance ratios) while keeping W̃ at the same scale as W.
    D_in = D_in / D_in.mean().clamp(min=eps_diag)
    D_out = D_out / D_out.mean().clamp(min=eps_diag)

    # Preconditioned target W̃ = D_out · W · D_in (paper Eq. 15)
    W_target = D_out.unsqueeze(1) * W_f * D_in.unsqueeze(0)

    # Initial U, V: small random, reproducible per-layer via `seed`.
    g = torch.Generator(device=device).manual_seed(seed)
    U = torch.randn(d_out, r, generator=g, device=device, dtype=torch.float32) / (r ** 0.5)
    V = torch.randn(d_in, r, generator=g, device=device, dtype=torch.float32) / (r ** 0.5)
    Z_U = torch.zeros_like(U)
    Z_V = torch.zeros_like(V)
    Lam_U = torch.zeros_like(U)
    Lam_V = torch.zeros_like(V)

    eye_r = torch.eye(r, device=device, dtype=torch.float32)
    schedule = _rho_schedule(K, rho_start, rho_end)

    for k in range(K):
        rho = schedule[k]

        # --- U update ---
        # (V^T V + (ρ+λ)I)  U^T  =  V^T W̃^T + ρ (Z_U − Λ_U)^T
        VtV = V.T @ V
        M = VtV + (rho + lam) * eye_r
        L = torch.linalg.cholesky(M)
        rhs = W_target @ V + rho * (Z_U - Lam_U)  # (d_out, r)
        U = torch.cholesky_solve(rhs.T, L).T

        # --- V update (symmetric, roles swapped) ---
        UtU = U.T @ U
        M = UtU + (rho + lam) * eye_r
        L = torch.linalg.cholesky(M)
        rhs = W_target.T @ U + rho * (Z_V - Lam_V)  # (d_in, r)
        V = torch.cholesky_solve(rhs.T, L).T

        # --- Z updates via SVID on scaled-dual consensus (paper Eq. 26) ---
        Z_U = svid(U + Lam_U)
        Z_V = svid(V + Lam_V)

        # --- Dual updates (scaled) ---
        Lam_U = Lam_U + (U - Z_U)
        Lam_V = Lam_V + (V - Z_V)

    # Magnitude balancing (paper Eq. 7-9).
    # Recover unscaled continuous proxies by undoing the preconditioner.
    U_hat = U / D_out.unsqueeze(1)  # (d_out, r)
    V_hat = V / D_in.unsqueeze(1)   # (d_in,  r)

    U_hat_norm = U_hat.norm()
    V_hat_norm = V_hat.norm()
    # eta = sqrt(||V_hat||_F / ||U_hat||_F) — guards against zero norm.
    eta = (V_hat_norm.clamp(min=1e-12) / U_hat_norm.clamp(min=1e-12)).sqrt()

    # Paper Eq. 9: U := η · Û (no further normalization). SVID's rank-1
    # magnitude projection already makes every row of U_latent proportional
    # to the same (a[i], b[k]) pattern, so dividing by per-row mean-abs
    # (as svd_init does) would concentrate every entry exactly at |x| = 1,
    # killing the clipped-STE gradient. Keep the magnitudes from Eq. 7-9.
    U_latent = eta * U_hat
    V_latent = V_hat / eta

    # s1[i] = mean over r of |η · û_i[k]|
    s1 = U_latent.abs().mean(dim=1).clamp(min=eps_diag)  # (d_out,)
    s2 = V_latent.abs().mean(dim=1).clamp(min=eps_diag)  # (d_in,)

    return (
        U_latent.to(target_dtype),
        V_latent.to(target_dtype),
        s1.to(target_dtype),
        s2.to(target_dtype),
    )
