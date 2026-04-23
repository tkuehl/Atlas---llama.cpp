"""Multi-rank-magnitude LittleBit Linear — capacity-test variant.

Extension of LittleBit's Dual-SVID format where |Up| and |Vp|
magnitudes are approximated at rank K >= 1 instead of the paper's rank 1.
For K=1 this is equivalent to LittleBitLinearHF's math; for K>1 we
retain more of the SVD magnitude spectrum at a small BPW cost.

Single-matrix results (see multirank_probe.json): K=2 reduces
activation rel-err from 0.88 → 0.70 for +0.05 BPW.  K=8 drops to
0.59 for +0.35 BPW.  The structural ceiling (K=rank) sits at 0.25,
better than gradient-trained QAT's 0.31 on the same single matrix.

Parameters per linear (at rank r, magnitude rank K):
  U_fp:       (d_out, r)        — sign shadow, same as LittleBitLinearHF
  V_fp:       (d_in,  r)        — sign shadow
  U_mag:      (d_out, K)        — "h" generalized to K columns
  V_mag_u:    (K,     r)        — "l_u" generalized
  V_mag_g:    (d_in,  K)        — "g" generalized
  V_mag_lv:   (K,     r)        — "l_v" generalized
  bias:       (d_out,)

Forward (simple materialized variant):
  |Up| = U_mag @ V_mag_u
  |Vp| = V_mag_g @ V_mag_lv
  W_hat = (sign(U_fp) * |Up|) @ (sign(V_fp) * |Vp|)^T
  y = x @ W_hat^T

K=1 reduces to LittleBit's original Dual-SVID if we stash ell = l_u * l_v
in V_mag_u's row; we keep the more-general structure here so K can be
swept without format changes.

This is the CAPACITY test — does end-to-end quality track the single-
matrix activation-error reduction?  No binary-matmul speedup yet;
forward materializes W.  Engineer the efficient K² binary path
afterward if the quality holds.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Reuse the same SmoothSign autograd for sign shadow weights.
from littlebit_qat_model import SmoothSignEfficient, smooth_sign


def _dual_svid_multirank_numpy(W_np, r, K):
    """Dual-SVID with rank-K magnitude approximation.  Returns the six
    factor arrays in fp32 numpy.  Caller converts to tensors + dtype
    as needed.

    Mirrors littlebit_init_multirank.dual_svid_multirank — kept separate
    so that script doesn't depend on importing this class.
    """
    d_out, d_in = W_np.shape
    r_eff = min(r, d_out, d_in)
    K_eff = min(K, r_eff)

    U_full, S_full, VT_full = np.linalg.svd(W_np, full_matrices=False)
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

    U_mag = (uU[:, :K_eff] * np.sqrt(sU[:K_eff])[None, :]).astype(np.float32)
    V_mag_u = (vtU[:K_eff, :] * np.sqrt(sU[:K_eff])[:, None]).astype(np.float32)
    V_mag_g = (uV[:, :K_eff] * np.sqrt(sV[:K_eff])[None, :]).astype(np.float32)
    V_mag_lv = (vtV[:K_eff, :] * np.sqrt(sV[:K_eff])[:, None]).astype(np.float32)

    return (Up.astype(np.float32), Vp.astype(np.float32),
            U_mag, V_mag_u, V_mag_g, V_mag_lv, r_eff, K_eff)


class LittleBitLinearMultiRankHF(nn.Module):
    """LittleBit-style linear with rank-K magnitude approximation.

    Forward materializes the reconstructed W in fp32 and does a plain
    matmul.  This trades away LittleBit's binary-matmul speedup for
    format flexibility — useful as a research harness; a fused kernel
    would be a follow-up.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        r: int,
        K: int,
        bias: bool,
        tau: float = 100.0,
        shadow_dtype: torch.dtype = torch.float32,
        debug: bool = False,
    ):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.r = r
        self.K = K
        self.tau = tau
        self.shadow_dtype = shadow_dtype
        self.debug = debug

        # Sign shadow weights
        self.U_fp = nn.Parameter(torch.empty(d_out, r, dtype=shadow_dtype))
        self.V_fp = nn.Parameter(torch.empty(d_in, r, dtype=shadow_dtype))

        # Multi-rank magnitude factors (fp32 for numerical headroom)
        self.U_mag = nn.Parameter(torch.empty(d_out, K, dtype=torch.float32))
        self.V_mag_u = nn.Parameter(torch.empty(K, r, dtype=torch.float32))
        self.V_mag_g = nn.Parameter(torch.empty(d_in, K, dtype=torch.float32))
        self.V_mag_lv = nn.Parameter(torch.empty(K, r, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.empty(d_out, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        if self.debug:
            print(f"[mrlin] init d_in={d_in} d_out={d_out} r={r} K={K} "
                  f"bias={bias} shadow_dtype={shadow_dtype}")

    @classmethod
    def from_linear(
        cls,
        lin: nn.Linear,
        r: int,
        K: int,
        tau: float = 100.0,
        shadow_dtype: torch.dtype = torch.float32,
        debug: bool = False,
    ) -> "LittleBitLinearMultiRankHF":
        """Build via Dual-SVID with rank-K magnitudes."""
        W_np = lin.weight.data.detach().to(torch.float64).cpu().numpy()
        d_out, d_in = W_np.shape

        if debug:
            print(f"[mrlin] from_linear shape={W_np.shape} r={r} K={K}  "
                  f"||W||_F={np.linalg.norm(W_np):.4f}")

        Up, Vp, U_mag, V_mag_u, V_mag_g, V_mag_lv, r_eff, K_eff = \
            _dual_svid_multirank_numpy(W_np, r, K)

        if debug:
            # Single-matrix reconstruction rel-err for this layer.
            Up_abs_approx = U_mag @ V_mag_u
            Vp_abs_approx = V_mag_g @ V_mag_lv
            Up_approx = np.sign(Up) * Up_abs_approx
            Vp_approx = np.sign(Vp) * Vp_abs_approx
            W_hat = Up_approx @ Vp_approx.T
            frob_rel = (np.linalg.norm(W_hat - W_np) /
                        (np.linalg.norm(W_np) + 1e-12))
            print(f"[mrlin]   reconstruction Frobenius rel-err = {frob_rel:.4f}  "
                  f"(r_eff={r_eff}, K_eff={K_eff})")

        out = cls(d_in=d_in, d_out=d_out, r=r_eff, K=K_eff,
                  bias=lin.bias is not None, tau=tau,
                  shadow_dtype=shadow_dtype, debug=debug)
        with torch.no_grad():
            out.U_fp.copy_(torch.from_numpy(Up).to(shadow_dtype))
            out.V_fp.copy_(torch.from_numpy(Vp).to(shadow_dtype))
            out.U_mag.copy_(torch.from_numpy(U_mag))
            out.V_mag_u.copy_(torch.from_numpy(V_mag_u))
            out.V_mag_g.copy_(torch.from_numpy(V_mag_g))
            out.V_mag_lv.copy_(torch.from_numpy(V_mag_lv))
            if lin.bias is not None:
                out.bias.copy_(lin.bias.data.detach().to(torch.float32).cpu())
        return out

    def reconstruct_W(self) -> torch.Tensor:
        """Materialize the reconstructed weight matrix (d_out, d_in)."""
        # SmoothSign forward returns ±1 with surrogate gradient.
        U_sign = smooth_sign(self.U_fp, self.tau)   # (d_out, r)
        V_sign = smooth_sign(self.V_fp, self.tau)   # (d_in, r)
        Up_abs = self.U_mag @ self.V_mag_u           # (d_out, r)
        Vp_abs = self.V_mag_g @ self.V_mag_lv        # (d_in, r)
        # Cast signs to fp32 to match magnitude dtype.  ±1 values are
        # exact across any float dtype.
        Up = U_sign.to(torch.float32) * Up_abs
        Vp = V_sign.to(torch.float32) * Vp_abs
        return Up @ Vp.T                             # (d_out, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain matmul after materializing W.  Preserves input dtype
        on output so the block-level dtype-preservation wrapper remains
        valid."""
        W_hat = self.reconstruct_W()       # fp32
        y = torch.nn.functional.linear(
            x.to(torch.float32), W_hat,
            self.bias,
        )
        return y.to(x.dtype)


def convert_block_to_multirank(
    block: nn.Module,
    rank: int,
    K: int,
    tau: float = 100.0,
    shadow_dtype: torch.dtype = torch.float32,
    debug: bool = False,
) -> nn.Module:
    """Walk a transformer block and replace every nn.Linear with a
    LittleBitLinearMultiRankHF.  Mirrors
    littlebit_qat_brecq.convert_block_to_littlebit.
    """
    targets = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                parent = block.get_submodule(parent_name)
            else:
                parent = block
                attr = name
            targets.append((parent, attr, module, name))

    if debug:
        print(f"  [mrblk] converting {len(targets)} linears to "
              f"LittleBit-MultiRank r={rank} K={K}:")

    for parent, attr, lin, full_name in targets:
        lb = LittleBitLinearMultiRankHF.from_linear(
            lin, r=rank, K=K, tau=tau, shadow_dtype=shadow_dtype,
            debug=debug,
        )
        setattr(parent, attr, lb)
        if debug:
            d_out, d_in = lin.weight.shape
            r_eff = min(rank, d_in, d_out)
            K_eff = min(K, r_eff)
            print(f"  [mrblk]   {full_name}: ({d_in}, {d_out}) "
                  f"r={r_eff} K={K_eff}")

    # Dtype preservation wrapper (same as convert_block_to_littlebit).
    _orig_forward = block.forward

    def _dtype_preserving_forward(hidden_states, *args, **kwargs):
        in_dtype = hidden_states.dtype
        out = _orig_forward(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            if out[0] is None:
                return out
            return (out[0].to(in_dtype),) + out[1:]
        return out.to(in_dtype)

    block.forward = _dtype_preserving_forward
    return block


def count_params(module: nn.Module) -> dict:
    """Debug helper: break down parameter count by LittleBit-MultiRank role."""
    n = {"U_fp": 0, "V_fp": 0, "U_mag": 0, "V_mag_u": 0,
         "V_mag_g": 0, "V_mag_lv": 0, "bias": 0, "other": 0}
    for name, pr in module.named_parameters():
        matched = False
        for role in ("U_fp", "V_fp", "U_mag", "V_mag_u", "V_mag_g",
                     "V_mag_lv", "bias"):
            if name.endswith(role):
                n[role] += pr.numel()
                matched = True
                break
        if not matched:
            n["other"] += pr.numel()
    n["total"] = sum(v for k, v in n.items() if k != "total")
    return n
