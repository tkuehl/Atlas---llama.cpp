"""Binary-factored linear with straight-through estimator.

Represents each weight as

    W  ≈  diag(s1) · sign(U_latent) · sign(V_latent)^T · diag(s2)
       U_latent ∈ ℝ^{d_out × r},  V_latent ∈ ℝ^{d_in × r}
       s1 ∈ ℝ^{d_out},  s2 ∈ ℝ^{d_in}

Forward never materializes the effective weight — the computation is

    y  =  s1 ⊙ (sign(U) @ (sign(V)^T @ (x ⊙ s2))) + bias

which is O((d_in + d_out) · r) per batch-element instead of O(d_in · d_out).

Phase 1 init is rank-r SVD of the FP weight. Phase 2+ will replace this
with the paper's LB-ADMM init; the rest of the module stays the same.

Backward uses **clipped** STE (Bengio-Courbariaux): identity in the range
|x| < 1, zero outside. Pure identity diverges in practice because latent
magnitudes grow unbounded when sign() is the only nonlinearity; the clip
is standard in BinaryNet / XNOR-Net / LittleBit / BitNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        gate = (x.abs() < 1.0).to(grad_output.dtype)
        return grad_output * gate


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return _SignSTE.apply(x)


def svd_init(W: torch.Tensor, r: int) -> tuple[torch.Tensor, ...]:
    """Rank-r SVD init for the factored-binary form.

    Returns (U_latent, V_latent, s1, s2) such that
        diag(s1) · sign(U_latent) · sign(V_latent)^T · diag(s2)
    approximates the rank-r truncation of W.

    SVD is done in float32 on the input device for numerical stability;
    callers cast the return values back to the target dtype.
    """
    W_f = W.detach().float()
    d_out, d_in = W_f.shape
    U_full, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
    sqrt_S = S[:r].sqrt()                       # (r,)
    U_raw = U_full[:, :r] * sqrt_S.unsqueeze(0)  # (d_out, r)
    V_raw = Vh[:r, :].T * sqrt_S.unsqueeze(0)    # (d_in, r)
    # Per-row mean-abs -> scale; normalize latent so |latent| ~ 1 keeps
    # the clipped-STE gradient alive across most entries after init.
    s1 = U_raw.abs().mean(dim=1).clamp(min=1e-8)
    s2 = V_raw.abs().mean(dim=1).clamp(min=1e-8)
    U_latent = U_raw / s1.unsqueeze(1)
    V_latent = V_raw / s2.unsqueeze(1)
    return U_latent, V_latent, s1, s2


class BinaryFactoredLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        fact = {"device": device, "dtype": dtype}
        self.U_latent = nn.Parameter(torch.empty(out_features, r, **fact))
        self.V_latent = nn.Parameter(torch.empty(in_features, r, **fact))
        self.s1 = nn.Parameter(torch.empty(out_features, **fact))
        self.s2 = nn.Parameter(torch.empty(in_features, **fact))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fact))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, r: int) -> "BinaryFactoredLinear":
        W = lin.weight.data
        d_out, d_in = W.shape
        mod = cls(
            in_features=d_in,
            out_features=d_out,
            r=r,
            bias=(lin.bias is not None),
            device=W.device,
            dtype=W.dtype,
        )
        U_latent, V_latent, s1, s2 = svd_init(W, r)
        tgt_dtype = W.dtype
        mod.U_latent.data.copy_(U_latent.to(tgt_dtype))
        mod.V_latent.data.copy_(V_latent.to(tgt_dtype))
        mod.s1.data.copy_(s1.to(tgt_dtype))
        mod.s2.data.copy_(s2.to(tgt_dtype))
        if lin.bias is not None:
            mod.bias.data.copy_(lin.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U_sign = sign_ste(self.U_latent)  # (d_out, r)
        V_sign = sign_ste(self.V_latent)  # (d_in, r)
        z = x * self.s2                    # (..., d_in)
        z = z @ V_sign                     # (..., r)
        z = z @ U_sign.T                   # (..., d_out)
        z = z * self.s1                    # (..., d_out)
        if self.bias is not None:
            z = z + self.bias
        return z

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" r={self.r}, bias={self.bias is not None}"
        )


def quant_params(module: nn.Module):
    """Iterator over parameters that should be trained during STE refinement.

    Training any other parameter (RMSNorm weights, etc.) silently drifts the
    block normalization and corrupts the output — this was one of the four
    bugs in the cross-layer-svd Stage 4 sprint. Enumerate explicitly.
    """
    for m in module.modules():
        if isinstance(m, BinaryFactoredLinear):
            yield m.U_latent
            yield m.V_latent
            yield m.s1
            yield m.s2
            if m.bias is not None:
                yield m.bias


def replace_linears_with_quant(
    module: nn.Module, r: int, skip_names: tuple[str, ...] = ()
) -> list[str]:
    """Recursively swap every nn.Linear under `module` with a BinaryFactoredLinear.

    Returns the list of fully-qualified names that were replaced. Names in
    `skip_names` (matched as suffixes of the module path) are left as
    plain Linear.
    """
    replaced: list[str] = []

    def _walk(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(
                path.endswith(s) for s in skip_names
            ):
                setattr(parent, name, BinaryFactoredLinear.from_linear(child, r))
                replaced.append(path)
            else:
                _walk(child, path)

    _walk(module, "")
    return replaced
