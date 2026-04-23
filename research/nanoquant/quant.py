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
    """Factored-binary linear with fp32 latent parameters.

    Latent tensors (U_latent, V_latent, s1, s2) are stored in **fp32** so
    the AdamW step has enough mantissa precision to accumulate small
    updates (~1e-4) at small magnitudes. Bias stays in the surrounding
    block's dtype so the residual-stream arithmetic is uniform.

    The forward casts the factor params to the input dtype (bf16 in
    Phase 2) before the matmul, so activations stay in the model's
    compute dtype. Autograd handles the cast on the backward path.

    This was the source of the Phase 2 smoke's zero-movement bug: with
    LB-ADMM producing |U_latent| ≈ 0.04, bf16's relative precision 2^-7
    gave absolute precision ~3e-4, above the per-step AdamW update ~1e-4.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        bias: bool = True,
        device=None,
        dtype=None,  # kept for API compatibility; used only for bias.
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        # Factor params live in fp32 regardless of the surrounding block dtype.
        f32 = {"device": device, "dtype": torch.float32}
        self.U_latent = nn.Parameter(torch.empty(out_features, r, **f32))
        self.V_latent = nn.Parameter(torch.empty(in_features, r, **f32))
        self.s1 = nn.Parameter(torch.empty(out_features, **f32))
        self.s2 = nn.Parameter(torch.empty(in_features, **f32))
        if bias:
            bf = {"device": device, "dtype": dtype if dtype is not None else torch.float32}
            self.bias = nn.Parameter(torch.empty(out_features, **bf))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, r: int) -> "BinaryFactoredLinear":
        U_latent, V_latent, s1, s2 = svd_init(lin.weight.data, r)
        return cls.from_factors(lin, r, U_latent, V_latent, s1, s2)

    @classmethod
    def from_factors(
        cls,
        lin: nn.Linear,
        r: int,
        U_latent: torch.Tensor,
        V_latent: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> "BinaryFactoredLinear":
        """Construct from precomputed factors (e.g. from LB-ADMM).

        Factor params are always stored as fp32 (see class docstring);
        bias keeps the source linear's dtype.
        """
        W = lin.weight.data
        d_out, d_in = W.shape
        mod = cls(
            in_features=d_in,
            out_features=d_out,
            r=r,
            bias=(lin.bias is not None),
            device=W.device,
            dtype=W.dtype,  # bias dtype
        )
        mod.U_latent.data.copy_(U_latent.to(device=W.device, dtype=torch.float32))
        mod.V_latent.data.copy_(V_latent.to(device=W.device, dtype=torch.float32))
        mod.s1.data.copy_(s1.to(device=W.device, dtype=torch.float32))
        mod.s2.data.copy_(s2.to(device=W.device, dtype=torch.float32))
        if lin.bias is not None:
            mod.bias.data.copy_(lin.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Factor params are fp32; cast to the input's dtype so matmul
        # accumulates in the block's native precision. Autograd handles
        # the cast on the backward path.
        cdtype = x.dtype
        U_sign = sign_ste(self.U_latent.to(cdtype))
        V_sign = sign_ste(self.V_latent.to(cdtype))
        z = x * self.s2.to(cdtype)
        z = z @ V_sign
        z = z @ U_sign.T
        z = z * self.s1.to(cdtype)
        if self.bias is not None:
            z = z + self.bias.to(cdtype)
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
    module: nn.Module,
    r: int,
    skip_names: tuple[str, ...] = (),
    init_fn=None,
) -> list[str]:
    """Recursively swap every nn.Linear under `module` with a BinaryFactoredLinear.

    `init_fn`, if given, is called as `init_fn(path, linear) -> (U, V, s1, s2)`
    and its output replaces the default SVD init. `path` is the module path
    relative to `module`.

    Returns the list of fully-qualified (relative) names that were replaced.
    """
    replaced: list[str] = []

    def _walk(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(
                path.endswith(s) for s in skip_names
            ):
                if init_fn is None:
                    new = BinaryFactoredLinear.from_linear(child, r)
                else:
                    U, V, s1, s2 = init_fn(path, child)
                    new = BinaryFactoredLinear.from_factors(child, r, U, V, s1, s2)
                setattr(parent, name, new)
                replaced.append(path)
            else:
                _walk(child, path)

    _walk(module, "")
    return replaced
