"""Single-matrix QAT diagnostic for LittleBit.

The sanity checks (littlebit_sanity.py) showed Dual-SVID init captures
~39% of rank-r subspace energy; the paper's quality claims require QAT
to recover the other ~61%. Before committing 8-16h to full-model QAT,
this script asks the local version of that question on ONE matrix:

  Given the LittleBit parameterization (U_fp for sign, V_fp for sign,
  and the three FP16 scale vectors h/g/l), trained with SmoothSign
  backward under plain Frobenius loss ||W - W_pri||_F^2, how close to
  the true W does the trained reconstruction get?

If local gradient descent can drive Frobenius error materially below
Dual-SVID's ~0.82 (at r=512 on Qwen 0.5B gate_proj L12), then the
format has the capacity and the init is the bottleneck. If it
plateaus near 0.82, the format itself is the ceiling and a full-model
QAT will not recover either — saving 8-16h of wasted compute.

This is a strict upper bound on what QAT can achieve on this matrix:
plain Frobenius is easier than KL+intermediate-MSE-through-model
(the real QAT objective), which cares about how much each weight
matters for downstream outputs. Any ceiling on Frobenius transfers
downward to activation-weighted objectives.

Usage:
  python littlebit_qat_single.py --model Qwen/Qwen2.5-0.5B \\
         --role mlp.gate_proj --layer 12 --rank 512 \\
         --steps 2000 --lr 1e-2 \\
         --out littlebit_qat_single.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SmoothSign(torch.autograd.Function):
    """Forward: sign(x). Backward: d/dx tanh(tau*x) (paper: tau=100)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tau: float):
        ctx.save_for_backward(x)
        ctx.tau = tau
        # sign with a 0 -> +1 convention; trained params are essentially
        # never exactly 0 but be defensive.
        out = torch.where(x >= 0,
                          torch.ones_like(x),
                          -torch.ones_like(x))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        tau = ctx.tau
        # d/dx tanh(tau * x) = tau * sech^2(tau * x) = tau * (1 - tanh^2)
        surrogate = tau * (1.0 - torch.tanh(tau * x) ** 2)
        return grad_output * surrogate, None


def smooth_sign(x: torch.Tensor, tau: float = 100.0) -> torch.Tensor:
    return SmoothSign.apply(x, tau)


class LittleBitLinear(nn.Module):
    """Trainable LittleBit factorization of a single target matrix W.

    Parameters:
      U_fp  : (d_out, r)   soft factor, sign taken each forward
      V_fp  : (d_in,  r)
      h     : (d_out,)     row scale
      g     : (d_in,)      column scale
      ell   : (r,)         latent-rank scale

    Forward returns the reconstructed d_out x d_in matrix, not an
    applied-to-activations output.  For this single-matrix QAT we
    care only about ||W - W_pri||_F^2.
    """

    def __init__(self, W: torch.Tensor, r: int, tau: float = 100.0):
        super().__init__()
        d_out, d_in = W.shape
        self.d_out = d_out
        self.d_in = d_in
        self.r = r
        self.tau = tau
        # Initialize via Dual-SVID, same logic as littlebit_sanity.py dual_svid().
        W_np = W.detach().to(torch.float64).cpu().numpy()
        U_full, S_full, VT_full = np.linalg.svd(W_np, full_matrices=False)
        Uk = U_full[:, :r]
        Sk = S_full[:r]
        Vk = VT_full[:r, :].T
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

        # Soft-sign parameters: initialize as Up / Vp themselves so that
        # sign(U_fp) == sign(Up).  Scale is arbitrary (only sign matters
        # in forward), but keeping the magnitudes gives SmoothSign a
        # sensible surrogate gradient magnitude.
        self.U_fp = nn.Parameter(torch.tensor(Up, dtype=torch.float32))
        self.V_fp = nn.Parameter(torch.tensor(Vp, dtype=torch.float32))
        self.h   = nn.Parameter(torch.tensor(h0,   dtype=torch.float32))
        self.g   = nn.Parameter(torch.tensor(g0,   dtype=torch.float32))
        self.ell = nn.Parameter(torch.tensor(ell0, dtype=torch.float32))

    def reconstruct(self) -> torch.Tensor:
        U_sign = smooth_sign(self.U_fp, self.tau)
        V_sign = smooth_sign(self.V_fp, self.tau)
        # diag(h) @ U_sign @ diag(ell) @ V_sign.T @ diag(g)
        M = (U_sign * self.ell[None, :]) @ V_sign.T   # d_out x d_in
        M = M * self.g[None, :]
        M = M * self.h[:, None]
        return M


def load_matrix(model_id: str, role: str, layer: int) -> torch.Tensor:
    from transformers import AutoModelForCausalLM
    print(f"loading {model_id} (bfloat16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")
    target = f"model.layers.{layer}.{role}"
    W = dict(model.named_modules())[target].weight.data.detach().clone()
    W = W.to(torch.float32)
    print(f"  extracted {target} [{W.shape[0]} x {W.shape[1]}]")
    return W


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--role", default="mlp.gate_proj")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default="littlebit_qat_single.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--freeze-signs", action="store_true",
                   help="Freeze U_fp, V_fp; train only h, g, ell")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")

    W = load_matrix(args.model, args.role, args.layer).to(device)
    W_norm = torch.linalg.norm(W).item()
    print(f"||W||_F = {W_norm:.4f}")

    layer = LittleBitLinear(W, r=args.rank, tau=args.tau).to(device)
    if args.freeze_signs:
        layer.U_fp.requires_grad_(False)
        layer.V_fp.requires_grad_(False)
        print("signs frozen: only h/g/ell are trained")

    with torch.no_grad():
        W0 = layer.reconstruct()
        init_rel = (torch.linalg.norm(W - W0) / W_norm).item()
        print(f"Dual-SVID init rel Frobenius err: {init_rel:.4f}")

    trainable = [p for p in layer.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    history = [{"step": 0, "rel_err": init_rel, "loss": None}]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        W_hat = layer.reconstruct()
        loss = torch.nn.functional.mse_loss(W_hat, W, reduction="sum")
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                W_hat_eval = layer.reconstruct()
                rel = (torch.linalg.norm(W - W_hat_eval) / W_norm).item()
            history.append({
                "step": step, "loss": loss.item(), "rel_err": rel,
            })
            print(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"rel_err={rel:.4f}  "
                  f"elapsed={time.time() - t0:.1f}s")

    best = min(history, key=lambda h: h["rel_err"])
    print(f"\nfinal: init={init_rel:.4f}  "
          f"best={best['rel_err']:.4f} (step {best['step']})  "
          f"improvement={init_rel - best['rel_err']:+.4f}")

    out = {
        "model": args.model, "role": args.role, "layer": args.layer,
        "W_shape": list(W.shape), "W_frob": W_norm,
        "rank": args.rank, "steps": args.steps, "lr": args.lr,
        "tau": args.tau,
        "init_rel_err": init_rel,
        "final_rel_err": history[-1]["rel_err"],
        "best_rel_err": best["rel_err"],
        "best_step": best["step"],
        "history": history,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
