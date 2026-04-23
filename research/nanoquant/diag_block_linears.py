"""Per-linear rank-r signed-SVD reconstruction error for suspect blocks.

Phase 1 found blocks 6 and 16 of Qwen3-4B with init MSE ~10× their
neighbors at r=2 SVD init. This script walks the 7 linears of each
block in a user-specified set, computes:

- `err_trunc`  — rel Frobenius error of rank-r SVD of W (no signing).
  This is the irreducible loss from rank truncation.
- `err_signed` — rel Frobenius error of `diag(s1)·sign(U)·sign(V)^T·diag(s2)`
  using the same SVD-init recipe `svd_init` uses. The delta over
  err_trunc is the signing cost.
- `top_r_energy` — fraction of squared singular energy captured by
  the top r singular values. A flat spectrum → near 0; a rapidly
  decaying spectrum → near 1.
- `cond_r` — σ_1 / σ_r. Large values suggest one direction dominates.

Reports are printed per (block, linear). To find a rogue layer, look
for rows where `err_signed` is much higher than in neighboring blocks
for the same linear name.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from quant import svd_init


LINEAR_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def _get_linear(block: torch.nn.Module, dotted: str) -> torch.nn.Linear:
    m = block
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


@torch.inference_mode()
def diagnose_linear(W: torch.Tensor, r: int) -> dict:
    W_f = W.detach().float()
    W_norm = W_f.norm().item()

    # Unbinarized rank-r truncation error + singular statistics
    U_full, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
    S = S.cpu()
    total_energy = (S * S).sum().item()
    top_r_energy = (S[:r] * S[:r]).sum().item() / total_energy
    cond_r = (S[0] / S[r - 1]).item() if S[r - 1] > 0 else float("inf")
    W_trunc = U_full[:, :r] @ torch.diag(S[:r].to(W_f.device)) @ Vh[:r, :]
    err_trunc = (W_f - W_trunc).norm().item() / W_norm

    # Signed-SVD reconstruction (matches svd_init + BinaryFactoredLinear forward)
    U_latent, V_latent, s1, s2 = svd_init(W, r)
    W_bin = (
        s1.unsqueeze(1) * torch.sign(U_latent)
    ) @ (torch.sign(V_latent).T * s2.unsqueeze(0))
    err_signed = (W_f - W_bin.float()).norm().item() / W_norm

    return {
        "shape": tuple(W.shape),
        "top_r_energy": top_r_energy,
        "cond_r": cond_r,
        "err_trunc": err_trunc,
        "err_signed": err_signed,
        "signing_cost": err_signed - err_trunc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[5, 6, 7, 15, 16, 17],
        help="block indices to diagnose",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[load] {args.model} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    hdr = (
        f"{'block':>5}  {'linear':<20}  {'shape':<15}  "
        f"{'top'+str(args.rank)+'E':>7}  "
        f"{'cond_r':>8}  {'truncE':>7}  {'signE':>7}  {'signCost':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for b in args.blocks:
        block = model.model.layers[b]
        for name in LINEAR_NAMES:
            lin = _get_linear(block, name)
            d = diagnose_linear(lin.weight, args.rank)
            print(
                f"{b:>5}  {name:<20}  "
                f"{str(d['shape']):<15}  "
                f"{d['top_r_energy']:>7.4f}  "
                f"{d['cond_r']:>8.2f}  "
                f"{d['err_trunc']:>7.4f}  "
                f"{d['err_signed']:>7.4f}  "
                f"{d['signing_cost']:>9.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
