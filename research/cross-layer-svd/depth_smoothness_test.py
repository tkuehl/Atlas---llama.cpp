"""Depth-smoothness hypothesis test — is the per-role weight evolution
across layers a smooth function of depth, compressible via continuous
(ODE-style) parameterization?

For each role R (q_proj, k_proj, ..., down_proj), form the adjacent-layer
difference tensor:
    D_i^R = W_{i+1}^R - W_i^R     for i = 0 .. L-2

Test two things about this sequence:

  1. MAGNITUDE: ||D_i||_F / ||W_i||_F.
     If ~0, layers are nearly identical (trivial sharing).
     If >= sqrt(2), layers are effectively independent (no smoothness).
     Between → some smoothness, magnitude bounds the residual "budget".

  2. SUBSPACE STRUCTURE: cumulative energy spectrum of the accumulated
     gramian G_diff^R = Σ_i (D_i D_i^T) projected into hidden-space.
     Low effective rank of G_diff → differences live in a compact
     subspace that could be parameterized as a small number of "flow
     directions" in depth. High effective rank → differences are as
     structurally rich as the weights themselves → ODE parameterization
     won't help.

Decision gate:
  * mean ||D||/||W|| < 0.5 AND diff-stacked effective rank is < 30% of
    per-matrix effective rank → VIABLE (depth parameterization worth
    pursuing as compression scheme).
  * diff-stacked ER ≈ per-matrix ER → DEAD (differences are independent
    of each other, no smoothness to exploit).

Same hidden-space projection as cross_matrix_svd_test.py so results are
directly comparable.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def effective_rank(eigvals: np.ndarray) -> float:
    p = eigvals / (eigvals.sum() + 1e-30)
    return float(1.0 / np.sum(p * p))


def cumulative_energy(eigvals: np.ndarray, ks) -> dict[int, float]:
    sorted_vals = np.sort(eigvals)[::-1]
    total = sorted_vals.sum()
    cum = np.cumsum(sorted_vals) / (total + 1e-30)
    return {k: float(cum[min(k, len(cum)) - 1]) for k in ks}


def _hidden_gramian(W: torch.Tensor, hidden: int) -> torch.Tensor:
    """Project a weight matrix into the hidden-space gramian (fp64).
    Matches the convention in cross_matrix_svd_test.py."""
    d_out, d_in = W.shape
    W32 = W.to(torch.float32)
    if d_out == hidden:
        G = (W32 @ W32.T).to(torch.float64)
    elif d_in == hidden:
        G = (W32.T @ W32).to(torch.float64)
    else:
        raise ValueError(
            f"weight shape {tuple(W.shape)} does not intersect hidden={hidden}")
    return G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="cpu")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--out", default="depth_smoothness_test.pkl")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype}, device_map={args.device_map})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"  loaded in {time.time() - t0:.1f}s")

    hidden = args.hidden_size or model.config.hidden_size
    print(f"hidden_size = {hidden}")

    # Collect per-role per-layer weights. Name format: "model.layers.{N}.{path}"
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    by_role: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        m = layer_re.search(name)
        if not m:
            continue  # skip lm_head etc. — no layer index
        layer_idx = int(m.group(1))
        # Role is the dotted tail after ".layers.N."
        role = name.split(f"layers.{layer_idx}.", 1)[1]
        W = module.weight.data.detach()
        if W.shape[0] != hidden and W.shape[1] != hidden:
            continue
        by_role[role][layer_idx] = W

    # Sort layers per role.
    role_chains: dict[str, list[torch.Tensor]] = {}
    for role, m in by_role.items():
        idxs = sorted(m)
        role_chains[role] = [m[i] for i in idxs]
        print(f"  {role:<30} {len(idxs)} layers  shape={tuple(m[idxs[0]].shape)}")

    ks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    results: dict[str, dict] = {}

    print("\nComputing depth-smoothness spectrum per role…")
    for role, chain in role_chains.items():
        if len(chain) < 2:
            continue
        L = len(chain)

        # Magnitude analysis.
        w_norms = [float(torch.linalg.norm(W.to(torch.float32))) for W in chain]
        diff_norms = []
        G_diff_sum = None  # fp64 on CPU
        G_W_sum = None     # for side-by-side reference
        G_i_list_eigvals = []

        for i in range(L):
            W_i = chain[i]
            G_i = _hidden_gramian(W_i, hidden).cpu().numpy()
            # Normalize per-matrix so per-matrix curves are comparable
            G_i_norm = G_i / (np.trace(G_i) + 1e-30)
            eig_i = np.clip(np.linalg.eigvalsh(G_i_norm), 0, None)
            G_i_list_eigvals.append(eig_i)

            if G_W_sum is None:
                G_W_sum = np.zeros_like(G_i, dtype=np.float64)
            G_W_sum += G_i

            if i == L - 1:
                continue
            D_i = chain[i + 1] - chain[i]
            diff_norms.append(
                float(torch.linalg.norm(D_i.to(torch.float32))))
            G_Di = _hidden_gramian(D_i, hidden).cpu().numpy()
            if G_diff_sum is None:
                G_diff_sum = np.zeros_like(G_Di, dtype=np.float64)
            G_diff_sum += G_Di

        # Per-matrix mean effective rank + mean cumulative curve
        per_mat_er = float(np.mean([effective_rank(e) for e in G_i_list_eigvals]))
        per_mat_ce = {k: float(np.mean(
            [cumulative_energy(e, [k])[k] for e in G_i_list_eigvals])) for k in ks}

        # Stacked-all-weights spectrum (should reproduce prior test)
        eig_Wsum = np.clip(np.linalg.eigvalsh(G_W_sum), 0, None)
        stacked_w_er = effective_rank(eig_Wsum)
        stacked_w_ce = cumulative_energy(eig_Wsum, ks)

        # Differences stacked
        eig_D = np.clip(np.linalg.eigvalsh(G_diff_sum), 0, None)
        diff_er = effective_rank(eig_D)
        diff_ce = cumulative_energy(eig_D, ks)

        # Magnitude ratio: typical diff norm vs typical matrix norm
        mean_w_norm = float(np.mean(w_norms))
        mean_d_norm = float(np.mean(diff_norms))
        d_over_w = mean_d_norm / (mean_w_norm + 1e-30)

        # Total energy ratio
        total_d_energy = float(np.sum(np.square(diff_norms)))
        total_w_energy = float(np.sum(np.square(w_norms)))
        energy_ratio = total_d_energy / (total_w_energy + 1e-30)

        results[role] = {
            "layers": L,
            "shape": tuple(chain[0].shape),
            "per_matrix_er": per_mat_er,
            "per_matrix_ce": per_mat_ce,
            "stacked_w_er": stacked_w_er,
            "stacked_w_ce": stacked_w_ce,
            "diff_er": diff_er,
            "diff_ce": diff_ce,
            "mean_w_norm": mean_w_norm,
            "mean_d_norm": mean_d_norm,
            "d_over_w": d_over_w,
            "energy_ratio": energy_ratio,
            "w_norms": w_norms,
            "d_norms": diff_norms,
            "diff_eigvals": eig_D.astype(np.float32),
            "w_eigvals": eig_Wsum.astype(np.float32),
        }

        print(f"  {role:<30} "
              f"mean ||D||/||W||={d_over_w:.3f}  "
              f"energy ratio={energy_ratio:.3f}  "
              f"per-mat ER={per_mat_er:.0f}  "
              f"diff ER={diff_er:.0f}  "
              f"diff k=128 cum={diff_ce[128]*100:.1f}%")

    # ---- Summary ----
    print("\n" + "=" * 88)
    print(f"Depth-smoothness report — Qwen 2.5 7B, hidden={hidden}")
    print("=" * 88)
    print(f"{'role':<30} {'||D||/||W||':>12} {'energy%':>8} "
          f"{'per-mat ER':>11} {'diff ER':>8} {'ratio':>7} "
          f"{'diff@k=64':>10} {'@k=128':>8} {'@k=256':>8}")
    for role in sorted(results):
        r = results[role]
        ratio = r["diff_er"] / max(r["per_matrix_er"], 1e-9)
        print(f"{role:<30} {r['d_over_w']:>12.3f} "
              f"{r['energy_ratio']*100:>7.1f}% "
              f"{r['per_matrix_er']:>11.0f} "
              f"{r['diff_er']:>8.0f} "
              f"{ratio:>7.2f} "
              f"{r['diff_ce'][64]*100:>9.1f}% "
              f"{r['diff_ce'][128]*100:>7.1f}% "
              f"{r['diff_ce'][256]*100:>7.1f}%")

    # Role-by-role verdict
    print("\n=== Per-role verdicts ===")
    for role in sorted(results):
        r = results[role]
        ratio = r["diff_er"] / max(r["per_matrix_er"], 1e-9)
        if r["d_over_w"] < 0.5 and ratio < 0.3:
            verdict = "VIABLE"
        elif r["d_over_w"] < 0.7 and ratio < 0.5:
            verdict = "WEAK-VIABLE"
        else:
            verdict = "NOT VIABLE"
        print(f"  {role:<30} "
              f"||D||/||W||={r['d_over_w']:.2f} "
              f"diff/matrix ER ratio={ratio:.2f}  →  {verdict}")

    Path(args.out).write_bytes(pickle.dumps(results))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
