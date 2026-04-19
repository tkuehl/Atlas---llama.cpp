"""Hypothesis test — is there compact cross-matrix shared structure?

Gate for the cross-matrix global-codebook research track. Tests whether
transformer linear weights share principal directions in the residual-
stream basis (hidden-size dimension).

For each nn.Linear in the model, project to the hidden-size subspace:
  * d_out == hidden  →  G_i = W_normalized @ W_normalized.T   (3584×3584)
  * d_in  == hidden  →  G_i = W_normalized.T @ W_normalized   (3584×3584)
  * neither          →  skipped (embedding/head are different coords)

Each W is normalized by its Frobenius norm so tr(G_i) = 1 for every matrix
— G_i is a unit-energy "direction distribution" over hidden-space.

Two summaries reported:

  * Stacked:     G_sum = Σ_i G_i  (shared-structure accumulator)
  * Baseline:    mean eigenvalue curve over per-matrix G_i's (null
                 hypothesis "no sharing, each matrix is independent")

If matrices share residual-stream directions, the stacked spectrum is
more concentrated than the per-matrix mean — top K captures proportionally
more energy. Conversely, if matrices are independent, the stacked
spectrum tracks the mean.

Decision rule (per memory project_unweight_research.md):
  * Top 1024 of G_sum captures ≥90% energy AND stacked effective rank is
    significantly smaller than the per-matrix mean effective rank
    → shared-dictionary thesis viable, pursue.
  * Stacked spectrum ≈ per-matrix mean
    → matrices genuinely independent, archive direction.

Usage:
  python cross_matrix_svd_test.py --model Qwen/Qwen2.5-7B
                                  [--out shared_basis_test.pkl]
                                  [--hidden-size 3584]
                                  [--roles q_proj,k_proj,...]
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def effective_rank(eigvals: np.ndarray) -> float:
    """participation ratio: 1 / Σ p_i² where p = eigval/sum(eigval).
    Measures how evenly mass is distributed over eigendirections."""
    p = eigvals / (eigvals.sum() + 1e-30)
    return float(1.0 / np.sum(p * p))


def cumulative_energy(eigvals: np.ndarray, ks) -> dict[int, float]:
    """Fraction of total energy captured by top-k eigenvalues."""
    sorted_vals = np.sort(eigvals)[::-1]
    total = sorted_vals.sum()
    cum = np.cumsum(sorted_vals) / (total + 1e-30)
    return {k: float(cum[min(k, len(cum)) - 1]) for k in ks}


def iter_linears_hidden_projected(model, hidden_size: int, role_filter=None):
    """Yield (name, role, G_i, direction) for each nn.Linear that
    intersects hidden_size on either axis."""
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        W = module.weight.data.detach()
        d_out, d_in = W.shape

        # Match role filter (e.g. "q_proj", "down_proj")
        role = name.split(".")[-1]
        if role_filter is not None and role not in role_filter:
            continue

        if d_out == hidden_size and d_in == hidden_size:
            direction = "both"  # square matrices: use output-basis by default
            W32 = W.to(torch.float32)
            frob = torch.linalg.norm(W32) + 1e-30
            Wn = W32 / frob
            G = (Wn @ Wn.T).to(torch.float64)
        elif d_out == hidden_size:
            direction = "out"
            W32 = W.to(torch.float32)
            frob = torch.linalg.norm(W32) + 1e-30
            Wn = W32 / frob
            G = (Wn @ Wn.T).to(torch.float64)
        elif d_in == hidden_size:
            direction = "in"
            W32 = W.to(torch.float32)
            frob = torch.linalg.norm(W32) + 1e-30
            Wn = W32 / frob
            G = (Wn.T @ Wn).to(torch.float64)
        else:
            # lm_head / embeddings / etc. — different coord system, skip
            continue

        yield name, role, G.cpu().numpy(), direction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--hidden-size", type=int, default=None,
                   help="override hidden size; auto-detected from config if unset")
    p.add_argument("--roles", default=None,
                   help="comma-separated role filter, e.g. q_proj,down_proj")
    p.add_argument("--out", default="shared_basis_test.pkl")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype}, device_map={args.device_map})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"  loaded in {time.time() - t0:.1f}s")

    hidden = args.hidden_size or model.config.hidden_size
    print(f"hidden_size = {hidden}")

    role_filter = None
    if args.roles:
        role_filter = set(args.roles.split(","))
        print(f"role filter: {sorted(role_filter)}")

    per_matrix: list[dict] = []
    per_role_sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((hidden, hidden), dtype=np.float64)
    )
    per_role_counts: dict[str, int] = defaultdict(int)
    global_sum = np.zeros((hidden, hidden), dtype=np.float64)
    total = 0

    ks = [32, 64, 128, 256, 512, 1024, 2048]

    t0 = time.time()
    for name, role, G, direction in iter_linears_hidden_projected(
            model, hidden, role_filter):
        eig = np.linalg.eigvalsh(G)
        eig = np.clip(eig, 0, None)
        ce = cumulative_energy(eig, ks)
        er = effective_rank(eig)
        per_matrix.append({
            "name": name, "role": role, "direction": direction,
            "effective_rank": er, "cumulative_energy": ce,
            "eigvals": eig.astype(np.float32),  # save all for plotting
        })
        per_role_sums[role] += G
        per_role_counts[role] += 1
        global_sum += G
        total += 1
        if total % 20 == 0:
            print(f"  {total} matrices processed "
                  f"({(time.time() - t0)/total:.2f}s/matrix)")

    if total == 0:
        raise SystemExit("No matrices matched. Check role filter / hidden size.")

    print(f"\n{total} matrices in {time.time() - t0:.1f}s")

    # === Stacked analysis ===
    eig_global = np.linalg.eigvalsh(global_sum)
    eig_global = np.clip(eig_global, 0, None)
    ce_global = cumulative_energy(eig_global, ks)
    er_global = effective_rank(eig_global)

    # Per-matrix mean curves (null hypothesis "no sharing")
    mean_ce = {k: float(np.mean([m["cumulative_energy"][k]
                                  for m in per_matrix])) for k in ks}
    mean_er = float(np.mean([m["effective_rank"] for m in per_matrix]))

    # Per-role stacked analysis
    role_stacked: dict[str, dict] = {}
    for role, G in per_role_sums.items():
        eig = np.linalg.eigvalsh(G)
        eig = np.clip(eig, 0, None)
        ce = cumulative_energy(eig, ks)
        role_stacked[role] = {
            "count": per_role_counts[role],
            "effective_rank": effective_rank(eig),
            "cumulative_energy": ce,
            "eigvals": eig.astype(np.float32),
        }

    # === Report ===
    print("\n" + "=" * 70)
    print(f"Hidden-space shared-basis test — {total} matrices, hidden={hidden}")
    print("=" * 70)

    print(f"\n{'':<14} {'er':>7}   ", end="")
    for k in ks:
        print(f"k={k:<4}", end="  ")
    print()

    print(f"{'stacked-all':<14} {er_global:>7.1f}   ", end="")
    for k in ks:
        print(f"{ce_global[k]*100:>5.1f}%", end="  ")
    print()
    print(f"{'per-mat mean':<14} {mean_er:>7.1f}   ", end="")
    for k in ks:
        print(f"{mean_ce[k]*100:>5.1f}%", end="  ")
    print()

    print("\n=== Per-role stacked ===")
    print(f"{'role':<14} {'n':>3} {'er':>7}   ", end="")
    for k in ks:
        print(f"k={k:<4}", end="  ")
    print()
    for role in sorted(role_stacked):
        rs = role_stacked[role]
        print(f"{role:<14} {rs['count']:>3} {rs['effective_rank']:>7.1f}   ",
              end="")
        for k in ks:
            print(f"{rs['cumulative_energy'][k]*100:>5.1f}%", end="  ")
        print()

    # === Verdict ===
    print("\n=== Decision gates ===")
    ce_1024_global = ce_global[1024]
    share_ratio = mean_er / max(er_global, 1e-9)
    print(f"Top-1024 of stacked captures {ce_1024_global*100:.1f}% "
          f"(gate ≥90% for 'pursue')")
    print(f"Stacked effective rank = {er_global:.1f}")
    print(f"Per-matrix mean ER     = {mean_er:.1f}")
    print(f"Compression ratio (mean/stacked) = {share_ratio:.2f}× "
          f"(>1 = shared structure; =1 = independent)")

    if ce_1024_global >= 0.90 and share_ratio > 1.5:
        verdict = "VIABLE — pursue shared-dictionary research bet"
    elif ce_1024_global >= 0.80 and share_ratio > 1.2:
        verdict = "WEAK-VIABLE — some shared structure, marginal"
    else:
        verdict = "DEAD — matrices are largely independent; archive direction"
    print(f"\nVerdict: {verdict}")

    # === Save ===
    output = {
        "model": args.model,
        "hidden_size": hidden,
        "total_matrices": total,
        "global_stacked": {
            "effective_rank": er_global,
            "cumulative_energy": ce_global,
            "eigvals": eig_global.astype(np.float32),
        },
        "per_matrix_mean": {
            "effective_rank": mean_er,
            "cumulative_energy": mean_ce,
        },
        "per_role_stacked": role_stacked,
        "per_matrix": per_matrix,
        "ks": ks,
        "verdict_ratio": float(share_ratio),
        "verdict_ce_1024": float(ce_1024_global),
    }
    Path(args.out).write_bytes(pickle.dumps(output))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
