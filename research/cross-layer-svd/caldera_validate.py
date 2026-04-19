"""Real-matrix CALDERA validation.

Loads a saved (W, Sigma) pair from balanced_snapshot.pkl (Qwen 2.5 0.5B
MLP weight + its activation gramian from calibration), then compares:

  1. Pure quant baseline        — W -> dequant(quantize(W))
  2. Pure Σ-weighted low-rank    — W -> L·R via activation-weighted SVD
  3. CALDERA (RPCD)             — W -> dequant(Q) + L·R, alternating fit

...at multiple (qtype, rank) points. Reports Σ-weighted relative error
and effective bits-per-weight for each config so we can read the
quality-vs-size curve directly.

Usage:
  python caldera_validate.py [--snapshot balanced_snapshot.pkl]
                             [--iters 3] [--device cuda] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

from basis_sharing import sqrt_and_inv  # noqa: E402
from caldera import (  # noqa: E402
    _QTYPE_BY_NAME,
    _baseline_quant,
    _baseline_svd,
    caldera_decompose,
)

# Bits per stored weight for GGUF quant formats (block overhead included)
_QTYPE_BPW = {
    "Q4_0": 4.5,   # 4-bit codes + fp16 scale per 32-weight block
    "Q8_0": 8.5,   # 8-bit codes + fp16 scale per 32-weight block
}


def effective_bpw(qtype_name: str, rank: int, d_out: int, d_in: int) -> float:
    """bpw of quantized W plus fp16 L·R overhead per weight."""
    base = _QTYPE_BPW[qtype_name]
    lr_bits = 2 * rank * (d_out + d_in) * 16 / (d_out * d_in)
    return base + lr_bits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="balanced_snapshot.pkl")
    p.add_argument("--ranks", default="32,64,128,256")
    p.add_argument("--qtypes", default="Q4_0,Q8_0")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    snap_path = Path(args.snapshot)
    with snap_path.open("rb") as fh:
        snap = pickle.load(fh)

    W = snap["W"].to(device=device, dtype=torch.float32)
    Sigma = snap["XTX"].to(device=device, dtype=torch.float32)
    d_out, d_in = W.shape

    print(f"Model      : {snap.get('model_id','?')}")
    print(f"Role/Layer : {snap.get('role','?')} / {snap.get('layer_idx','?')}")
    print(f"W shape    : [{d_out}, {d_in}]")
    print(f"Sigma shape: {tuple(Sigma.shape)}")
    print(f"Calib samps: {snap.get('calib_samples','?')}")
    print(f"Device     : {device}")
    print(f"Iters      : {args.iters}\n")

    ranks = [int(r) for r in args.ranks.split(",") if r]
    qtypes = [q.strip() for q in args.qtypes.split(",") if q.strip()]

    result = {"snapshot": str(snap_path), "W_shape": [d_out, d_in],
              "rows": []}

    # --- Pure quant baselines ---
    print("=== Pure quant baselines ===")
    print(f"{'qtype':<8} {'err':>8} {'bpw':>6}")
    for qt in qtypes:
        q = _QTYPE_BY_NAME[qt]
        err = _baseline_quant(W, Sigma, q)
        bpw = _QTYPE_BPW[qt]
        print(f"{qt:<8} {err:>8.4f} {bpw:>6.2f}")
        result["rows"].append({"method": "pure_quant", "qtype": qt,
                               "rank": None, "err": err, "bpw": bpw})

    # --- Pure LR baseline per rank ---
    print("\n=== Pure Σ-weighted low-rank ===")
    print(f"{'rank':<6} {'err':>8} {'bpw':>6}")
    for r in ranks:
        err = _baseline_svd(W, Sigma, r)
        bpw = 2 * r * (d_out + d_in) * 16 / (d_out * d_in)
        print(f"{r:<6} {err:>8.4f} {bpw:>6.2f}")
        result["rows"].append({"method": "pure_lr", "qtype": None,
                               "rank": r, "err": err, "bpw": bpw})

    # --- CALDERA sweep ---
    print("\n=== CALDERA (RPCD) ===")
    print(f"{'qtype':<8} {'rank':<6} {'err':>8} {'bpw':>6}  history")
    for qt in qtypes:
        q = _QTYPE_BY_NAME[qt]
        for r in ranks:
            res = caldera_decompose(W, Sigma, r, q,
                                    n_iters=args.iters, device=device)
            bpw = effective_bpw(qt, r, d_out, d_in)
            hist = [round(e, 4) for e in res.rel_err_history]
            print(f"{qt:<8} {r:<6} {res.rel_err_history[-1]:>8.4f} "
                  f"{bpw:>6.2f}  {hist}")
            result["rows"].append({"method": "caldera", "qtype": qt,
                                   "rank": r, "err": res.rel_err_history[-1],
                                   "bpw": bpw, "history": hist})

    # Quick summary: best non-quant baseline vs best CALDERA at each rank
    print("\n=== Summary ===")
    print(f"{'config':<28} {'err':>8} {'bpw':>6}  {'vs pure_quant':>15}")
    pure_q = {row["qtype"]: row["err"] for row in result["rows"]
              if row["method"] == "pure_quant"}
    for row in result["rows"]:
        if row["method"] != "caldera":
            continue
        pq = pure_q.get(row["qtype"])
        delta_pct = (row["err"] - pq) / pq * 100 if pq else None
        tag = f"CALDERA {row['qtype']} r={row['rank']}"
        print(f"{tag:<28} {row['err']:>8.4f} {row['bpw']:>6.2f}  "
              f"{'improvement' if delta_pct and delta_pct < 0 else 'regression'} "
              f"{delta_pct:+.1f}%" if delta_pct is not None else "")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
