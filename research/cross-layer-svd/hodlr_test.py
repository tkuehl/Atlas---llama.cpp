"""HODLR (Hierarchically Off-Diagonal Low-Rank) on a single weight matrix.

Tests the core hypothesis: non-uniform rank allocation (high rank near the
diagonal, low rank off-diagonal) beats uniform-rank flat SVD at matched storage.

We compare on ONE MLP weight (e.g., layer-12 gate_proj from Qwen 2.5 0.5B) at
depth in {1, 2} and several off-diagonal ranks. For each HODLR configuration we
compute storage, then run flat SVD at the rank that uses the same storage, and
report the reconstruction error for both.

Runs activation-aware (ASVD-style) and plain variants. No model forward passes
here — weight + XTX are pickled by extract_one_matrix().

Usage:
    python hodlr_test.py extract --model Qwen/Qwen2.5-0.5B --role mlp.gate_proj --layer 12
    python hodlr_test.py sweep
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import torch

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

HERE = Path(__file__).resolve().parent


# ---------- HODLR core ----------

def low_rank_svd(M, rank):
    """Rank-r SVD of M. Returns (U_r @ diag(sigma_r), Vt_r) so that UV ≈ M."""
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, S.numel())
    return (U[:, :r] * S[:r].unsqueeze(0)).contiguous(), Vt[:r, :].contiguous()


def hodlr_decompose(W, depth, offdiag_rank, weighted_offdiag=None):
    """Recursive HODLR. Off-diagonal blocks are low-rank, diagonal blocks recurse.

    weighted_offdiag: callable(block, row_slice, col_slice) -> weighted_block.
        Used for activation-aware SVD on off-diagonal blocks. The weighting must
        be applied only to the SVD input; the stored factors still reconstruct W.
        For now we pass the full XTX and slice the diagonal per block (not ideal
        — ignores cross-block covariance — but fine for a first test).
    """
    m, n = W.shape
    if depth == 0:
        return {"type": "dense", "W": W.contiguous(),
                "shape": (m, n)}

    m2 = m // 2
    n2 = n // 2
    # Handle odd dimensions: put the extra row/col on the bottom-right diagonal block
    W11 = W[:m2, :n2]
    W12 = W[:m2, n2:]
    W21 = W[m2:, :n2]
    W22 = W[m2:, n2:]

    def _offdiag(block, row_off, col_off):
        h, w = block.shape
        if weighted_offdiag is not None:
            M = weighted_offdiag(block, slice(row_off, row_off + h),
                                  slice(col_off, col_off + w))
            U, Vt = low_rank_svd(M, offdiag_rank)
            # Projection-form reconstruction: we want to approximate the UNweighted
            # block, so use U_r U_r^T @ block (the S drops out in the projection).
            U_r = U / (U.norm(dim=0, keepdim=True) + 1e-12)  # re-orthonormalize
            A = U_r.T @ block
            return {"type": "lowrank", "U": U_r.contiguous(), "V": A.contiguous()}
        else:
            U, Vt = low_rank_svd(block, offdiag_rank)
            return {"type": "lowrank", "U": U.contiguous(), "V": Vt.contiguous()}

    return {
        "type": "hodlr",
        "shape": (m, n),
        "m2": m2, "n2": n2,
        "W11": hodlr_decompose(W11, depth - 1, offdiag_rank, weighted_offdiag),
        "W22": hodlr_decompose(W22, depth - 1, offdiag_rank, weighted_offdiag),
        "W12": _offdiag(W12, 0, n2),
        "W21": _offdiag(W21, m2, 0),
    }


def hodlr_reconstruct(node):
    """Rebuild W from the decomposition."""
    if node["type"] == "dense":
        return node["W"]
    if node["type"] == "lowrank":
        return node["U"] @ node["V"]
    W11 = hodlr_reconstruct(node["W11"])
    W22 = hodlr_reconstruct(node["W22"])
    W12 = hodlr_reconstruct(node["W12"])
    W21 = hodlr_reconstruct(node["W21"])
    top = torch.cat([W11, W12], dim=1)
    bot = torch.cat([W21, W22], dim=1)
    return torch.cat([top, bot], dim=0)


def hodlr_storage(node):
    """Parameter count of a HODLR tree."""
    if node["type"] == "dense":
        return node["W"].numel()
    if node["type"] == "lowrank":
        return node["U"].numel() + node["V"].numel()
    return (hodlr_storage(node["W11"]) + hodlr_storage(node["W22"])
            + hodlr_storage(node["W12"]) + hodlr_storage(node["W21"]))


# ---------- Flat SVD baseline ----------

def flat_svd_reconstruct(W, rank, XTX=None):
    """Flat SVD reconstruction at a given rank. ASVD if XTX provided."""
    if XTX is not None:
        evals, evecs = torch.linalg.eigh(XTX.to(torch.float64)
                                         + 1e-6 * XTX.diag().mean().item()
                                         * torch.eye(XTX.shape[0], dtype=torch.float64))
        evals = evals.clamp(min=1e-12)
        S = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T
        S = S.to(W.dtype)
        W_weighted = W @ S
        U, _, _ = torch.linalg.svd(W_weighted, full_matrices=False)
        r = min(rank, U.shape[1])
        U_r = U[:, :r]
        return U_r @ (U_r.T @ W)
    else:
        U, sigma, Vt = torch.linalg.svd(W, full_matrices=False)
        r = min(rank, sigma.numel())
        return (U[:, :r] * sigma[:r].unsqueeze(0)) @ Vt[:r, :]


def flat_svd_storage(shape, rank):
    m, n = shape
    return (m + n) * rank


# ---------- Error metrics ----------

def rel_err(W, W_hat, weight=None):
    """||W - W_hat|| / ||W||. If weight provided, weighted Frobenius."""
    if weight is not None:
        diff = (W - W_hat) @ weight
        norm = W @ weight
    else:
        diff = W - W_hat
        norm = W
    return (torch.linalg.norm(diff) / (torch.linalg.norm(norm) + 1e-12)).item()


def storage_to_flat_rank(shape, storage_target):
    """Given a byte budget, what flat-SVD rank fits?"""
    m, n = shape
    return storage_target // (m + n)


# ---------- Calibration snapshot I/O ----------

def extract_one_matrix(model_id, role, layer_idx, out_path, calib_samples=64):
    """Load model, run calibration, snapshot one weight + its XTX to disk."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    model.eval()

    target_name = f"model.layers.{layer_idx}.{role}"
    target_mod = dict(model.named_modules())[target_name]

    xtx = torch.zeros(target_mod.in_features, target_mod.in_features,
                      dtype=torch.float32)

    def hook(mod, inputs):
        x = inputs[0]
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        xtx.add_((flat.T @ flat).cpu())

    h = target_mod.register_forward_pre_hook(hook)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    from tqdm import tqdm
    count = 0
    with torch.no_grad():
        for row in tqdm(ds, desc="calib"):
            t = row["text"].strip()
            if len(t) < 400:
                continue
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            ids = enc.input_ids.to("cuda")
            if ids.shape[1] < 8:
                continue
            model(ids)
            count += 1
            if count >= calib_samples:
                break

    h.remove()
    W = target_mod.weight.data.cpu().clone()
    Path(out_path).write_bytes(pickle.dumps({
        "model_id": model_id,
        "role": role,
        "layer_idx": layer_idx,
        "W": W,
        "XTX": xtx,
        "calib_samples": count,
    }))
    print(f"wrote {out_path}: W={tuple(W.shape)}, XTX diag mean={xtx.diag().mean().item():.3f}")


# ---------- Sweep ----------

def sweep(snapshot_path, depth_list, offdiag_ranks):
    data = pickle.loads(Path(snapshot_path).read_bytes())
    W = data["W"].to(torch.float32)
    XTX = data["XTX"]
    print(f"\nmatrix: {data['role']} layer {data['layer_idx']} shape {tuple(W.shape)}")
    print(f"original storage = {W.numel():,} params = {W.numel()*2/1e6:.2f} MB fp16\n")

    # Per-block XTX slice: each off-diagonal block [h, w] with column range
    # [c0, c0+w] uses XTX[c0:c0+w, c0:c0+w] as its whitening target. This ignores
    # cross-block covariance but is a reasonable per-block proxy.
    def weighted_offdiag(block, row_slice, col_slice):
        sub = XTX[col_slice, col_slice].to(torch.float64)
        evals, evecs = torch.linalg.eigh(
            sub + 1e-6 * sub.diag().mean().item() * torch.eye(sub.shape[0], dtype=torch.float64))
        evals = evals.clamp(min=1e-12)
        S_sub = ((evecs * evals.sqrt().unsqueeze(0)) @ evecs.T).to(block.dtype)
        return block @ S_sub

    # Full-XTX flat SVD baseline
    flat_rec_asvd_cache = {}

    print(f"{'mode':<30} {'params':>12} {'bytes MB':>10} {'rel_err':>10} {'wgt_err':>10}")
    print("-" * 78)

    # HODLR sweep
    for depth in depth_list:
        for r in offdiag_ranks:
            t0 = time.time()
            # ASVD HODLR
            node_asvd = hodlr_decompose(W, depth, r, weighted_offdiag)
            W_hat_asvd = hodlr_reconstruct(node_asvd)
            params = hodlr_storage(node_asvd)
            err_plain = rel_err(W, W_hat_asvd)
            err_weighted = rel_err(W, W_hat_asvd,
                                   weight=_whiten(XTX, W.dtype))
            print(f"HODLR d={depth} r={r:<4} ASVD  "
                  f"{params:>12,} {params*2/1e6:>10.2f} {err_plain:>10.4f} {err_weighted:>10.4f}  ({time.time()-t0:.1f}s)")

            # Matched flat SVD
            flat_rank = storage_to_flat_rank(W.shape, params)
            key = (flat_rank, "asvd")
            if key not in flat_rec_asvd_cache:
                flat_rec_asvd_cache[key] = flat_svd_reconstruct(W, flat_rank, XTX)
            W_flat = flat_rec_asvd_cache[key]
            flat_params = flat_svd_storage(W.shape, flat_rank)
            err_flat = rel_err(W, W_flat)
            err_flat_w = rel_err(W, W_flat, weight=_whiten(XTX, W.dtype))
            print(f"  flat SVD ASVD r={flat_rank:<4} (matched bytes)   "
                  f"{flat_params:>12,} {flat_params*2/1e6:>10.2f} {err_flat:>10.4f} {err_flat_w:>10.4f}")


def _whiten(XTX, dtype):
    """Compute S = (XTX + eps*I)^(1/2) for weighted-error computation."""
    xtx = XTX.to(torch.float64)
    eps = 1e-6 * xtx.diag().mean().item()
    reg = xtx + eps * torch.eye(xtx.shape[0], dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(reg)
    evals = evals.clamp(min=eps)
    return ((evecs * evals.sqrt().unsqueeze(0)) @ evecs.T).to(dtype)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="snapshot one weight + XTX from a model")
    p_extract.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_extract.add_argument("--role", default="mlp.gate_proj")
    p_extract.add_argument("--layer", type=int, default=12)
    p_extract.add_argument("--out", default=str(HERE / "hodlr_snapshot.pkl"))
    p_extract.add_argument("--calib-samples", type=int, default=64)

    p_sweep = sub.add_parser("sweep", help="HODLR vs flat SVD sweep")
    p_sweep.add_argument("--snapshot", default=str(HERE / "hodlr_snapshot.pkl"))
    p_sweep.add_argument("--depth", nargs="+", type=int, default=[1, 2])
    p_sweep.add_argument("--ranks", nargs="+", type=int, default=[32, 64, 128, 256, 512])

    args = p.parse_args()

    if args.cmd == "extract":
        extract_one_matrix(args.model, args.role, args.layer, args.out, args.calib_samples)
    elif args.cmd == "sweep":
        sweep(args.snapshot, args.depth, args.ranks)


if __name__ == "__main__":
    main()
