"""Balanced truncation on a single weight matrix.

Tests whether adding output-side gradient covariance weighting (balanced POD
from control theory) beats plain activation-aware SVD (ASVD) at matched rank
on the weighted-error metric.

Three reconstructions compared at each rank:
  plain SVD  - no weighting, rank-r truncation of W
  ASVD       - input weighting only, rank-r projection via U from SVD(W @ S_in)
  balanced   - input+output weighting, rank-r truncation of S_out @ W @ S_in
               then un-weight: W_r = S_out^{-1} U_r Sigma_r V_r^T S_in^{-1}

We also report an "actionable" metric: ||S_out (W - W_r) S_in||_F / ||S_out W S_in||_F,
which measures error in the coordinate system where both input importance (what
activations hit which input channels) and output importance (how much output
channels affect the loss) are accounted for. This is the metric that best
predicts downstream PPL impact.
"""

import argparse
import pickle
import time
from pathlib import Path

import torch

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

HERE = Path(__file__).resolve().parent


# ---------- gramian sqrt helpers ----------

def sqrt_and_inv(M, eps_ratio=1e-5):
    """Compute S and S^{-1} such that S^2 ≈ M + eps*I, symmetric PSD."""
    M64 = M.to(torch.float64)
    eps = eps_ratio * M64.diag().mean().item() + 1e-12
    M_reg = M64 + eps * torch.eye(M64.shape[0], dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(M_reg)
    evals = evals.clamp(min=eps)
    sqrt_e = evals.sqrt()
    inv_sqrt_e = 1.0 / sqrt_e
    S = (evecs * sqrt_e.unsqueeze(0)) @ evecs.T
    S_inv = (evecs * inv_sqrt_e.unsqueeze(0)) @ evecs.T
    return S, S_inv


# ---------- three reconstructions ----------

def plain_svd(W, rank):
    U, sigma, Vt = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    r = min(rank, sigma.numel())
    W_r = (U[:, :r] * sigma[:r].unsqueeze(0)) @ Vt[:r, :]
    return W_r.to(W.dtype)


def asvd(W, XTX, rank):
    """Input-weighted ASVD: W_r = U_r U_r^T W where U_r is top-r of SVD(W @ S_in)."""
    S_in, _ = sqrt_and_inv(XTX)
    W64 = W.to(torch.float64)
    M = W64 @ S_in
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, U.shape[1])
    U_r = U[:, :r]
    W_r = U_r @ (U_r.T @ W64)
    return W_r.to(W.dtype)


def balanced(W, XTX, GGT, rank):
    """Balanced truncation: rank-r of S_out @ W @ S_in, un-weighted."""
    S_in, S_in_inv = sqrt_and_inv(XTX)
    S_out, S_out_inv = sqrt_and_inv(GGT)
    W64 = W.to(torch.float64)
    M = S_out @ W64 @ S_in
    U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, sigma.numel())
    M_r = (U[:, :r] * sigma[:r].unsqueeze(0)) @ Vt[:r, :]
    W_r = S_out_inv @ M_r @ S_in_inv
    return W_r.to(W.dtype)


def obs_repair_svd(W, XTX, rank, eps_ratio=1e-5):
    """Plain-SVD rank-r truncation with OBS-style input-side repair.

    Adapted from SparseGPT/OBS math (our derivation; paper doesn't directly
    address SVD truncation): let V_k be the d_in - r bottom right singular
    vectors of W (the dropped input directions). With H = XTX + eps*I:

        W' = W - (W V_k) (V_k^T H^-1 V_k)^-1 (H^-1 V_k)^T

    Intuition: instead of orthogonally projecting out V_k (which plain SVD
    does), this projection is in the H^-1 inner product — accounting for
    which input directions the calibration data actually activates.
    """
    W64 = W.to(torch.float64)
    d_out, d_in = W64.shape

    # Plain SVD, same as plain_svd() — kept here so we can reuse V.
    U, sigma, Vt = torch.linalg.svd(W64, full_matrices=False)
    r = min(rank, sigma.numel())
    if r >= Vt.shape[0]:
        return W.clone()

    V_k = Vt[r:, :].T                   # [d_in, d_in - r] (dropped directions)

    # H = XTX + eps*I
    xtx64 = XTX.to(torch.float64)
    eps = eps_ratio * xtx64.diag().mean().item() + 1e-8
    H = xtx64 + eps * torch.eye(d_in, dtype=torch.float64)

    # Stable H^-1 @ V_k via Cholesky solve (don't materialize H^-1).
    L = torch.linalg.cholesky(H)                     # H = L L^T
    Hinv_V = torch.cholesky_solve(V_k, L)            # [d_in, d_in - r]

    # Gram: V_k^T H^-1 V_k
    Gram = V_k.T @ Hinv_V                            # [d_in - r, d_in - r]
    Gram_inv_HinvV_T = torch.linalg.solve(Gram, Hinv_V.T)  # [d_in - r, d_in]

    correction = (W64 @ V_k) @ Gram_inv_HinvV_T      # [d_out, d_in]
    W_repaired = W64 - correction
    return W_repaired.to(W.dtype)


# ---------- error metrics ----------

def err_plain(W, W_hat):
    diff = W - W_hat
    return (torch.linalg.norm(diff.to(torch.float64))
            / (torch.linalg.norm(W.to(torch.float64)) + 1e-12)).item()


def err_input_weighted(W, W_hat, S_in):
    W64 = W.to(torch.float64)
    Wh64 = W_hat.to(torch.float64)
    diff = (W64 - Wh64) @ S_in
    ref = W64 @ S_in
    return (torch.linalg.norm(diff) / (torch.linalg.norm(ref) + 1e-12)).item()


def err_balanced_weighted(W, W_hat, S_in, S_out):
    W64 = W.to(torch.float64)
    Wh64 = W_hat.to(torch.float64)
    diff = S_out @ (W64 - Wh64) @ S_in
    ref = S_out @ W64 @ S_in
    return (torch.linalg.norm(diff) / (torch.linalg.norm(ref) + 1e-12)).item()


# ---------- extract (one matrix + X^T X + G^T G) ----------

def extract(model_id, role, layer_idx, out_path, calib_samples=32, seq_len=512):
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # fp32 weights so backward pass has stable gradients
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target_name = f"model.layers.{layer_idx}.{role}"
    target_mod = dict(model.named_modules())[target_name]
    target_mod.weight.requires_grad_(True)  # needed for backward hook to fire

    d_in = target_mod.in_features
    d_out = target_mod.out_features
    xtx = torch.zeros(d_in, d_in, dtype=torch.float32)
    ggt = torch.zeros(d_out, d_out, dtype=torch.float32)

    def fwd_hook(mod, inputs):
        x = inputs[0].detach()
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        xtx.add_((flat.T @ flat).cpu())

    def bwd_hook(mod, grad_input, grad_output):
        g = grad_output[0].detach()
        flat = g.reshape(-1, g.shape[-1]).to(torch.float32)
        ggt.add_((flat.T @ flat).cpu())

    h_fwd = target_mod.register_forward_pre_hook(fwd_hook)
    h_bwd = target_mod.register_full_backward_hook(bwd_hook)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    count = 0
    for row in tqdm(ds, desc="calib+bwd"):
        t = row["text"].strip()
        if len(t) < 400:
            continue
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=seq_len)
        ids = enc.input_ids.to("cuda")
        if ids.shape[1] < 8:
            continue
        out = model(ids, labels=ids)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        count += 1
        if count >= calib_samples:
            break

    h_fwd.remove()
    h_bwd.remove()

    W = target_mod.weight.data.cpu().clone()
    Path(out_path).write_bytes(pickle.dumps({
        "model_id": model_id,
        "role": role,
        "layer_idx": layer_idx,
        "W": W,
        "XTX": xtx,
        "GGT": ggt,
        "calib_samples": count,
    }))
    print(f"wrote {out_path}: W={tuple(W.shape)} "
          f"XTX diag mean={xtx.diag().mean().item():.3f} "
          f"GGT diag mean={ggt.diag().mean().item():.3e}")


# ---------- sweep ----------

def sweep(snapshot_path, ranks):
    data = pickle.loads(Path(snapshot_path).read_bytes())
    W = data["W"].to(torch.float32)
    XTX = data["XTX"]
    GGT = data["GGT"]
    print(f"\nmatrix: {data['role']} layer {data['layer_idx']} shape {tuple(W.shape)}")
    print(f"XTX diag mean={XTX.diag().mean().item():.3f}  "
          f"GGT diag mean={GGT.diag().mean().item():.3e}\n")

    S_in, _ = sqrt_and_inv(XTX)
    S_out, _ = sqrt_and_inv(GGT)

    print(f"{'method':<16} {'rank':>6} {'plain_err':>10} {'in_wgt':>10} {'bal_wgt':>10}  {'time':>8}")
    print("-" * 72)

    for r in ranks:
        for name, fn in [("plain-SVD",    lambda r=r: plain_svd(W, r)),
                         ("SVD + OBS",    lambda r=r: obs_repair_svd(W, XTX, r)),
                         ("ASVD",         lambda r=r: asvd(W, XTX, r)),
                         ("balanced",     lambda r=r: balanced(W, XTX, GGT, r))]:
            t0 = time.time()
            W_hat = fn()
            dt = time.time() - t0
            ep = err_plain(W, W_hat)
            ei = err_input_weighted(W, W_hat, S_in)
            eb = err_balanced_weighted(W, W_hat, S_in, S_out)
            print(f"{name:<16} {r:>6} {ep:>10.4f} {ei:>10.4f} {eb:>10.4f}  {dt:>7.1f}s")
        print()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_e = sub.add_parser("extract", help="snapshot W, XTX, GGT from a chosen layer")
    p_e.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_e.add_argument("--role", default="mlp.gate_proj")
    p_e.add_argument("--layer", type=int, default=12)
    p_e.add_argument("--out", default=str(HERE / "balanced_snapshot.pkl"))
    p_e.add_argument("--calib-samples", type=int, default=32)

    p_s = sub.add_parser("sweep", help="compare plain SVD vs ASVD vs balanced")
    p_s.add_argument("--snapshot", default=str(HERE / "balanced_snapshot.pkl"))
    p_s.add_argument("--ranks", nargs="+", type=int,
                     default=[64, 128, 256, 384, 512, 640, 768])

    args = p.parse_args()
    if args.cmd == "extract":
        extract(args.model, args.role, args.layer, args.out, args.calib_samples)
    elif args.cmd == "sweep":
        sweep(args.snapshot, args.ranks)


if __name__ == "__main__":
    main()
