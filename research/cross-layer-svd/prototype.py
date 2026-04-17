"""
Activation-aware cross-layer SVD prototype.

Gates the factored-inference research direction: if rank ~30-50% of full
gives acceptable PPL, the C++/CUDA streaming scheme is viable.

Decomposition (per weight role, across all L transformer layers):
    stack W_i in W_stack [d_out, L * d_in]
    whiten each layer by its calibrated (X_i^T X_i)^(1/2)
    SVD -> U [d_out, r] (shared basis), V_i [r, d_in] (per-layer coeffs)
    reconstruct: W_i_hat = U @ diag(sigma) @ V_i @ S_i^{-1}

Sweeps rank and reports PPL on WikiText-2.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Weight roles we factor. Names match Qwen2/Llama-family HF modules.
ROLES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def find_target_linears(model):
    """Return {role: [(layer_idx, module), ...]} for every decomposable linear."""
    groups = defaultdict(list)
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        for role in ROLES:
            if name.endswith(role):
                # Extract layer index from names like "model.layers.12.self_attn.q_proj"
                parts = name.split(".")
                try:
                    idx = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    continue
                groups[role].append((idx, name, mod))
                break
    for role in groups:
        groups[role].sort(key=lambda t: t[0])
    return groups


@torch.no_grad()
def collect_activation_stats(model, tokenizer, texts, device, seq_len):
    """Run calibration texts and accumulate X^T X for each target linear.

    Accumulators live on CPU fp32 to avoid VRAM blowup on down_proj (d_in=hidden*mlp_ratio).
    """
    groups = find_target_linears(model)
    stats = {}  # {module_name: X^T X tensor on CPU}
    hooks = []

    def make_hook(name, d_in):
        stats[name] = torch.zeros(d_in, d_in, dtype=torch.float32)

        def hook(mod, inputs):
            x = inputs[0]
            flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            stats[name].add_((flat.T @ flat).cpu())

        return hook

    for role, entries in groups.items():
        for _, name, mod in entries:
            d_in = mod.in_features
            hooks.append(mod.register_forward_pre_hook(make_hook(name, d_in)))

    model.eval()
    pbar = tqdm(texts, desc="calibration")
    for text in pbar:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 8:
            continue
        model(input_ids)

    for h in hooks:
        h.remove()
    return stats, groups


def whiten_matrices(xtx, eps_ratio=1e-4):
    """Return (S, S_inv) where S @ S ≈ XTX + eps*I, both symmetric PSD."""
    xtx = xtx.to(torch.float64)
    n = xtx.shape[0]
    mean_diag = xtx.diag().mean().item()
    eps = eps_ratio * mean_diag + 1e-8
    reg = xtx + eps * torch.eye(n, dtype=torch.float64)
    # Symmetric eigendecomposition (PSD by construction)
    evals, evecs = torch.linalg.eigh(reg)
    evals = evals.clamp(min=eps)
    sqrt_evals = evals.sqrt()
    inv_sqrt_evals = 1.0 / sqrt_evals
    S = (evecs * sqrt_evals.unsqueeze(0)) @ evecs.T
    S_inv = (evecs * inv_sqrt_evals.unsqueeze(0)) @ evecs.T
    return S.to(torch.float32), S_inv.to(torch.float32)


def precompute_factors(weights, xtx_list, mode, activation_aware=True):
    """One-shot factorization. `mode` selects the decomposition geometry:
      - 'cross-layer-h': stack W_i horizontally; shared output basis U [d_out, r].
      - 'cross-layer-v': stack W_i vertically;   shared input  basis V [d_in, r].
      - 'per-matrix':    per-matrix SVD, no sharing.
    Reconstruction uses the numerically stable projection form in all cases.
    """
    L = len(weights)
    d_out, d_in = weights[0].shape
    assert all(w.shape == (d_out, d_in) for w in weights), "weight shapes must match"

    weights_fp32 = [w.to(torch.float32).contiguous() for w in weights]

    if activation_aware:
        S_list = [whiten_matrices(x)[0] for x in xtx_list]
        weighted = [w @ S for w, S in zip(weights_fp32, S_list)]
    else:
        weighted = weights_fp32

    f = {
        "mode": mode,
        "weights_fp32": weights_fp32,
        "d_out": d_out, "d_in": d_in, "L": L,
        "activation_aware": activation_aware,
        "dtype": weights[0].dtype,
    }

    if mode == "cross-layer-h":
        W_cat = torch.cat(weighted, dim=1)  # [d_out, L * d_in]
        U, _, _ = torch.linalg.svd(W_cat, full_matrices=False)
        f["U"] = U  # [d_out, min(d_out, L*d_in)]
    elif mode == "cross-layer-v":
        W_cat = torch.cat(weighted, dim=0)  # [L * d_out, d_in]
        _, _, Vt = torch.linalg.svd(W_cat, full_matrices=False)
        f["V"] = Vt.T  # [d_in, min(d_in, L*d_out)]
    elif mode == "per-matrix":
        Us = []
        Vts = []
        for w_weighted, w_orig in zip(weighted, weights_fp32):
            U, _, _ = torch.linalg.svd(w_weighted, full_matrices=False)
            Us.append(U)
        f["U_per_layer"] = Us  # each [d_out, min(d_out, d_in)]
    else:
        raise ValueError(f"unknown mode: {mode}")

    return f


def reconstruct_at_rank(factors, rank):
    """Slice cached factors to `rank` and rebuild per-layer weights via projection."""
    mode = factors["mode"]
    dtype = factors["dtype"]
    recons = []

    if mode == "cross-layer-h":
        r = min(rank, factors["U"].shape[1])
        U_r = factors["U"][:, :r]
        for W_i in factors["weights_fp32"]:
            A_i = U_r.T @ W_i
            recons.append((U_r @ A_i).to(dtype))
    elif mode == "cross-layer-v":
        r = min(rank, factors["V"].shape[1])
        V_r = factors["V"][:, :r]
        for W_i in factors["weights_fp32"]:
            # Project onto top-r input directions: W_i @ V_r @ V_r^T
            recons.append(((W_i @ V_r) @ V_r.T).to(dtype))
    elif mode == "per-matrix":
        r_effective = None
        for U_i, W_i in zip(factors["U_per_layer"], factors["weights_fp32"]):
            r = min(rank, U_i.shape[1])
            r_effective = r
            U_r = U_i[:, :r]
            A_i = U_r.T @ W_i
            recons.append((U_r @ A_i).to(dtype))
        r = r_effective
    else:
        raise ValueError(f"unknown mode: {mode}")

    return r, recons


def swap_weights(modules, new_weights):
    """Replace .weight on each module (in-place). Returns originals for restoration."""
    originals = []
    for mod, w in zip(modules, new_weights):
        originals.append(mod.weight.data.clone())
        mod.weight.data.copy_(w.to(mod.weight.device, mod.weight.dtype))
    return originals


def restore_weights(modules, originals):
    for mod, w in zip(modules, originals):
        mod.weight.data.copy_(w)


@torch.no_grad()
def evaluate_ppl(model, tokenizer, text, device, stride=2048, n_tokens=None):
    """Sliding-window PPL on concatenated text."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    if n_tokens is not None and input_ids.shape[1] > n_tokens:
        input_ids = input_ids[:, :n_tokens]
    max_len = getattr(model.config, "max_position_embeddings", 2048)
    window = min(stride, max_len)
    nll_sum = 0.0
    n = 0
    prev_end = 0
    for begin in range(0, input_ids.shape[1], window):
        end = min(begin + window, input_ids.shape[1])
        chunk = input_ids[:, begin:end]
        target = chunk.clone()
        # Predict from position 1 onward (standard causal LM PPL).
        target[:, : max(0, prev_end - begin)] = -100
        out = model(chunk, labels=target)
        valid = (target != -100).sum().item()
        if valid > 0:
            nll_sum += out.loss.item() * valid
            n += valid
        prev_end = end
        if end == input_ids.shape[1]:
            break
    return float(torch.exp(torch.tensor(nll_sum / max(n, 1))))


def load_calibration_texts(tokenizer, n, min_chars=400):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars:
            out.append(t)
            if len(out) >= n:
                break
    return out


def load_eval_text():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(r["text"] for r in ds if r["text"].strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--eval-tokens", type=int, default=8192)
    p.add_argument("--ranks", nargs="+", type=int,
                   default=[32, 64, 128, 192, 256, 384, 512, 768])
    p.add_argument("--mode", default="cross-layer-h",
                   choices=["cross-layer-h", "cross-layer-v", "per-matrix"])
    p.add_argument("--no-asvd", action="store_true",
                   help="disable activation-aware weighting (naive SVD baseline)")
    p.add_argument("--roles", nargs="+", default=None,
                   help="subset of ROLES to decompose (default: all)")
    p.add_argument("--out", default="results.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    roles_to_factor = args.roles or ROLES

    print(f"loading {args.model} in {args.dtype} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    calib = load_calibration_texts(tokenizer, args.calib_samples)
    print(f"calibration corpus: {len(calib)} samples")
    t0 = time.time()
    xtx_by_name, groups = collect_activation_stats(model, tokenizer, calib, device, args.calib_seq_len)
    print(f"calibration pass done in {time.time()-t0:.1f}s")

    eval_text = load_eval_text()

    # Baseline PPL
    print("evaluating baseline PPL")
    t0 = time.time()
    baseline_ppl = evaluate_ppl(model, tokenizer, eval_text, device, n_tokens=args.eval_tokens)
    print(f"baseline PPL = {baseline_ppl:.3f} ({time.time()-t0:.1f}s)")

    results = {
        "model": args.model,
        "mode": args.mode,
        "calib_samples": args.calib_samples,
        "eval_tokens": args.eval_tokens,
        "roles_factored": roles_to_factor,
        "activation_aware": not args.no_asvd,
        "baseline_ppl": baseline_ppl,
        "sweeps": [],
    }

    # One-shot factorization per role, cache, then slice cheaply per rank.
    print(f"\nprecomputing factors per role (mode={args.mode})")
    role_cache = {}
    for role in roles_to_factor:
        entries = groups.get(role, [])
        if not entries:
            continue
        _, names, modules = zip(*entries)
        weights = [m.weight.data.cpu() for m in modules]
        xtx_list = [xtx_by_name[n] for n in names]
        t0 = time.time()
        factors = precompute_factors(weights, xtx_list, args.mode,
                                     activation_aware=not args.no_asvd)
        dt = time.time() - t0
        role_cache[role] = {"factors": factors, "modules": modules, "names": names}
        d_out, d_in = factors["d_out"], factors["d_in"]
        if args.mode == "cross-layer-h":
            max_r = factors["U"].shape[1]
        elif args.mode == "cross-layer-v":
            max_r = factors["V"].shape[1]
        else:
            max_r = factors["U_per_layer"][0].shape[1]
        print(f"  {role}: {d_out}x{d_in} x{len(modules)}  max_rank={max_r}  ({dt:.1f}s)")

    for r in args.ranks:
        print(f"\n--- rank = {r} ---")
        role_summary = {}
        all_originals = []
        all_modules = []
        for role, entry in role_cache.items():
            factors = entry["factors"]
            modules = entry["modules"]
            effective_r, recons = reconstruct_at_rank(factors, r)
            # Diagnostic: reconstruction error and magnitude sanity
            rel_errs = []
            max_abs = 0.0
            nonfinite = False
            for W_orig_fp32, W_recon in zip(factors["weights_fp32"], recons):
                W_recon_fp32 = W_recon.to(torch.float32)
                if not torch.isfinite(W_recon_fp32).all():
                    nonfinite = True
                err = torch.linalg.norm(W_orig_fp32 - W_recon_fp32).item()
                nrm = torch.linalg.norm(W_orig_fp32).item()
                rel_errs.append(err / max(nrm, 1e-12))
                max_abs = max(max_abs, W_recon_fp32.abs().max().item())
            originals = swap_weights(modules, recons)
            all_originals.append(originals)
            all_modules.append(modules)
            role_summary[role] = {
                "shape": [factors["d_out"], factors["d_in"]],
                "n_layers": factors["L"],
                "rank_applied": effective_r,
                "rel_err_min": min(rel_errs),
                "rel_err_max": max(rel_errs),
                "recon_max_abs": max_abs,
                "nonfinite": nonfinite,
            }
            print(f"  {role}: r={effective_r}  rel_err {min(rel_errs):.3f}-{max(rel_errs):.3f}"
                  f"  max|W_r|={max_abs:.2f}  {'NONFINITE!' if nonfinite else ''}")

        t0 = time.time()
        ppl = evaluate_ppl(model, tokenizer, eval_text, device, n_tokens=args.eval_tokens)
        eval_dt = time.time() - t0
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"PPL = {ppl:.3f}  (delta {delta:+.2f}% vs baseline, eval {eval_dt:.1f}s)")

        results["sweeps"].append({
            "rank": r,
            "ppl": ppl,
            "delta_pct": delta,
            "roles": role_summary,
        })

        for modules, originals in zip(all_modules, all_originals):
            restore_weights(modules, originals)

        # Restoration sanity check: PPL after restore should match baseline
        restore_ppl = evaluate_ppl(model, tokenizer, eval_text, device, n_tokens=1024)
        # Re-eval on fewer tokens for speed; compare to a same-sized baseline on first iteration
        print(f"  [post-restore PPL on 1K tokens = {restore_ppl:.3f}]")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
