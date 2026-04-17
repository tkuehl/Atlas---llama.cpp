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
    """Run calibration texts and accumulate X^T X for each target linear."""
    groups = find_target_linears(model)
    stats = {}
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
    for text in tqdm(texts, desc="calibration"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 8:
            continue
        model(input_ids)

    for h in hooks:
        h.remove()
    return stats, groups


def collect_activation_and_grad_stats(model, tokenizer, texts, device, seq_len):
    """Run forward+backward on calibration texts and accumulate:
       - X^T X (input covariance) per target linear
       - G^T G (output gradient covariance) per target linear
    Needed for balanced truncation weighting.
    """
    groups = find_target_linears(model)
    xtx = {}
    ggt = {}
    hooks = []

    # Freeze everything, then re-enable grads only on target weights.
    # The backward hook needs the target's output to be in the grad graph,
    # which requires at least one parameter in or upstream of the module to have grad.
    for p in model.parameters():
        p.requires_grad_(False)
    for role, entries in groups.items():
        for _, _, mod in entries:
            mod.weight.requires_grad_(True)

    def make_fwd_hook(name, d_in):
        xtx[name] = torch.zeros(d_in, d_in, dtype=torch.float32)

        def hook(mod, inputs):
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            xtx[name].add_((flat.T @ flat).cpu())

        return hook

    def make_bwd_hook(name, d_out):
        ggt[name] = torch.zeros(d_out, d_out, dtype=torch.float32)

        def hook(mod, grad_input, grad_output):
            g = grad_output[0].detach()
            flat = g.reshape(-1, g.shape[-1]).to(torch.float32)
            ggt[name].add_((flat.T @ flat).cpu())

        return hook

    for role, entries in groups.items():
        for _, name, mod in entries:
            hooks.append(mod.register_forward_pre_hook(make_fwd_hook(name, mod.in_features)))
            hooks.append(mod.register_full_backward_hook(make_bwd_hook(name, mod.out_features)))

    model.eval()
    for text in tqdm(texts, desc="calibration+bwd"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 8:
            continue
        out = model(input_ids, labels=input_ids)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    for h in hooks:
        h.remove()
    for p in model.parameters():
        p.requires_grad_(False)
    return xtx, ggt, groups


def _stable_svd_left(W, device):
    """Return left singular vectors of W. Uses GPU only for smaller matrices;
    falls back to CPU for big MLP matrices where cuSolver hits workspace issues
    or runs out of VRAM. Threshold is empirical."""
    rows, cols = W.shape
    if device == "cuda" and rows * cols <= 5_000_000:
        try:
            w_gpu = W.to(device)
            U, _, _ = torch.linalg.svd(w_gpu, full_matrices=False, driver="gesvd")
            return U
        except RuntimeError as e:
            if "cusolver" not in str(e).lower() and "cuda" not in str(e).lower() \
               and "memory" not in str(e).lower():
                raise
            # fall through to CPU
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    return U


def whiten_matrices(xtx, eps_ratio=1e-4, device=None):
    """Return (S, S_inv) where S @ S ≈ XTX + eps*I, both symmetric PSD.

    Tries GPU eigh for speed; falls back to CPU on cuSolver failures that we've
    seen intermittently on Blackwell + CUDA 12.8 nightly.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n = xtx.shape[0]
    mean_diag = xtx.to(torch.float64).diag().mean().item()
    eps = eps_ratio * mean_diag + 1e-8

    def _compute(dev):
        xtx64 = xtx.to(device=dev, dtype=torch.float64)
        reg = xtx64 + eps * torch.eye(n, device=dev, dtype=torch.float64)
        evals, evecs = torch.linalg.eigh(reg)
        evals = evals.clamp(min=eps)
        S = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T
        S_inv = (evecs * (1.0 / evals.sqrt()).unsqueeze(0)) @ evecs.T
        return S, S_inv

    if device == "cuda":
        try:
            S, S_inv = _compute("cuda")
            return S.cpu().to(torch.float32), S_inv.cpu().to(torch.float32)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cusolver" not in msg and "cuda" not in msg and "memory" not in msg:
                raise
            # fall through
    S, S_inv = _compute("cpu")
    return S.to(torch.float32), S_inv.to(torch.float32)


def precompute_factors(weights, xtx_list, mode, weighting="asvd", ggt_list=None):
    """One-shot factorization. `mode` selects the decomposition geometry:
      - 'cross-layer-h': stack W_i horizontally; shared output basis.
      - 'cross-layer-v': stack W_i vertically;   shared input  basis.
      - 'per-matrix':    per-matrix SVD, no sharing.
    `weighting` selects the inner-product the SVD is taken in:
      - 'none':     plain Frobenius (no weighting)
      - 'asvd':     input-activation weighted
      - 'balanced': input AND output-gradient weighted (per-matrix only)
    """
    L = len(weights)
    d_out, d_in = weights[0].shape
    assert all(w.shape == (d_out, d_in) for w in weights), "weight shapes must match"

    weights_fp32 = [w.to(torch.float32).contiguous() for w in weights]

    f = {
        "mode": mode,
        "weighting": weighting,
        "weights_fp32": weights_fp32,
        "d_out": d_out, "d_in": d_in, "L": L,
        "dtype": weights[0].dtype,
    }

    if mode == "per-matrix" and weighting == "balanced":
        assert ggt_list is not None, "balanced weighting requires ggt_list"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        per_layer = []

        def _gpu_whiten(M, eps_ratio=1e-5):
            M64 = M.to(device=device, dtype=torch.float64)
            n = M64.shape[0]
            eps = eps_ratio * M64.diag().mean().item() + 1e-8
            reg = M64 + eps * torch.eye(n, device=device, dtype=torch.float64)
            evals, evecs = torch.linalg.eigh(reg)
            evals = evals.clamp(min=eps)
            S = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T
            S_inv = (evecs * (1.0 / evals.sqrt()).unsqueeze(0)) @ evecs.T
            return S, S_inv

        for W, XTX, GGT in zip(weights_fp32, xtx_list, ggt_list):
            S_in, S_in_inv = _gpu_whiten(XTX)
            S_out, S_out_inv = _gpu_whiten(GGT)
            W_g = W.to(device=device, dtype=torch.float64)
            M = S_out @ W_g @ S_in
            U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
            # Move everything back to CPU fp64 for storage; GPU VRAM stays bounded.
            per_layer.append({
                "U": U.cpu(), "sigma": sigma.cpu(), "Vt": Vt.cpu(),
                "S_in_inv": S_in_inv.cpu(),
                "S_out_inv": S_out_inv.cpu(),
            })
            del S_in, S_in_inv, S_out, S_out_inv, W_g, M, U, sigma, Vt
            if device == "cuda":
                torch.cuda.empty_cache()
        f["per_layer"] = per_layer
        return f

    # Non-balanced paths: optionally apply activation weighting, then SVD.
    # SVD on GPU fp32 for the big MLP matrices on larger models.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if weighting == "asvd":
        S_list = [whiten_matrices(x)[0] for x in xtx_list]
        weighted = [w @ S for w, S in zip(weights_fp32, S_list)]
    elif weighting == "none":
        weighted = weights_fp32
    else:
        raise ValueError(f"weighting={weighting} not supported for mode={mode}")

    if mode == "cross-layer-h":
        W_cat = torch.cat(weighted, dim=1).to(device)
        U, _, _ = torch.linalg.svd(W_cat, full_matrices=False)
        f["U"] = U.cpu()
        del W_cat, U
        if device == "cuda":
            torch.cuda.empty_cache()
    elif mode == "cross-layer-v":
        W_cat = torch.cat(weighted, dim=0).to(device)
        _, _, Vt = torch.linalg.svd(W_cat, full_matrices=False)
        f["V"] = Vt.T.cpu()
        del W_cat, Vt
        if device == "cuda":
            torch.cuda.empty_cache()
    elif mode == "per-matrix":
        Us = []
        for w_weighted in weighted:
            U = _stable_svd_left(w_weighted, device)
            Us.append(U.cpu())
            del U
            if device == "cuda":
                torch.cuda.empty_cache()
        f["U_per_layer"] = Us
    else:
        raise ValueError(f"unknown mode: {mode}")

    return f


def reconstruct_at_rank(factors, rank):
    """Slice cached factors to `rank` and rebuild per-layer weights."""
    mode = factors["mode"]
    weighting = factors.get("weighting", "asvd")
    dtype = factors["dtype"]
    recons = []

    if mode == "per-matrix" and weighting == "balanced":
        r_effective = None
        for layer_f in factors["per_layer"]:
            U = layer_f["U"]; sigma = layer_f["sigma"]; Vt = layer_f["Vt"]
            S_in_inv = layer_f["S_in_inv"]; S_out_inv = layer_f["S_out_inv"]
            r = min(rank, sigma.numel())
            r_effective = r
            M_r = (U[:, :r] * sigma[:r].unsqueeze(0)) @ Vt[:r, :]
            W_r = S_out_inv @ M_r @ S_in_inv
            recons.append(W_r.to(dtype))
        return r_effective, recons

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
    p.add_argument("--weighting", default="asvd",
                   choices=["none", "asvd", "balanced"],
                   help="SVD inner-product weighting. 'balanced' is per-matrix only "
                        "and requires a backward-pass calibration (slower).")
    p.add_argument("--no-asvd", action="store_true",
                   help="deprecated alias for --weighting none")
    p.add_argument("--roles", nargs="+", default=None,
                   help="subset of ROLES to decompose (default: all)")
    p.add_argument("--out", default="results.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    roles_to_factor = args.roles or ROLES

    # Legacy flag compatibility
    if args.no_asvd:
        args.weighting = "none"

    if args.weighting == "balanced" and args.mode != "per-matrix":
        p.error("--weighting balanced is only supported with --mode per-matrix")

    # Backward pass needs fp32 weights for stable gradients on small calibration sets.
    load_dtype = torch.float32 if args.weighting == "balanced" else dtype
    print(f"loading {args.model} in {load_dtype} on {device} "
          f"(mode={args.mode}, weighting={args.weighting})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=load_dtype).to(device)
    model.eval()

    calib = load_calibration_texts(tokenizer, args.calib_samples)
    print(f"calibration corpus: {len(calib)} samples")
    t0 = time.time()
    ggt_by_name = None
    if args.weighting == "balanced":
        xtx_by_name, ggt_by_name, groups = collect_activation_and_grad_stats(
            model, tokenizer, calib, device, args.calib_seq_len)
    else:
        xtx_by_name, groups = collect_activation_stats(
            model, tokenizer, calib, device, args.calib_seq_len)
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
        "weighting": args.weighting,
        "calib_samples": args.calib_samples,
        "eval_tokens": args.eval_tokens,
        "roles_factored": roles_to_factor,
        "baseline_ppl": baseline_ppl,
        "sweeps": [],
    }

    # One-shot factorization per role, cache, then slice cheaply per rank.
    # Move the model to CPU during SVD so GPU VRAM is free for cuSolver workspace
    # (matters on larger models where model + SVD workspace > VRAM).
    print(f"\nprecomputing factors per role (mode={args.mode} weighting={args.weighting})")
    if device == "cuda":
        model.cpu()
        torch.cuda.empty_cache()
    role_cache = {}
    for role in roles_to_factor:
        entries = groups.get(role, [])
        if not entries:
            continue
        _, names, modules = zip(*entries)
        weights = [m.weight.data.cpu() for m in modules]
        xtx_list = [xtx_by_name[n] for n in names]
        ggt_list = [ggt_by_name[n] for n in names] if ggt_by_name else None
        t0 = time.time()
        factors = precompute_factors(weights, xtx_list, args.mode,
                                     weighting=args.weighting,
                                     ggt_list=ggt_list)
        dt = time.time() - t0
        role_cache[role] = {"factors": factors, "modules": modules, "names": names}
        d_out, d_in = factors["d_out"], factors["d_in"]
        if args.mode == "cross-layer-h":
            max_r = factors["U"].shape[1]
        elif args.mode == "cross-layer-v":
            max_r = factors["V"].shape[1]
        elif args.mode == "per-matrix" and args.weighting == "balanced":
            max_r = factors["per_layer"][0]["sigma"].numel()
        else:
            max_r = factors["U_per_layer"][0].shape[1]
        print(f"  {role}: {d_out}x{d_in} x{len(modules)}  max_rank={max_r}  ({dt:.1f}s)")

    # Move the model back to GPU for the PPL eval loop.
    if device == "cuda":
        model.to(device)

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
