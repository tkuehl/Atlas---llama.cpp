"""K-FAC diagonal preconditioner collection (NanoQuant Phase 1 of Algorithm 1).

Per-linear, we need

    D_in[j]  ≈ sqrt(E[x_j²])              (input-activation RMS)
    D_out[i] ≈ sqrt(E[(∂L/∂y_i)²])         (output-gradient RMS)

so that L(Ŵ) ≈ ‖D_out (W − Ŵ) D_in‖_F² approximates the diagonal K-FAC
Hessian of the task loss (paper Eq. 2).

Raw estimates are then put through **ROBUSTDIAG**: the paper clips each
entry to a running τ_max threshold derived from a percentile of the
per-linear distribution (Appendix B.2, cumulative maximum update rule).
Finally **Ledoit-Wolf shrinkage** (paper Eq. 3) pulls entries toward the
linear's mean:

    [D̃]_ii ← (1 − γ)[D]_ii + γ · mean(D)

γ = 0.2 is paper's recommendation for Qwen/Llama.

For the task loss we use next-token prediction cross-entropy on the same
calibration sequences used for the activation cache. This gives a single
forward+backward per calibration sample.

Storage: Qwen3-4B has ~252 linears under `model.model.layers.*`. All
D_in + D_out combined is a few MB of fp32 — dominated by the largest
linear (9728 entries). Saved to disk so the (5-minute-ish) backward pass
isn't re-run on every Phase 2 invocation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import calibration_samples


PRECOND_FILE = "preconditioners.pt"
PRECOND_META = "preconditioners_meta.json"


def _linear_paths(model) -> list[str]:
    """Every nn.Linear under model.model.layers — the set we quantize."""
    paths: list[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.startswith("model.layers"):
            paths.append(name)
    return paths


def _resolve(model, dotted: str) -> nn.Module:
    m = model
    for part in dotted.split("."):
        m = getattr(m, part) if not part.isdigit() else m[int(part)]
    return m


@dataclass
class PreconditionerMeta:
    n_samples: int
    seq_len: int
    seed: int
    gamma: float
    percentile: float
    model_hf_id: str


def collect_preconditioners(
    model,
    tokenizer,
    cache_dir: str | Path,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 0,
    gamma: float = 0.2,
    percentile: float = 0.99,
    chunk_size: int = 1,
    device: str | torch.device = "cuda",
    model_hf_id: str = "",
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Forward + backward through `n_samples` calibration sequences, hook every
    linear under `model.model.layers`, accumulate x² and (∂L/∂y)² per
    input/output channel, then apply ROBUSTDIAG percentile clipping and
    Ledoit-Wolf shrinkage.

    Returns {module_path: (D_in, D_out)} with both vectors in fp32 on CPU.
    Also persists to `cache_dir / preconditioners.pt`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    calib = calibration_samples(tokenizer, n=n_samples, seq_len=seq_len, seed=seed)
    input_ids = calib.input_ids

    paths = _linear_paths(model)
    # Pre-allocate fp32 GPU-resident accumulators per linear. One .cpu()
    # happens at the end; hook fires stay on-device, so the 252-linear x
    # per-sample hot path is pure compute.
    stats_in_acc: dict[str, torch.Tensor] = {}
    stats_in_count: dict[str, int] = {}
    stats_out_acc: dict[str, torch.Tensor] = {}
    stats_out_count: dict[str, int] = {}

    for path in paths:
        mod = _resolve(model, path)
        stats_in_acc[path] = torch.zeros(
            mod.in_features, device=device, dtype=torch.float32
        )
        stats_in_count[path] = 0
        stats_out_acc[path] = torch.zeros(
            mod.out_features, device=device, dtype=torch.float32
        )
        stats_out_count[path] = 0

    def make_fwd(path: str):
        def hook(module, inputs):
            x = inputs[0]
            x_flat = x.detach().reshape(-1, x.shape[-1])
            # Accumulate in fp32 on the same device — no .cpu() in the hot path.
            stats_in_acc[path].add_(x_flat.to(torch.float32).pow(2).sum(dim=0))
            stats_in_count[path] += x_flat.shape[0]
        return hook

    def make_bwd(path: str):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return
            g_flat = g.detach().reshape(-1, g.shape[-1])
            stats_out_acc[path].add_(g_flat.to(torch.float32).pow(2).sum(dim=0))
            stats_out_count[path] += g_flat.shape[0]
        return hook

    handles = []
    for path in paths:
        mod = _resolve(model, path)
        handles.append(mod.register_forward_pre_hook(make_fwd(path)))
        handles.append(mod.register_full_backward_hook(make_bwd(path)))

    # bf16/fp16 model params need requires_grad for backward hooks to fire.
    # We don't update anything — just propagate gradients so the hooks run.
    for p in model.parameters():
        p.requires_grad_(False)
    # Gradient checkpointing makes full-model backward fit in 16 GB.
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    model.eval()  # Qwen3 has no dropout — eval mode is fine for autograd.

    # We still need grads flowing through activations to trigger our hooks.
    # torch.enable_grad() around the forward is sufficient even with
    # requires_grad=False on params, since hooks fire on activation tensors.
    try:
        for start in tqdm(
            range(0, n_samples, chunk_size),
            desc="precond",
            total=(n_samples + chunk_size - 1) // chunk_size,
        ):
            ids = input_ids[start : start + chunk_size].to(device)
            with torch.enable_grad():
                # Enable grads on the embedding output so backward hooks fire.
                # Easiest: set inputs_embeds requires_grad via a small trick —
                # ask the model to compute loss and do `loss.backward(inputs=[])`.
                # Simpler alternative: briefly flip the embedding to require grad.
                emb = model.get_input_embeddings()
                emb.weight.requires_grad_(True)
                try:
                    out = model(ids, labels=ids, use_cache=False)
                    out.loss.backward()
                finally:
                    emb.weight.requires_grad_(False)
                    model.zero_grad(set_to_none=True)
    finally:
        for h in handles:
            h.remove()
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    # Aggregate → D_in / D_out as RMS (sqrt of mean-square).
    precond: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for path in paths:
        n_x = stats_in_count[path]
        n_g = stats_out_count[path]
        if n_x == 0 or n_g == 0:
            continue
        d_in = (stats_in_acc[path] / n_x).sqrt().cpu()
        d_out = (stats_out_acc[path] / n_g).sqrt().cpu()

        # ROBUSTDIAG: clip each entry to the `percentile`-th quantile of
        # this linear's distribution. Approximates the paper's cumulative-
        # max threshold with a single-shot percentile — cheaper and matches
        # the bound property (Appendix B.2 Lemma 1) if the final τ_max
        # equals the final-iteration percentile.
        tau_in = torch.quantile(d_in, percentile)
        tau_out = torch.quantile(d_out, percentile)
        d_in = d_in.clamp(max=tau_in)
        d_out = d_out.clamp(max=tau_out)

        # Ledoit-Wolf shrinkage toward the linear-wise mean.
        d_in = (1.0 - gamma) * d_in + gamma * d_in.mean()
        d_out = (1.0 - gamma) * d_out + gamma * d_out.mean()

        precond[path] = (d_in, d_out)

    torch.save(precond, cache_dir / PRECOND_FILE)
    with open(cache_dir / PRECOND_META, "w") as f:
        json.dump(
            {
                "n_samples": n_samples,
                "seq_len": seq_len,
                "seed": seed,
                "gamma": gamma,
                "percentile": percentile,
                "model_hf_id": model_hf_id,
                "n_linears": len(precond),
            },
            f,
            indent=2,
        )
    return precond


def load_preconditioners(
    cache_dir: str | Path,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cache_dir = Path(cache_dir)
    return torch.load(
        cache_dir / PRECOND_FILE, map_location="cpu", weights_only=True
    )
