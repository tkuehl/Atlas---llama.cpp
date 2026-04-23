"""Teacher activation cache for block-wise PTQ.

Runs the FP model once on calibration data, saves the hidden states at each
transformer-block boundary to disk. Training block `b` then loads

    X_b = boundary_b         (input  to block b)
    Z_b = boundary_{b+1}     (output of block b)

and refines the student block in isolation, pure-teacher-forcing.

Also captures the auxiliary kwargs (rotary `position_embeddings`, attention
mask, etc.) the model passes to a decoder layer, so block-wise training can
reconstruct the forward without threading them through manually.

Layout:
    cache_dir/
        meta.json
        aux_kwargs.pt          # dict of tensors shared across all samples
        boundary_000.pt        # (n_samples, seq_len, d_model) fp16
        boundary_001.pt
        ...
        boundary_N.pt          # N = num_hidden_layers

At seq_len=2048, d_model=2560, FP16: ~335 MB per boundary at n_samples=32,
~1.3 GB at n_samples=128. A 36-block model at 128 samples needs ~50 GB disk;
keep n_samples modest for Phase 1 (32–64) and scale up once the pipeline
is validated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from data import calibration_samples


META_FILE = "meta.json"
AUX_FILE = "aux_kwargs.pt"


@dataclass
class Cache:
    dir: Path
    n_samples: int
    seq_len: int
    d_model: int
    n_blocks: int

    @classmethod
    def load(cls, cache_dir: str | Path) -> "Cache":
        cache_dir = Path(cache_dir)
        with open(cache_dir / META_FILE, "r") as f:
            meta = json.load(f)
        return cls(dir=cache_dir, **meta)

    def boundary_path(self, b: int) -> Path:
        return self.dir / f"boundary_{b:03d}.pt"

    def load_boundary(self, b: int) -> torch.Tensor:
        return torch.load(self.boundary_path(b), map_location="cpu", weights_only=True)

    def load_aux_kwargs(self, device: str | torch.device) -> dict:
        kw = torch.load(self.dir / AUX_FILE, map_location="cpu", weights_only=False)
        out = {}
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in v):
                out[k] = type(v)(t.to(device) for t in v)
            else:
                out[k] = v
        return out


def _capture_aux_kwargs(model, sample_ids: torch.Tensor, device) -> dict:
    """Forward one calibration sample; hook layer 0 to snapshot its input kwargs."""
    captured: dict = {}

    def pre_hook(mod, args, kwargs):
        if "kwargs" in captured:
            return
        snap = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                snap[k] = v.detach().cpu()
            elif isinstance(v, (tuple, list)) and all(
                isinstance(t, torch.Tensor) for t in v
            ):
                snap[k] = type(v)(t.detach().cpu() for t in v)
            # skip scalars/ints that are per-forward state (past_key_values, etc.)
        captured["kwargs"] = snap

    h = model.model.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            model(sample_ids.to(device))
    finally:
        h.remove()
    return captured.get("kwargs", {})


def build_cache(
    model,
    tokenizer,
    cache_dir: str | Path,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 0,
    chunk_size: int = 2,
    device: str | torch.device = "cuda",
) -> Cache:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    calib = calibration_samples(tokenizer, n=n_samples, seq_len=seq_len, seed=seed)
    input_ids = calib.input_ids  # (n_samples, seq_len)

    # Snapshot aux kwargs from a single-sample forward so their shapes are
    # known (attention_mask, position_embeddings) before we start accumulating.
    aux_kwargs = _capture_aux_kwargs(model, input_ids[:1], device)
    torch.save(aux_kwargs, cache_dir / AUX_FILE)

    n_blocks = len(model.model.layers)
    d_model = model.config.hidden_size
    boundary_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_blocks + 1)]

    model.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, n_samples, chunk_size),
            desc="cache",
            total=(n_samples + chunk_size - 1) // chunk_size,
        ):
            ids = input_ids[start : start + chunk_size].to(device)
            out = model(ids, output_hidden_states=True, use_cache=False)
            for b, hs in enumerate(out.hidden_states):
                boundary_chunks[b].append(hs.to(torch.float16).cpu())

    for b in range(n_blocks + 1):
        tensor = torch.cat(boundary_chunks[b], dim=0)
        torch.save(tensor, cache_dir / f"boundary_{b:03d}.pt")
        boundary_chunks[b] = []  # free

    meta = {
        "n_samples": n_samples,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_blocks": n_blocks,
    }
    with open(cache_dir / META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    return Cache(
        dir=cache_dir,
        n_samples=n_samples,
        seq_len=seq_len,
        d_model=d_model,
        n_blocks=n_blocks,
    )
