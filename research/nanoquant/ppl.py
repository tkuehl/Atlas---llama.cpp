"""Sliding-window perplexity on a pre-tokenized stream.

Matches the GPTQ/BiLLM convention: non-overlapping windows of `seq_len`
tokens, per-token NLL averaged across the whole stream, PPL = exp(mean NLL).
A trailing partial window is dropped (same as the reference implementations).

Returns enough metadata for results.json (window count, token count, mean NLL)
so a run is fully characterized.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class PPLResult:
    ppl: float
    nll_mean: float
    num_tokens: int
    num_windows: int
    seq_len: int
    stride: int


@torch.inference_mode()
def sliding_window_ppl(
    model,
    token_stream: torch.Tensor,
    seq_len: int = 2048,
    stride: int | None = None,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
) -> PPLResult:
    """Non-overlapping (stride == seq_len) sliding-window PPL by default."""
    if stride is None:
        stride = seq_len

    ids = token_stream.to(device)
    total = ids.numel()
    n_windows = (total - 1) // stride  # drop trailing partial
    if n_windows <= 0:
        raise ValueError(f"stream has {total} tokens, need >= {seq_len + 1}")

    nll_sum = 0.0
    tok_count = 0

    model.eval()
    it = range(n_windows)
    if show_progress:
        it = tqdm(it, desc="ppl", leave=False)

    for i in it:
        start = i * stride
        end = start + seq_len
        if end + 1 > total:
            break
        window = ids[start : end + 1].unsqueeze(0)  # (1, seq_len+1)
        inputs = window[:, :-1]
        targets = window[:, 1:]
        logits = model(inputs).logits  # (1, seq_len, V)
        # sum NLL over the window, then aggregate — numerically stabler than
        # averaging per-window means when windows are equal length anyway.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        tok_count += targets.numel()

    nll_mean = nll_sum / tok_count
    return PPLResult(
        ppl=float(torch.tensor(nll_mean).exp().item()),
        nll_mean=nll_mean,
        num_tokens=tok_count,
        num_windows=n_windows,
        seq_len=seq_len,
        stride=stride,
    )
