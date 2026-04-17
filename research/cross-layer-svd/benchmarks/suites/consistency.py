"""Consistency metrics: cheap, dense signal for picking rank budgets.

Given a baseline and a candidate backend (same tokenizer, same model family),
we measure how close the candidate's output distribution is to the baseline on
held-out text. These metrics run in seconds on 4K tokens and correlate with
downstream task degradation.

Metrics:
    kl_div_bits:     mean KL(P_baseline || P_candidate) in bits per token
    top1_agree:      fraction of positions where top-1 predicted token matches
    top5_agree:      same for top-5 set
    entropy_delta:   mean H(P_candidate) - H(P_baseline), shows confidence drift
"""

from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class ConsistencyResult:
    n_tokens: int
    kl_div_bits: float
    top1_agree: float
    top5_agree: float
    entropy_delta: float

    def to_dict(self):
        return asdict(self)


def _held_out_text(n_chars_min=400, n_rows=20):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    chunks = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= n_chars_min:
            chunks.append(t)
            if len(chunks) >= n_rows:
                break
    return "\n\n".join(chunks)


@torch.no_grad()
def run(baseline, candidate, n_tokens: int = 4096, window: int = 1024,
        text: str = None) -> ConsistencyResult:
    """Compute consistency of candidate vs baseline on held-out text.

    Assumes both backends use the same tokenizer (same model family, different weights).
    """
    text = text or _held_out_text()
    ids = baseline.tokenize(text)[:n_tokens]
    if len(ids) < 32:
        raise ValueError(f"need at least 32 tokens, got {len(ids)}")

    kl_accum = 0.0
    top1_hits = 0
    top5_hits = 0
    entropy_delta_accum = 0.0
    scored = 0

    for begin in tqdm(range(0, len(ids), window), desc="consistency"):
        chunk = ids[begin:begin + window]
        if len(chunk) < 8:
            continue
        b_logits = baseline.logits(chunk).float()   # [seq, vocab]
        c_logits = candidate.logits(chunk).float()

        b_logp = F.log_softmax(b_logits, dim=-1)
        c_logp = F.log_softmax(c_logits, dim=-1)
        b_p = b_logp.exp()
        c_p = c_logp.exp()

        # KL(P_baseline || P_candidate) in bits
        kl = (b_p * (b_logp - c_logp)).sum(dim=-1) / torch.log(torch.tensor(2.0))
        kl_accum += kl.sum().item()

        # Top-k agreement
        b_top1 = b_logits.argmax(dim=-1)
        c_top1 = c_logits.argmax(dim=-1)
        top1_hits += (b_top1 == c_top1).sum().item()

        b_top5 = b_logits.topk(5, dim=-1).indices
        c_top5 = c_logits.topk(5, dim=-1).indices
        # fraction of positions where top-1 of baseline is in candidate's top-5
        top5_hits += (b_top1.unsqueeze(-1) == c_top5).any(dim=-1).sum().item()

        # Entropy delta (candidate - baseline)
        b_ent = -(b_p * b_logp).sum(dim=-1)
        c_ent = -(c_p * c_logp).sum(dim=-1)
        entropy_delta_accum += (c_ent - b_ent).sum().item()

        scored += len(chunk)

    return ConsistencyResult(
        n_tokens=scored,
        kl_div_bits=kl_accum / max(scored, 1),
        top1_agree=top1_hits / max(scored, 1),
        top5_agree=top5_hits / max(scored, 1),
        entropy_delta=entropy_delta_accum / max(scored, 1),
    )
