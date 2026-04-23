"""WikiText-2 loaders — calibration and evaluation.

Two callers:

- `calibration_samples(tokenizer, n=128, seq_len=2048, seed=0)` — deterministic
  calibration set, matching the paper's recipe (128 × 2048 tokens from
  WikiText-2 train).
- `eval_token_stream(tokenizer)` — full test split concatenated and tokenized
  once; `ppl.py` slides a fixed window over this.

Convention: we use `wikitext-2-raw-v1` (the raw text split). The joining
convention ("\\n\\n" vs joined-as-loaded) varies across quantization papers;
we follow the GPTQ/BiLLM lineage of concatenating with "\\n\\n" because that
is what most published PPL numbers use.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from datasets import load_dataset


DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


def _load_split(split: str) -> str:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    return "\n\n".join(t for t in ds["text"] if t)


def eval_token_stream(tokenizer) -> torch.Tensor:
    """Full WikiText-2 test split tokenized as a single 1-D LongTensor."""
    text = _load_split("test")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids


@dataclass
class CalibSet:
    input_ids: torch.Tensor  # (n, seq_len) LongTensor
    seq_len: int
    seed: int


def calibration_samples(
    tokenizer,
    n: int = 128,
    seq_len: int = 2048,
    seed: int = 0,
) -> CalibSet:
    """Deterministically sample `n` windows of `seq_len` tokens from the train split.

    Matches the convention used in GPTQ / BiLLM / HBLLM calibration: tokenize
    the whole train split once, then draw random contiguous windows.
    """
    text = _load_split("train")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total = ids.numel()
    if total < seq_len + 1:
        raise ValueError(
            f"train split has {total} tokens, need at least {seq_len + 1}"
        )

    rng = random.Random(seed)
    starts = [rng.randint(0, total - seq_len - 1) for _ in range(n)]
    windows = torch.stack([ids[s : s + seq_len] for s in starts], dim=0)
    return CalibSet(input_ids=windows, seq_len=seq_len, seed=seed)
