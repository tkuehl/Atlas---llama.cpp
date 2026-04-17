"""Throughput suite: tok/s, TTFT, peak VRAM for a backend.

Uses a fixed set of prompts at varying lengths to exercise both prefill and decode paths.
Reports median across trials to damp jitter.
"""

from dataclasses import dataclass, asdict, field
from statistics import median, mean
from typing import List

import torch


DEFAULT_PROMPTS = [
    # (label, prompt, max_new_tokens)
    ("short_qa", "What is the capital of France? Answer in one sentence.", 32),
    ("short_reasoning", "A train leaves Boston at 10am going 60 mph. Another leaves NYC at 11am going 80 mph in the opposite direction. Who is farther from Boston at 2pm?", 128),
    ("long_prefill", "Below is a passage. Summarize it in three bullet points.\n\n" + "The quick brown fox jumps over the lazy dog. " * 80, 128),
    ("code", "Write a Python function that computes the nth Fibonacci number using memoization.", 256),
    ("long_decode", "List 50 common English verbs, one per line.", 512),
]


@dataclass
class ThroughputTrial:
    label: str
    prompt_tokens: int
    decode_tokens: int
    ttft_ms: float
    decode_tok_per_s: float
    total_s: float


@dataclass
class ThroughputResult:
    trials: List[ThroughputTrial] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    summary: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "trials": [asdict(t) for t in self.trials],
            "peak_vram_mb": self.peak_vram_mb,
            "summary": self.summary,
        }


def run(backend, prompts=None, repeats: int = 3, warmup: int = 1) -> ThroughputResult:
    """Exercise backend across prompts and return aggregated throughput result."""
    prompts = prompts or DEFAULT_PROMPTS

    # Warmup to avoid cold-cache CUDA kernels skewing first measurement
    for _ in range(warmup):
        backend.generate(prompts[0][1], max_new_tokens=8)

    res = ThroughputResult()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for label, prompt, mn in prompts:
        trials = []
        prompt_tokens = len(backend.tokenize(prompt))
        for _ in range(repeats):
            g = backend.generate(prompt, max_new_tokens=mn)
            decode_s = max(g["total_s"] - g["ttft_s"], 1e-6)
            tok_per_s = g["n_tokens"] / decode_s if g["n_tokens"] > 0 else 0.0
            trials.append(ThroughputTrial(
                label=label,
                prompt_tokens=prompt_tokens,
                decode_tokens=g["n_tokens"],
                ttft_ms=g["ttft_s"] * 1000,
                decode_tok_per_s=tok_per_s,
                total_s=g["total_s"],
            ))
        # Keep median across repeats
        trials.sort(key=lambda t: t.decode_tok_per_s)
        res.trials.append(trials[len(trials) // 2])

    if torch.cuda.is_available():
        res.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    res.summary = {
        "median_decode_tok_per_s": median(t.decode_tok_per_s for t in res.trials),
        "mean_ttft_ms": mean(t.ttft_ms for t in res.trials),
    }
    return res
