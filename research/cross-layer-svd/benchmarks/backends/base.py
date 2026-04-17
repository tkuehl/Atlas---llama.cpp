"""Backend abstraction: any benchmark suite runs against these two primitives."""

from abc import ABC, abstractmethod
from typing import List


class Backend(ABC):
    """Minimal interface a backend must expose to be benchmarkable."""

    name: str

    @abstractmethod
    def logits(self, token_ids: List[int]) -> "torch.Tensor":
        """Return logits [seq_len, vocab_size] for the given input tokens."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int,
                 temperature: float = 0.0) -> dict:
        """Generate text. Returns dict with keys: text, ttft_s, total_s, n_tokens."""

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize to ids."""

    @abstractmethod
    def close(self) -> None:
        """Free resources."""
