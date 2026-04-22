"""Offline teacher cache for LittleBit QAT distillation.

Sprint 3 infrastructure per
[scale_to_30b_architecture.md §2](scale_to_30b_architecture.md).

The teacher's contribution to training (top-k logits for KL and
hidden states for MSE) is static given a fixed corpus + fixed
sampling trajectory. We extract it once, store it on disk, and
mmap it at training time — eliminating the teacher from GPU memory
and from the per-step forward cost.

Layout on disk:

  <cache-dir>/
    metadata.json            - config + hash for validation
    topk_values.bin          - (n_microsteps, B, T, k) fp16 flat
    topk_indices.bin         - (n_microsteps, B, T, k) int32 flat
    hidden/
      layer_00.bin           - (n_microsteps, B, T, H) uint16 (bf16 bits)
      layer_01.bin
      ...
      layer_(N-1).bin        - post-final-norm tensor (matches HF
                                output_hidden_states[-1])

Access pattern: numpy.memmap for zero-copy reads.  All files are
flat binary; the metadata carries the shape and dtype info.

Cache is strictly positional — extraction and training must use the
same seed, seq_len, batch_size, grad_accum, c4_samples, and
tokenizer.  We encode those in metadata and validate on load.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


# -------------------------------------------------------------------
# Metadata
# -------------------------------------------------------------------

@dataclass
class CacheMetadata:
    """Schema for metadata.json.  Drives both extract-time allocation
    and train-time validation."""

    teacher_model: str
    tokenizer_model: str
    vocab_size: int
    hidden_size: int
    n_layers: int                    # hidden-state layers cached (N)
    k: int                           # top-k logit count
    seq_len: int
    batch_size: int
    grad_accum_steps: int            # for metadata completeness
    n_microsteps: int                # total cached micro-steps
    opt_steps: int                   # = n_microsteps // grad_accum
    c4_samples: int
    seed: int
    # Which layers' hidden states are cached.  If this lists [0..N-1]
    # it means every decoder layer's output (first N-1) plus the
    # post-final-norm tensor (last one), matching HF's
    # output_hidden_states[1:] after our hook-fix.
    hidden_layer_indices: list[int]
    # Hash over the token stream's first 1 MB, guards against
    # tokenizer / corpus drift between extract and train.
    corpus_hash: str
    # Teacher's PPL on wikitext-2 test — cached so the training
    # script can display it without having to load the teacher.
    # Optional; None if the extraction script skipped the PPL step.
    teacher_ppl: float | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CacheMetadata":
        return cls(**json.loads(s))


# -------------------------------------------------------------------
# Writer (used by littlebit_teacher_extract.py)
# -------------------------------------------------------------------

class TeacherCacheWriter:
    """Extract-time writer.

    Pre-allocates memmap files sized for the full run, then fills
    them chunk by chunk.  Call `write_microstep(idx, ...)` once per
    teacher forward.
    """

    def __init__(self, cache_dir: Path, meta: CacheMetadata):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "hidden").mkdir(exist_ok=True)
        self.meta = meta

        # Shape: (N, B, T, k) for top-k; (N, B, T, H) for hidden.
        shape_topk = (meta.n_microsteps, meta.batch_size, meta.seq_len, meta.k)
        shape_hidden = (meta.n_microsteps, meta.batch_size,
                        meta.seq_len, meta.hidden_size)

        self.topk_values = np.memmap(
            self.cache_dir / "topk_values.bin",
            dtype=np.float16, mode="w+", shape=shape_topk,
        )
        self.topk_indices = np.memmap(
            self.cache_dir / "topk_indices.bin",
            dtype=np.int32, mode="w+", shape=shape_topk,
        )
        self.hidden_files = []
        for li in meta.hidden_layer_indices:
            m = np.memmap(
                self.cache_dir / "hidden" / f"layer_{li:02d}.bin",
                dtype=np.uint16, mode="w+", shape=shape_hidden,
            )
            self.hidden_files.append(m)

    def write_microstep(self, idx: int,
                        topk_values: torch.Tensor,
                        topk_indices: torch.Tensor,
                        hidden_states: Sequence[torch.Tensor]) -> None:
        """Write one micro-step's worth of teacher output.

        topk_values: (B, T, k) fp16
        topk_indices: (B, T, k) int32 or int64
        hidden_states: list of N tensors, each (B, T, H) bf16
        """
        self.topk_values[idx] = topk_values.detach().to(
            dtype=torch.float16, device="cpu").numpy()
        self.topk_indices[idx] = topk_indices.detach().to(
            dtype=torch.int32, device="cpu").numpy()
        for out, h in zip(self.hidden_files, hidden_states):
            # bf16 has no numpy dtype; view bf16 bits as uint16 and
            # write those 2 bytes.
            h_cpu = h.detach().to(dtype=torch.bfloat16, device="cpu")
            out[idx] = h_cpu.view(torch.uint16).numpy()

    def finalize(self) -> None:
        """Flush memmaps to disk and write metadata.json."""
        # Memmaps flush on deletion / exit, but be explicit.
        self.topk_values.flush()
        self.topk_indices.flush()
        for m in self.hidden_files:
            m.flush()
        (self.cache_dir / "metadata.json").write_text(self.meta.to_json())


# -------------------------------------------------------------------
# Reader (used by training)
# -------------------------------------------------------------------

class TeacherCacheReader:
    """Train-time reader via numpy.memmap.

    Usage:
        cache = TeacherCacheReader(Path("/path/to/cache"))
        cache.validate_config(args)  # raises if mismatched
        for step in range(steps):
            for micro in range(accum):
                idx = step * accum + micro
                topk_vals, topk_idx, hiddens = cache.get(idx, device)
                # ... use in loss
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        if not (self.cache_dir / "metadata.json").exists():
            raise FileNotFoundError(
                f"No metadata.json in {cache_dir}; did extraction complete?"
            )
        self.meta = CacheMetadata.from_json(
            (self.cache_dir / "metadata.json").read_text()
        )

        shape_topk = (self.meta.n_microsteps, self.meta.batch_size,
                      self.meta.seq_len, self.meta.k)
        shape_hidden = (self.meta.n_microsteps, self.meta.batch_size,
                        self.meta.seq_len, self.meta.hidden_size)

        self.topk_values = np.memmap(
            self.cache_dir / "topk_values.bin",
            dtype=np.float16, mode="r", shape=shape_topk,
        )
        self.topk_indices = np.memmap(
            self.cache_dir / "topk_indices.bin",
            dtype=np.int32, mode="r", shape=shape_topk,
        )
        self.hidden_memmaps = []
        for li in self.meta.hidden_layer_indices:
            m = np.memmap(
                self.cache_dir / "hidden" / f"layer_{li:02d}.bin",
                dtype=np.uint16, mode="r", shape=shape_hidden,
            )
            self.hidden_memmaps.append(m)

    def validate_config(self, *, seed: int, seq_len: int, batch_size: int,
                        grad_accum_steps: int, c4_samples: int,
                        teacher_model: str, corpus_hash: str,
                        required_steps: int) -> None:
        """Raise if the cache wasn't produced with the given config.

        `required_steps` is the number of opt-steps training will
        consume; the cache must contain at least that many.
        """
        m = self.meta
        mismatches = []
        if m.seed != seed:
            mismatches.append(f"seed: cache={m.seed} vs train={seed}")
        if m.seq_len != seq_len:
            mismatches.append(f"seq_len: cache={m.seq_len} vs train={seq_len}")
        if m.batch_size != batch_size:
            mismatches.append(
                f"batch_size: cache={m.batch_size} vs train={batch_size}")
        if m.grad_accum_steps != grad_accum_steps:
            mismatches.append(
                f"grad_accum_steps: cache={m.grad_accum_steps} "
                f"vs train={grad_accum_steps}")
        if m.c4_samples != c4_samples:
            mismatches.append(
                f"c4_samples: cache={m.c4_samples} vs train={c4_samples}")
        if m.teacher_model != teacher_model:
            mismatches.append(
                f"teacher_model: cache={m.teacher_model} vs train={teacher_model}")
        if m.corpus_hash != corpus_hash:
            mismatches.append(
                f"corpus_hash: cache={m.corpus_hash} vs train={corpus_hash} "
                "(tokenizer or corpus changed between extract and train)")
        if m.opt_steps < required_steps:
            mismatches.append(
                f"cache has {m.opt_steps} opt-steps, training wants "
                f"{required_steps}")
        if mismatches:
            raise RuntimeError(
                "TeacherCacheReader: config mismatch — cache is not "
                "usable for this run:\n  " + "\n  ".join(mismatches))

    def get(self, microstep_idx: int, device: torch.device):
        """Return (topk_values_fp16, topk_indices_int64, hiddens_bf16)
        for a given micro-step index."""
        tv = torch.from_numpy(
            np.ascontiguousarray(self.topk_values[microstep_idx])
        ).to(device, non_blocking=True)
        ti = torch.from_numpy(
            np.ascontiguousarray(self.topk_indices[microstep_idx])
        ).to(device, dtype=torch.int64, non_blocking=True)
        hiddens = []
        for mm in self.hidden_memmaps:
            u16 = torch.from_numpy(
                np.ascontiguousarray(mm[microstep_idx])
            ).to(device, non_blocking=True)
            hiddens.append(u16.view(torch.bfloat16))
        return tv, ti, hiddens


# -------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------

def kl_topk_loss(student_logits: torch.Tensor,
                 teacher_topk_values: torch.Tensor,
                 teacher_topk_indices: torch.Tensor) -> torch.Tensor:
    """Top-k truncated KL distillation loss.

    student_logits:         (B, T, V) — full-vocab student logits
    teacher_topk_values:    (B, T, k) — top-k teacher logit values
    teacher_topk_indices:   (B, T, k) — top-k teacher vocabulary indices

    Loss formulation: the teacher distribution is approximated as
    softmax over its top-k (all tail mass dropped).  The student is
    softmaxed over the FULL vocab — we then gather the student's
    log-probs at teacher's top-k indices.  KL is the truncated sum.

    This matters a lot.  An earlier implementation softmaxed the
    student over only the top-k indices too, which gave the student
    gradient no signal to push mass between top-k and the tail —
    measured 2x PPL regression on a 500-step 0.5B smoke.  Using the
    full-vocab student softmax, the student gradient pushes top-k
    logits up relative to all V-k non-topk logits as expected, and
    the gradient matches the full-KL direction to within ~3%.

    Known residual gap (2026-04-22, 0.5B, 1000 steps): 7-8% PPL
    worse than live-teacher full-vocab KL, even at k=1024.  Source
    is the re-normalisation: `softmax(topk_values)` sums to 1 over
    top-k, which implicitly divides every probability by the top-k
    coverage ratio (~0.93 at k=256, ~0.97 at k=1024).  Effectively
    scales the KL gradient by 1/coverage — a systematic LR bias on
    KL relative to MSE.  Fix TODO(C): cache `log_sum_exp` of the
    teacher's full-vocab logits (4 bytes/position, trivial) and
    compute `t_log_probs = topk_values - lse` directly.  This gives
    the true teacher log-probs on the top-k support (sums to
    coverage, not 1), which eliminates the renormalisation bias.
    Expected to drop the gap into the 1-3% range.
    """
    # Teacher distribution over top-k only (tail mass dropped — this
    # is the standard top-k distillation approximation).
    t_log_probs = F.log_softmax(teacher_topk_values, dim=-1)    # (B, T, k)
    t_probs = t_log_probs.exp()

    # Student distribution over full vocab, evaluated at teacher's
    # top-k indices.  Using the full-vocab softmax is what gives the
    # student gradient access to "push topk up vs tail".
    s_log_probs_full = F.log_softmax(
        student_logits.to(t_log_probs.dtype), dim=-1
    )                                                            # (B, T, V)
    s_log_probs_at_topk = s_log_probs_full.gather(
        -1, teacher_topk_indices
    )                                                            # (B, T, k)

    # Truncated KL: sum over top-k of p_t * (log p_t − log p_s).
    # Note: the two log-prob spaces live in different normalisations
    # (teacher: over top-k; student: over full vocab) but that's OK
    # here — we're not computing exact KL(p_t || p_s), we're computing
    # a distillation loss whose gradient pushes student toward teacher
    # on top-k while respecting full-vocab normalisation.
    kl = (t_probs * (t_log_probs - s_log_probs_at_topk)).sum(-1)
    return kl.mean()


# -------------------------------------------------------------------
# Utility: corpus hash
# -------------------------------------------------------------------

def compute_corpus_hash(tokens: torch.Tensor, n_prefix: int = 1_000_000) -> str:
    """Hash the first `n_prefix` tokens as a sanity fingerprint.

    Used to detect cache / corpus drift between extract time and
    train time without hashing the full multi-million-token stream.
    """
    prefix = tokens[:n_prefix].detach().cpu().numpy().tobytes()
    return hashlib.sha256(prefix).hexdigest()[:16]
