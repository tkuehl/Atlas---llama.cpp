"""Basis Sharing converter (Phase 1).

Implements the decomposition pinned in DESIGN.md:
  - Window=2 Basis Sharing (arxiv:2410.03765) for Q/K/V/gate/up
  - Per-matrix balanced truncation for O/down (paper says these don't share)
  - OUR CONTRIBUTION: closed-form weighted-LS refit of per-layer coefficients
    given the fixed shared basis, using output-gradient covariance S_out
    (combines Basis Sharing's cross-layer basis with balanced truncation's
    output-aware weighting)

For Phase 1 we materialize the decomposed weights back to dense in-place
and measure PPL against baseline. Phase 2 emits a factored GGUF; Phase 3
adds the streaming runtime.

Usage:
    python basis_sharing.py --model Qwen/Qwen2.5-0.5B --target-ratio 0.5
"""

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Pin cuSOLVER as the CUDA linalg backend. Empirically MAGMA is deprecated in
# recent PyTorch nightlies — setting it succeeds but cuSOLVER still runs under
# the hood per a torch warning. Pinning cuSOLVER explicitly lets us pass
# driver="gesvd" (classical Golub-Reinsch) which is stable on Blackwell, vs
# the default gesvdj (Jacobi) which hits CUSOLVER_STATUS_INVALID_VALUE.
_CUDA_LINALG_BACKEND = "default"
if torch.cuda.is_available():
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
        _CUDA_LINALG_BACKEND = "cusolver"
    except Exception:
        pass


SHARED_ROLES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
]
PERMATRIX_ROLES = [
    "self_attn.o_proj",
    "mlp.down_proj",
]


# ---------- gramian helpers ----------

_CUDA_FALLBACK_LOGGED = {"eigh": 0, "svd": 0}


def _stable_svd(M):
    """SVD of M on its current device. Chooses driver based on backend:
      - cuSOLVER backend on CUDA: pass driver='gesvd' (more stable than gesvdj)
      - MAGMA backend on CUDA: no driver (MAGMA uses its own routine)
      - CPU: standard SVD
    Returns a tuple on the SAME DEVICE as input M (never silently moves to CPU
    except on an error; caller gets a tensor they can still use in device-local
    ops). On CUDA failure, falls back to CPU and returns CPU tensors — caller
    must check result devices in sensitive paths.
    """
    if M.is_cuda:
        kwargs = {"full_matrices": False}
        if _CUDA_LINALG_BACKEND == "cusolver":
            kwargs["driver"] = "gesvd"
        try:
            return torch.linalg.svd(M, **kwargs)
        except RuntimeError as e:
            msg = str(e).lower()
            if not any(s in msg for s in ("cusolver", "cuda", "memory")):
                raise
            _cuda_fallback_log("svd", e)
            M = M.cpu()
    return torch.linalg.svd(M, full_matrices=False)


def _stable_svd_topk(M, k, oversample=10):
    """Truncated SVD returning approximate top-(k+oversample) components.

    Wraps `torch.svd_lowrank` (Halko-Martinsson-Tropp randomized SVD) with a
    fallback to the full `_stable_svd`. Returns (U, S, Vt) matching
    _stable_svd's convention.

    Skips lowrank when the requested q is close to the full dim — lowrank's
    O(m·n·q) cost ties the full O(m·n·min(m,n)) cost at q ≈ min(m,n)/2 and
    has a higher constant factor, so full SVD actually wins on small
    matrices or when k is nearly full rank. The 2x-dim heuristic is
    conservative; real speedup shows up at k < min_dim/2 (e.g. target_ratio
    ≤ 0.8 on permatrix down_proj). At r=1.0 lowrank is barely faster but
    still numerically equivalent to the required precision, so we always
    use it above the threshold rather than gate on target_ratio.

    For the balanced-truncation and ASVD paths the caller discards ranks
    beyond `rank`, so feeding `k = rank + oversample` yields the same
    final factors as full SVD to O(eps_fp32) precision with no accuracy
    loss at our usage pattern.
    """
    m, n = M.shape
    min_dim = min(m, n)
    q = min(k + oversample, min_dim)
    # Heuristic: if q is within 1.5× of min_dim, full SVD is competitive
    # (randomized methods have a constant-factor overhead that dominates
    # when k is near full rank).
    if q * 3 >= min_dim * 2:
        return _stable_svd(M)
    try:
        # niter=2 is the HMT-recommended default — small enough to stay
        # cheap, big enough that the top-q singular values match full SVD
        # to 4-6 decimal places on realistic LLM whitened matrices.
        U, S, V = torch.svd_lowrank(M, q=q, niter=2)
        # svd_lowrank returns V (not Vt) — transpose for API parity with
        # torch.linalg.svd (which all of our decomposers expect).
        return U, S, V.transpose(-2, -1).contiguous()
    except RuntimeError as e:
        msg = str(e).lower()
        if not any(s in msg for s in ("cuda", "memory", "cusolver", "magma")):
            raise
        _cuda_fallback_log("svd", e)
        return _stable_svd(M)


def _cuda_fallback_log(op, err):
    """Log CUDA → CPU fallbacks the first few times so we can diagnose."""
    if _CUDA_FALLBACK_LOGGED[op] < 3:
        print(f"  [linalg] CUDA {op} failed, falling back to CPU: {str(err)[:120]}")
        _CUDA_FALLBACK_LOGGED[op] += 1


def sqrt_and_inv(M, eps_ratio=1e-5, device="cuda", need_inv=True,
                 skip_cholesky=False):
    """Return (S, S_inv) such that S @ S^T ≈ M + eps*I.

    Primary path: Cholesky in fp32 on GPU — ~5-10x faster than fp64 eigh for
    d≈11008 because (a) O(n^3/3) vs O(n^3) + smaller constant, (b) fp32 uses
    full tensor-core throughput on Blackwell (fp64 is 1/64 on consumer cards).

    The returned S is lower-triangular (not symmetric) in the Cholesky path.
    For ASVD compression that's fine: left singular vectors of (W @ S) span
    the same subspace regardless of sqrt choice, since (WS)(WS)^T = W M W^T
    is identical.

    need_inv=False skips the triangular-solve for L^-1 — another ~2x savings
    when callers only use S (common in our ASVD path).

    Falls back to eigh on fp64 CPU for near-singular matrices or CUDA errors.

    `skip_cholesky=True` bypasses the Cholesky ladder entirely and goes
    straight to GPU fp64 eigh + eigenvalue clamp. Use when the caller knows
    the gramian is rank-deficient (e.g. d > calib_rows for wide-d roles) —
    every Cholesky tier is mathematically guaranteed to fail on such
    gramians, so probing them just wastes seconds per layer. Produces the
    same S as the CPU-eigh terminal fallback, just faster.
    """
    if not torch.isfinite(M).all():
        M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    n = M.shape[0]
    mean_diag = M.to(torch.float32).diag().abs().mean().item()

    def _chol(dev, dt, eps_factor):
        """Cholesky at a specific regularization strength (multiplier on mean_diag)."""
        eps = eps_factor * mean_diag + 1e-8
        M_t = M.to(device=dev, dtype=dt)
        reg = M_t + eps * torch.eye(n, device=dev, dtype=dt)
        L = torch.linalg.cholesky(reg)
        L_inv = None
        if need_inv:
            I = torch.eye(n, device=dev, dtype=dt)
            L_inv = torch.linalg.solve_triangular(L, I, upper=False)
        return L, L_inv

    def _eigh(dev, dt=torch.float64):
        eps = eps_ratio * mean_diag + 1e-8
        M_t = M.to(device=dev, dtype=dt)
        reg = M_t + eps * torch.eye(n, device=dev, dtype=dt)
        evals, evecs = torch.linalg.eigh(reg)
        evals = evals.clamp(min=eps)
        S = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T
        S_inv = None
        if need_inv:
            S_inv = (evecs * (1.0 / evals.sqrt()).unsqueeze(0)) @ evecs.T
        return S, S_inv

    def _to_out(S, S_inv, dt=torch.float32):
        return (S.to(dt), S_inv.to(dt) if S_inv is not None else None)

    # Fast path for known-rank-deficient gramians: skip the fp32/fp64
    # probe tiers (both guaranteed to fail with "not positive definite"
    # when calibration rows < d) and jump straight to the tier we've
    # empirically observed succeeds — GPU fp64 Cholesky with eps×10
    # regularization. Saves ~1-2s per layer of failed probes on 7B
    # down_proj. The original attempt to use GPU fp64 eigh OOMs at
    # d=18944 on a 12 GB card (eigh workspace is ~10 GB), which caused
    # the skip path to fall all the way to the very slow CPU eigh tier.
    # fp64 Cholesky workspace is much smaller (~3x the matrix) so it
    # fits cleanly.
    if skip_cholesky and device == "cuda":
        try:
            result = _to_out(*_chol("cuda", torch.float64, eps_ratio * 10))
            return result
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                _cuda_fallback_log("eigh", e)
            # Fall through to the CPU tiers below on failure.

    # Precision-first fallback ladder. Rank-deficient gramians (common for
    # down_proj when calibration tokens < d_in) break fp32 Cholesky; bumping
    # eps aggressively distorts the whitening matrix and tanks ASVD quality.
    # We'd rather pay fp64 precision than eat quality loss from over-regularizing.
    #   1) GPU fp32 Cholesky @ eps×1    — fastest path (usually works on full-rank)
    #   2) GPU fp64 Cholesky @ eps×1    — same regularization, higher precision
    #   3) GPU fp64 Cholesky @ eps×10   — if even fp64 fails, mild extra damping
    #   4) CPU fp64 eigh                — last resort (handles singular cleanly)
    tried = []
    if device == "cuda" and not skip_cholesky:
        for tier_name, dev, dt, factor in [
            ("gpu-fp32",       "cuda", torch.float32, eps_ratio),
            ("gpu-fp64",       "cuda", torch.float64, eps_ratio),
            ("gpu-fp64-eps10", "cuda", torch.float64, eps_ratio * 10),
        ]:
            try:
                result = _to_out(*_chol(dev, dt, factor))
                if len(tried) > 0:
                    print(f"    [linalg] n={n}: fell through to {tier_name} "
                          f"(tried: {tried})", flush=True)
                return result
            except RuntimeError as e:
                tried.append(tier_name)
                msg = str(e).lower()
                if "out of memory" in msg:
                    _cuda_fallback_log("eigh", e)
                    break
                if not any(s in msg for s in ("cholesky", "not positive",
                                              "singular", "cusolver", "cuda")):
                    raise
    # CPU path
    for tier_name, dt, factor in [
        ("cpu-fp64",       torch.float64, eps_ratio),
        ("cpu-fp64-eps10", torch.float64, eps_ratio * 10),
    ]:
        try:
            result = _to_out(*_chol("cpu", dt, factor))
            print(f"    [linalg] n={n}: cpu fallback to {tier_name} "
                  f"(tried: {tried})", flush=True)
            return result
        except RuntimeError:
            tried.append(tier_name)
    # Final: eigh on CPU fp64 handles near-singular PSD cleanly via eigenvalue clamp.
    print(f"    [linalg] n={n}: final fallback to cpu-eigh-fp64 "
          f"(tried: {tried})", flush=True)
    return _to_out(*_eigh("cpu", torch.float64))


# ---------- module discovery ----------

def find_layers(model, roles):
    """Return {role: [(layer_idx, name, module)], ...} sorted by layer_idx."""
    groups = defaultdict(list)
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        for role in roles:
            if name.endswith(role):
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


def make_windows(L, window_size):
    """Non-overlapping adjacent windows. Trailing short window if L % w != 0."""
    out = []
    i = 0
    while i < L:
        out.append(list(range(i, min(i + window_size, L))))
        i += window_size
    return out


# ---------- memory safety + gramian storage ----------
#
# Calibration accumulates d×d fp32 gramians (xtx, optionally ggt) for every
# linear module we're about to factor. For tall intermediate-dim matrices like
# down_proj on 7B+ (d_in=18944), a single gramian is 1.4 GB and the full model
# needs ~40 GB — more than fits in RAM alongside the model. XtxStore solves
# that by optionally memmap-backing the accumulators onto NVMe, so page cache
# absorbs the hot set and RAM pressure stays bounded.

def _require_ram_headroom(bytes_needed: int, min_free_gb: float, label: str):
    """Safety check (Option A): refuse the allocation if it would put free RAM
    below the configured OS headroom. Raises with a message that names the
    offending role and suggests --xtx-backend disk as the mitigation."""
    avail = psutil.virtual_memory().available
    reserve = int(min_free_gb * 1e9)
    if avail - bytes_needed < reserve:
        raise MemoryError(
            f"refusing to allocate {bytes_needed/1e9:.2f} GB for {label}: "
            f"only {avail/1e9:.2f} GB free, would leave less than the "
            f"{min_free_gb:.1f} GB reserved for the OS. "
            f"Pass --xtx-backend disk (or auto) to spill gramians to NVMe."
        )


def _require_disk_headroom(bytes_needed: int, target_dir: Path,
                           max_pct: float, label: str):
    """Refuse the allocation if it would push used-disk above max_pct of
    the drive's total size. Mirrors the RAM headroom check so runs that
    trip against disk fail fast with a clear message instead of the generic
    OSError(28) from the filesystem."""
    usage = shutil.disk_usage(str(target_dir))
    used_after = (usage.total - usage.free) + bytes_needed
    cap_bytes = int(usage.total * max_pct / 100.0)
    if used_after > cap_bytes:
        raise OSError(
            f"refusing to allocate {bytes_needed/1e9:.2f} GB on "
            f"{target_dir} for {label}: would push used disk to "
            f"{used_after/1e9:.1f} GB vs cap "
            f"{cap_bytes/1e9:.1f} GB ({max_pct:.0f}% of "
            f"{usage.total/1e9:.1f} GB total). Free space or raise "
            f"--disk-max-pct."
        )


class XtxStore:
    """Allocates and owns d×d gramian accumulators for calibration.

    Modes:
      - "ram":  torch.zeros on CPU (current behavior)
      - "disk": np.memmap → torch.from_numpy; addmm_ writes through to NVMe.
                The OS page cache handles residency; this just caps RAM.
      - "auto": per-tensor choice. If the allocation would trip the
                --min-free-ram-gb threshold, fall back to disk; else RAM.

    Precision policy: gramians are allocated as float64 when d ≥
    `accum_fp64_threshold` and float32 otherwise. Wide gramians (down_proj
    on 7B+ has d=18944) accumulate ~10^3–10^4 hook updates worth of
    fp32-matmul deltas, and the low-eigenvalue tail drowns in fp32
    running-sum noise long before the Cholesky whitening step runs. Keeping
    the accumulator in fp64 costs 2× memory (handled by disk backing on big
    models) and zero GPU compute — the matmul stays fp32; PyTorch's
    in-place `.add_()` auto-promotes to the destination dtype. The cast for
    small-d roles stays fp32 to preserve the existing RAM/checkpoint path.

    All returned tensors are CPU and safe to hand to downstream `addmm_` and
    SVD code unchanged (`sqrt_and_inv` re-casts internally). Call `cleanup`
    when done to delete temp files.
    """

    def __init__(self, mode: str, temp_dir: Path | None, min_free_gb: float,
                 gpu_workspace_cap_gb: float = 4.0,
                 accum_fp64_threshold: int = 4096,
                 disk_max_pct: float = 90.0,
                 gpu_vram_max_pct: float = 90.0,
                 gpu_accum_budget_gb: float = 4.0):
        assert mode in ("ram", "disk", "auto")
        self.mode = mode
        self.min_free_gb = min_free_gb
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        if mode != "ram":
            if self.temp_dir is None:
                raise ValueError("disk-backed XtxStore requires a temp_dir")
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._memmaps: dict[str, np.memmap] = {}
        self._paths: list[Path] = []
        # Per-d_in GPU scratch buffers for the `flat.T @ flat` matmul that
        # produces each hook's gramian delta. We pre-allocate one workspace
        # per unique d value (typically just 2 for decoder-only LLMs:
        # hidden_size and intermediate_size) so the hot path is alloc-free.
        # Cap is a soft budget; individual workspaces larger than this fall
        # back to the CPU addmm path in collect_stats.
        self._gpu_workspaces: dict[int, torch.Tensor] = {}
        self._gpu_enabled = torch.cuda.is_available()
        self._gpu_workspace_cap_bytes = int(gpu_workspace_cap_gb * 1e9)
        self.accum_fp64_threshold = int(accum_fp64_threshold)
        self.disk_max_pct = float(disk_max_pct)
        self.gpu_vram_max_pct = float(gpu_vram_max_pct)
        # Total GPU-resident gramian budget, in bytes. When > 0, alloc()
        # places fp32 gramians directly on GPU so the hook path can do an
        # in-place `addmm_` into them — avoiding the per-hook ~5 ms D2H +
        # CPU add_ round trip that dominates calibration wall time for
        # models where model-forward is cheap relative to hook overhead.
        # 0 disables GPU residency entirely (legacy CPU addmm path).
        self._gpu_accum_budget_bytes = int(gpu_accum_budget_gb * 1e9)
        self._gpu_accum_bytes_used = 0
        self._gpu_accum_tensors: dict[str, torch.Tensor] = {}

    def _accum_dtype(self, d: int) -> torch.dtype:
        return torch.float64 if d >= self.accum_fp64_threshold else torch.float32

    def _accum_nbytes(self, d: int) -> int:
        return 8 if d >= self.accum_fp64_threshold else 4

    def _use_disk(self, bytes_needed: int) -> bool:
        if self.mode == "ram":
            return False
        if self.mode == "disk":
            return True
        # auto: go to disk if this allocation would cross the safety line OR
        # if free RAM is already within one allocation of the reserve. The
        # double-reserve margin biases us toward disk earlier, before RAM
        # pressure cascades: better to spill a few extra gramians to NVMe
        # than to fill RAM and stall the next allocation.
        avail = psutil.virtual_memory().available
        reserve = int(self.min_free_gb * 1e9)
        return bytes_needed > avail - 2 * reserve

    def _try_alloc_gpu(self, name: str, d: int, dtype: torch.dtype,
                       bytes_needed: int) -> torch.Tensor | None:
        """Attempt GPU-resident allocation under the configured budget.

        Returns a CUDA tensor on success; None if disabled, dtype is fp64
        (which is impractically slow on consumer GPUs for the matmul path
        we're optimizing), budget is exhausted, or cudaMemGetInfo reports
        insufficient headroom against --gpu-vram-max-pct. Caller falls
        through to the CPU path in the latter cases."""
        if (not self._gpu_enabled
                or self._gpu_accum_budget_bytes <= 0
                or dtype == torch.float64):
            return None
        if (self._gpu_accum_bytes_used + bytes_needed
                > self._gpu_accum_budget_bytes):
            return None
        free_vram, total_vram = torch.cuda.mem_get_info()
        cap_bytes = int(total_vram * self.gpu_vram_max_pct / 100.0)
        used_after = (total_vram - free_vram) + bytes_needed
        if used_after > cap_bytes:
            return None
        try:
            t = torch.zeros(d, d, dtype=dtype, device="cuda")
        except torch.cuda.OutOfMemoryError:
            return None
        self._gpu_accum_tensors[name] = t
        self._gpu_accum_bytes_used += bytes_needed
        return t

    def alloc(self, name: str, d: int) -> torch.Tensor:
        dtype = self._accum_dtype(d)
        nbytes = self._accum_nbytes(d)
        bytes_needed = d * d * nbytes
        # GPU-resident shared-role accumulators: lets the hook do an
        # in-place `addmm_` into the gramian on device, skipping the
        # workspace + D2H + CPU add_ round trip per hook. fp64 deferred
        # to CPU path (1/64 throughput on consumer cards, not worth it).
        gpu_tensor = self._try_alloc_gpu(name, d, dtype, bytes_needed)
        if gpu_tensor is not None:
            return gpu_tensor
        use_disk = self._use_disk(bytes_needed)
        if use_disk:
            # No RAM safety check on the disk path. Memmap writes go to the
            # OS page cache, which Windows auto-reclaims under pressure — so
            # even when free RAM looks low, we are not actually allocating
            # new process memory. Guarding this on virtual_memory().available
            # produces false positives once the page cache fills up.
            # Disk headroom *is* checked: memmap w+ pre-allocates the full
            # file (Windows semantics), and running the drive past the cap
            # trips OSError(28) mid-calibration as we saw on the first 7B
            # fp64 attempt. Fail early with a clear message instead.
            _require_disk_headroom(bytes_needed, self.temp_dir,
                                   self.disk_max_pct, name)
            safe = name.replace("/", "_").replace(".", "_")
            suffix = "f64" if nbytes == 8 else "f32"
            np_dtype = "float64" if nbytes == 8 else "float32"
            path = self.temp_dir / f"xtx_{safe}.{suffix}"
            # mode='w+' truncates; freshly-extended pages are zero-initialized
            # by the OS, so we don't need (and shouldn't pay for) an explicit
            # arr[:] = 0 that dirties every page up front.
            arr = np.memmap(str(path), dtype=np_dtype, mode="w+", shape=(d, d))
            self._memmaps[name] = arr
            self._paths.append(path)
            return torch.from_numpy(arr)
        _require_ram_headroom(bytes_needed, self.min_free_gb, name)
        return torch.zeros(d, d, dtype=dtype)

    def get_gpu_workspace(self, d: int) -> torch.Tensor | None:
        """Return (lazily-allocated) CUDA fp32 d×d buffer for delta matmul.

        Returns None when CUDA is unavailable, the tensor would exceed the
        configured cap, or allocation fails (e.g. VRAM exhausted). Callers
        in the hook path fall back to CPU-side addmm in that case.

        Two independent caps gate the allocation:
        - `gpu_workspace_cap_gb`: absolute byte budget for any single
          workspace (kept for backward compat; skips this d entirely if
          the single tensor would exceed it).
        - `gpu_vram_max_pct`: total-VRAM ceiling. Checks current free VRAM
          via cudaMemGetInfo and allocates only if the resulting usage
          stays under the cap. Lets multiple small workspaces coexist but
          refuses to push VRAM past the configured percentage.
        """
        if not self._gpu_enabled:
            return None
        if d in self._gpu_workspaces:
            return self._gpu_workspaces[d]
        bytes_needed = d * d * 4
        if bytes_needed > self._gpu_workspace_cap_bytes:
            return None
        # VRAM percent cap: refuse the alloc if it would push used-VRAM
        # above the configured fraction of total. free_vram is reported
        # after the model + any prior workspaces have already been
        # allocated, so this is an honest "what's left" signal.
        free_vram, total_vram = torch.cuda.mem_get_info()
        cap_bytes = int(total_vram * self.gpu_vram_max_pct / 100.0)
        used_after = (total_vram - free_vram) + bytes_needed
        if used_after > cap_bytes:
            return None
        try:
            ws = torch.empty(d, d, dtype=torch.float32, device="cuda")
        except torch.cuda.OutOfMemoryError:
            return None
        self._gpu_workspaces[d] = ws
        return ws

    def free_gpu_workspaces(self):
        """Release all pre-allocated GPU buffers. Call at the end of
        calibration so downstream compute_factors has the full VRAM to
        work with during SVD."""
        self._gpu_workspaces.clear()
        if self._gpu_enabled:
            torch.cuda.empty_cache()

    def consolidate_to_cpu(self, xtx: dict[str, torch.Tensor]) -> None:
        """Move any GPU-resident gramian accumulators into CPU tensors
        in `xtx` so downstream code sees uniform CPU tensors. Called
        once after calibration completes. Frees VRAM the decomposition
        will use for SVD workspace."""
        if not self._gpu_accum_tensors:
            return
        for name, t in list(self._gpu_accum_tensors.items()):
            if name in xtx and xtx[name].is_cuda:
                xtx[name] = t.cpu().contiguous()
        self._gpu_accum_tensors.clear()
        self._gpu_accum_bytes_used = 0
        if self._gpu_enabled:
            torch.cuda.empty_cache()

    def has_disk_backed(self) -> bool:
        """True if any of the accumulators allocated via this store live on
        an np.memmap. checkpoint_save pickles the whole xtx dict, which
        materializes memmaps into RAM — so we disable the checkpoint when
        this is True (the resulting .pt would be tens of GB and the reload
        would OOM)."""
        return bool(self._memmaps)

    def cleanup(self):
        # Drop strong refs so the OS can unlink the files on Windows.
        self.free_gpu_workspaces()
        self._memmaps.clear()
        for p in self._paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        self._paths.clear()
        if self.temp_dir is not None and self.temp_dir.exists():
            try:
                self.temp_dir.rmdir()
            except OSError:
                pass  # dir not empty (leftover files or other artifacts)


# ---------- streaming activation collector ----------
#
# For wide-d roles at 7B+ (mlp.down_proj with d_in=18944), an fp64 d×d
# gramian is ~2.87 GB per layer. Keeping all 28 layers' gramians live
# during calibration (80 GB) forces disk-backed memmaps whose random-access
# write pattern pathologically thrashes Windows' page cache — we observed
# 10-30 min per calibration sample on an HDD and 5-15 min/sample on NVMe
# before the system stalled.
#
# This class sidesteps that: the forward hook captures raw activations to
# a per-layer pinned-host buffer (bf16 to halve memory vs fp32). After all
# calibration samples run, `finalize()` iterates layers, uploads each
# layer's concatenated activations to GPU, computes the fp64 gramian via
# chunked fp32 matmul with fp64 accumulate, and D2Hs the result into the
# normal XtxStore accumulator.
#
# Memory vs the live-accumulate path, per role of d_in=D, L layers,
# N samples at batch×seq=T tokens:
#   live fp64 gramian: L × D × D × 8 bytes            (80 GB at 7B)
#   streamed activations (bf16): L × N × T × D × 2     (17 GB at 7B/8smpl)
# Win is specifically when D is big enough that L×D² dominates L×N×T×D,
# i.e. D > N×T. For N×T ~ 16K and D=18944, streaming is ~5× less memory.
# For D=3584 the math inverts and the regular hook path is cheaper.

class StreamingCollector:
    """Activation cache + post-pass gramian computer for wide-d roles.

    Hook captures activations into pinned-host bf16 buffers keyed by
    module name. `finalize()` computes fp64 gramians on GPU (chunked)
    and writes them into the provided XtxStore so downstream code sees
    them exactly like regular live-accumulated gramians.

    Raises MemoryError if total pinned-host usage would cross the
    configured budget — caller should reduce calib samples or narrow
    `--streaming-roles`.
    """

    def __init__(self, xtx_store: "XtxStore", max_pinned_ram_gb: float,
                 attn_mask_holder: list | None = None):
        self.xtx_store = xtx_store
        self.max_pinned_bytes = int(max_pinned_ram_gb * 1e9)
        self.buffers: dict[str, list[torch.Tensor]] = {}
        self._bytes_held = 0
        # Holder reference shared with collect_stats. When batched
        # calibration is active, the mask tensor lives at
        # attn_mask_holder[0]; each capture zeros padded positions before
        # stashing the activation. None means batch=1 (no padding).
        self._attn_mask_holder = attn_mask_holder
        # Running accumulators (xtx_store-backed fp64 gramians). Lazily
        # allocated on the first drain that sees each name. Between drains
        # we accumulate new deltas into these, so peak pinned-host usage
        # stays bounded by the budget regardless of sample count.
        self.accumulators: dict[str, torch.Tensor] = {}
        # Rows contributed per name (for logging). Reset when we drain
        # at finalize, useful only for the summary line.
        self._rows_seen: dict[str, int] = {}
        # Incremented per-drain; used only for log framing.
        self._drain_idx = 0
        # Largest per-sample byte growth seen so far. Used by should_drain
        # to guarantee we always leave room for one more forward — if we
        # let the buffer fill to the cap minus less-than-one-sample, the
        # next forward's hooks trip the hard MemoryError before drain
        # gets a chance to run. Grows monotonically.
        self._peak_sample_bytes = 0
        self._bytes_before_forward = 0

    def register(self, name: str) -> None:
        self.buffers.setdefault(name, [])
        self._rows_seen.setdefault(name, 0)

    def capture_hook(self, name: str):
        """Return a forward_pre_hook that stashes activations for `name`."""
        def hook(mod, inputs):
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1])
            # Batched calibration: mask padded positions to zero before
            # we stash them. Otherwise finalize() would add padded
            # positions' outer products to the fp64 gramian, distorting
            # the spectrum used by the downstream SVD.
            if (self._attn_mask_holder is not None
                    and self._attn_mask_holder[0] is not None):
                m = self._attn_mask_holder[0].reshape(-1, 1).to(
                    dtype=flat.dtype, device=flat.device)
                flat = flat * m
            # Preserve the activation's native dtype. For Qwen loaded as
            # fp16 (default), this retains 11-bit mantissa precision;
            # hardcoding bf16 here would downcast to 7 bits and contaminate
            # the gramian with ~1% relative input noise before the matmul
            # even runs. Pinned so the D2H back to GPU in finalize() is
            # overlap-friendly. Allocate + copy is fast vs the model's
            # own per-layer compute, so the hook stays off the hot path.
            pinned = torch.empty_like(flat, device="cpu", pin_memory=True)
            pinned.copy_(flat)
            self.buffers[name].append(pinned)
            self._bytes_held += pinned.numel() * pinned.element_size()
            if self._bytes_held > self.max_pinned_bytes:
                raise MemoryError(
                    f"StreamingCollector exceeded pinned-host budget: "
                    f"{self._bytes_held/1e9:.1f} GB vs cap "
                    f"{self.max_pinned_bytes/1e9:.1f} GB. Reduce "
                    f"--calib-samples or increase drain frequency."
                )
        return hook

    def note_forward_start(self) -> None:
        """Mark the pinned-bytes level before the next forward. Pair with
        `note_forward_end` / `should_drain` so the collector can size the
        drain window against the actual per-sample growth, not a guess."""
        self._bytes_before_forward = self._bytes_held

    def note_forward_end(self) -> None:
        """Update the running max-per-sample tally. Called after each
        forward so should_drain knows how much room the next sample needs."""
        grew = self._bytes_held - self._bytes_before_forward
        if grew > self._peak_sample_bytes:
            self._peak_sample_bytes = grew

    def should_drain(self) -> bool:
        """True when the next forward would exceed the budget. Uses the
        peak per-sample growth seen so far as the guard — before the first
        forward finishes, peak is 0 so this returns False and the first
        sample fills freely."""
        # Require room for one more sample plus a 10% safety margin.
        projected = self._bytes_held + int(self._peak_sample_bytes * 1.1)
        return projected > self.max_pinned_bytes

    def drain(self, device: str = "cuda", chunk_rows: int = 4096) -> None:
        """Compute gramian contributions from current buffers, add into the
        running per-name accumulators, release pinned memory. Can be called
        repeatedly during calibration to keep peak pinned bounded across
        many samples; the final call should be `finalize()` which also
        returns the dict for collect_stats to merge."""
        gpu_ok = device == "cuda" and torch.cuda.is_available()
        drained_bytes = 0
        for name, bufs in list(self.buffers.items()):
            if not bufs:
                continue
            d = bufs[0].shape[1]
            # Lazy-alloc the running accumulator on first drain for this
            # name. Allocating later means we never reserve xtx_store space
            # for a streaming role whose buffers never received any data.
            if name not in self.accumulators:
                self.accumulators[name] = self.xtx_store.alloc(name, d)
            dest = self.accumulators[name]
            rows_in_drain = sum(b.shape[0] for b in bufs)
            self._rows_seen[name] = self._rows_seen.get(name, 0) + rows_in_drain
            if gpu_ok:
                gram = torch.zeros(d, d, dtype=torch.float64, device="cuda")
                for b in bufs:
                    for chunk in b.split(chunk_rows):
                        chunk_gpu = chunk.to(device="cuda",
                                             dtype=torch.float32,
                                             non_blocking=True)
                        delta = chunk_gpu.T @ chunk_gpu
                        gram.add_(delta)
                        del chunk_gpu, delta
                # Accumulate: dest is fp64, gram is fp64 → in-place add
                # preserves dest dtype and semantics across multiple drains.
                dest.add_(gram.cpu().to(dest.dtype))
                del gram
                torch.cuda.empty_cache()
            else:
                for b in bufs:
                    chunk_cpu = b.to(dtype=dest.dtype)
                    dest.addmm_(chunk_cpu.T, chunk_cpu)
                    del chunk_cpu
            # Release this drain's buffers and subtract their weight from
            # the pinned-bytes tally. _bytes_held is a tracking int, not
            # directly tied to tensor lifetimes, so we update it manually.
            self.buffers[name] = []
            drained_bytes += sum(b.shape[0] for b in bufs) * d * (
                bufs[0].element_size())
        self._bytes_held = max(0, self._bytes_held - drained_bytes)
        self._drain_idx += 1

    def finalize(self, device: str = "cuda",
                 chunk_rows: int = 4096) -> dict[str, torch.Tensor]:
        """Flush remaining buffers, log a summary, return the accumulator
        dict for collect_stats to merge into xtx."""
        self.drain(device=device, chunk_rows=chunk_rows)
        total = len(self.accumulators)
        for idx, (name, dest) in enumerate(self.accumulators.items()):
            d = dest.shape[0]
            rows = self._rows_seen.get(name, 0)
            print(f"    [streaming] finalized {idx+1}/{total} {name}: "
                  f"d={d} rows={rows}", flush=True)
        return dict(self.accumulators)


# ---------- calibration (forward + optional backward) ----------

def collect_stats(model, tokenizer, texts, device, seq_len, role_groups, need_grad,
                  xtx_store: XtxStore, ggt_store: XtxStore | None = None,
                  streaming_roles: set[str] | None = None,
                  streaming_max_ram_gb: float = 20.0,
                  batch_size: int = 1):
    """Run calibration and accumulate X^T X (input cov) and optionally G^T G
    (output-gradient cov) for every linear we care about.

    `role_groups` is a list-of-lists: each inner list is one calibration pass
    whose hooks fire only for those roles. A single-pass calibration is
    expressed as `[all_roles]` and is the cheapest when gramians fit in RAM.
    Splitting into multiple passes (e.g. [shared_roles, [o_proj], [down_proj]])
    caps peak RAM per pass, which is how we make 14B+ fit on a 64 GB box.

    When the XtxStore has GPU workspaces available, hooks push the tall
    (batch*seq × d) activation to the GPU and run the d×d delta matmul
    there — then copy the small(-ish) delta back to the CPU accumulator.
    For d=18944 on RTX 5070 this is ~3× faster than the pure-CPU addmm
    path, and the per-hook PCIe traffic (~1.4 GB for down_proj) stays
    bounded by the already-resident gramian size. On machines without
    CUDA or when a workspace would exceed the cap, hooks fall back to the
    original CPU addmm path.
    """
    xtx = {}
    ggt = {} if need_grad else None
    streaming_roles = streaming_roles or set()
    # Batched calibration needs to zero out padding-token activations
    # before they're summed into the gramian — otherwise padded positions
    # add their (meaningless) outer products to X^T X, distorting the
    # spectrum the downstream SVD relies on. We share the current batch's
    # attention mask through this holder so every hook (standard,
    # streaming, backward) can mask its activations in place. None means
    # batch=1 mode — no padding.
    _attn_mask_holder: list = [None]
    # The streaming collector is a no-op if no role is flagged for it.
    # When present, forward hooks for those roles cache activations
    # to pinned host bf16; finalize() after all passes computes fp64
    # gramians on GPU in chunks, writing through the same xtx_store.
    sc = (StreamingCollector(xtx_store, streaming_max_ram_gb,
                             attn_mask_holder=_attn_mask_holder)
          if streaming_roles else None)
    if streaming_roles and need_grad:
        # Backward-gradient streaming isn't implemented — ggt accumulation
        # for streamed roles would need the same treatment and we haven't
        # needed it (refit is off for wide-d scale tests). Fail loud
        # instead of silently dropping ggt samples.
        raise NotImplementedError(
            "streaming_roles with need_grad=True is not supported yet — "
            "the backward hook path still uses live ggt accumulation. "
            "Pass --no-refit or extend StreamingCollector to cache "
            "gradients as well."
        )
    # Every module we'll touch across every pass — we grad-enable the whole
    # set upfront so the backward hooks installed later don't hit frozen
    # params.
    all_groups: dict = {}
    for pass_roles in role_groups:
        for role, entries in find_layers(model, pass_roles).items():
            all_groups.setdefault(role, entries)
    if need_grad:
        for p in model.parameters():
            p.requires_grad_(False)
        for role, entries in all_groups.items():
            for _, _, mod in entries:
                mod.weight.requires_grad_(True)

    def fwd_hook_factory(name, d_in):
        xtx[name] = xtx_store.alloc(name, d_in)

        # GPU-resident accumulator path — zero-copy addmm into the gramian
        # that already lives on device. Eliminates the ~5ms D2H per hook
        # (and matching CPU add_) that dominates calibration when model
        # forward is cheap relative to hook overhead.
        if xtx[name].is_cuda:
            gram = xtx[name]
            gram_dtype = gram.dtype
            def hook(mod, inputs):
                x = inputs[0].detach()
                flat = x.reshape(-1, x.shape[-1]).to(
                    device="cuda", dtype=gram_dtype, non_blocking=True)
                m = _attn_mask_holder[0]
                if m is not None:
                    mask_flat = m.reshape(-1, 1).to(device="cuda",
                                                     dtype=gram_dtype,
                                                     non_blocking=True)
                    flat = flat * mask_flat
                gram.addmm_(flat.T, flat)
            return hook

        ws = xtx_store.get_gpu_workspace(d_in)

        if ws is not None:
            def hook(mod, inputs):
                x = inputs[0].detach()
                flat = x.reshape(-1, x.shape[-1])
                flat_gpu = flat.to(device="cuda", dtype=torch.float32, non_blocking=True)
                # Batched calibration: mask padded positions to zero so
                # they don't contribute to the gramian. Mask shape is
                # (B, T); reshape to (B*T, 1) to broadcast across d.
                m = _attn_mask_holder[0]
                if m is not None:
                    mask_flat = m.reshape(-1, 1).to(device="cuda",
                                                     dtype=torch.float32,
                                                     non_blocking=True)
                    flat_gpu = flat_gpu * mask_flat
                # In-place matmul into the pre-allocated workspace, then D2H
                # into a fresh CPU tensor. .cpu() is synchronous in torch's
                # default stream, so the next hook's overwrite of `ws` is
                # safely ordered after this copy completes. `.add_()`
                # promotes fp32→fp64 when the accumulator is fp64 (wide-d
                # roles), so a single workspace dtype covers both tracks.
                torch.mm(flat_gpu.T, flat_gpu, out=ws)
                xtx[name].add_(ws.cpu())
            return hook

        # CPU fallback — identical semantics, just slower for big d.
        # Match the flat activation dtype to the accumulator's dtype so
        # `addmm_` works (it does not do dtype promotion). For fp64
        # accumulators this also means the matmul itself runs in fp64, which
        # on CPU is only ~2× slower than fp32 and is numerically stricter.
        dest_dtype = xtx[name].dtype
        def hook(mod, inputs):
            x = inputs[0].detach()
            flat_cpu = x.reshape(-1, x.shape[-1]).to(device="cpu", dtype=dest_dtype)
            m = _attn_mask_holder[0]
            if m is not None:
                flat_cpu = flat_cpu * m.reshape(-1, 1).to(dtype=dest_dtype,
                                                           device="cpu")
            xtx[name].addmm_(flat_cpu.T, flat_cpu)
        return hook

    def bwd_hook_factory(name, d_out):
        ggt[name] = ggt_store.alloc(name, d_out)

        # GPU-resident ggt accumulator — symmetric to the forward path.
        if ggt[name].is_cuda:
            gram = ggt[name]
            gram_dtype = gram.dtype
            def hook(mod, grad_input, grad_output):
                g = grad_output[0].detach()
                flat = g.reshape(-1, g.shape[-1]).to(
                    device="cuda", dtype=gram_dtype, non_blocking=True)
                m = _attn_mask_holder[0]
                if m is not None:
                    mask_flat = m.reshape(-1, 1).to(device="cuda",
                                                     dtype=gram_dtype,
                                                     non_blocking=True)
                    flat = flat * mask_flat
                gram.addmm_(flat.T, flat)
            return hook

        ws = ggt_store.get_gpu_workspace(d_out)

        if ws is not None:
            def hook(mod, grad_input, grad_output):
                g = grad_output[0].detach()
                flat = g.reshape(-1, g.shape[-1])
                flat_gpu = flat.to(device="cuda", dtype=torch.float32, non_blocking=True)
                m = _attn_mask_holder[0]
                if m is not None:
                    mask_flat = m.reshape(-1, 1).to(device="cuda",
                                                     dtype=torch.float32,
                                                     non_blocking=True)
                    flat_gpu = flat_gpu * mask_flat
                torch.mm(flat_gpu.T, flat_gpu, out=ws)
                ggt[name].add_(ws.cpu())
            return hook

        dest_dtype = ggt[name].dtype
        def hook(mod, grad_input, grad_output):
            g = grad_output[0].detach()
            flat_cpu = g.reshape(-1, g.shape[-1]).to(device="cpu", dtype=dest_dtype)
            m = _attn_mask_holder[0]
            if m is not None:
                flat_cpu = flat_cpu * m.reshape(-1, 1).to(dtype=dest_dtype,
                                                           device="cpu")
            ggt[name].addmm_(flat_cpu.T, flat_cpu)
        return hook

    model.eval()
    n_passes = len(role_groups)
    for pass_idx, pass_roles in enumerate(role_groups):
        pass_groups = find_layers(model, pass_roles)
        hooks = []
        for role, entries in pass_groups.items():
            is_streamed = role in streaming_roles
            for _, name, mod in entries:
                if is_streamed:
                    sc.register(name)
                    hooks.append(mod.register_forward_pre_hook(
                        sc.capture_hook(name)))
                else:
                    hooks.append(mod.register_forward_pre_hook(
                        fwd_hook_factory(name, mod.in_features)))
                # ggt is only needed for LS-refit, which streaming doesn't
                # support yet — we raised earlier if both were requested.
                if need_grad and not is_streamed:
                    hooks.append(mod.register_full_backward_hook(
                        bwd_hook_factory(name, mod.out_features)))

        # Report accumulator precision per role so the fp64 promotion is
        # visible in logs. Grouped by (role, d, dtype) since all layers of a
        # role share the same d_in → same dtype. Streaming roles report
        # dtype as "stream/fp64" since the accumulator isn't allocated
        # until finalize() runs.
        precision_summary = []
        for role, entries in pass_groups.items():
            if not entries:
                continue
            _, sample_name, sample_mod = entries[0]
            d_in = sample_mod.in_features
            if role in streaming_roles:
                precision_summary.append(f"{role}(d={d_in},stream->fp64)")
            else:
                dtype = xtx[sample_name].dtype
                precision_summary.append(f"{role}(d={d_in},{str(dtype).replace('torch.','')})")
        if precision_summary:
            print(f"  [xtx] pass {pass_idx+1}/{n_passes} accumulators: "
                  f"{', '.join(precision_summary)}", flush=True)

        desc = (f"calib pass {pass_idx+1}/{n_passes} "
                f"({sum(len(e) for e in pass_groups.values())} modules)")
        # Batched mode (batch_size > 1) tokenizes + pads in groups. At
        # batch_size=1 the old codepath is preserved (no padding, no mask
        # in the hook). Qwen's tokenizer pads on the right by default
        # with the eos_token as pad; attention_mask zeros those tokens
        # for attention, and our hooks zero them for gramian contribution.
        if batch_size < 1:
            batch_size = 1
        n_texts = len(texts)
        batches = [texts[i:i + batch_size] for i in range(0, n_texts, batch_size)]
        for batch_texts in tqdm(batches, desc=desc):
            if batch_size == 1:
                enc = tokenizer(batch_texts[0], return_tensors="pt",
                                truncation=True, max_length=seq_len)
                input_ids = enc.input_ids.to(device)
                attn_mask = None
                _attn_mask_holder[0] = None
            else:
                enc = tokenizer(batch_texts, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=seq_len)
                input_ids = enc.input_ids.to(device)
                attn_mask = enc.attention_mask.to(device)
                _attn_mask_holder[0] = attn_mask
            if input_ids.shape[1] < 8:
                _attn_mask_holder[0] = None
                continue
            if sc is not None:
                sc.note_forward_start()
            if need_grad:
                out = model(input_ids, attention_mask=attn_mask,
                            labels=input_ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    model(input_ids, attention_mask=attn_mask)
            _attn_mask_holder[0] = None  # clear between batches
            if sc is not None:
                sc.note_forward_end()
                # Drain if the NEXT sample would overflow the pinned cap.
                # The collector tracks the largest per-sample growth seen
                # so far, so this is safe starting from sample 2.
                if sc.should_drain():
                    sc.drain(device=device)

        for h in hooks:
            h.remove()

    # Streaming finalize: convert cached activations into fp64 gramians
    # per-layer on GPU and merge into xtx. Runs once, after the last
    # calibration sample goes through.
    if sc is not None:
        streamed = sc.finalize(device=device)
        xtx.update(streamed)

    # GPU-resident accumulators → CPU: decomposition's SVD path needs VRAM
    # headroom. Move gramians to CPU so downstream code sees uniform CPU
    # tensors and GPU workspace is free for `_stable_svd` / sqrt_and_inv.
    xtx_store.consolidate_to_cpu(xtx)
    if ggt_store is not None and ggt is not None:
        ggt_store.consolidate_to_cpu(ggt)

    if need_grad:
        for p in model.parameters():
            p.requires_grad_(False)
    return xtx, ggt, all_groups


# ---------- rank computation ----------

def rank_shared(d_out, d_in, n_layers, target_ratio):
    """Target-ratio rank for a shared-basis window of n_layers same-role weights.
    Paper's formula: k = r * n * d_out * d_in / (d_out + n * d_in)."""
    return max(1, int(target_ratio * n_layers * d_out * d_in / (d_out + n_layers * d_in)))


def rank_permatrix(d_out, d_in, target_ratio):
    """Target-ratio rank for per-matrix low-rank factoring."""
    return max(1, int(target_ratio * d_out * d_in / (d_out + d_in)))


def _whitened_svdvals(W_list, XTX, device="cuda", skip_cholesky_above_d=0):
    """Compute σ(W_stack) where W_stack = cat([W_i @ S_in], dim=1) and
    S_in = sqrt(XTX + eps*I). Used by both the shared-window analyzer
    (len(W_list) == n) and the permatrix analyzer (len(W_list) == 1) so
    both tracks apply the SAME regularized whitening — the water-fill
    allocator was producing pathological results when the two tracks
    used inconsistent spectra (unregularized vs regularized).

    Returns 1D fp64 tensor of singular values in descending order."""
    skip_chol = (XTX.shape[0] > skip_cholesky_above_d
                 if skip_cholesky_above_d > 0 else False)
    S_in, _ = sqrt_and_inv(XTX, device=device, need_inv=False,
                           skip_cholesky=skip_chol)
    # sqrt_and_inv may fall back to CPU (cpu-eigh-fp64 terminal tier) for
    # ill-conditioned gramians, so pin downstream ops to S_in's actual
    # device, not the requested one.
    work_device = S_in.device
    work_dtype = S_in.dtype
    parts = [W.to(device=work_device, dtype=work_dtype) @ S_in for W in W_list]
    W_stack = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    del parts, S_in
    if not torch.isfinite(W_stack).all():
        W_stack = torch.nan_to_num(W_stack, nan=0.0, posinf=0.0, neginf=0.0)
    # svdvals skips U and Vt — ~30% faster than full SVD.
    try:
        sigma = torch.linalg.svdvals(W_stack)
    except RuntimeError:
        sigma = torch.linalg.svdvals(W_stack.cpu())
    return sigma.to(torch.float64).cpu()


def _analyze_spectrum_permatrix(W, XTX, device="cuda",
                                skip_cholesky_above_d=0,
                                eps_ratio=1e-5):
    """σ(W @ S_in) via direct eigh, no sqrt_and_inv needed.

    Identity: σ²(W @ S_in) = eigenvalues of W @ (S_in S_in^T) @ W^T
                          = W @ (XTX + eps*I) @ W^T
                          = W@XTX@W^T + eps * W@W^T

    For permatrix roles where d_out < d_in (e.g. down_proj at 7B:
    3584 < 18944), this computes eigh on a d_out × d_out matrix in
    well under a second — vs sqrt_and_inv(XTX) at d_in=18944 which
    ladders through multiple tiers and can take minutes per call on
    rank-deficient gramians. Regularization matches sqrt_and_inv's
    default (`eps_ratio * mean_diag`), so the resulting spectrum is
    consistent with what the shared-window analyzer sees via
    sqrt_and_inv + svdvals. Only the spectrum tail near-or-below eps
    differs between paths, and water-fill never allocates rank there."""
    W64 = W.to(device=device, dtype=torch.float64)
    XTX64 = XTX.to(device=device, dtype=torch.float64)
    mean_diag = XTX64.diag().abs().mean().item()
    eps = eps_ratio * mean_diag + 1e-8
    # W@XTX@W^T is d_out × d_out (cheap when d_out < d_in). The eps·W@W^T
    # term absorbs the `sqrt(XTX + eps*I)` regularization without ever
    # forming S_in.
    M = W64 @ XTX64 @ W64.T + eps * (W64 @ W64.T)
    try:
        evals = torch.linalg.eigvalsh(M)
    except RuntimeError:
        evals = torch.linalg.eigvalsh(M.cpu())
    evals = evals.flip(0).clamp(min=0.0)
    return evals.sqrt().to(torch.float64).cpu()


def _analyze_spectrum_shared_window(W_list, xtx_list, device="cuda",
                                    skip_cholesky_above_d=0):
    """σ(W_stack) for a shared window via the unified helper."""
    XTX_joint = sum(xtx_list)
    return _whitened_svdvals(W_list, XTX_joint, device=device,
                             skip_cholesky_above_d=skip_cholesky_above_d)


def allocate_ranks_waterfill(specs, total_budget_elements,
                             min_rank_frac=0.75, log=True):
    """Greedy rank allocation across matrices by marginal utility.

    specs: dict keyed by matrix id, each value a dict with:
      - sigma: 1D descending torch.Tensor of singular values
      - cost: int, bytes (elements) added per unit rank
      - max_rank: int, cap from matrix geometry
      - baseline_rank: int, the rank the global formula would use
    total_budget_elements: sum over matrices of baseline_rank * cost
    min_rank_frac: floor each matrix at `min_rank_frac × baseline`. This
      prevents the allocator from reducing any single matrix's rank below
      a safe fraction — water-fill on sum-of-σ² is a proxy for total
      reconstruction error, which isn't a reliable proxy for PPL when any
      one matrix is decimated. Empirically 0.5 preserves per-layer
      coherence while still letting water-fill redistribute ~50% of
      total budget where it's most useful.

    Returns {id: rank} with sum(rank[i] * cost[i]) ≤ total_budget_elements.

    Strategy: classic water-filling. Start each matrix at its floor. At
    each step, add 1 rank to the matrix maximizing (σ_{r_i}² / cost_i) —
    biggest error reduction per byte spent. Uses a heap for O(log N)
    lookup. Converges in O(total_ranks_added × log N)."""
    import heapq
    ids = list(specs.keys())
    ranks = {mid: max(1, min(
        int(min_rank_frac * specs[mid]["baseline_rank"]),
        specs[mid]["max_rank"])) for mid in ids}
    spent = sum(ranks[mid] * specs[mid]["cost"] for mid in ids)
    remaining = max(0, total_budget_elements - spent)

    # Max-heap keyed by -utility so Python's min-heap gives max.
    heap: list = []
    for mid in ids:
        spec = specs[mid]
        if ranks[mid] < spec["max_rank"]:
            sigma = spec["sigma"]
            if ranks[mid] < sigma.numel():
                u = float(sigma[ranks[mid]]) ** 2 / max(1, spec["cost"])
                heapq.heappush(heap, (-u, mid))

    while remaining > 0 and heap:
        neg_u, mid = heapq.heappop(heap)
        spec = specs[mid]
        if ranks[mid] >= spec["max_rank"]:
            continue
        cost = spec["cost"]
        if remaining < cost:
            break
        ranks[mid] += 1
        remaining -= cost
        if ranks[mid] < spec["max_rank"]:
            sigma = spec["sigma"]
            if ranks[mid] < sigma.numel():
                u = float(sigma[ranks[mid]]) ** 2 / max(1, cost)
                heapq.heappush(heap, (-u, mid))

    if log:
        bumped = sum(1 for mid in ids if ranks[mid] > specs[mid]["baseline_rank"])
        cut = sum(1 for mid in ids if ranks[mid] < specs[mid]["baseline_rank"])
        print(f"  [rank-alloc] {bumped} matrices gained rank, "
              f"{cut} matrices lost rank, {len(ids)-bumped-cut} unchanged; "
              f"{remaining} elements unused of {total_budget_elements}",
              flush=True)
    return ranks


# ---------- decomposition ----------

def decompose_shared_window(W_list, xtx_list, ggt_list, rank, refit=True, device="cpu",
                             skip_cholesky_above_d=0):
    """Window decomposition.

    Paper's method (refit=False): single SVD of horizontally concatenated
    activation-weighted stack; per-layer coeffs are the corresponding slice
    of U_k^T @ W_i.

    Our method (refit=True): same shared basis U_k, but per-layer coefficients
    are refit via closed-form weighted LS with the output-gradient covariance
    (balanced-truncation style). This is the OBS-adjacent step that the paper
    doesn't do.
    """
    n = len(W_list)
    d_out, d_in = W_list[0].shape

    # Sum XTX across the window to get the joint calibration covariance.
    XTX_joint = sum(xtx_list)
    skip_chol = (XTX_joint.shape[0] > skip_cholesky_above_d
                 if skip_cholesky_above_d > 0 else False)
    S_in, _ = sqrt_and_inv(XTX_joint, device=device, need_inv=False,
                           skip_cholesky=skip_chol)

    # Stack activation-weighted matrices horizontally and SVD.
    # Work in fp32 (S_in is fp32 from the Cholesky path); fp32 SVD is stable
    # for these shapes and ~2x faster than fp64.
    work_dtype = S_in.dtype
    weighted = [W.to(device=device, dtype=work_dtype) @ S_in for W in W_list]
    W_stack = torch.cat(weighted, dim=1)                   # [d_out, n*d_in]

    # Sanity check: NaN in W_stack → cuSolver raises CUSOLVER_STATUS_INVALID_VALUE.
    # Upstream source is usually S_in having NaN from a near-singular gramian.
    if not torch.isfinite(W_stack).all():
        print(f"  [warn] W_stack has non-finite values (likely from S_in); "
              f"clamping and retrying")
        W_stack = torch.nan_to_num(W_stack, nan=0.0, posinf=0.0, neginf=0.0)

    # Partial SVD: we only ever take the top-rank U columns below, so
    # computing the remaining (min_dim - rank) singular vectors is pure
    # waste. _stable_svd_topk falls back to full SVD when rank is close
    # to full, preserving correctness at r ≈ 1.0.
    U, _, _ = _stable_svd_topk(W_stack, rank)
    B = U[:, :rank].contiguous()                           # [d_out, rank]
    # SVD may have fallen back to CPU. Lock all subsequent ops to B's device.
    work_device = B.device
    del W_stack, weighted

    coeffs = []
    for i, W_i in enumerate(W_list):
        W_i_t = W_i.to(device=work_device, dtype=work_dtype)
        if refit and ggt_list is not None:
            # Balanced-weighted LS refit: A_i = (B^T S_out^T S_out B)^-1 B^T S_out^T S_out W_i
            ggt_skip = (ggt_list[i].shape[0] > skip_cholesky_above_d
                        if skip_cholesky_above_d > 0 else False)
            S_out_i, _ = sqrt_and_inv(ggt_list[i], device=work_device, need_inv=False,
                                       skip_cholesky=ggt_skip)
            M = S_out_i.to(device=work_device, dtype=work_dtype) @ B
            gram = M.T @ M                                 # [rank, rank]
            rhs = M.T @ (S_out_i.to(device=work_device, dtype=work_dtype) @ W_i_t)
            A_i = torch.linalg.solve(gram, rhs)
            del S_out_i, M, gram, rhs
        else:
            # Plain projection: A_i = B^T W_i (paper's method after un-whitening)
            A_i = B.T @ W_i_t
        coeffs.append(A_i.to("cpu"))
        del W_i_t

    return B.to("cpu"), coeffs


def decompose_permatrix(W, XTX, GGT, rank, device="cpu", skip_cholesky_above_d=0):
    """Per-matrix balanced truncation. If GGT is None, fall back to input-only
    ASVD (no output-gradient weighting) — cheaper calibration, still respects
    activation distribution.
    """
    skip_in = (XTX.shape[0] > skip_cholesky_above_d
               if skip_cholesky_above_d > 0 else False)
    S_in, S_in_inv = sqrt_and_inv(XTX, device=device, skip_cholesky=skip_in)
    W64 = W.to(device=device, dtype=torch.float64)

    if GGT is None:
        # ASVD path: project onto top-r output directions of (W @ S_in).
        U, _, _ = _stable_svd_topk(W64 @ S_in, rank)
        r = min(rank, U.shape[1])
        U_r = U[:, :r]
        W_r = U_r @ (U_r.T @ W64)
        # Return U and V separately: U_r as "U", (U_r^T @ W) as "V"
        U_out = U_r.clone()
        V_out = U_r.T @ W64
        return U_out.cpu(), V_out.cpu()

    # Balanced truncation with both input and output weighting
    skip_out = (GGT.shape[0] > skip_cholesky_above_d
                if skip_cholesky_above_d > 0 else False)
    S_out, S_out_inv = sqrt_and_inv(GGT, device=device, skip_cholesky=skip_out)
    M = S_out @ W64 @ S_in
    U, sigma, Vt = _stable_svd_topk(M, rank)
    work = U.device
    if work != M.device:
        S_in_inv = S_in_inv.to(work)
        S_out_inv = S_out_inv.to(work)
    r = min(rank, sigma.numel())
    U_r = S_out_inv @ U[:, :r]
    V_r = sigma[:r].unsqueeze(1) * (Vt[:r, :] @ S_in_inv)
    return U_r.cpu(), V_r.cpu()


# ---------- apply to model ----------

# ---------- checkpointing ----------
#
# These long 3B+ runs were painful to restart from scratch when we tweak
# downstream code. Split into resumable stages:
#   checkpoint/calibration.pt         — {xtx: {name: tensor}, ggt: {name: tensor}}
#   checkpoint/factors_shared.pt      — full factors dict for shared roles
#   checkpoint/factors_permatrix.pt   — full factors dict for permatrix roles
#   checkpoint/completed.json         — list of completed phases
# Phases complete monotonically: calibration → factors_shared → factors_permatrix.
# On re-run, main() checks what's already done and resumes.


def _ckpt_manifest_path(ckpt_dir):
    return Path(ckpt_dir) / "completed.json"


def checkpoint_has(ckpt_dir, phase):
    if ckpt_dir is None:
        return False
    mp = _ckpt_manifest_path(ckpt_dir)
    if not mp.exists():
        return False
    return phase in json.loads(mp.read_text())


def checkpoint_save(ckpt_dir, phase, data):
    if ckpt_dir is None:
        return
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(data, p / f"{phase}.pt")
    mp = _ckpt_manifest_path(p)
    manifest = json.loads(mp.read_text()) if mp.exists() else []
    if phase not in manifest:
        manifest.append(phase)
        mp.write_text(json.dumps(manifest, indent=2))
    print(f"  [ckpt] saved {phase}", flush=True)


def checkpoint_load(ckpt_dir, phase):
    if ckpt_dir is None:
        return None
    p = Path(ckpt_dir) / f"{phase}.pt"
    if not p.exists():
        return None
    print(f"  [ckpt] loading {phase}", flush=True)
    return torch.load(p, weights_only=False, map_location="cpu")


def _gpu_mem_str():
    if not torch.cuda.is_available():
        return ""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    return f"[gpu {used:.1f}/{total/1024**3:.1f}GB]"


def _weight_to_cpu(mod):
    """Safely fetch a module's weight as a CPU tensor.

    With accelerate's device_map offloading, a module's parameters can live
    on the `meta` device (pure metadata, no data). `mod.weight.data.cpu()`
    raises NotImplementedError on those. `mod.state_dict()` triggers the
    accelerate hooks that materialize the weights, so we fall back to it.
    """
    w = mod.weight
    if w.device.type == "meta":
        return mod.state_dict()["weight"].cpu()
    return w.data.cpu()


def _install_weight(mod, new_weight):
    """Write `new_weight` into `mod.weight`, coping with meta-device params.

    In-place `mod.weight.data.copy_(...)` fails when the parameter is a
    meta placeholder (no storage). In that case we swap in a fresh
    Parameter so the module has a real weight backing the forward pass.
    Non-meta params use the normal in-place copy so we don't perturb
    accelerate's offload hooks.
    """
    if mod.weight.device.type == "meta":
        mod.weight = torch.nn.Parameter(new_weight.detach(), requires_grad=False)
    else:
        mod.weight.data.copy_(new_weight.to(mod.weight.device))


def compute_factors(model, xtx, ggt, groups_shared, groups_permatrix,
                    window_size, target_ratio, device, refit=True,
                    skip_cholesky_above_d=0,
                    rank_alloc="per-matrix",
                    factor_output_dir=None,
                    materialize_dtype=torch.float16):
    """Compute all decomposition factors WITHOUT modifying the model.

    Returns a dict:
      {
        "shared": {
          role: {
            "d_out": ..., "d_in": ..., "windows": [
              {"window_id": w, "layers": [i, i+1, ...], "rank": r,
               "basis": Tensor[d_out, r], "coeffs": [Tensor[r, d_in], ...]}
            ]
          }
        },
        "permatrix": {
          role: {
            "d_out": ..., "d_in": ..., "rank": r, "layers": [
              {"layer": i, "U": Tensor[d_out, r], "V": Tensor[r, d_in]}
            ]
          }
        }
      }
    All factor tensors are CPU fp32 at this stage; caller casts as needed.
    """
    result = {"shared": {}, "permatrix": {}}
    t_start = time.time()

    # Streaming factor output: instead of collecting every U/V/basis/coeff
    # into `result` (~22 GB peak for 7B at r=0.8), write each factor to a
    # small per-window/per-layer .pt file as soon as it's computed and
    # install the reconstructed dense weight into `model` in place. The
    # result dict then holds only metadata + file paths. save_factored
    # assembles the final single safetensors from those files.
    streaming = factor_output_dir is not None
    if streaming:
        factor_output_dir = Path(factor_output_dir)
        factor_output_dir.mkdir(parents=True, exist_ok=True)
        # Clean any stale streamed tensors from a previous run's tmpdir
        # so we don't mix old factors into the new safetensors assembly.
        for old in factor_output_dir.glob("*.pt"):
            try:
                os.unlink(old)
            except OSError:
                pass
        # Flag downstream that model.weight is already holding reconstructed
        # factored weights — materialize_factors_in_place should no-op.
        result["already_materialized"] = True
        result["streamed_tmpdir"] = str(factor_output_dir)

    def _stream_shared(role, w_idx, layer_indices, mods, B_cpu, coeffs_cpu, rank, mode):
        """Inline-install reconstructed weights into the model, persist
        factors to per-window .pt files, free locals. Returns the metadata
        entry for result['shared'][role]['windows']."""
        for i_local, layer_i in enumerate(layer_indices):
            W_recon = (B_cpu @ coeffs_cpu[i_local]).to(materialize_dtype)
            _install_weight(mods[layer_i], W_recon)
            del W_recon
        safe_role = role.replace(".", "_")
        basis_path = factor_output_dir / f"shared_{safe_role}_w{w_idx:03d}_basis.pt"
        torch.save(B_cpu.contiguous(), str(basis_path))
        coeffs_paths = []
        for i_local, layer_i in enumerate(layer_indices):
            cp = (factor_output_dir /
                  f"shared_{safe_role}_w{w_idx:03d}_coeffs_{layer_i:03d}.pt")
            torch.save(coeffs_cpu[i_local].contiguous(), str(cp))
            coeffs_paths.append(str(cp))
        return {
            "window_id": w_idx,
            "layers": list(layer_indices),
            "rank": rank,
            "basis_path": str(basis_path),
            "coeffs_paths": coeffs_paths,
            "mode": mode,
        }

    def _stream_permatrix(role, layer_i, mod, U_cpu, V_cpu, rank):
        """Inline-install U @ V into the model, persist U/V to .pt files,
        return metadata entry."""
        W_recon = (U_cpu @ V_cpu).to(materialize_dtype)
        _install_weight(mod, W_recon)
        del W_recon
        safe_role = role.replace(".", "_")
        u_path = factor_output_dir / f"permatrix_{safe_role}_layer{layer_i:03d}_U.pt"
        v_path = factor_output_dir / f"permatrix_{safe_role}_layer{layer_i:03d}_V.pt"
        torch.save(U_cpu.contiguous(), str(u_path))
        torch.save(V_cpu.contiguous(), str(v_path))
        return {"layer": layer_i, "rank": rank,
                "U_path": str(u_path), "V_path": str(v_path)}

    # Phase 1+2: per-matrix rank allocation via water-filling. We compute
    # each matrix's singular-value spectrum (cheap — eigh of a small
    # gramian), then greedily assign rank under the same total byte budget
    # the global formula would use. The existing decomposition loop below
    # consumes `allocated_ranks` when set, falling back to the formula
    # otherwise. Global mode is preserved for regression comparison.
    allocated_ranks: dict | None = None
    if rank_alloc == "per-matrix":
        specs: dict = {}
        total_budget = 0
        have_ggt_global = ggt is not None
        for role, entries in groups_shared.items():
            _, names, mods = zip(*entries)
            L = len(mods)
            xtx_list_all = [xtx[n] for n in names]
            d_out, d_in = _weight_to_cpu(mods[0]).shape
            windows = make_windows(L, window_size)
            for w_idx, layer_indices in enumerate(windows):
                n = len(layer_indices)
                if n == 1:
                    # Singleton shared window falls through to the
                    # permatrix-fallback code path. Allocate like a
                    # permatrix layer: spectrum from W @ XTX @ W^T.
                    i = layer_indices[0]
                    W_i = _weight_to_cpu(mods[i])
                    sigma = _analyze_spectrum_permatrix(
                        W_i, xtx_list_all[i], device=device,
                        skip_cholesky_above_d=skip_cholesky_above_d)
                    cost = d_out + d_in
                    r_base = rank_permatrix(d_out, d_in, target_ratio)
                    specs[("shared", role, w_idx)] = {
                        "sigma": sigma, "cost": cost,
                        "max_rank": min(d_out, d_in),
                        "baseline_rank": r_base,
                    }
                    total_budget += r_base * cost
                else:
                    W_sub = [_weight_to_cpu(mods[i]) for i in layer_indices]
                    xtx_sub = [xtx_list_all[i] for i in layer_indices]
                    sigma = _analyze_spectrum_shared_window(
                        W_sub, xtx_sub, device=device,
                        skip_cholesky_above_d=skip_cholesky_above_d)
                    cost = d_out + n * d_in
                    r_base = rank_shared(d_out, d_in, n, target_ratio)
                    specs[("shared", role, w_idx)] = {
                        "sigma": sigma, "cost": cost,
                        "max_rank": min(d_out, n * d_in),
                        "baseline_rank": r_base,
                    }
                    total_budget += r_base * cost
        for role, entries in groups_permatrix.items():
            _, names, mods = zip(*entries)
            d_out, d_in = _weight_to_cpu(mods[0]).shape
            r_base = rank_permatrix(d_out, d_in, target_ratio)
            cost = d_out + d_in
            for i, (name, mod) in enumerate(zip(names, mods)):
                W_i = _weight_to_cpu(mod)
                sigma = _analyze_spectrum_permatrix(
                    W_i, xtx[name], device=device,
                    skip_cholesky_above_d=skip_cholesky_above_d)
                specs[("permatrix", role, i)] = {
                    "sigma": sigma, "cost": cost,
                    "max_rank": min(d_out, d_in),
                    "baseline_rank": r_base,
                }
                total_budget += r_base * cost
        print(f"[{time.time()-t_start:7.1f}s] rank-alloc phase 1 done "
              f"({len(specs)} matrices, total_budget={total_budget:.2e} "
              f"elements)", flush=True)
        allocated_ranks = allocate_ranks_waterfill(specs, total_budget)
        # Emit a per-role summary so we can see where rank moved.
        from collections import defaultdict as _dd
        role_summary = _dd(lambda: {"n": 0, "base_sum": 0, "alloc_sum": 0})
        for (mtype, mrole, _idx), spec in specs.items():
            r = role_summary[f"{mtype}/{mrole}"]
            r["n"] += 1
            r["base_sum"] += spec["baseline_rank"]
            r["alloc_sum"] += allocated_ranks[(mtype, mrole, _idx)]
        for key in sorted(role_summary.keys()):
            r = role_summary[key]
            avg_base = r["base_sum"] / r["n"]
            avg_alloc = r["alloc_sum"] / r["n"]
            delta = (avg_alloc - avg_base) / avg_base * 100 if avg_base else 0
            print(f"  [rank-alloc] {key}: avg rank {avg_base:.0f} -> "
                  f"{avg_alloc:.0f} ({delta:+.1f}%)", flush=True)

    for role, entries in groups_shared.items():
        _, names, mods = zip(*entries)
        L = len(mods)
        W_list_all = [_weight_to_cpu(m) for m in mods]
        xtx_list_all = [xtx[n] for n in names]
        ggt_list_all = [ggt[n] for n in names] if ggt else None
        d_out, d_in = W_list_all[0].shape
        windows = make_windows(L, window_size)
        print(f"[{time.time()-t_start:7.1f}s] shared {role}: "
              f"{d_out}x{d_in} x{L} -> {len(windows)} windows {_gpu_mem_str()}",
              flush=True)

        role_windows = []
        for w_idx, layer_indices in enumerate(windows):
            n = len(layer_indices)
            if n == 1:
                # Trailing short window — emit a synthetic 1-layer "window".
                # If GGT is available use balanced truncation; else ASVD.
                i = layer_indices[0]
                r = (allocated_ranks[("shared", role, w_idx)]
                     if allocated_ranks is not None
                     else rank_permatrix(d_out, d_in, target_ratio))
                have_ggt = ggt_list_all is not None
                skip_in = (xtx_list_all[i].shape[0] > skip_cholesky_above_d
                            if skip_cholesky_above_d > 0 else False)
                S_in, S_in_inv = sqrt_and_inv(xtx_list_all[i], device=device,
                                              need_inv=have_ggt,
                                              skip_cholesky=skip_in)
                W_t = W_list_all[i].to(device=S_in.device, dtype=S_in.dtype)
                if have_ggt:
                    skip_out = (ggt_list_all[i].shape[0] > skip_cholesky_above_d
                                 if skip_cholesky_above_d > 0 else False)
                    S_out, S_out_inv = sqrt_and_inv(ggt_list_all[i], device=device,
                                                    need_inv=True,
                                                    skip_cholesky=skip_out)
                    S_out = S_out.to(dtype=S_in.dtype)
                    S_out_inv = S_out_inv.to(dtype=S_in.dtype)
                    M = S_out @ W_t @ S_in
                    U, sigma, Vt = _stable_svd_topk(M, r)
                    if U.device != M.device:
                        S_in_inv = S_in_inv.to(U.device)
                        S_out_inv = S_out_inv.to(U.device)
                    r_eff = min(r, sigma.numel())
                    U_r = S_out_inv @ U[:, :r_eff]
                    V_r = sigma[:r_eff].unsqueeze(1) * (Vt[:r_eff, :] @ S_in_inv)
                else:
                    # ASVD path
                    M = W_t @ S_in
                    U, _, _ = _stable_svd_topk(M, r)
                    r_eff = min(r, U.shape[1])
                    U_r = U[:, :r_eff]
                    V_r = U_r.T @ W_t.to(U_r.device)
                U_r_cpu = U_r.cpu().to(torch.float32)
                V_r_cpu = V_r.cpu().to(torch.float32)
                if streaming:
                    entry = _stream_shared(role, w_idx, [i], mods,
                                           U_r_cpu, [V_r_cpu], r_eff,
                                           "permatrix_fallback")
                    role_windows.append(entry)
                    del U_r, V_r, U_r_cpu, V_r_cpu
                else:
                    role_windows.append({
                        "window_id": w_idx,
                        "layers": [i],
                        "rank": r_eff,
                        "basis": U_r_cpu,
                        "coeffs": [V_r_cpu],
                        "mode": "permatrix_fallback",
                    })
                continue

            r = (allocated_ranks[("shared", role, w_idx)]
                 if allocated_ranks is not None
                 else rank_shared(d_out, d_in, n, target_ratio))
            W_list = [W_list_all[i] for i in layer_indices]
            xtx_list = [xtx_list_all[i] for i in layer_indices]
            ggt_list_w = [ggt_list_all[i] for i in layer_indices] if ggt_list_all else None
            t_win = time.time()
            B, coeffs = decompose_shared_window(W_list, xtx_list, ggt_list_w, r,
                                                refit=refit, device=device,
                                                skip_cholesky_above_d=skip_cholesky_above_d)
            print(f"[{time.time()-t_start:7.1f}s]   win {w_idx+1}/{len(windows)} "
                  f"layers={layer_indices} r={r} ({time.time()-t_win:.1f}s) "
                  f"{_gpu_mem_str()}", flush=True)
            B_cpu = B.to(torch.float32)
            coeffs_cpu = [c.to(torch.float32) for c in coeffs]
            if streaming:
                entry = _stream_shared(role, w_idx, layer_indices, mods,
                                       B_cpu, coeffs_cpu, r, "shared")
                role_windows.append(entry)
                del B, coeffs, B_cpu, coeffs_cpu
            else:
                role_windows.append({
                    "window_id": w_idx,
                    "layers": list(layer_indices),
                    "rank": r,
                    "basis": B_cpu,
                    "coeffs": coeffs_cpu,
                    "mode": "shared",
                })

        result["shared"][role] = {"L": L, "d_out": d_out, "d_in": d_in,
                                  "windows": role_windows}

    for role, entries in groups_permatrix.items():
        _, names, mods = zip(*entries)
        d_out, d_in = mods[0].weight.shape
        r_default = rank_permatrix(d_out, d_in, target_ratio)
        if allocated_ranks is not None:
            role_ranks = [allocated_ranks[("permatrix", role, i)]
                          for i in range(len(mods))]
            print(f"[{time.time()-t_start:7.1f}s] permatrix {role}: "
                  f"{d_out}x{d_in} x{len(mods)} "
                  f"rank range={min(role_ranks)}-{max(role_ranks)} "
                  f"(base={r_default}) {_gpu_mem_str()}", flush=True)
        else:
            role_ranks = [r_default] * len(mods)
            print(f"[{time.time()-t_start:7.1f}s] permatrix {role}: "
                  f"{d_out}x{d_in} x{len(mods)} rank={r_default} "
                  f"{_gpu_mem_str()}", flush=True)
        layer_factors = []
        t_role_start = time.time()
        for i, (name, mod) in enumerate(zip(names, mods)):
            t_layer = time.time()
            r = role_ranks[i]
            have_ggt = ggt is not None
            skip_in = (xtx[name].shape[0] > skip_cholesky_above_d
                        if skip_cholesky_above_d > 0 else False)
            S_in, S_in_inv = sqrt_and_inv(xtx[name], device=device,
                                          need_inv=have_ggt,
                                          skip_cholesky=skip_in)
            W_t = _weight_to_cpu(mod).to(device=S_in.device, dtype=S_in.dtype)
            if have_ggt:
                skip_out = (ggt[name].shape[0] > skip_cholesky_above_d
                             if skip_cholesky_above_d > 0 else False)
                S_out, S_out_inv = sqrt_and_inv(ggt[name], device=device,
                                                need_inv=True,
                                                skip_cholesky=skip_out)
                S_out = S_out.to(dtype=S_in.dtype)
                S_out_inv = S_out_inv.to(dtype=S_in.dtype)
                M = S_out @ W_t @ S_in
                U, sigma, Vt = _stable_svd_topk(M, r)
                if U.device != M.device:
                    S_in_inv = S_in_inv.to(U.device)
                    S_out_inv = S_out_inv.to(U.device)
                r_eff = min(r, sigma.numel())
                U_r = S_out_inv @ U[:, :r_eff]
                V_r = sigma[:r_eff].unsqueeze(1) * (Vt[:r_eff, :] @ S_in_inv)
            else:
                # ASVD path — no output gradient weighting
                M = W_t @ S_in
                U, _, _ = _stable_svd_topk(M, r)
                r_eff = min(r, U.shape[1])
                U_r = U[:, :r_eff]
                V_r = U_r.T @ W_t.to(U_r.device)
            U_cpu = U_r.cpu().to(torch.float32)
            V_cpu = V_r.cpu().to(torch.float32)
            if streaming:
                layer_factors.append(_stream_permatrix(role, i, mod,
                                                        U_cpu, V_cpu, r_eff))
                del U_r, V_r, U_cpu, V_cpu
            else:
                layer_factors.append({
                    "layer": i,
                    "rank": r_eff,
                    "U": U_cpu,
                    "V": V_cpu,
                })
            if (i + 1) % 8 == 0 or i == len(mods) - 1:
                print(f"[{time.time()-t_start:7.1f}s]   {role} layer "
                      f"{i+1}/{len(mods)} done r={r_eff} "
                      f"({time.time()-t_layer:.1f}s last) "
                      f"{_gpu_mem_str()}", flush=True)
        # Role-level "rank" is the baseline (what the global formula
        # would have assigned). Per-layer allocated rank lives on each
        # layer_factors entry for readers that care.
        result["permatrix"][role] = {"L": len(mods), "d_out": d_out, "d_in": d_in,
                                     "rank": r_default, "layers": layer_factors}

    return result


def materialize_factors_in_place(model, factors, groups_shared, groups_permatrix,
                                  dtype=torch.float16):
    """Apply factor reconstruction back into model.weight tensors in place.
    Used for PPL evaluation of the factored model before GGUF emission.
    Returns a summary report for logging.

    No-op when `compute_factors` streamed its output (factors dict has
    `already_materialized=True`). Weights were installed inline during
    decomposition, so the model is already at the factored state."""
    if factors.get("already_materialized"):
        # Emit a matching report so callers that inspect it stay happy.
        report = {}
        for role, info in factors["shared"].items():
            role_report = []
            for win in info["windows"]:
                role_report.append({"window": win["window_id"],
                                    "layers": win["layers"],
                                    "mode": win["mode"],
                                    "rank": win["rank"]})
            report[role] = role_report
        for role, info in factors["permatrix"].items():
            role_report = []
            for lf in info["layers"]:
                role_report.append({"layer": lf["layer"],
                                    "rank": lf.get("rank", info["rank"])})
            report[role] = role_report
        return report

    report = {}

    for role, entries in groups_shared.items():
        _, _, mods = zip(*entries)
        role_report = []
        for win in factors["shared"][role]["windows"]:
            B = win["basis"]
            for i_local, layer_i in enumerate(win["layers"]):
                W_recon = (B @ win["coeffs"][i_local]).to(dtype)
                _install_weight(mods[layer_i], W_recon)
            role_report.append({"window": win["window_id"], "layers": win["layers"],
                                "mode": win["mode"], "rank": win["rank"]})
        report[role] = role_report

    for role, entries in groups_permatrix.items():
        _, _, mods = zip(*entries)
        role_report = []
        pm_info = factors["permatrix"][role]
        for lf in pm_info["layers"]:
            i = lf["layer"]
            W_r = (lf["U"] @ lf["V"]).to(dtype)
            _install_weight(mods[i], W_r)
            role_report.append({"layer": i,
                                "rank": lf.get("rank", pm_info["rank"])})
        report[role] = role_report

    return report


# ---------- persistence ----------
#
# Intermediate format: safetensors + JSON manifest. Maps to the GGUF naming
# convention in DESIGN.md §4 but stored more conveniently for Python roundtrip:
#
#   out_dir/
#     manifest.json           model_id, window, target_ratio, role->shape map
#     untouched.safetensors   every weight that wasn't factored (embeddings,
#                             norms, lm_head, unfactored roles)
#     factored.safetensors    all factor tensors with structured names:
#                               shared.{role}.w{W}.basis
#                               shared.{role}.w{W}.coeffs.{LAYER_IDX}
#                               permatrix.{role}.{LAYER_IDX}.U
#                               permatrix.{role}.{LAYER_IDX}.V
#
# This is the interchange format between Phase 1 (Python converter) and Phase 2
# (C++ loader). A thin wrapper in Phase 2 will emit a real GGUF from this data;
# keeping our own format lets us iterate independently of upstream GGUF changes.


def _role_tag(role):
    """Map 'self_attn.q_proj' -> 'attn_q', 'mlp.gate_proj' -> 'ffn_gate' etc.,
    matching llama.cpp's tensor-naming conventions."""
    mapping = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }
    return mapping[role]


def save_factored(model, factors, groups_shared, groups_permatrix,
                   out_dir, model_id, window_size, target_ratio, refit):
    """Write factors + untouched weights to the intermediate format."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Names of every factored linear module, so we can skip them when saving "untouched"
    factored_names = set()
    for role_entries in [groups_shared, groups_permatrix]:
        for entries in role_entries.values():
            for _, name, _ in entries:
                factored_names.add(name + ".weight")

    # Untouched tensors
    untouched = {}
    for name, tensor in model.state_dict().items():
        if name in factored_names:
            continue
        untouched[name] = tensor.detach().cpu().contiguous()
    save_file(untouched, str(out / "untouched.safetensors"))

    # Factored tensors with structured names. When factors came from
    # streaming compute_factors, each tensor lives in a per-window/per-
    # layer .pt file rather than in the dict — load each right before
    # stuffing into `factored_tensors`. Peak RAM during this step is the
    # same as before (safetensors.save_file is atomic), but outside this
    # step we're not holding the 22 GB of factors any more.
    def _fetch(entry, key):
        """Return tensor for `key` from a dict entry; load from .pt when
        streamed."""
        if key in entry:
            return entry[key]
        # Streamed: lookup the matching *_path
        path_key = key + "_path"
        if path_key in entry:
            return torch.load(entry[path_key], map_location="cpu",
                              weights_only=True)
        raise KeyError(f"no tensor or path for '{key}' in entry")

    factored_tensors = {}
    streamed_paths: list[str] = []  # to unlink after safetensors write
    manifest_shared = {}
    for role, info in factors["shared"].items():
        tag = _role_tag(role)
        windows_manifest = []
        for win in info["windows"]:
            key_base = f"shared.{tag}.w{win['window_id']:03d}"
            if "basis" in win:
                factored_tensors[f"{key_base}.basis"] = win["basis"].contiguous()
                coeffs_list = win["coeffs"]
            else:
                factored_tensors[f"{key_base}.basis"] = torch.load(
                    win["basis_path"], map_location="cpu",
                    weights_only=True).contiguous()
                streamed_paths.append(win["basis_path"])
                coeffs_list = [torch.load(p, map_location="cpu",
                                           weights_only=True)
                               for p in win["coeffs_paths"]]
                streamed_paths.extend(win["coeffs_paths"])
            for i_local, layer_i in enumerate(win["layers"]):
                factored_tensors[f"{key_base}.coeffs.{layer_i:03d}"] = \
                    coeffs_list[i_local].contiguous()
            windows_manifest.append({
                "window_id": win["window_id"],
                "layers": win["layers"],
                "rank": win["rank"],
                "mode": win["mode"],
                "basis_key": f"{key_base}.basis",
                "coeffs_keys": [f"{key_base}.coeffs.{li:03d}" for li in win["layers"]],
            })
        manifest_shared[role] = {
            "tag": tag, "d_out": info["d_out"], "d_in": info["d_in"],
            "L": info["L"], "windows": windows_manifest,
        }

    manifest_permatrix = {}
    for role, info in factors["permatrix"].items():
        tag = _role_tag(role)
        layers_manifest = []
        for lf in info["layers"]:
            i = lf["layer"]
            u_key = f"permatrix.{tag}.{i:03d}.U"
            v_key = f"permatrix.{tag}.{i:03d}.V"
            if "U" in lf:
                factored_tensors[u_key] = lf["U"].contiguous()
                factored_tensors[v_key] = lf["V"].contiguous()
            else:
                factored_tensors[u_key] = torch.load(
                    lf["U_path"], map_location="cpu",
                    weights_only=True).contiguous()
                factored_tensors[v_key] = torch.load(
                    lf["V_path"], map_location="cpu",
                    weights_only=True).contiguous()
                streamed_paths.extend([lf["U_path"], lf["V_path"]])
            layers_manifest.append({"layer": i, "U_key": u_key, "V_key": v_key})
        manifest_permatrix[role] = {
            "tag": tag, "d_out": info["d_out"], "d_in": info["d_in"],
            "L": info["L"], "rank": info["rank"], "layers": layers_manifest,
        }

    save_file(factored_tensors, str(out / "factored.safetensors"))
    # With streaming, the .pt tmp files have served their purpose — unlink
    # them and free the ~22 GB of disk staging once safetensors is on disk.
    del factored_tensors
    for p in streamed_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    # Also try to remove the streaming tmpdir if empty
    if factors.get("streamed_tmpdir"):
        try:
            Path(factors["streamed_tmpdir"]).rmdir()
        except OSError:
            pass  # non-empty (unexpected) or already gone

    manifest = {
        "format_version": 1,
        "model_id": model_id,
        "window_size": window_size,
        "target_ratio": target_ratio,
        "refit": refit,
        "shared_roles": manifest_shared,
        "permatrix_roles": manifest_permatrix,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_factored_reconstruct(out_dir):
    """Load factored tensors and reconstruct full dense weights.
    Returns {tensor_name: tensor} ready to stuff into a model.state_dict.
    Used for roundtrip testing — in Phase 3 the runtime loads factors directly.
    """
    from safetensors import safe_open
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())

    state = {}
    with safe_open(str(out / "untouched.safetensors"), framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)

    with safe_open(str(out / "factored.safetensors"), framework="pt") as f:
        for role, info in manifest["shared_roles"].items():
            for win in info["windows"]:
                B = f.get_tensor(win["basis_key"])
                for i_local, layer_i in enumerate(win["layers"]):
                    C = f.get_tensor(win["coeffs_keys"][i_local])
                    W = (B @ C).to(torch.float32)
                    # Reconstruct HF-style tensor name
                    hf_name = f"model.layers.{layer_i}.{role}.weight"
                    state[hf_name] = W
        for role, info in manifest["permatrix_roles"].items():
            for lf in info["layers"]:
                U = f.get_tensor(lf["U_key"])
                V = f.get_tensor(lf["V_key"])
                W = (U @ V).to(torch.float32)
                hf_name = f"model.layers.{lf['layer']}.{role}.weight"
                state[hf_name] = W

    return state, manifest


# ---------- evaluation ----------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, text, device, n_tokens=2048, window=1024):
    """Sliding-window PPL on the concatenated text."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if n_tokens and ids.shape[1] > n_tokens:
        ids = ids[:, :n_tokens]
    nll_sum = 0.0
    n = 0
    prev_end = 0
    max_len = min(window, getattr(model.config, "max_position_embeddings", window))
    for begin in range(0, ids.shape[1], max_len):
        end = min(begin + max_len, ids.shape[1])
        chunk = ids[:, begin:end]
        target = chunk.clone()
        target[:, : max(0, prev_end - begin)] = -100
        out = model(chunk, labels=target)
        valid = (target != -100).sum().item()
        if valid > 0:
            nll_sum += out.loss.item() * valid
            n += valid
        prev_end = end
    return float(torch.exp(torch.tensor(nll_sum / max(n, 1))))


def load_calib_texts(tokenizer, n, min_chars=400):
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


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--target-ratio", type=float, default=1.0,
                   help="target compression ratio; k chosen to hit this. "
                        "Default 1.0 is the lossless-ish validation point "
                        "(rank ≈ dense-equivalent storage, small residual "
                        "from truncation). Production runs typically use "
                        "0.8; r=0.5 is a stress test that catastrophically "
                        "over-compresses at every scale tested.")
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--calib-batch-size", type=int, default=1,
                   help="Samples per forward pass during calibration. "
                        "Batch>1 pads to longest-in-batch and masks padded "
                        "positions out of the gramian accumulation, so the "
                        "numerics are equivalent to batch=1 processing. "
                        "Biggest wall-clock lever on calibration — batch=8 "
                        "typically cuts calibration time 4-6× vs batch=1 "
                        "since the GPU isn't compute-bound at small batch.")
    p.add_argument("--eval-tokens", type=int, default=2048)
    p.add_argument("--no-refit", action="store_true",
                   help="use paper's SVD-slice coeffs (disables LS refit)")
    p.add_argument("--permatrix-only", action="store_true",
                   help="baseline: apply balanced truncation everywhere (no basis sharing)")
    p.add_argument("--out", default="bs_result.json")
    p.add_argument("--save-dir", default=None,
                   help="directory to save factored tensors + manifest (Phase 2a format)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="directory to save/resume intermediate stages (calibration, factors). "
                        "Defaults to {save-dir}/checkpoint if --save-dir is set.")
    p.add_argument("--verify-roundtrip", action="store_true",
                   help="after saving, reload from save-dir and verify weights match")
    p.add_argument("--eigh-device", default="cuda",
                   help="device for gramian eigendecompositions + SVDs "
                        "(cuda preferred; falls back to CPU on cuSolver failures)")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device-map", default=None,
                   help="HuggingFace device_map, e.g. 'auto' for transformers-managed "
                        "GPU+CPU dispatch on models that don't fit entirely in VRAM. "
                        "When set, skips the manual .to(device)/.cpu() moves.")
    p.add_argument("--min-free-ram-gb", type=float, default=0.0,
                   help="Legacy absolute lower bound on free RAM during "
                        "gramian allocation. Effective reserve is the max "
                        "of this and the reserve derived from "
                        "--cpu-ram-max-pct. Leave at 0 to rely on the "
                        "percentage cap alone.")
    p.add_argument("--cpu-ram-max-pct", type=float, default=90.0,
                   help="Max %% of total system RAM this run may hold. "
                        "Auto-backend spills to disk when approaching the "
                        "cap; ram-backend refuses the allocation. Default 90.")
    p.add_argument("--disk-max-pct", type=float, default=90.0,
                   help="Max %% of total disk space on the xtx temp dir's "
                        "drive that memmap allocations may consume. Refuses "
                        "the allocation loudly instead of crashing mid-run "
                        "with OSError(28). Default 90.")
    p.add_argument("--gpu-vram-max-pct", type=float, default=90.0,
                   help="Max %% of total GPU VRAM the calibration workspaces "
                        "may occupy (on top of the model already loaded). "
                        "Workspaces that would cross the cap fall back to "
                        "the CPU addmm hook path. Default 90.")
    p.add_argument("--xtx-backend", default="auto", choices=["auto", "ram", "disk"],
                   help="Where to store d*d calibration gramians. 'ram' matches "
                        "legacy behavior; 'disk' memmaps them to NVMe; 'auto' "
                        "picks per-tensor based on the RAM cap.")
    p.add_argument("--xtx-temp-dir", default=None,
                   help="Directory for disk-backed gramians (default: "
                        "{checkpoint-dir}/xtx or {save-dir}/xtx).")
    p.add_argument("--xtx-gpu-workspace-gb", type=float, default=4.0,
                   help="Absolute per-workspace byte cap (single d*d delta-"
                        "matmul buffer). Complements --gpu-vram-max-pct: a "
                        "workspace must pass BOTH gates. Set to 0 to force "
                        "the CPU addmm path.")
    p.add_argument("--gpu-accum-budget-gb", type=float, default=4.0,
                   help="VRAM budget for GPU-resident gramian accumulators. "
                        "When > 0, fp32 gramians are allocated directly on "
                        "GPU so the hook can do in-place addmm_ without "
                        "per-hook D2H; saves ~40%% of calibration wall time "
                        "on mid-size models where hook path dominates. "
                        "3B shared+o_proj fits under 4 GB; larger models "
                        "spill extras to the CPU path automatically. "
                        "Set to 0 to disable (legacy all-CPU behavior).")
    p.add_argument("--accum-fp64-threshold", type=int, default=4096,
                   help="Allocate gramian accumulators in float64 when d >= "
                        "this value; float32 otherwise. Wide gramians "
                        "(down_proj on 7B+ has d=18944) lose the "
                        "low-eigenvalue tail in fp32 accumulation noise "
                        "across thousands of hook updates, which wrecks the "
                        "Cholesky whitening and the downstream SVD. Pass a "
                        "very large value (e.g. 1000000) to disable fp64.")
    p.add_argument("--calib-passes", default="single",
                   choices=["single", "by-role"],
                   help="single: accumulate all gramians in one forward sweep "
                        "(fastest when RAM fits). by-role: run separate sweeps "
                        "for shared roles, o_proj, and down_proj so only one "
                        "role's gramians co-exist in RAM — enables 14B+ models.")
    p.add_argument("--streaming-roles", default="",
                   help="Comma-separated role suffixes (e.g. 'mlp.down_proj') "
                        "for which the calibration hook caches activations in "
                        "pinned host bf16 instead of live-accumulating a d*d "
                        "gramian. Post-pass, the fp64 gramian is computed on "
                        "GPU in chunks. Use for wide-d roles whose fp64 "
                        "gramian wouldn't fit RAM — eliminates the disk "
                        "memmap thrash seen at 7B down_proj. Empty = legacy "
                        "path everywhere.")
    p.add_argument("--streaming-max-ram-gb", type=float, default=20.0,
                   help="Pinned-host budget for activation caching across all "
                        "streaming roles. Trip this and the run fails fast "
                        "with a clear message. Default 20 GB fits 8 samples "
                        "of 7B down_proj comfortably (17 GB peak).")
    p.add_argument("--no-stream-factors", action="store_true",
                   help="Disable factor-output streaming. By default, "
                        "compute_factors writes each decomposed matrix's "
                        "U/V/basis/coeffs to a per-layer .pt file on disk "
                        "and installs the reconstructed dense weight into "
                        "the model inline — keeping peak RAM during the "
                        "multi-minute decomposition phase below ~1 GB of "
                        "factor staging (vs ~22 GB at 7B r=0.8 without "
                        "streaming). save_factored then assembles the "
                        "final safetensors from those .pt files. Pass "
                        "this flag to force the legacy all-in-RAM path.")
    p.add_argument("--rank-alloc", default="global",
                   choices=["global", "per-matrix"],
                   help="How to assign rank across matrices. 'global' "
                        "(default) uses the target_ratio formula per "
                        "matrix. 'per-matrix' computes each matrix's "
                        "singular-value spectrum and water-fills rank by "
                        "marginal utility (σ²/bytes_per_rank) under the same "
                        "total byte budget. EXPERIMENTAL: helped 0.5B at "
                        "r=0.8 (PPL 162→114) but hurt 7B (22.28→28.32) — "
                        "sum-of-σ² is not a reliable PPL proxy at larger "
                        "scale. Needs a better utility function before "
                        "this is the default.")
    p.add_argument("--skip-cholesky-above-d", type=int, default=15000,
                   help="Skip the fp32/fp64 Cholesky probe tiers in "
                        "sqrt_and_inv for any gramian whose d exceeds this "
                        "value; jump directly to GPU fp64 Cholesky with "
                        "eps×10 regularization (the tier we empirically "
                        "observe succeeds on rank-deficient wide-d "
                        "gramians). Catches 7B+ down_proj (d=18944) and "
                        "leaves 3B (d=11008) and smaller on the full "
                        "ladder. Saves ~5% of 7B decomposition wall time "
                        "by avoiding failed probes. 0 = never skip.")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GGT is only needed for the LS-refit path (our extension) and per-matrix
    # balanced truncation. Paper's Basis Sharing (--no-refit) uses input cov only.
    need_grad = not args.no_refit

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None and args.save_dir:
        ckpt_dir = str(Path(args.save_dir) / "checkpoint")

    print(f"loading {args.model} on {device}, window={args.window}, "
          f"target_ratio={args.target_ratio}, refit={not args.no_refit}")
    print(f"  cuda linalg backend: {_CUDA_LINALG_BACKEND}, "
          f"eigh/svd device: {args.eigh_device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # bf16 for model weights — fp32 is too big for models >0.5B on 12 GB VRAM
    # once we add backward. Gramian accumulators are fp32 regardless (we cast
    # inside the hooks), so calibration statistics stay precise.
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    use_device_map = args.device_map is not None
    if use_device_map:
        print(f"  using device_map='{args.device_map}' (transformers-managed dispatch)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=load_dtype, device_map=args.device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=load_dtype).to(device)
    model.eval()
    # When device_map is in use, the script's `device` variable (derived from
    # torch.cuda.is_available) may not match where the model's inputs should
    # land — e.g. device_map="cpu" wants inputs on CPU even though CUDA is
    # available. Use the model's own device for forward passes in that case.
    input_device = str(model.device) if use_device_map else device

    # Baseline PPL first (untouched weights)
    eval_text = load_eval_text()
    print("baseline PPL")
    t0 = time.time()
    baseline_ppl = evaluate_ppl(model, tokenizer, eval_text, input_device, n_tokens=args.eval_tokens)
    print(f"  = {baseline_ppl:.3f} ({time.time()-t0:.1f}s)")

    # Calibration
    calib = load_calib_texts(tokenizer, args.calib_samples)
    all_roles = SHARED_ROLES + PERMATRIX_ROLES

    # Calibration — checkpoint-resumable. xtx_store/ggt_store are created
    # only on a fresh calibration pass; reused-checkpoint paths use in-memory
    # tensors loaded from the pickle and have nothing to clean up.
    xtx_store: XtxStore | None = None
    ggt_store: XtxStore | None = None
    cached_calib = checkpoint_load(ckpt_dir, "calibration") if ckpt_dir else None
    calib_policy_invalidated = False
    if cached_calib is not None:
        xtx, ggt = cached_calib["xtx"], cached_calib["ggt"]
        # Reject caches that predate a precision-policy change. If any
        # gramian whose d would now demand fp64 is on disk as fp32, the
        # running-sum noise baked into that accumulator is the exact bug
        # we're fixing — reusing it would negate the change. Recalibrate
        # AND force a factors recompute (the saved factors were derived
        # from the stale gramians and are equally invalid).
        mismatches = []
        for name, t in xtx.items():
            expected = (torch.float64 if t.shape[0] >= args.accum_fp64_threshold
                        else torch.float32)
            if t.dtype != expected:
                mismatches.append((name, t.shape[0], t.dtype, expected))
        if mismatches:
            sample = mismatches[0]
            print(f"  [stale] cached calibration has {len(mismatches)} "
                  f"accumulator(s) with dtype mismatching "
                  f"--accum-fp64-threshold={args.accum_fp64_threshold}; "
                  f"example {sample[0]} d={sample[1]} got={sample[2]} "
                  f"want={sample[3]}. Recalibrating and discarding "
                  f"any cached factors.")
            cached_calib = None
            calib_policy_invalidated = True
        else:
            print(f"  reused cached calibration ({len(xtx)} modules)")
    if cached_calib is None:
        # Resolve the temp dir for disk-backed gramians. Falls through a
        # preference chain so single-shot runs without a checkpoint dir still
        # have somewhere reasonable to spill to.
        if args.xtx_temp_dir is not None:
            xtx_temp_dir = Path(args.xtx_temp_dir)
        elif ckpt_dir is not None:
            xtx_temp_dir = Path(ckpt_dir) / "xtx"
        elif args.save_dir is not None:
            xtx_temp_dir = Path(args.save_dir) / "xtx"
        else:
            xtx_temp_dir = Path(".") / "xtx_tmp"
        avail_gb = psutil.virtual_memory().available / 1e9
        # Translate the percentage caps to the absolutes the XtxStore and
        # friends consume. The RAM reserve is max(legacy absolute, percent-
        # derived) so a stricter absolute floor always wins; the disk and
        # VRAM caps flow through directly.
        total_ram_gb = psutil.virtual_memory().total / 1e9
        pct_ram_reserve_gb = total_ram_gb * (100.0 - args.cpu_ram_max_pct) / 100.0
        effective_min_free_gb = max(args.min_free_ram_gb, pct_ram_reserve_gb)
        total_vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                         if torch.cuda.is_available() else 0.0)
        # Walk up to an existing ancestor before querying disk_usage —
        # --xtx-temp-dir on a fresh save-dir has several non-existent
        # parents and shutil rejects paths it can't stat.
        probe = xtx_temp_dir
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        disk_usage = shutil.disk_usage(str(probe))
        disk_total_gb = disk_usage.total / 1e9
        disk_free_gb = disk_usage.free / 1e9
        # Per-role pass selection. `single` keeps all hooks live for one sweep
        # through the calibration texts (fastest when RAM fits). `by-role`
        # splits into three sweeps so only one role's gramians are resident
        # at a time — the trade is 3× forward-pass cost for ~1/3 the peak
        # RAM. Down_proj gets its own pass because d_in=18944 dominates the
        # per-gramian cost on 7B+.
        if args.calib_passes == "by-role":
            role_groups = [
                list(SHARED_ROLES),
                ["self_attn.o_proj"],
                ["mlp.down_proj"],
            ]
        else:
            role_groups = [SHARED_ROLES + PERMATRIX_ROLES]
        print(f"calibration ({len(calib)} samples, need_grad={need_grad}); "
              f"passes={args.calib_passes} ({len(role_groups)} sweep(s)), "
              f"xtx backend={args.xtx_backend}, "
              f"temp_dir={xtx_temp_dir}")
        print(f"  caps: cpu-ram<={args.cpu_ram_max_pct:.0f}% of "
              f"{total_ram_gb:.1f} GB (reserve {effective_min_free_gb:.1f} GB; "
              f"free now {avail_gb:.1f} GB) | "
              f"disk<={args.disk_max_pct:.0f}% of "
              f"{disk_total_gb:.1f} GB (free now {disk_free_gb:.1f} GB) | "
              f"gpu-vram<={args.gpu_vram_max_pct:.0f}% of "
              f"{total_vram_gb:.1f} GB | "
              f"gpu-workspace-abs<={args.xtx_gpu_workspace_gb:.1f} GB")
        xtx_store = XtxStore(args.xtx_backend, xtx_temp_dir,
                             effective_min_free_gb,
                             gpu_workspace_cap_gb=args.xtx_gpu_workspace_gb,
                             accum_fp64_threshold=args.accum_fp64_threshold,
                             disk_max_pct=args.disk_max_pct,
                             gpu_vram_max_pct=args.gpu_vram_max_pct,
                             gpu_accum_budget_gb=args.gpu_accum_budget_gb)
        ggt_store = (XtxStore(args.xtx_backend, xtx_temp_dir,
                              effective_min_free_gb,
                              gpu_workspace_cap_gb=args.xtx_gpu_workspace_gb,
                              accum_fp64_threshold=args.accum_fp64_threshold,
                              disk_max_pct=args.disk_max_pct,
                              gpu_vram_max_pct=args.gpu_vram_max_pct,
                              gpu_accum_budget_gb=args.gpu_accum_budget_gb)
                     if need_grad else None)
        streaming_role_set = {r.strip() for r in args.streaming_roles.split(",")
                              if r.strip()}
        if streaming_role_set:
            print(f"  streaming roles: {sorted(streaming_role_set)} "
                  f"(pinned-host budget {args.streaming_max_ram_gb:.1f} GB)",
                  flush=True)
        t0 = time.time()
        try:
            xtx, ggt, _ = collect_stats(model, tokenizer, calib, input_device,
                                        args.calib_seq_len, role_groups, need_grad,
                                        xtx_store=xtx_store,
                                        ggt_store=ggt_store,
                                        streaming_roles=streaming_role_set,
                                        streaming_max_ram_gb=args.streaming_max_ram_gb,
                                        batch_size=args.calib_batch_size)
            print(f"  done in {time.time()-t0:.1f}s")
            # Free GPU scratch buffers now so compute_factors gets full VRAM.
            xtx_store.free_gpu_workspaces()
            if ggt_store is not None:
                ggt_store.free_gpu_workspaces()
            # Checkpoint only when gramians are fully in RAM. Memmap-backed
            # tensors pickle by materializing into host memory (torch.save
            # reads the entire file), producing multi-GB .pt files that
            # can't be reloaded without OOM — worse than skipping the
            # checkpoint outright.
            if ckpt_dir and not (xtx_store.has_disk_backed() or
                                 (ggt_store is not None and ggt_store.has_disk_backed())):
                checkpoint_save(ckpt_dir, "calibration", {"xtx": xtx, "ggt": ggt})
            elif ckpt_dir:
                print("  skipping calibration checkpoint "
                      "(disk-backed gramians aren't safely picklable)")
        except BaseException:
            # Best-effort cleanup of disk-backed accumulators on failure so a
            # crashed run doesn't leave 40 GB of temp files behind.
            xtx_store.cleanup()
            if ggt_store is not None:
                ggt_store.cleanup()
            raise

    # Compute factors (no model mutation yet)
    shared_roles_use = [] if args.permatrix_only else SHARED_ROLES
    groups_shared = find_layers(model, shared_roles_use) if shared_roles_use else {}
    pm_roles_use = all_roles if args.permatrix_only else PERMATRIX_ROLES
    groups_permatrix = find_layers(model, pm_roles_use)

    print(f"computing factors (shared={len(groups_shared)} roles, "
          f"permatrix={len(groups_permatrix)} roles)")
    cached_factors = (checkpoint_load(ckpt_dir, "factors")
                      if ckpt_dir and not calib_policy_invalidated else None)
    if cached_factors is not None:
        factors = cached_factors
        print(f"  reused cached factors")
    else:
        # Move model to CPU during decomposition — frees ~6 GB of VRAM for the
        # SVD/Cholesky workspace (big win on 12 GB GPUs with 3B+ models).
        # Skip when device_map is in use: accelerate-dispatched models can't be
        # naively .to()-ed, and parts are already on CPU anyway.
        if device == "cuda" and not use_device_map:
            model.cpu()
            torch.cuda.empty_cache()
            print(f"  model moved to CPU for compute_factors {_gpu_mem_str()}", flush=True)
        t0 = time.time()
        # Streaming factor output: dumps each decomposed matrix's U/V
        # straight to a per-layer .pt file, avoiding a multi-GB dict in
        # RAM during the long decomposition phase. Only disabled by the
        # explicit --no-stream-factors flag.
        factor_tmp_dir = None
        if not args.no_stream_factors:
            factor_tmp_dir = (Path(args.save_dir) / ".streamed_factors"
                              if args.save_dir
                              else Path("./.streamed_factors"))
        factors = compute_factors(model, xtx, ggt, groups_shared, groups_permatrix,
                                  args.window, args.target_ratio, args.eigh_device,
                                  refit=not args.no_refit,
                                  skip_cholesky_above_d=args.skip_cholesky_above_d,
                                  rank_alloc=args.rank_alloc,
                                  factor_output_dir=factor_tmp_dir,
                                  materialize_dtype=dtype)
        print(f"  done in {time.time()-t0:.1f}s")
        if ckpt_dir and not factors.get("already_materialized"):
            # Streaming compute_factors keeps tensors in per-layer .pt
            # files that save_factored unlinks. Checkpointing a dict of
            # disk paths would leave dangling references after resume,
            # so we skip the factor checkpoint in streaming mode. Resume
            # replays compute_factors from cached calibration, which is
            # cheap relative to the factor-disk-IO savings.
            checkpoint_save(ckpt_dir, "factors", factors)
        elif ckpt_dir:
            print("  skipping factors checkpoint "
                  "(streaming mode - per-layer .pt files aren't safely "
                  "re-usable after save_factored assembles them)")
        # Gramians are no longer needed — factors are cached. Drop strong refs
        # and delete any disk-backed memmaps before PPL eval and save_factored
        # load the model back into RAM.
        xtx = {}
        ggt = {} if ggt is not None else None
        if xtx_store is not None:
            xtx_store.cleanup()
        if ggt_store is not None:
            ggt_store.cleanup()
        # Move model back for PPL eval
        if device == "cuda" and not use_device_map:
            model.to(device)

    # Persist factors BEFORE mutating model (so we save the clean decomposition)
    if args.save_dir:
        print(f"saving factored tensors to {args.save_dir}")
        manifest = save_factored(model, factors, groups_shared, groups_permatrix,
                                  args.save_dir, args.model, args.window,
                                  args.target_ratio, not args.no_refit)
        print(f"  manifest: {len(manifest['shared_roles'])} shared roles, "
              f"{len(manifest['permatrix_roles'])} permatrix roles")

    # Materialize factors back into model for PPL eval
    report = materialize_factors_in_place(model, factors, groups_shared,
                                          groups_permatrix, dtype=dtype)
    # Skip global .to(dtype) for device_map models — accelerate dispatch hooks
    # don't survive the move. materialize_factors_in_place already casts the
    # rewritten linear weights to `dtype`; other params stay at load_dtype.
    if not use_device_map:
        model = model.to(dtype)

    print("factored PPL")
    t0 = time.time()
    factored_ppl = evaluate_ppl(model, tokenizer, eval_text, input_device, n_tokens=args.eval_tokens)
    print(f"  = {factored_ppl:.3f} ({time.time()-t0:.1f}s)  "
          f"delta {(factored_ppl-baseline_ppl)/baseline_ppl*100:+.1f}% vs baseline")

    # Roundtrip check: reload from disk, verify weights match what's in-memory
    roundtrip_ok = None
    if args.save_dir and args.verify_roundtrip:
        print("verifying roundtrip (reload from disk + compare)")
        state_reloaded, _ = load_factored_reconstruct(args.save_dir)
        max_err = 0.0
        for name, t_reloaded in state_reloaded.items():
            if name in dict(model.state_dict()):
                t_in_model = model.state_dict()[name].cpu().to(torch.float32)
                diff = (t_reloaded.to(torch.float32) - t_in_model).abs().max().item()
                max_err = max(max_err, diff)
        roundtrip_ok = max_err < 1e-3
        print(f"  max abs diff vs in-memory model: {max_err:.2e}  "
              f"{'OK' if roundtrip_ok else 'FAIL'}")

    result = {
        "model": args.model,
        "window": args.window,
        "target_ratio": args.target_ratio,
        "refit": not args.no_refit,
        "permatrix_only": args.permatrix_only,
        "calib_samples": args.calib_samples,
        "eval_tokens": args.eval_tokens,
        "baseline_ppl": baseline_ppl,
        "factored_ppl": factored_ppl,
        "delta_pct": (factored_ppl - baseline_ppl) / baseline_ppl * 100,
        "save_dir": args.save_dir,
        "roundtrip_verified": roundtrip_ok,
        "report": report,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
