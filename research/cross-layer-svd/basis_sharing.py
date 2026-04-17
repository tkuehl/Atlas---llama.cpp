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


def _cuda_fallback_log(op, err):
    """Log CUDA → CPU fallbacks the first few times so we can diagnose."""
    if _CUDA_FALLBACK_LOGGED[op] < 3:
        print(f"  [linalg] CUDA {op} failed, falling back to CPU: {str(err)[:120]}")
        _CUDA_FALLBACK_LOGGED[op] += 1


def sqrt_and_inv(M, eps_ratio=1e-5, device="cuda", need_inv=True):
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

    # Precision-first fallback ladder. Rank-deficient gramians (common for
    # down_proj when calibration tokens < d_in) break fp32 Cholesky; bumping
    # eps aggressively distorts the whitening matrix and tanks ASVD quality.
    # We'd rather pay fp64 precision than eat quality loss from over-regularizing.
    #   1) GPU fp32 Cholesky @ eps×1    — fastest path (usually works on full-rank)
    #   2) GPU fp64 Cholesky @ eps×1    — same regularization, higher precision
    #   3) GPU fp64 Cholesky @ eps×10   — if even fp64 fails, mild extra damping
    #   4) CPU fp64 eigh                — last resort (handles singular cleanly)
    tried = []
    if device == "cuda":
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


class XtxStore:
    """Allocates and owns d×d fp32 accumulators for calibration.

    Modes:
      - "ram":  torch.zeros on CPU (current behavior)
      - "disk": np.memmap → torch.from_numpy; addmm_ writes through to NVMe.
                The OS page cache handles residency; this just caps RAM.
      - "auto": per-tensor choice. If the allocation would trip the
                --min-free-ram-gb threshold, fall back to disk; else RAM.

    All tensors are CPU fp32 and safe to hand to downstream addmm_ and SVD
    code unchanged. Call cleanup() when done to delete temp files.
    """

    def __init__(self, mode: str, temp_dir: Path | None, min_free_gb: float,
                 gpu_workspace_cap_gb: float = 4.0):
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

    def alloc(self, name: str, d: int) -> torch.Tensor:
        bytes_needed = d * d * 4  # fp32
        use_disk = self._use_disk(bytes_needed)
        if use_disk:
            # No RAM safety check on the disk path. Memmap writes go to the
            # OS page cache, which Windows auto-reclaims under pressure — so
            # even when free RAM looks low, we are not actually allocating
            # new process memory. Guarding this on virtual_memory().available
            # produces false positives once the page cache fills up.
            safe = name.replace("/", "_").replace(".", "_")
            path = self.temp_dir / f"xtx_{safe}.f32"
            # mode='w+' truncates; freshly-extended pages are zero-initialized
            # by the OS, so we don't need (and shouldn't pay for) an explicit
            # arr[:] = 0 that dirties every page up front.
            arr = np.memmap(str(path), dtype="float32", mode="w+", shape=(d, d))
            self._memmaps[name] = arr
            self._paths.append(path)
            return torch.from_numpy(arr)
        _require_ram_headroom(bytes_needed, self.min_free_gb, name)
        return torch.zeros(d, d, dtype=torch.float32)

    def get_gpu_workspace(self, d: int) -> torch.Tensor | None:
        """Return (lazily-allocated) CUDA fp32 d×d buffer for delta matmul.

        Returns None when CUDA is unavailable, the tensor would exceed the
        configured cap, or allocation fails (e.g. VRAM exhausted). Callers
        in the hook path fall back to CPU-side addmm in that case.
        """
        if not self._gpu_enabled:
            return None
        if d in self._gpu_workspaces:
            return self._gpu_workspaces[d]
        bytes_needed = d * d * 4
        if bytes_needed > self._gpu_workspace_cap_bytes:
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


# ---------- calibration (forward + optional backward) ----------

def collect_stats(model, tokenizer, texts, device, seq_len, role_groups, need_grad,
                  xtx_store: XtxStore, ggt_store: XtxStore | None = None):
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
        ws = xtx_store.get_gpu_workspace(d_in)

        if ws is not None:
            def hook(mod, inputs):
                x = inputs[0].detach()
                flat = x.reshape(-1, x.shape[-1])
                flat_gpu = flat.to(device="cuda", dtype=torch.float32, non_blocking=True)
                # In-place matmul into the pre-allocated workspace, then D2H
                # into a fresh CPU tensor. .cpu() is synchronous in torch's
                # default stream, so the next hook's overwrite of `ws` is
                # safely ordered after this copy completes.
                torch.mm(flat_gpu.T, flat_gpu, out=ws)
                xtx[name].add_(ws.cpu())
            return hook

        # CPU fallback — identical semantics, just slower for big d.
        def hook(mod, inputs):
            x = inputs[0].detach()
            flat_cpu = x.reshape(-1, x.shape[-1]).to(device="cpu", dtype=torch.float32)
            xtx[name].addmm_(flat_cpu.T, flat_cpu)
        return hook

    def bwd_hook_factory(name, d_out):
        ggt[name] = ggt_store.alloc(name, d_out)
        ws = ggt_store.get_gpu_workspace(d_out)

        if ws is not None:
            def hook(mod, grad_input, grad_output):
                g = grad_output[0].detach()
                flat = g.reshape(-1, g.shape[-1])
                flat_gpu = flat.to(device="cuda", dtype=torch.float32, non_blocking=True)
                torch.mm(flat_gpu.T, flat_gpu, out=ws)
                ggt[name].add_(ws.cpu())
            return hook

        def hook(mod, grad_input, grad_output):
            g = grad_output[0].detach()
            flat_cpu = g.reshape(-1, g.shape[-1]).to(device="cpu", dtype=torch.float32)
            ggt[name].addmm_(flat_cpu.T, flat_cpu)
        return hook

    model.eval()
    n_passes = len(role_groups)
    for pass_idx, pass_roles in enumerate(role_groups):
        pass_groups = find_layers(model, pass_roles)
        hooks = []
        for role, entries in pass_groups.items():
            for _, name, mod in entries:
                hooks.append(mod.register_forward_pre_hook(
                    fwd_hook_factory(name, mod.in_features)))
                if need_grad:
                    hooks.append(mod.register_full_backward_hook(
                        bwd_hook_factory(name, mod.out_features)))

        desc = (f"calib pass {pass_idx+1}/{n_passes} "
                f"({sum(len(e) for e in pass_groups.values())} modules)")
        for text in tqdm(texts, desc=desc):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            input_ids = enc.input_ids.to(device)
            if input_ids.shape[1] < 8:
                continue
            if need_grad:
                out = model(input_ids, labels=input_ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    model(input_ids)

        for h in hooks:
            h.remove()

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


# ---------- decomposition ----------

def decompose_shared_window(W_list, xtx_list, ggt_list, rank, refit=True, device="cpu"):
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
    S_in, _ = sqrt_and_inv(XTX_joint, device=device, need_inv=False)

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

    U, _, _ = _stable_svd(W_stack)
    B = U[:, :rank].contiguous()                           # [d_out, rank]
    # SVD may have fallen back to CPU. Lock all subsequent ops to B's device.
    work_device = B.device
    del W_stack, weighted

    coeffs = []
    for i, W_i in enumerate(W_list):
        W_i_t = W_i.to(device=work_device, dtype=work_dtype)
        if refit and ggt_list is not None:
            # Balanced-weighted LS refit: A_i = (B^T S_out^T S_out B)^-1 B^T S_out^T S_out W_i
            S_out_i, _ = sqrt_and_inv(ggt_list[i], device=work_device, need_inv=False)
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


def decompose_permatrix(W, XTX, GGT, rank, device="cpu"):
    """Per-matrix balanced truncation. If GGT is None, fall back to input-only
    ASVD (no output-gradient weighting) — cheaper calibration, still respects
    activation distribution.
    """
    S_in, S_in_inv = sqrt_and_inv(XTX, device=device)
    W64 = W.to(device=device, dtype=torch.float64)

    if GGT is None:
        # ASVD path: project onto top-r output directions of (W @ S_in).
        U, _, _ = _stable_svd(W64 @ S_in)
        r = min(rank, U.shape[1])
        U_r = U[:, :r]
        W_r = U_r @ (U_r.T @ W64)
        # Return U and V separately: U_r as "U", (U_r^T @ W) as "V"
        U_out = U_r.clone()
        V_out = U_r.T @ W64
        return U_out.cpu(), V_out.cpu()

    # Balanced truncation with both input and output weighting
    S_out, S_out_inv = sqrt_and_inv(GGT, device=device)
    M = S_out @ W64 @ S_in
    U, sigma, Vt = _stable_svd(M)
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
                    window_size, target_ratio, device, refit=True):
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
                r = rank_permatrix(d_out, d_in, target_ratio)
                have_ggt = ggt_list_all is not None
                S_in, S_in_inv = sqrt_and_inv(xtx_list_all[i], device=device,
                                              need_inv=have_ggt)
                W_t = W_list_all[i].to(device=S_in.device, dtype=S_in.dtype)
                if have_ggt:
                    S_out, S_out_inv = sqrt_and_inv(ggt_list_all[i], device=device,
                                                    need_inv=True)
                    S_out = S_out.to(dtype=S_in.dtype)
                    S_out_inv = S_out_inv.to(dtype=S_in.dtype)
                    M = S_out @ W_t @ S_in
                    U, sigma, Vt = _stable_svd(M)
                    if U.device != M.device:
                        S_in_inv = S_in_inv.to(U.device)
                        S_out_inv = S_out_inv.to(U.device)
                    r_eff = min(r, sigma.numel())
                    U_r = S_out_inv @ U[:, :r_eff]
                    V_r = sigma[:r_eff].unsqueeze(1) * (Vt[:r_eff, :] @ S_in_inv)
                else:
                    # ASVD path
                    M = W_t @ S_in
                    U, _, _ = _stable_svd(M)
                    r_eff = min(r, U.shape[1])
                    U_r = U[:, :r_eff]
                    V_r = U_r.T @ W_t.to(U_r.device)
                role_windows.append({
                    "window_id": w_idx,
                    "layers": [i],
                    "rank": r_eff,
                    "basis": U_r.cpu().to(torch.float32),
                    "coeffs": [V_r.cpu().to(torch.float32)],
                    "mode": "permatrix_fallback",
                })
                continue

            r = rank_shared(d_out, d_in, n, target_ratio)
            W_list = [W_list_all[i] for i in layer_indices]
            xtx_list = [xtx_list_all[i] for i in layer_indices]
            ggt_list_w = [ggt_list_all[i] for i in layer_indices] if ggt_list_all else None
            t_win = time.time()
            B, coeffs = decompose_shared_window(W_list, xtx_list, ggt_list_w, r,
                                                refit=refit, device=device)
            print(f"[{time.time()-t_start:7.1f}s]   win {w_idx+1}/{len(windows)} "
                  f"layers={layer_indices} r={r} ({time.time()-t_win:.1f}s) "
                  f"{_gpu_mem_str()}", flush=True)
            role_windows.append({
                "window_id": w_idx,
                "layers": list(layer_indices),
                "rank": r,
                "basis": B.to(torch.float32),
                "coeffs": [c.to(torch.float32) for c in coeffs],
                "mode": "shared",
            })

        result["shared"][role] = {"L": L, "d_out": d_out, "d_in": d_in,
                                  "windows": role_windows}

    for role, entries in groups_permatrix.items():
        _, names, mods = zip(*entries)
        d_out, d_in = mods[0].weight.shape
        r = rank_permatrix(d_out, d_in, target_ratio)
        print(f"[{time.time()-t_start:7.1f}s] permatrix {role}: "
              f"{d_out}x{d_in} x{len(mods)} rank={r} {_gpu_mem_str()}", flush=True)
        layer_factors = []
        t_role_start = time.time()
        for i, (name, mod) in enumerate(zip(names, mods)):
            t_layer = time.time()
            have_ggt = ggt is not None
            S_in, S_in_inv = sqrt_and_inv(xtx[name], device=device,
                                          need_inv=have_ggt)
            W_t = _weight_to_cpu(mod).to(device=S_in.device, dtype=S_in.dtype)
            if have_ggt:
                S_out, S_out_inv = sqrt_and_inv(ggt[name], device=device,
                                                need_inv=True)
                S_out = S_out.to(dtype=S_in.dtype)
                S_out_inv = S_out_inv.to(dtype=S_in.dtype)
                M = S_out @ W_t @ S_in
                U, sigma, Vt = _stable_svd(M)
                if U.device != M.device:
                    S_in_inv = S_in_inv.to(U.device)
                    S_out_inv = S_out_inv.to(U.device)
                r_eff = min(r, sigma.numel())
                U_r = S_out_inv @ U[:, :r_eff]
                V_r = sigma[:r_eff].unsqueeze(1) * (Vt[:r_eff, :] @ S_in_inv)
            else:
                # ASVD path — no output gradient weighting
                M = W_t @ S_in
                U, _, _ = _stable_svd(M)
                r_eff = min(r, U.shape[1])
                U_r = U[:, :r_eff]
                V_r = U_r.T @ W_t.to(U_r.device)
            layer_factors.append({
                "layer": i,
                "U": U_r.cpu().to(torch.float32),
                "V": V_r.cpu().to(torch.float32),
            })
            if (i + 1) % 8 == 0 or i == len(mods) - 1:
                print(f"[{time.time()-t_start:7.1f}s]   {role} layer "
                      f"{i+1}/{len(mods)} done ({time.time()-t_layer:.1f}s last) "
                      f"{_gpu_mem_str()}", flush=True)
        result["permatrix"][role] = {"L": len(mods), "d_out": d_out, "d_in": d_in,
                                     "rank": r, "layers": layer_factors}

    return result


def materialize_factors_in_place(model, factors, groups_shared, groups_permatrix,
                                  dtype=torch.float16):
    """Apply factor reconstruction back into model.weight tensors in place.
    Used for PPL evaluation of the factored model before GGUF emission.
    Returns a summary report for logging."""
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
            role_report.append({"layer": i, "rank": pm_info["rank"]})
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

    # Factored tensors with structured names
    factored_tensors = {}
    manifest_shared = {}
    for role, info in factors["shared"].items():
        tag = _role_tag(role)
        windows_manifest = []
        for win in info["windows"]:
            key_base = f"shared.{tag}.w{win['window_id']:03d}"
            factored_tensors[f"{key_base}.basis"] = win["basis"].contiguous()
            for i_local, layer_i in enumerate(win["layers"]):
                factored_tensors[f"{key_base}.coeffs.{layer_i:03d}"] = \
                    win["coeffs"][i_local].contiguous()
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
            factored_tensors[u_key] = lf["U"].contiguous()
            factored_tensors[v_key] = lf["V"].contiguous()
            layers_manifest.append({"layer": i, "U_key": u_key, "V_key": v_key})
        manifest_permatrix[role] = {
            "tag": tag, "d_out": info["d_out"], "d_in": info["d_in"],
            "L": info["L"], "rank": info["rank"], "layers": layers_manifest,
        }

    save_file(factored_tensors, str(out / "factored.safetensors"))

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
    p.add_argument("--target-ratio", type=float, default=0.5,
                   help="target compression ratio; k chosen to hit this")
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--calib-seq-len", type=int, default=512)
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
    p.add_argument("--min-free-ram-gb", type=float, default=4.0,
                   help="OS headroom: refuse gramian allocations that would leave "
                        "less than this much free RAM. Safety net against OOM "
                        "locking up the machine.")
    p.add_argument("--xtx-backend", default="auto", choices=["auto", "ram", "disk"],
                   help="Where to store d*d calibration gramians. 'ram' matches "
                        "legacy behavior; 'disk' memmaps them to NVMe; 'auto' "
                        "picks per-tensor based on --min-free-ram-gb.")
    p.add_argument("--xtx-temp-dir", default=None,
                   help="Directory for disk-backed gramians (default: "
                        "{checkpoint-dir}/xtx or {save-dir}/xtx).")
    p.add_argument("--xtx-gpu-workspace-gb", type=float, default=4.0,
                   help="Max VRAM budget for the d*d delta-matmul workspace "
                        "used during gramian accumulation. Set to 0 to force "
                        "the CPU addmm path.")
    p.add_argument("--calib-passes", default="single",
                   choices=["single", "by-role"],
                   help="single: accumulate all gramians in one forward sweep "
                        "(fastest when RAM fits). by-role: run separate sweeps "
                        "for shared roles, o_proj, and down_proj so only one "
                        "role's gramians co-exist in RAM — enables 14B+ models.")
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
    if cached_calib is not None:
        xtx, ggt = cached_calib["xtx"], cached_calib["ggt"]
        print(f"  reused cached calibration ({len(xtx)} modules)")
    else:
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
              f"xtx backend={args.xtx_backend}, min-free-ram={args.min_free_ram_gb:.1f} GB, "
              f"gpu-workspace={args.xtx_gpu_workspace_gb:.1f} GB, "
              f"free now={avail_gb:.1f} GB, temp_dir={xtx_temp_dir}")
        xtx_store = XtxStore(args.xtx_backend, xtx_temp_dir,
                             args.min_free_ram_gb,
                             gpu_workspace_cap_gb=args.xtx_gpu_workspace_gb)
        ggt_store = (XtxStore(args.xtx_backend, xtx_temp_dir,
                              args.min_free_ram_gb,
                              gpu_workspace_cap_gb=args.xtx_gpu_workspace_gb)
                     if need_grad else None)
        t0 = time.time()
        try:
            xtx, ggt, _ = collect_stats(model, tokenizer, calib, input_device,
                                        args.calib_seq_len, role_groups, need_grad,
                                        xtx_store=xtx_store,
                                        ggt_store=ggt_store)
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
    cached_factors = checkpoint_load(ckpt_dir, "factors") if ckpt_dir else None
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
        factors = compute_factors(model, xtx, ggt, groups_shared, groups_permatrix,
                                  args.window, args.target_ratio, args.eigh_device,
                                  refit=not args.no_refit)
        print(f"  done in {time.time()-t0:.1f}s")
        if ckpt_dir:
            checkpoint_save(ckpt_dir, "factors", factors)
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
