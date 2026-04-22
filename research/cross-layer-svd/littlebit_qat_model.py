"""Full-model QAT for LittleBit on Qwen 2.5 0.5B.

Wraps every nn.Linear in the base model (except lm_head) with a
LittleBitLinear initialized via Dual-SVID, then trains with KL
divergence (+ optional intermediate MSE) against a frozen fp16
teacher on wikitext. Evaluates PPL on wikitext-2 test.

The single-matrix activation-weighted experiment
(littlebit_qat_activation.py) showed the format has ~90% activation-
energy capture at r=512 locally. This script tests whether that
compounds usably through 24 layers.

Kill criteria (from littlebit_math.md §13.6):
  - loss not decreasing after 500 warmup steps → format breaks
    under compounding, stop
  - PPL > 200 at first eval (step 500) → below archived FP-SVD
    floor, stop

Usage:
  # Phase 1 smoke (5 min, verifies the pipeline)
  python littlebit_qat_model.py --steps 50 --eval-every 50 \\
         --out littlebit_qat_model_smoke.json

  # Phase 2 real run (4-8 hours)
  python littlebit_qat_model.py --steps 8000 --eval-every 500 \\
         --out littlebit_qat_model_run1.json
"""

from __future__ import annotations

# --- Windows / torch.compile bootstrap ---
# torch.compile's inductor backend writes generated Python/Triton
# source to its on-disk cache using the default Python text-mode
# encoding, which on Windows is cp1252.  Inductor's emitted code
# contains non-ASCII characters (e.g. Greek theta in RoPE kernel
# comments) that cp1252 can't encode, so the first compiled
# forward dies with a UnicodeEncodeError from codecache.write_atomic.
# Python's UTF-8 mode (PEP 540) forces every `open(...)` to use
# UTF-8, which fixes this at the source.  The flag must be set
# before the interpreter starts, so we re-launch via a subprocess
# on first entry when running on Windows without it already set.
# Note: subprocess (not os.execv) because os.execv on Windows
# routes through CreateProcess and silently drops empty-string
# argv entries (like `--best-ckpt ""`), whereas subprocess quotes
# them properly via list2cmdline.
import os as _os
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":
    import subprocess as _subprocess
    _env = dict(_os.environ)
    _env["PYTHONUTF8"] = "1"
    _sys.exit(_subprocess.run(
        [_sys.executable] + _sys.argv, env=_env
    ).returncode)

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from littlebit_qat_single import smooth_sign as _smooth_sign_fp32


class SmoothSignEfficient(torch.autograd.Function):
    """Memory-efficient SmoothSign.

    Forward: sign(x)  (same as paper).
    Backward: grad_output * tau * (1 - tanh(tau*x)**2)

    Key difference vs littlebit_qat_single.SmoothSign: saves the
    precomputed surrogate gradient in bfloat16 instead of saving
    the full-precision input x.  Halves the activation memory
    stored for backward at the cost of one extra tanh in forward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, tau: float):
        with torch.no_grad():
            tanh_tx = torch.tanh(tau * x)
            surrogate = tau * (1.0 - tanh_tx * tanh_tx)
            ctx.save_for_backward(surrogate.to(torch.bfloat16))
        out = torch.where(x >= 0,
                          torch.ones_like(x),
                          -torch.ones_like(x))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (surrogate_bf16,) = ctx.saved_tensors
        surrogate = surrogate_bf16.to(grad_output.dtype)
        return grad_output * surrogate, None


def smooth_sign(x: torch.Tensor, tau: float = 100.0) -> torch.Tensor:
    return SmoothSignEfficient.apply(x, tau)


class LittleBitLinearHF(nn.Module):
    """Drop-in replacement for nn.Linear under the LittleBit factorization.

    Forward uses Prop. 1 efficient form (Eq. 5).  Parameters:
      U_fp (d_out, r), V_fp (d_in, r) — sign-shadow matrices
      h (d_out), g (d_in), ell (r)     — FP32 scale vectors
      bias (d_out)                     — if original Linear had one

    Shadow dtype: U_fp / V_fp can be stored as bf16 instead of fp32
    (see `shadow_dtype` arg).  These are the "shadow weights" that
    drive SmoothSign's forward + backward — the actual ±1 signs are
    recomputed from them every forward.  Halving their precision
    halves the VRAM spent on sign matrices, their gradients, and
    (as a follow-on) their optimizer state.  At 0.5B that's ~600 MB
    saved; at 7B it's ~5.5 GB; at 30B it's ~24 GB — the savings
    scale with model size, which makes this the cheapest memory
    lever for going up the scale ladder.

    Scale vectors (h, g, ell) and bias stay FP32: they're small
    (O(d) each, not O(d·r)) so the memory saving wouldn't be
    meaningful, and they enter the forward as multiplicative
    scales where precision matters more than for the binary U/V.
    """

    def __init__(self, d_in: int, d_out: int, r: int, bias: bool,
                 tau: float = 100.0,
                 shadow_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.r = r
        self.tau = tau
        self.shadow_dtype = shadow_dtype
        self.U_fp = nn.Parameter(torch.empty(d_out, r, dtype=shadow_dtype))
        self.V_fp = nn.Parameter(torch.empty(d_in, r, dtype=shadow_dtype))
        self.h = nn.Parameter(torch.empty(d_out))
        self.g = nn.Parameter(torch.empty(d_in))
        self.ell = nn.Parameter(torch.empty(r))
        if bias:
            self.bias = nn.Parameter(torch.empty(d_out))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, r: int,
                    tau: float = 100.0,
                    shadow_dtype: torch.dtype = torch.float32
                    ) -> "LittleBitLinearHF":
        """Initialize via Dual-SVID from an FP linear."""
        W = lin.weight.data.detach().to(torch.float64).cpu().numpy()
        d_out, d_in = W.shape
        r_eff = min(r, d_out, d_in)

        U_full, S_full, VT_full = np.linalg.svd(W, full_matrices=False)
        Uk = U_full[:, :r_eff]
        Sk = S_full[:r_eff]
        Vk = VT_full[:r_eff, :].T
        sqrt_S = np.sqrt(Sk)
        Up = Uk * sqrt_S[None, :]
        Vp = Vk * sqrt_S[None, :]

        U_abs = np.abs(Up)
        V_abs = np.abs(Vp)
        uU, sU, vtU = np.linalg.svd(U_abs, full_matrices=False)
        uV, sV, vtV = np.linalg.svd(V_abs, full_matrices=False)
        h0 = uU[:, 0] * np.sqrt(sU[0])
        l_u0 = vtU[0, :] * np.sqrt(sU[0])
        g0 = uV[:, 0] * np.sqrt(sV[0])
        l_v0 = vtV[0, :] * np.sqrt(sV[0])
        if h0.sum() < 0:
            h0 = -h0; l_u0 = -l_u0
        if g0.sum() < 0:
            g0 = -g0; l_v0 = -l_v0
        ell0 = l_u0 * l_v0

        out = cls(d_in=d_in, d_out=d_out, r=r_eff,
                  bias=lin.bias is not None, tau=tau,
                  shadow_dtype=shadow_dtype)
        with torch.no_grad():
            # Dual-SVID is computed in fp64 for numerical stability,
            # then cast to the shadow dtype.  Note that fp64 → bf16 is
            # a large precision drop (56-bit → 7-bit mantissa) but the
            # *signs* are what matter for the binary factorization;
            # the exact magnitudes will be recovered by gradient
            # descent within a few hundred steps.
            out.U_fp.copy_(torch.tensor(Up, dtype=shadow_dtype))
            out.V_fp.copy_(torch.tensor(Vp, dtype=shadow_dtype))
            out.h.copy_(torch.tensor(h0,   dtype=torch.float32))
            out.g.copy_(torch.tensor(g0,   dtype=torch.float32))
            out.ell.copy_(torch.tensor(ell0, dtype=torch.float32))
            if lin.bias is not None:
                out.bias.copy_(lin.bias.data.detach().to(torch.float32).cpu())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. 5 efficient form: Y = ((((X*g) @ V_sign) * ell) @ U_sign.T) * h
        # x: (..., d_in)
        # SmoothSign output inherits the shadow dtype.  Cast the signs
        # to the activation dtype (y picks that up from x * self.g,
        # with g fp32 so y is fp32) — ±1 values are lossless across
        # any float dtype, so this is free.  Keeps the matmul in the
        # activation precision; the memory savings come entirely from
        # storing U_fp / V_fp (and their grads / Adam state) in bf16.
        U_sign = smooth_sign(self.U_fp, self.tau)
        V_sign = smooth_sign(self.V_fp, self.tau)
        y = x * self.g
        y = y @ V_sign.to(y.dtype)
        y = y * self.ell
        y = y @ U_sign.to(y.dtype).T
        y = y * self.h
        if self.bias is not None:
            y = y + self.bias
        return y


class HiddenCapture:
    """Per-layer hidden-state collector via forward hooks.

    Replaces output_hidden_states=True.  HF's list-based return
    retains references across backward and partially defeats gradient
    checkpointing; hooks let us capture, use for MSE, then explicitly
    release, letting checkpointing drop the activations.

    Semantic equivalence with output_hidden_states=True:
    HF's `Qwen2Model.forward` builds hidden_states as:
        [ embed, post-layer-0, post-layer-1, ..., post-layer-(N-2),
          post-final-norm(post-layer-(N-1)) ]
    `hidden_states[1:]` drops the embedding, giving N tensors where
    the last one is **post** the final RMSNorm.  A naive hook bank
    on `model.layers` captures N decoder-layer outputs, but its last
    entry is **pre** the final norm — different tensor, different
    magnitude (measured 2.3x at Qwen2.5-0.5B), and with a rank-
    compressed student the pre-norm MSE blows up and dominates the
    24-term sum, steering all gradient toward fixing a representation
    the final norm would have rescaled away.  Measured impact: ~2x
    worse step-500 PPL vs output_hidden_states=True
    ([JOURNAL.md](JOURNAL.md) 2026-04-22).

    Fix: hook layers[0 .. N-2] and also hook model.norm, so the final
    entry is post-final-norm — matching HF exactly.
    """

    def __init__(self):
        self.states: list[torch.Tensor] = []

    def _layer_hook(self, _module, _inputs, output):
        # Decoder layers return a tuple (hidden_states, ...) or just a
        # tensor depending on HF version.
        h = output[0] if isinstance(output, tuple) else output
        self.states.append(h)

    def _post_norm_hook(self, _module, _inputs, output):
        # Final RMSNorm returns a tensor, not a tuple.
        self.states.append(output)

    def clear(self) -> None:
        self.states.clear()

    def install(self, model: nn.Module) -> list:
        """Register hooks on layers[0..N-2] and on the final norm.

        The first N-1 hooks capture post-layer outputs (pre-norm).
        The final hook captures the post-final-norm tensor so
        self.states aligns with output_hidden_states=True's
        hidden_states[1:] byte-for-byte.
        """
        layers = None
        final_norm = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            # Qwen2 / Llama-like: model.model.norm is the final RMSNorm.
            final_norm = getattr(model.model, "norm", None)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
            # GPT-2 / Neo-style: model.transformer.ln_f is the final norm.
            final_norm = getattr(model.transformer, "ln_f", None)
        else:
            raise RuntimeError(
                "HiddenCapture: couldn't find decoder layers on "
                "model; expected model.model.layers or "
                "model.transformer.h"
            )
        if final_norm is None:
            raise RuntimeError(
                "HiddenCapture: couldn't find final-norm module "
                "(looked for model.model.norm / model.transformer.ln_f).  "
                "Either add a case for your architecture or fall back "
                "to --no-mse-via-hooks."
            )
        handles = [layer.register_forward_hook(self._layer_hook)
                   for layer in layers[:-1]]
        handles.append(final_norm.register_forward_hook(self._post_norm_hook))
        return handles


def wrap_model_littlebit_shapes(model: nn.Module, r: int,
                                tau: float = 100.0,
                                skip: tuple[str, ...] = ("lm_head",),
                                shadow_dtype: torch.dtype = torch.float32
                                ) -> int:
    """Replace nn.Linear with empty LittleBitLinearHF modules (no init).

    Used by the init-cache fast path: we need the module structure to
    match for load_state_dict, but skip the expensive Dual-SVID SVDs
    because the cached state_dict has the initialized parameters.
    """
    count = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in skip):
                continue
            d_out = child.out_features
            d_in = child.in_features
            r_eff = min(r, d_out, d_in)
            new = LittleBitLinearHF(d_in=d_in, d_out=d_out, r=r_eff,
                                    bias=child.bias is not None, tau=tau,
                                    shadow_dtype=shadow_dtype)
            setattr(module, child_name, new)
            count += 1
    return count


def wrap_model_littlebit(model: nn.Module, r: int,
                         tau: float = 100.0,
                         skip: tuple[str, ...] = ("lm_head",),
                         log_every: int = 10,
                         shadow_dtype: torch.dtype = torch.float32) -> int:
    """Recursively replace nn.Linear with LittleBitLinearHF in-place.

    Returns the number of layers wrapped.
    """
    # Collect all (parent, child_name, full_name) to wrap first, so we
    # can track progress accurately.
    targets = []
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in skip):
                continue
            targets.append((module, child_name, full, child))
    print(f"  wrapping {len(targets)} Linear layers...", flush=True)

    t0 = time.time()
    for i, (parent, child_name, full, child) in enumerate(targets, start=1):
        t_i = time.time()
        new = LittleBitLinearHF.from_linear(child, r=r, tau=tau,
                                            shadow_dtype=shadow_dtype)
        setattr(parent, child_name, new)
        if i % log_every == 0 or i == len(targets):
            per = (time.time() - t0) / i
            remain = per * (len(targets) - i)
            print(f"    {i:3d}/{len(targets)}  "
                  f"last={full} [{child.out_features}x{child.in_features}] "
                  f"{time.time() - t_i:.1f}s  "
                  f"total={time.time() - t0:.0f}s  "
                  f"eta={remain:.0f}s",
                  flush=True)
    return len(targets)


@torch.no_grad()
def wikitext_ppl(model, tokenizer, split: str = "test",
                 seq_len: int = 2048, max_tokens: int = 100_000,
                 device: torch.device = torch.device("cuda")) -> float:
    """Standard sliding-window PPL on wikitext-2.

    Concatenate all test text, tokenize once, slide a window of
    `seq_len` with full stride (non-overlapping). Compute mean NLL
    across non-padding positions, then exp() for PPL. Caps total
    tokens for speed during training-time eval.

    Compile-safety: this function is called between training steps
    on a torch.compile-wrapped student, so naive implementations
    poison dynamo's compile cache and cost 2-3x per training step
    for the rest of the run. Three specific guards:

      1. Do NOT flip `model.training` (no `.eval()` / `.train()`).
         Dynamo guards specialize on `self.training`, so every mode
         flip invalidates the cached train-mode graph and forces a
         recompile burst (measured 3.6x slowdown in isolation).
         @torch.no_grad() already disables grad accumulation — the
         only other thing `.eval()` changes is dropout, and Qwen2
         runs attention_dropout=0.0 by default, so the mode flip is
         functionally a no-op but carries the full guard cost.
      2. Force eager mode for the eval forwards themselves via
         `torch.compiler.set_stance("force_eager")`, so the train-
         mode compiled graph is never evaluated with eval-shaped
         inputs (which would create an additional specialization).
      3. Drop any tail chunk whose shape doesn't match the stride
         — different shapes spawn new dynamo guards.  Loses ~0.3%
         of eval tokens at max_tokens=50k, seq_len=512, well within
         stochastic variance.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"] for row in ds)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]

    nll_sum = 0.0
    count = 0
    with torch.compiler.set_stance("force_eager"):
        for i in range(0, len(ids) - 1, seq_len):
            chunk = ids[i:i + seq_len].to(device).unsqueeze(0)
            if chunk.shape[1] != seq_len:
                break
            logits = model(chunk, output_hidden_states=False).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            nll_sum += loss.item()
            count += shift_labels.numel()
    return math.exp(nll_sum / max(1, count))


def prepare_train_stream(tokenizer, c4_samples: int = 0) -> torch.Tensor:
    """Build the combined wikitext-2 + optional C4 token stream.

    Split out from iter_batches() so the caller owns the random
    generator (enables checkpoint-resume).
    """
    from datasets import load_dataset
    print("  preparing train token stream...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row["text"] for row in ds)
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    n_wiki = tokens.shape[0]
    print(f"  wikitext-2: {n_wiki:,} tokens", flush=True)

    if c4_samples > 0:
        print(f"  streaming {c4_samples} C4 samples...", flush=True)
        t0 = time.time()
        c4 = load_dataset("allenai/c4", "en", split="train",
                          streaming=True)
        c4_texts = []
        for i, row in enumerate(c4):
            if i >= c4_samples:
                break
            c4_texts.append(row["text"])
        c4_stream = "\n\n".join(c4_texts)
        c4_tokens = tokenizer(c4_stream, return_tensors="pt").input_ids[0]
        n_c4 = c4_tokens.shape[0]
        tokens = torch.cat([tokens, c4_tokens], dim=0)
        print(f"  c4: {n_c4:,} tokens in {time.time() - t0:.0f}s; "
              f"combined {tokens.shape[0]:,}",
              flush=True)

    n = tokens.shape[0]
    print(f"  train stream: {n:,} tokens",
          flush=True)
    return tokens


def iter_batches(tokens: torch.Tensor, generator: torch.Generator,
                 seq_len: int, batch_size: int):
    """Yield (batch_size, seq_len) tensors forever.

    Uses `generator` for randint — generator state is owned by the
    caller so it can be saved to a checkpoint and restored.
    """
    n = tokens.shape[0]
    while True:
        batch = []
        for _ in range(batch_size):
            start = int(torch.randint(0, n - seq_len - 1, (1,),
                                      generator=generator).item())
            batch.append(tokens[start:start + seq_len])
        yield torch.stack(batch, dim=0)


def iter_train_samples(tokenizer, seq_len: int, batch_size: int,
                       seed: int = 0, c4_samples: int = 0):
    """Backwards-compatible wrapper.  Callers that want checkpoint-
    resume should use prepare_train_stream + iter_batches directly."""
    tokens = prepare_train_stream(tokenizer, c4_samples=c4_samples)
    n = tokens.shape[0]
    print(f"  train stream: {n:,} tokens, {n // seq_len:,} "
          f"non-overlapping {seq_len}-token windows",
          flush=True)
    g = torch.Generator()
    g.manual_seed(seed)
    yield from iter_batches(tokens, g, seq_len, batch_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Gradient accumulation steps.  Effective batch "
                        "= batch_size * grad_accum_steps.  Use this to "
                        "emulate paper's batch=4 on a tight memory budget.")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=2000,
                   help="Run PPL eval every N opt-steps.  Default 2000 "
                        "to minimise the ~5-7s/eval tax on long runs; "
                        "drop to 500 for tighter early-convergence "
                        "resolution during debugging.")
    p.add_argument("--eval-max-tokens", type=int, default=50_000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--early-stop-window", type=int, default=5,
                   help="Stop training if the PPL range across the last "
                        "N evals is below --early-stop-min-delta. "
                        "0 disables plateau detection.")
    p.add_argument("--early-stop-min-delta", type=float, default=2.0,
                   help="Minimum PPL range across the recent eval window "
                        "to keep training. Below this = converged.")
    p.add_argument("--early-stop-min-steps", type=int, default=4000,
                   help="Earliest step at which plateau detection can "
                        "trigger.  Prevents stopping during early "
                        "convergence before LR decay has bitten.")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay.  Paper-default-adjacent; 0 disables.")
    p.add_argument("--optimizer", default="adamw8bit",
                   choices=("adamw", "adamw8bit"),
                   help="Student optimizer.  adamw8bit uses bitsandbytes "
                        "(saves ~9 GB Adam state at 7B); adamw is torch default.")
    p.add_argument("--grad-checkpoint", action="store_true", default=True,
                   help="Enable gradient checkpointing on the student "
                        "(trades ~30%% per-step compute for activation memory). "
                        "Use --no-grad-checkpoint to disable.")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint",
                   action="store_false")
    p.add_argument("--c4-samples", type=int, default=0,
                   help="Number of C4 samples to mix into the train stream. "
                        "0 disables C4.  Paper uses 'selected partitions "
                        "from C4'; ~2000 adds ~600k tokens of diversity.")
    p.add_argument("--inter-mse-weight", type=float, default=10.0,
                   help="Weight for intermediate hidden-state MSE. "
                        "Paper's value is 10.0; set 0 to disable. "
                        "Loss contribution is l2l_weight * sum_{i in layers[1:]} "
                        "MSE(student_h_i, teacher_h_i).")
    p.add_argument("--out", default="littlebit_qat_model.json")
    p.add_argument("--gpu-mem-fraction", type=float, default=0.80,
                   help="Cap PyTorch GPU memory at this fraction of total "
                        "(e.g. 0.80 = 12.8 GB on 16 GB).  Allocations past "
                        "this raise CUDA OOM, which is recoverable, instead "
                        "of risking system-level OOM / driver hang.")
    p.add_argument("--checkpoint", default="littlebit_qat_checkpoint.pt",
                   help="End-of-training state_dict path (always saved)")
    p.add_argument("--init-cache", default="littlebit_qat_init_cache.pt",
                   help="Cache Dual-SVID-initialized student here. If the "
                        "file exists, skip the wrap and load instead "
                        "(saves ~5 min on re-runs).")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="Save rolling training checkpoint every N opt-steps. "
                        "0 disables (only end-of-training save).  Set to "
                        "--eval-every or 2*eval-every for pause/resume.")
    p.add_argument("--rolling-ckpt",
                   default="littlebit_qat_rolling.pt",
                   help="Path for rolling periodic checkpoint (overwritten "
                        "each save).  Includes model + optimizer + RNG.")
    p.add_argument("--best-ckpt",
                   default="littlebit_qat_best.pt",
                   help="Path for best-PPL checkpoint.  Empty string "
                        "disables; otherwise updated whenever eval PPL "
                        "improves.")
    p.add_argument("--resume", default="",
                   help="Resume training from this checkpoint path.  "
                        "Restores model + optimizer + step + RNG so the "
                        "run continues bit-identically to an uninterrupted "
                        "run.  Empty disables.")
    p.add_argument("--save-dtype", default="bf16",
                   choices=("fp32", "bf16"),
                   help="Precision for saved checkpoints.  bf16 halves "
                        "file size with no measurable quality loss on "
                        "round-trip (load auto-casts to fp32 for training).")
    p.add_argument("--compile", action="store_true", default=True,
                   help="Enable torch.compile on the student model. "
                        "Potential 30-50%% per-step speedup via graph "
                        "fusion (measured ~2.1x on 0.5B forward on "
                        "Blackwell).  Requires Triton; on Windows, "
                        "install `triton-windows` (see requirements.txt). "
                        "Gracefully falls back to eager mode if "
                        "compilation fails.  --no-compile to disable.")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--compile-mode", default="default",
                   choices=("default", "reduce-overhead",
                            "max-autotune", "max-autotune-no-cudagraphs"),
                   help="torch.compile optimisation mode.  `default` "
                        "applies inductor kernel fusion without CUDA "
                        "graphs — safe for training.  `reduce-overhead` "
                        "adds CUDA graphs but reuses output buffers "
                        "across replays, which is incompatible with "
                        "training loops that thread activations into "
                        "a later loss (raises 'accessing tensor output "
                        "of CUDAGraphs that has been overwritten').  "
                        "Only use it for pure inference.")
    p.add_argument("--liger", action="store_true", default=True,
                   help="Apply Liger Kernel fused RMSNorm + RoPE to the "
                        "student.  Excludes SwiGLU (our LittleBit wrap "
                        "replaces the Linears in the MLP) and fused "
                        "cross-entropy (we use soft-target KL, not CE). "
                        "Requires Triton; on Windows install "
                        "`triton-windows` and `pip install --no-deps "
                        "liger-kernel` (see requirements.txt). "
                        "--no-liger to disable.")
    p.add_argument("--no-liger", dest="liger", action="store_false")
    p.add_argument("--mse-via-hooks", action="store_true", default=False,
                   help="Capture hidden states for MSE loss via forward "
                        "hooks instead of output_hidden_states=True. "
                        "Correct but off by default: the 2026-04-22 "
                        "benchmark on 0.5B showed zero measurable "
                        "memory win vs output_hidden_states=True "
                        "(10.17 GB peak both ways) and a ~3.5%% wall "
                        "tax from hook side-effects (701 vs 677 s for "
                        "1000 opt-steps).  The earlier 2x-PPL "
                        "regression was a real bug — hooks captured "
                        "pre-final-norm for the last layer instead of "
                        "post-final-norm — and is now fixed, so "
                        "--mse-via-hooks is safe if someone wants to "
                        "re-benchmark it at 7B+ where activation "
                        "footprint scales larger and the savings may "
                        "actually materialize.  "
                        "--no-mse-via-hooks is the shipping default.")
    p.add_argument("--no-mse-via-hooks", dest="mse_via_hooks",
                   action="store_false")
    p.add_argument("--tf32", action="store_true", default=True,
                   help="Enable TF32 on matmul + cuDNN (Ampere+ / Blackwell). "
                        "~13%% matmul speedup.  --no-tf32 to disable "
                        "for ablation.")
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--shadow-dtype", default="fp32",
                   choices=("fp32", "bf16"),
                   help="Precision of the U_fp / V_fp shadow matrices "
                        "inside each LittleBitLinear.  fp32 is the "
                        "conservative default that matches the paper.  "
                        "bf16 halves the shadow-weight memory + their "
                        "gradient + 8-bit-Adam state — saves ~600 MB at "
                        "0.5B, ~5.5 GB at 7B, ~24 GB at 30B — the one "
                        "lever in our stack that scales proportionally "
                        "with model size.  Numerical risk: SmoothSign's "
                        "tanh(tau*x) surrogate is sharp near x=0; bf16's "
                        "7-bit mantissa may alias nearby x values, "
                        "though in practice the saturated region (|x| >> "
                        "1/tau) is the dominant regime and is fine.  "
                        "Scales h/g/ell/bias stay fp32 regardless.")
    p.add_argument("--teacher-cache", default=None,
                   help="Path to an offline teacher cache produced by "
                        "littlebit_teacher_extract.py.  When provided, "
                        "the teacher model is NOT loaded; per-micro-step "
                        "top-k logits and hidden states are read from the "
                        "cache via mmap, eliminating the teacher from GPU "
                        "memory and from the per-step forward cost.  "
                        "Cache is trajectory-keyed — training must use "
                        "the same seed / seq_len / batch_size / "
                        "grad_accum_steps / c4_samples / teacher model "
                        "as extraction (validated on load).")
    p.add_argument("--delete-init-cache-after-start", action="store_true",
                   default=False,
                   help="Delete the init-cache file once training's first "
                        "step completes.  Saves 1.5-30 GB depending on "
                        "scale but loses the cache for future runs at the "
                        "same model+rank.  Default off (preserves cache).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Free wall-time knobs (Sprint 0 Tier A).  TF32 on Ampere+ gives
    # ~13% matmul speedup for fp32 with essentially no quality cost.
    # cudnn.benchmark autotunes conv kernel selection for fixed shapes.
    if device.type == "cuda" and args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("tf32: on (matmul), cudnn.benchmark: on", flush=True)
    elif device.type == "cuda":
        # Explicit disable — force fp32 matmul to measure TF32 contribution
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        print("tf32: OFF (ablation)", flush=True)

    # Cap GPU memory so allocations past the cap OOM cleanly rather
    # than taking the whole system down with a driver hang.
    if device.type == "cuda" and 0 < args.gpu_mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap_gb = total_gb * args.gpu_mem_fraction
        print(f"gpu memory cap: {args.gpu_mem_fraction * 100:.0f}% "
              f"of {total_gb:.1f} GB = {cap_gb:.1f} GB", flush=True)

    def _gpu_mem(label: str) -> None:
        if device.type != "cuda":
            return
        used_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"  [mem:{label}] allocated={used_gb:.2f} GB "
              f"reserved={reserved_gb:.2f} GB", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Teacher path: either live (bf16 on GPU) or cached (mmap from disk).
    # Cached path eliminates ~1 GB VRAM at 0.5B and ~14 GB at 7B, plus
    # the per-step teacher forward cost.
    teacher = None
    teacher_cache = None
    if args.teacher_cache:
        from teacher_cache import TeacherCacheReader, compute_corpus_hash
        print(f"using teacher cache: {args.teacher_cache}", flush=True)
        teacher_cache = TeacherCacheReader(Path(args.teacher_cache))
        # We haven't built the token stream yet; defer config validation
        # until after prepare_train_stream + corpus_hash below.
        teacher_ppl = (teacher_cache.meta.teacher_ppl
                       if teacher_cache.meta.teacher_ppl is not None
                       else float("nan"))
        print(f"  teacher PPL (from cache metadata): {teacher_ppl:.3f}"
              if teacher_cache.meta.teacher_ppl is not None
              else "  teacher PPL: n/a (cache did not record it)",
              flush=True)
    else:
        print(f"loading teacher {args.model} (bfloat16)...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
        ).to(device)
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print("teacher loaded.")
        _gpu_mem("after teacher")

        # Baseline PPL (teacher)
        print("evaluating teacher PPL...")
        t0 = time.time()
        teacher_ppl = wikitext_ppl(teacher, tokenizer,
                                   max_tokens=args.eval_max_tokens,
                                   seq_len=args.seq_len, device=device)
        print(f"  teacher PPL = {teacher_ppl:.3f}  "
              f"({time.time() - t0:.1f}s)")

    # Student: separate copy, then wrap, then move to device.
    # (Wrap creates new CPU tensors via np.linalg.svd; must .to(device)
    # the whole model after wrapping.)
    _shadow_dtype = {
        "fp32": torch.float32, "bf16": torch.bfloat16,
    }[args.shadow_dtype]
    init_cache_path = Path(args.init_cache)
    print(f"loading student copy and wrapping at r={args.rank}...")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    )

    # Liger fused kernels BEFORE LittleBit wrap.  Liger patches
    # RMSNorm / RoPE at the class-instance level; our LittleBit
    # wrap then replaces specific nn.Linear instances.  Order
    # matters: if we wrap first, some Liger patches may fail to
    # detect their target modules.  Skip SwiGLU (our LittleBit
    # Linears live inside the MLP) and cross-entropy (our loss is
    # soft-target KL, not standard CE).
    if args.liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True,
                rms_norm=True,
                swiglu=False,          # MLP contains our wrapped Linears
                cross_entropy=False,   # we use KL
                fused_linear_cross_entropy=False,
                model=student,
            )
            print("liger kernels: rope + rms_norm applied to student",
                  flush=True)
        except ImportError:
            print("liger not installed; continuing with PyTorch native",
                  flush=True)
        except Exception as e:
            print(f"liger apply failed ({e}); continuing without",
                  flush=True)

    if init_cache_path.exists():
        # Fast path: skip Dual-SVID SVDs, load cached init instead.
        print(f"  init-cache hit: {init_cache_path} "
              f"(skipping Dual-SVID wrap)",
              flush=True)
        # We still need to wrap to put LittleBitLinearHF modules in
        # place; then load their learned parameters from cache.
        wrapped = wrap_model_littlebit_shapes(student, r=args.rank,
                                              tau=args.tau,
                                              shadow_dtype=_shadow_dtype)
        state = torch.load(init_cache_path, map_location="cpu",
                           weights_only=True)
        # The cached init may be fp32 even when we're training with
        # bf16 shadows.  Cast U_fp / V_fp tensors to the target dtype
        # at load time — scales stay fp32 regardless.
        if _shadow_dtype is not torch.float32:
            for name, tensor in list(state.items()):
                if name.endswith(".U_fp") or name.endswith(".V_fp"):
                    state[name] = tensor.to(_shadow_dtype)
        student.load_state_dict(state)
        print(f"  loaded cached init in {time.time() - t0:.1f}s "
              f"({wrapped} layers; shadow_dtype={args.shadow_dtype})",
              flush=True)
    else:
        wrapped = wrap_model_littlebit(student, r=args.rank, tau=args.tau,
                                       shadow_dtype=_shadow_dtype)
        print(f"  wrapped {wrapped} Linear layers  "
              f"({time.time() - t0:.1f}s; shadow_dtype={args.shadow_dtype})")
        # Cache for future runs.  Save on CPU so the file is portable.
        try:
            torch.save(student.state_dict(), init_cache_path)
            print(f"  cached Dual-SVID init to {init_cache_path}",
                  flush=True)
        except Exception as e:
            print(f"  warning: init cache save failed: {e}", flush=True)
    student.to(device)

    # Gradient checkpointing: trades ~30% per-step compute for
    # activation memory.  Required for seq=1024 at batch>=2 on
    # 16 GB GPU.  Safe to keep on even at smaller configs.
    if args.grad_checkpoint:
        try:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # HF training mode requires input grads for checkpointing
            # to work with teacher forcing.  Only enable if the model
            # supports the helper.
            if hasattr(student, "enable_input_require_grads"):
                student.enable_input_require_grads()
            print("gradient checkpointing: enabled on student")
        except Exception as e:
            print(f"gradient_checkpointing_enable failed ({e}); continuing")

    # Count student trainable params.
    trainable_params = sum(p.numel() for p in student.parameters()
                           if p.requires_grad)
    print(f"  student trainable params: {trainable_params:,}")
    _gpu_mem("after student")

    # Hidden-state capture hooks for MSE (installed BEFORE compile
    # so they persist through module wrapping).  Only register when
    # both MSE is active and --mse-via-hooks is enabled.
    student_hidden = None
    teacher_hidden = None
    use_hook_mse = bool(args.inter_mse_weight) and args.mse_via_hooks
    if use_hook_mse:
        student_hidden = HiddenCapture()
        student_hidden.install(student)
        # Teacher hooks only when the teacher is live.  Under --teacher-cache
        # the teacher's hidden states are already materialised on disk
        # (stored as post-layer outputs plus post-final-norm, matching the
        # hook-fix layout) and will be loaded per micro-step.
        if teacher is not None:
            teacher_hidden = HiddenCapture()
            teacher_hidden.install(teacher)
        print(f"hidden-state capture: hook-based "
              f"({len(student.model.layers)} decoder layers"
              f"{'; teacher side via cache' if teacher is None else ''})",
              flush=True)

    # torch.compile: inductor kernel fusion.  PyTorch's inductor
    # backend requires Triton; on Windows install `triton-windows`
    # (see requirements.txt).  Detect at startup so we don't fail
    # on the first forward pass mid-training.
    #
    # Mode choice: `default` (kernel fusion only) rather than
    # `reduce-overhead` (fusion + CUDA graphs).  CUDA graphs reuse
    # the same memory buffers across replays, so any tensor kept
    # alive past the next compiled forward (e.g. for backward or
    # for our hidden-state MSE) gets overwritten and raises
    # `accessing tensor output of CUDAGraphs that has been
    # overwritten`.  That's fatal for training loops that thread
    # activations into a later loss; safe only for pure inference.
    if args.compile:
        try:
            import triton  # noqa: F401
            triton_available = True
        except ImportError:
            triton_available = False

        if not triton_available:
            print("torch.compile: skipped (triton not importable; "
                  "on Windows install `triton-windows`).  "
                  "--no-compile to silence this message.",
                  flush=True)
        else:
            try:
                compile_t0 = time.time()
                student = torch.compile(student, mode=args.compile_mode,
                                        fullgraph=False, dynamic=False)
                print(f"torch.compile: wrapped student "
                      f"(mode={args.compile_mode}, "
                      f"graph_breaks_allowed=True) in "
                      f"{time.time() - compile_t0:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"torch.compile failed ({e}); running eager",
                      flush=True)

    # Initial (post-init, pre-QAT) PPL
    print("evaluating student PPL post-init (pre-QAT)...")
    t0 = time.time()
    init_ppl = wikitext_ppl(student, tokenizer,
                            max_tokens=args.eval_max_tokens,
                            seq_len=args.seq_len, device=device)
    print(f"  init PPL = {init_ppl:.3f}  "
          f"({time.time() - t0:.1f}s)")

    history = [{
        "step": 0, "loss": None,
        "ppl": init_ppl, "teacher_ppl": teacher_ppl,
    }]

    # Kill-criterion early-exit if init PPL is unmeasurable.
    if not math.isfinite(init_ppl) or init_ppl > 1e6:
        print(f"init PPL {init_ppl} is already broken; saving and exiting.")
        Path(args.out).write_text(json.dumps({
            "status": "init_broken",
            "teacher_ppl": teacher_ppl, "init_ppl": init_ppl,
            "args": vars(args),
            "history": history,
        }, indent=2))
        return

    params = [p for p in student.parameters() if p.requires_grad]
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=args.lr,
                                      weight_decay=args.weight_decay)
            print(f"optimizer: bnb.AdamW8bit  "
                  f"(weight_decay={args.weight_decay})")
        except Exception as e:
            print(f"bitsandbytes unavailable ({e}); falling back to torch AdamW")
            opt = torch.optim.AdamW(params, lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr,
                                weight_decay=args.weight_decay)
        print(f"optimizer: torch.AdamW  "
              f"(weight_decay={args.weight_decay})")

    # Simple cosine LR with warmup
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / \
                   max(1, args.steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    kl = nn.KLDivLoss(reduction="batchmean", log_target=False)
    # Sampler with externally-owned generator, for checkpoint-resume.
    train_tokens = prepare_train_stream(tokenizer,
                                        c4_samples=args.c4_samples)
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(0)
    print(f"  train stream: {train_tokens.shape[0]:,} tokens, "
          f"{train_tokens.shape[0] // args.seq_len:,} "
          f"non-overlapping {args.seq_len}-token windows",
          flush=True)
    it = iter_batches(train_tokens, sampler_gen,
                      args.seq_len, args.batch_size)

    accum = max(1, args.grad_accum_steps)
    effective_batch = args.batch_size * accum

    # Validate teacher-cache config now that the token stream is built
    # (we need its corpus_hash).  Fails fast if cache was produced for
    # a different trajectory.
    if teacher_cache is not None:
        from teacher_cache import compute_corpus_hash
        teacher_cache.validate_config(
            seed=0,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_accum_steps=accum,
            c4_samples=args.c4_samples,
            teacher_model=args.model,
            corpus_hash=compute_corpus_hash(train_tokens),
            required_steps=args.steps,
        )
        print(f"  teacher cache validated: {teacher_cache.meta.n_microsteps} "
              f"micro-steps available for {args.steps} opt-steps × "
              f"{accum} accum", flush=True)

    # Resume-from-checkpoint support: if --resume points to a valid
    # rolling checkpoint, restore model / optimizer / RNG / step /
    # history and skip ahead to the saved step.
    start_step = 1
    best_ppl = float("inf")
    best_step_from_ckpt: int | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"--resume {resume_path} does not exist"
            )
        print(f"resuming from {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, map_location="cpu",
                          weights_only=False)
        # Cast checkpoint tensors back to student's native dtype.
        # Student is fp32 in memory for SmoothSign backward stability;
        # saved tensors may be bf16 (new default) or fp32 (legacy).
        native_dtype = next(student.parameters()).dtype
        model_sd = {k: (v.to(native_dtype) if torch.is_tensor(v)
                        and v.is_floating_point() else v)
                    for k, v in ckpt["model"].items()}
        student.load_state_dict(model_sd)
        opt.load_state_dict(ckpt["opt"])
        torch.set_rng_state(ckpt["rng_torch"])
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        sampler_gen.set_state(ckpt["rng_sampler"])
        start_step = int(ckpt["step"]) + 1
        history = ckpt.get("history", history)
        best_ppl = float(ckpt.get("best_ppl", float("inf")))
        best_step_from_ckpt = ckpt.get("best_step")
        print(f"  resumed at step {start_step - 1}; "
              f"continuing to {args.steps} "
              f"(best_ppl so far: {best_ppl:.3f})", flush=True)

    print(f"training: {args.steps} opt-steps, lr={args.lr}, "
          f"batch={args.batch_size} × accum={accum} "
          f"= effective {effective_batch}, seq_len={args.seq_len}")
    _gpu_mem("pre-train-loop")

    def _cast_state_dict(sd, dtype):
        """Cast all floating-point tensors to `dtype` for save."""
        if dtype is None:
            return sd
        out = {}
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                out[k] = v.to(dtype)
            else:
                out[k] = v
        return out

    _save_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.save_dtype]

    def save_rolling(step_now: int, history_snapshot: list) -> None:
        """Single-file rolling checkpoint, overwrites each save."""
        tmp = Path(f"{args.rolling_ckpt}.tmp")
        final = Path(args.rolling_ckpt)
        payload = {
            "step": step_now,
            "model": _cast_state_dict(student.state_dict(), _save_dtype),
            "opt": opt.state_dict(),  # keep optimizer native precision
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                          if torch.cuda.is_available() else None),
            "rng_sampler": sampler_gen.get_state(),
            "args": vars(args),
            "history": history_snapshot,
            "best_ppl": best_ppl,
            "best_step": best_step_from_ckpt,
            "wrapped_layers": wrapped,
            "save_dtype": args.save_dtype,
        }
        torch.save(payload, tmp)
        # Atomic-ish rename — on Windows this fails if `final` is
        # open.  os.replace handles the overwrite cleanly.
        import os
        os.replace(tmp, final)

    loss_recent = []
    t0 = time.time()
    for step in range(start_step, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(accum):
            batch = next(it).to(device)
            microstep_idx = (step - 1) * accum + micro
            # Clear hook captures from previous micro-step.
            if use_hook_mse:
                student_hidden.clear()
                if teacher_hidden is not None:
                    teacher_hidden.clear()
            # When hooks are active, output_hidden_states is False
            # to avoid doubling work.
            want_hidden = bool(args.inter_mse_weight) and not use_hook_mse

            # Teacher signal: either live forward or from cache.
            if teacher_cache is not None:
                t_topk_vals, t_topk_idx, t_cached_hidden = teacher_cache.get(
                    microstep_idx, device
                )
                t_logits = None
            else:
                t_topk_vals = t_topk_idx = t_cached_hidden = None
                with torch.no_grad():
                    t_out = teacher(batch, output_hidden_states=want_hidden)
                    t_logits = t_out.logits
                    t_hidden = t_out.hidden_states if want_hidden else None

            # Student forward
            s_out = student(batch, output_hidden_states=want_hidden)
            s_logits = s_out.logits

            if teacher_cache is not None:
                # Top-k truncated KL against cached teacher logits.
                from teacher_cache import kl_topk_loss
                l_kl = kl_topk_loss(s_logits, t_topk_vals, t_topk_idx)
                del s_logits
            else:
                log_p_s = torch.nn.functional.log_softmax(s_logits, dim=-1)
                with torch.no_grad():
                    p_t = torch.nn.functional.softmax(t_logits, dim=-1).to(log_p_s.dtype)
                l_kl = kl(
                    log_p_s.view(-1, log_p_s.size(-1)),
                    p_t.view(-1, p_t.size(-1)),
                )
                del log_p_s, p_t, s_logits, t_logits
            loss = l_kl

            if args.inter_mse_weight:
                if use_hook_mse:
                    # Hook-captured states: aligned with
                    # output_hidden_states[1:], one per decoder layer.
                    s_hidden = student_hidden.states
                else:
                    s_hidden = s_out.hidden_states[1:]
                if teacher_cache is not None:
                    t_hidden_list = t_cached_hidden
                elif use_hook_mse:
                    t_hidden_list = teacher_hidden.states
                else:
                    t_hidden_list = t_hidden[1:]
                l_inter = 0.0
                for sh, th in zip(s_hidden, t_hidden_list):
                    l_inter = l_inter + torch.nn.functional.mse_loss(
                        sh, th.to(sh.dtype)
                    )
                loss = loss + args.inter_mse_weight * l_inter
                # Release hook references so gradient checkpointing
                # can actually drop these activations during backward.
                if use_hook_mse:
                    student_hidden.clear()
                    if teacher_hidden is not None:
                        teacher_hidden.clear()

            # Average micro-step losses so the effective gradient
            # matches a single batch=effective_batch forward.
            (loss / accum).backward()
            step_loss += loss.item()
            del loss

        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        loss_recent.append(step_loss / accum)
        if step == 1:
            _gpu_mem("after step 1 (peak)")
            # Optionally reclaim init-cache disk.  Only meaningful if
            # this run just created or used a cache file — by step 1
            # the wrapped state is in GPU memory so the cache is
            # redundant for *this* run.  Off by default to preserve
            # the cache for re-use across runs at same model+rank.
            if args.delete_init_cache_after_start and init_cache_path.exists():
                try:
                    size_mb = init_cache_path.stat().st_size / (1024 * 1024)
                    init_cache_path.unlink()
                    print(f"  deleted init cache at step 1 "
                          f"({size_mb:.0f} MB freed)",
                          flush=True)
                except Exception as e:
                    print(f"  warn: init-cache delete failed: {e}",
                          flush=True)
        if step % args.log_every == 0:
            recent = float(np.mean(loss_recent[-args.log_every:]))
            print(f"  step {step:5d}  lr={lr_at(step-1):.2e}  "
                  f"loss={recent:.4f}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            print(f"  evaluating PPL at step {step}...", flush=True)
            te = time.time()
            ppl = wikitext_ppl(student, tokenizer,
                                max_tokens=args.eval_max_tokens,
                                seq_len=args.seq_len, device=device)
            recent = float(np.mean(loss_recent[-args.log_every:]))
            history.append({
                "step": step, "loss": recent, "ppl": ppl,
                "teacher_ppl": teacher_ppl,
            })
            print(f"    PPL={ppl:.3f}  (teacher={teacher_ppl:.3f}, "
                  f"init={init_ppl:.3f})  eval took {time.time()-te:.0f}s")
            # Track best PPL checkpoint (for early-stop recovery)
            if ppl < best_ppl:
                best_ppl = ppl
                best_step_from_ckpt = step
                if args.best_ckpt:
                    try:
                        torch.save(
                            {"step": step, "ppl": ppl,
                             "model": _cast_state_dict(
                                 student.state_dict(), _save_dtype),
                             "save_dtype": args.save_dtype,
                             "config": {
                                 "model": args.model, "rank": args.rank,
                                 "tau": args.tau, "steps": args.steps,
                                 "seq_len": args.seq_len,
                                 "wrapped_layers": wrapped,
                             }},
                            args.best_ckpt,
                        )
                    except Exception as e:
                        print(f"    warn: best-ckpt save failed: {e}")
            # Kill on runaway PPL
            if step >= args.warmup_steps and ppl > max(200.0, init_ppl * 1.5):
                print("    kill criterion: PPL runaway, stopping")
                history[-1]["killed"] = "ppl_runaway"
                break
            # Plateau early-stop: if the most recent `window` evals
            # have a PPL range below `min_delta`, training has
            # converged and further steps won't meaningfully help.
            if args.early_stop_window > 0 and step >= args.early_stop_min_steps:
                eval_hist = [h for h in history[1:]
                             if h.get("ppl") is not None]
                if len(eval_hist) >= args.early_stop_window:
                    window_ppls = [h["ppl"]
                                   for h in eval_hist[-args.early_stop_window:]]
                    ppl_range = max(window_ppls) - min(window_ppls)
                    if ppl_range < args.early_stop_min_delta:
                        print(f"    plateau early-stop: last "
                              f"{args.early_stop_window} evals "
                              f"ranged only {ppl_range:.2f} PPL "
                              f"(< {args.early_stop_min_delta:.2f}), "
                              f"stopping at step {step}")
                        history[-1]["killed"] = "plateau"
                        # Save final rolling ckpt so we can resume if
                        # we change our mind.
                        if args.ckpt_every > 0:
                            try:
                                save_rolling(step, history)
                            except Exception:
                                pass
                        break

            # Note: no student.train() — wikitext_ppl does not call
            # .eval() anymore (see the compile-safety docstring there),
            # so student.training stays True throughout training.  A
            # .train() call here would trigger a dynamo cache evict on
            # the next compiled forward.

        # Rolling periodic checkpoint (overwrite-in-place).  Fires
        # independently of eval, so pausing is possible between evals.
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            try:
                save_rolling(step, history)
            except Exception as e:
                print(f"    warn: rolling ckpt save failed at step "
                      f"{step}: {e}", flush=True)

    best = min(history[1:], key=lambda h: h.get("ppl", float("inf")),
               default=history[0])
    summary = {
        "model": args.model, "rank": args.rank, "tau": args.tau,
        "steps": args.steps, "lr": args.lr,
        "batch_size": args.batch_size, "seq_len": args.seq_len,
        "inter_mse_weight": args.inter_mse_weight,
        "teacher_ppl": teacher_ppl,
        "init_ppl": init_ppl,
        "final_ppl": history[-1].get("ppl"),
        "best_ppl": best.get("ppl"),
        "best_step": best.get("step"),
        "history": history,
        "wrapped_layers": wrapped,
        "trainable_params": trainable_params,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

    # Save end-of-training checkpoint so downstream eval / resume
    # runs don't have to retrain.
    try:
        ckpt_path = Path(args.checkpoint)
        torch.save({
            "state_dict": _cast_state_dict(student.state_dict(), _save_dtype),
            "save_dtype": args.save_dtype,
            "config": {
                "model": args.model, "rank": args.rank, "tau": args.tau,
                "steps": args.steps, "seq_len": args.seq_len,
                "wrapped_layers": wrapped,
            },
            "summary": {k: summary[k] for k in
                        ("teacher_ppl", "init_ppl", "final_ppl",
                         "best_ppl", "best_step")},
        }, ckpt_path)
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"checkpoint: {ckpt_path}  ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"warning: checkpoint save failed: {e}")

    print(f"teacher {teacher_ppl:.3f}  init {init_ppl:.3f}  "
          f"best {best.get('ppl', float('inf')):.3f} "
          f"(step {best.get('step')})")


if __name__ == "__main__":
    main()
