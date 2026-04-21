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
      U_fp (d_out, r), V_fp (d_in, r) — soft factors, sign each step
      h (d_out), g (d_in), ell (r)     — FP16 scale vectors
      bias (d_out)                     — if original Linear had one
    """

    def __init__(self, d_in: int, d_out: int, r: int, bias: bool,
                 tau: float = 100.0):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.r = r
        self.tau = tau
        self.U_fp = nn.Parameter(torch.empty(d_out, r))
        self.V_fp = nn.Parameter(torch.empty(d_in, r))
        self.h = nn.Parameter(torch.empty(d_out))
        self.g = nn.Parameter(torch.empty(d_in))
        self.ell = nn.Parameter(torch.empty(r))
        if bias:
            self.bias = nn.Parameter(torch.empty(d_out))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, r: int,
                    tau: float = 100.0) -> "LittleBitLinearHF":
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
                  bias=lin.bias is not None, tau=tau)
        with torch.no_grad():
            out.U_fp.copy_(torch.tensor(Up, dtype=torch.float32))
            out.V_fp.copy_(torch.tensor(Vp, dtype=torch.float32))
            out.h.copy_(torch.tensor(h0,   dtype=torch.float32))
            out.g.copy_(torch.tensor(g0,   dtype=torch.float32))
            out.ell.copy_(torch.tensor(ell0, dtype=torch.float32))
            if lin.bias is not None:
                out.bias.copy_(lin.bias.data.detach().to(torch.float32).cpu())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. 5 efficient form: Y = ((((X*g) @ V_sign) * ell) @ U_sign.T) * h
        # x: (..., d_in)
        U_sign = smooth_sign(self.U_fp, self.tau)
        V_sign = smooth_sign(self.V_fp, self.tau)
        y = x * self.g
        y = y @ V_sign
        y = y * self.ell
        y = y @ U_sign.T
        y = y * self.h
        if self.bias is not None:
            y = y + self.bias
        return y


def wrap_model_littlebit_shapes(model: nn.Module, r: int,
                                tau: float = 100.0,
                                skip: tuple[str, ...] = ("lm_head",)) -> int:
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
                                    bias=child.bias is not None, tau=tau)
            setattr(module, child_name, new)
            count += 1
    return count


def wrap_model_littlebit(model: nn.Module, r: int,
                         tau: float = 100.0,
                         skip: tuple[str, ...] = ("lm_head",),
                         log_every: int = 10) -> int:
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
        new = LittleBitLinearHF.from_linear(child, r=r, tau=tau)
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
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"] for row in ds)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]

    model.eval()
    nll_sum = 0.0
    count = 0
    for i in range(0, len(ids) - 1, seq_len):
        chunk = ids[i:i + seq_len].to(device).unsqueeze(0)
        if chunk.shape[1] < 2:
            break
        logits = model(chunk).logits  # (1, T, V)
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
    p.add_argument("--eval-every", type=int, default=500)
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
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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

    print(f"loading teacher {args.model} (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    init_cache_path = Path(args.init_cache)
    print(f"loading student copy and wrapping at r={args.rank}...")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    )
    if init_cache_path.exists():
        # Fast path: skip Dual-SVID SVDs, load cached init instead.
        print(f"  init-cache hit: {init_cache_path} "
              f"(skipping Dual-SVID wrap)",
              flush=True)
        # We still need to wrap to put LittleBitLinearHF modules in
        # place; then load their learned parameters from cache.
        wrapped = wrap_model_littlebit_shapes(student, r=args.rank,
                                              tau=args.tau)
        state = torch.load(init_cache_path, map_location="cpu",
                           weights_only=True)
        student.load_state_dict(state)
        print(f"  loaded cached init in {time.time() - t0:.1f}s "
              f"({wrapped} layers)", flush=True)
    else:
        wrapped = wrap_model_littlebit(student, r=args.rank, tau=args.tau)
        print(f"  wrapped {wrapped} Linear layers  "
              f"({time.time() - t0:.1f}s)")
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
        student.load_state_dict(ckpt["model"])
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

    def save_rolling(step_now: int, history_snapshot: list) -> None:
        """Single-file rolling checkpoint, overwrites each save."""
        tmp = Path(f"{args.rolling_ckpt}.tmp")
        final = Path(args.rolling_ckpt)
        payload = {
            "step": step_now,
            "model": student.state_dict(),
            "opt": opt.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                          if torch.cuda.is_available() else None),
            "rng_sampler": sampler_gen.get_state(),
            "args": vars(args),
            "history": history_snapshot,
            "best_ppl": best_ppl,
            "best_step": best_step_from_ckpt,
            "wrapped_layers": wrapped,
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
            # Teacher forward (no grad, bf16 logits/hidden)
            with torch.no_grad():
                t_out = teacher(batch, output_hidden_states=bool(args.inter_mse_weight))
                t_logits = t_out.logits
                t_hidden = t_out.hidden_states if args.inter_mse_weight else None

            # Student forward
            s_out = student(batch, output_hidden_states=bool(args.inter_mse_weight))
            s_logits = s_out.logits

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
                s_hidden = s_out.hidden_states[1:]
                t_hidden_list = t_hidden[1:]
                l_inter = 0.0
                for sh, th in zip(s_hidden, t_hidden_list):
                    l_inter = l_inter + torch.nn.functional.mse_loss(
                        sh, th.to(sh.dtype)
                    )
                loss = loss + args.inter_mse_weight * l_inter

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
        if step % args.log_every == 0:
            recent = float(np.mean(loss_recent[-args.log_every:]))
            print(f"  step {step:5d}  lr={lr_at(step-1):.2e}  "
                  f"loss={recent:.4f}  elapsed={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"  evaluating PPL at step {step}...")
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
                             "model": student.state_dict(),
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

            student.train()

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
            "state_dict": student.state_dict(),
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
