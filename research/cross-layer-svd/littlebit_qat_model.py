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

from littlebit_qat_single import smooth_sign


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


def iter_train_samples(tokenizer, seq_len: int, batch_size: int,
                       seed: int = 0):
    """Yield (batch_size, seq_len) tensors from wikitext-2 train,
    looping forever.

    Wikitext rows are short (max ~500 tokens with Qwen tokenizer), so
    we concatenate the whole train split into one stream and slide
    random windows.  Same pattern as wikitext_ppl() in this file.
    """
    from datasets import load_dataset
    print("  preparing train token stream...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row["text"] for row in ds)
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    n = tokens.shape[0]
    print(f"  train stream: {n:,} tokens, {n // seq_len:,} "
          f"non-overlapping {seq_len}-token windows",
          flush=True)
    g = torch.Generator()
    g.manual_seed(seed)
    while True:
        batch = []
        for _ in range(batch_size):
            start = int(torch.randint(0, n - seq_len - 1, (1,),
                                      generator=g).item())
            batch.append(tokens[start:start + seq_len])
        yield torch.stack(batch, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-max-tokens", type=int, default=50_000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--inter-mse-weight", type=float, default=0.0,
                   help="Weight for intermediate hidden-state MSE; 0 disables")
    p.add_argument("--out", default="littlebit_qat_model.json")
    p.add_argument("--checkpoint", default="littlebit_qat_checkpoint.pt",
                   help="End-of-training state_dict path (always saved)")
    p.add_argument("--init-cache", default="littlebit_qat_init_cache.pt",
                   help="Cache Dual-SVID-initialized student here. If the "
                        "file exists, skip the wrap and load instead "
                        "(saves ~5 min on re-runs).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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

    # Count student trainable params.
    trainable_params = sum(p.numel() for p in student.parameters()
                           if p.requires_grad)
    print(f"  student trainable params: {trainable_params:,}")

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

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Simple cosine LR with warmup
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / \
                   max(1, args.steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    kl = nn.KLDivLoss(reduction="batchmean", log_target=False)
    it = iter_train_samples(tokenizer, args.seq_len, args.batch_size)

    print(f"training: {args.steps} steps, lr={args.lr}, "
          f"batch={args.batch_size}, seq_len={args.seq_len}")
    loss_recent = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)

        batch = next(it).to(device)
        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = teacher(batch, output_hidden_states=bool(args.inter_mse_weight))
            t_logits = t_out.logits.float()
            t_hidden = t_out.hidden_states if args.inter_mse_weight else None

        # Student forward
        s_out = student(batch, output_hidden_states=bool(args.inter_mse_weight))
        s_logits = s_out.logits.float()

        # KL(student || teacher) with log-softmax on student, softmax on teacher
        l_kl = kl(
            torch.nn.functional.log_softmax(s_logits, dim=-1).view(-1, s_logits.size(-1)),
            torch.nn.functional.softmax(t_logits, dim=-1).view(-1, t_logits.size(-1)),
        )
        loss = l_kl

        if args.inter_mse_weight:
            s_hidden = s_out.hidden_states
            l_inter = 0.0
            for sh, th in zip(s_hidden, t_hidden):
                l_inter = l_inter + torch.nn.functional.mse_loss(
                    sh.float(), th.float()
                )
            l_inter = l_inter / max(1, len(s_hidden))
            loss = loss + args.inter_mse_weight * l_inter

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        loss_recent.append(loss.item())
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
            # Kill on runaway PPL
            if step >= args.warmup_steps and ppl > max(200.0, init_ppl * 1.5):
                print("    kill criterion: PPL runaway, stopping")
                history[-1]["killed"] = "ppl_runaway"
                break

            student.train()

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
