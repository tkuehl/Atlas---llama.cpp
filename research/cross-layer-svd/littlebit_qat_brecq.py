"""BRECQ-style per-block QAT for LittleBit.

Implements Stage 4 of the one-shot PTQ sprint (see `stage_4_brecq_plan.md`).
Each transformer block is calibrated independently against the FP32
teacher's block output, using a Fisher-weighted MSE objective:

    L_b = E[ || f_b^{1/2} * (Z_b^student - Z_b^teacher) ||^2 ]

where f_b is the per-coordinate Fisher diagonal from `littlebit_fisher.py`.

Design decisions (locked 2026-04-22, per `stage_4_brecq_plan.md §2`):
  - Signs: RELAXED. U_fp, V_fp learnable via SmoothSign at the training LR.
  - Block input: PURE TEACHER. X_b from teacher forward; never propagated
    from student predecessor output.
  - Parameter scope: JOINT. All 7 linears (q, k, v, o, gate, up, down)
    of the block trained simultaneously against the block-output loss.

S4.0 (first experiment): single block 12 on Qwen 2.5 0.5B, r=512,
500 steps, BRECQ-relaxed.  Measures whether Fisher-weighted block MSE
does meaningful work per-block at modest step count; go signal is
block-output rel-err <= 0.3 at step 500.

Usage:
    python littlebit_qat_brecq.py --model Qwen/Qwen2.5-0.5B \\
        --block 12 --rank 512 --steps 500 \\
        --fisher qwen05b_fisher.pt \\
        --calib-samples 32 --seq-len 2048 \\
        --out s4_0_block12.json
"""

from __future__ import annotations

# Windows UTF-8 bootstrap (same as other scripts in this directory).
import os as _os
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":
    import subprocess as _subprocess
    _env = dict(_os.environ)
    _env["PYTHONUTF8"] = "1"
    _sys.exit(_subprocess.run(
        [_sys.executable] + _sys.argv, env=_env
    ).returncode)

# Force line-buffered stdout/stderr so progress prints stream immediately
# through `tee` / file redirects / background-task output files.  Without
# this, Python's default block buffering can hide progress for many
# minutes when stdout isn't a TTY.  `-u` on the command line does the
# same, but this is belt-and-suspenders.
try:
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn

from littlebit_qat_model import LittleBitLinearHF


# LittleBit-specific parameter name suffixes.  Used to decide which
# parameters in a converted block should receive gradients.  Everything
# else in the block (RMSNorms, any stray weights/biases not in a
# LittleBitLinear) stays frozen at teacher values — this matches the
# LittleBit paper's convention of freezing norms at FP16 and was being
# violated in the initial implementation, contributing to the S4.1/S4.2
# composition catastrophe (see audit 2026-04-22).
_LB_PARAM_SUFFIXES = ("U_fp", "V_fp", ".h", ".g", ".ell", ".bias")


def enable_littlebit_grads(
    module: nn.Module,
    freeze_signs: bool = False,
) -> tuple[int, int]:
    """Set `requires_grad=True` only on LittleBit-specific parameters
    (U_fp, V_fp, h, g, ell, bias) within `module`; freeze everything
    else (RMSNorm weights, teacher-inherited weights/biases not yet
    replaced, etc.).

    If `freeze_signs=True`, also freeze `U_fp`/`V_fp` — BRECQ-strict mode.

    Returns (n_trainable_params, n_frozen_params).
    """
    n_trainable = 0
    n_frozen = 0
    for name, pr in module.named_parameters():
        is_lb = any(name.endswith(s) for s in _LB_PARAM_SUFFIXES)
        if not is_lb:
            pr.requires_grad_(False)
            n_frozen += pr.numel()
            continue
        if freeze_signs and name.endswith(("U_fp", "V_fp")):
            pr.requires_grad_(False)
            n_frozen += pr.numel()
            continue
        pr.requires_grad_(True)
        n_trainable += pr.numel()
    return n_trainable, n_frozen


# --- Benchmark cache: teacher PPL + known checkpoint PPLs per base model ---
#
# Teacher PPL for a given (model_id, seq_len, max_tokens, dataset) is
# deterministic.  Re-computing it for every experiment wastes ~5s per run
# and clutters logs.  This cache stores it once and reads it subsequently.
#
# Structure (JSON at BENCHMARK_CACHE_PATH):
#     {
#       "Qwen/Qwen2.5-0.5B": {
#         "teacher_ppl": {
#           "seq=2048,tokens=25000,wikitext-2-raw-v1-test": 12.260,
#           ...
#         },
#         "known_results": {
#           "s15_kl_qat_8000": {"ppl": 83.3, "notes": "..."},
#           "s4_2_propagated_fixed": {"ppl": 678.97, ...},
#           ...
#         }
#       },
#       ...
#     }

BENCHMARK_CACHE_PATH = "benchmarks.json"


def _benchmark_key(seq_len: int, max_tokens: int,
                   dataset: str = "wikitext-2-raw-v1-test") -> str:
    return f"seq={seq_len},tokens={max_tokens},{dataset}"


def load_benchmarks(path: str = BENCHMARK_CACHE_PATH) -> dict:
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_benchmarks(data: dict, path: str = BENCHMARK_CACHE_PATH) -> None:
    import json
    import os
    from pathlib import Path
    try:
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, p)
    except Exception:
        pass


def get_or_eval_teacher_ppl(
    model,
    tokenizer,
    device,
    model_id: str,
    seq_len: int = 2048,
    max_tokens: int = 25000,
    dataset: str = "wikitext-2-raw-v1-test",
    force_recompute: bool = False,
) -> float:
    """Return teacher PPL from cache if available, otherwise evaluate and
    cache.  Call-compatible with `eval_ppl` return's `.ppl` field (float).

    The caller is expected to have already frozen / set eval() on `model`.
    `eval_ppl` is imported lazily to avoid circular imports (it lives in
    `littlebit_qat_brecq_full`).
    """
    key = _benchmark_key(seq_len, max_tokens, dataset)
    data = load_benchmarks()
    entry = data.get(model_id, {}).get("teacher_ppl", {}).get(key)
    if entry is not None and not force_recompute:
        print(f"[bench] teacher PPL cache hit: {model_id} @ {key} = {entry:.3f}")
        return float(entry)

    # Cache miss — evaluate.
    from littlebit_qat_brecq_full import eval_ppl
    ppl, _ = eval_ppl(model, tokenizer, device,
                      seq_len=seq_len, max_tokens=max_tokens)
    data.setdefault(model_id, {}).setdefault("teacher_ppl", {})[key] = ppl
    save_benchmarks(data)
    print(f"[bench] teacher PPL cached: {model_id} @ {key} = {ppl:.3f}")
    return ppl


def record_known_result(
    model_id: str,
    name: str,
    ppl: float,
    notes: str = "",
    extras: dict | None = None,
) -> None:
    """Append a named student-result reference to the benchmark cache.

    Useful for documenting baseline PPLs (e.g. §15 KL-QAT endpoint) so
    future sessions can look them up without re-running.
    """
    data = load_benchmarks()
    entry = {"ppl": ppl, "notes": notes}
    if extras:
        entry.update(extras)
    data.setdefault(model_id, {}).setdefault("known_results", {})[name] = entry
    save_benchmarks(data)


def write_status(path: str, payload: dict) -> None:
    """Atomically overwrite a JSON status file so external observers can
    check run progress at any moment without parsing the log.

    Writes to `path.tmp` then renames to `path` — the rename is atomic,
    so a reader never sees a half-written file.  Silent on failure
    (status is advisory; should never break training).
    """
    import json
    import os
    from pathlib import Path
    try:
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, p)
    except Exception:
        pass


def fisher_weighted_mse(
    Z_student: torch.Tensor,
    Z_target: torch.Tensor,
    f_b: torch.Tensor,
) -> torch.Tensor:
    """BRECQ Eq. 10 adapted: Fisher-diagonal-weighted MSE at the block
    output.  Shapes: Z_student, Z_target: (batch, seq, d_model);
    f_b: (d_model,).

    Returns a scalar loss = E_tokens[ Σ_i f_i · (Z_s - Z_t)_i² ].
    Note this sums over d_model then averages over (batch, seq) — the
    naive `.mean()` over all axes undercounts by a factor of 1/d_model,
    which scales the effective LR inconsistently.
    """
    diff = Z_student.float() - Z_target.float()
    return (f_b * diff.pow(2)).sum(-1).mean()


def convert_block_to_littlebit(
    block: nn.Module,
    rank: int,
    tau: float = 100.0,
    shadow_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Walk a transformer block and replace every nn.Linear (except lm_head,
    which isn't inside a block anyway) with a LittleBitLinearHF initialized
    via Dual-SVID from the original weight.

    Returns the mutated block.  Modifies in place.  Also wraps the block's
    forward to cast its output back to the input hidden_states dtype —
    LittleBitLinearHF returns fp32 internally (its scale vectors h/g/ell
    are fp32), so without this cast the block output dtype drifts and
    downstream modules (RMSNorm, lm_head) raise dtype mismatch errors.
    """
    # Collect (parent, attr, linear) triples so we can mutate without
    # disturbing the iterator.
    targets = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                parent = block.get_submodule(parent_name)
            else:
                parent = block
                attr = name
            targets.append((parent, attr, module, name))

    print(f"  converting {len(targets)} linear layers to LittleBit:")
    for parent, attr, lin, full_name in targets:
        lb = LittleBitLinearHF.from_linear(
            lin, r=rank, tau=tau, shadow_dtype=shadow_dtype,
        )
        setattr(parent, attr, lb)
        print(f"    {full_name}: ({lin.in_features}, {lin.out_features}) "
              f"r={min(rank, lin.in_features, lin.out_features)}")

    # Wrap forward to preserve input dtype on the returned hidden state.
    _orig_forward = block.forward

    def _dtype_preserving_forward(hidden_states, *args, **kwargs):
        in_dtype = hidden_states.dtype
        out = _orig_forward(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            if out[0] is None:
                return out
            return (out[0].to(in_dtype),) + out[1:]
        return out.to(in_dtype)

    block.forward = _dtype_preserving_forward
    return block


def capture_block_io(teacher, block_idx):
    """Register pre-hook + forward hook on teacher's block_idx.

    Returns (captured_dict, handles) where captured_dict will be
    populated each forward with X_b, Z_b, and the kwargs the block
    was called with.  Caller is responsible for removing handles
    when done.
    """
    captured = {}
    block = teacher.model.layers[block_idx]

    def pre_hook(_mod, args, kwargs):
        captured["X_b"] = args[0].detach()
        captured["args"] = tuple(
            a.detach() if torch.is_tensor(a) else a for a in args[1:]
        )
        captured["kwargs"] = {
            k: (v.detach() if torch.is_tensor(v) else
                tuple(t.detach() if torch.is_tensor(t) else t for t in v)
                if isinstance(v, tuple) else v)
            for k, v in kwargs.items()
        }

    def post_hook(_mod, _inputs, output):
        z = output[0] if isinstance(output, tuple) else output
        captured["Z_b"] = z.detach()

    h1 = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = block.register_forward_hook(post_hook)
    return captured, [h1, h2]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """|| a - b ||_F / || b ||_F, computed in fp32."""
    a32 = a.float()
    b32 = b.float()
    num = torch.linalg.norm(a32 - b32)
    den = torch.linalg.norm(b32)
    return float((num / (den + 1e-12)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--block", type=int, default=12,
                   help="Transformer block index to calibrate")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--calib-samples", type=int, default=32,
                   help="Calibration sequences drawn per epoch (cycled)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Sequences per opt-step")
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--fisher", default="qwen05b_fisher.pt",
                   help="Path to Fisher diagonals .pt file")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out", default="s4_0_block12.json")
    p.add_argument("--ablation-scales-only", action="store_true",
                   help="Freeze U_fp/V_fp; train scales only (BRECQ-strict)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load teacher -----
    print(f"[brecq] loading teacher {args.model} (bfloat16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    print(f"[brecq]   loaded in {time.time() - t0:.1f}s, "
          f"{len(teacher.model.layers)} blocks")

    # ----- Load Fisher for target block -----
    print(f"[brecq] loading Fisher from {args.fisher}")
    fisher_data = torch.load(args.fisher, weights_only=False)
    f_b = fisher_data["fisher"][args.block].to(device)  # (d_model,)
    f_b_sqrt = f_b.sqrt()
    print(f"[brecq]   block {args.block}: Fisher sum={f_b.sum().item():.3f}, "
          f"mean={f_b.mean().item():.3e}, max={f_b.max().item():.3e}")

    # ----- Build student block (LittleBit-factored copy of teacher block) -----
    print(f"[brecq] building student block {args.block} (r={args.rank})")
    # Deep copy the teacher block's nn.Module tree, then convert.
    # Deep copy avoids mutating the teacher we use for targets.
    teacher_block = teacher.model.layers[args.block]
    student_block = copy.deepcopy(teacher_block)
    student_block = convert_block_to_littlebit(
        student_block, rank=args.rank, tau=args.tau,
    ).to(device)

    # Enable gradients ONLY on LittleBit-specific params; keep norms etc.
    # frozen (see `enable_littlebit_grads` docstring — earlier versions
    # mistakenly trained RMSNorm weights, contributing to composition
    # failure).
    n_params, n_frozen = enable_littlebit_grads(
        student_block, freeze_signs=args.ablation_scales_only,
    )
    print(f"[brecq]   trainable params: {n_params}, frozen: {n_frozen}")

    # ----- Capture teacher I/O for the block -----
    captured, handles = capture_block_io(teacher, args.block)

    # ----- Load calibration data -----
    print(f"[brecq] preparing calibration pool: "
          f"{args.calib_samples} samples, seq_len={args.seq_len}")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_ids = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < args.min_chars:
            continue
        enc = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=args.seq_len,
        )
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        calib_ids.append(ids)
        if len(calib_ids) >= args.calib_samples:
            break
    print(f"[brecq]   {len(calib_ids)} calibration sequences ready")

    # ----- Build the student-block forward helper -----
    def run_student(X_b, captured_args, captured_kwargs):
        """Forward student block with the same args/kwargs teacher saw."""
        out = student_block(X_b, *captured_args, **captured_kwargs)
        return out[0] if isinstance(out, tuple) else out

    # ----- Measure Dual-SVID init quality (step 0) -----
    print(f"[brecq] measuring Dual-SVID init quality...")
    init_block_rel_errs = []
    with torch.no_grad():
        for ids in calib_ids[: min(8, len(calib_ids))]:
            _ = teacher(ids, use_cache=False)
            X_b = captured["X_b"]
            Z_b_teacher = captured["Z_b"]
            Z_b_student = run_student(X_b, captured["args"],
                                      captured["kwargs"])
            init_block_rel_errs.append(
                rel_err(Z_b_student, Z_b_teacher)
            )
    init_rel = sum(init_block_rel_errs) / len(init_block_rel_errs)
    print(f"[brecq]   init block-output rel-err = {init_rel:.4f} "
          f"(avg over {len(init_block_rel_errs)} samples)")

    # ----- Training loop -----
    # weight_decay=0.0: LittleBit params (U_fp/V_fp shadow weights, and
    # scale vectors h/g/ell/bias) should not be decayed — shrinking
    # scale vectors distorts the block's output magnitude and fights
    # against the block-output MSE objective.
    opt = torch.optim.AdamW(
        [pr for pr in student_block.parameters() if pr.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

    history = [{
        "step": 0,
        "loss": None,
        "block_rel_err": init_rel,
        "elapsed_s": 0.0,
    }]

    print(f"[brecq] training: {args.steps} steps, "
          f"lr={args.lr}, tau={args.tau}")
    t_train_start = time.time()

    for step in range(1, args.steps + 1):
        # Sample one calibration example (cycled).
        ids = calib_ids[(step - 1) % len(calib_ids)]

        # Teacher forward (no_grad) to capture X_b, Z_b, kwargs.
        with torch.no_grad():
            _ = teacher(ids, use_cache=False)
        X_b = captured["X_b"]
        Z_b_teacher = captured["Z_b"]
        cap_args = captured["args"]
        cap_kwargs = captured["kwargs"]

        # Student forward (with grad).
        Z_b_student = run_student(X_b, cap_args, cap_kwargs)

        # Fisher-weighted MSE, BRECQ Eq. 10 form:
        # E_tokens[ Σ_i f_i · (Z_s - Z_t)_i² ]
        loss = fisher_weighted_mse(Z_b_student, Z_b_teacher, f_b)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            # Validation: avg block-output rel-err over 4 held-out samples.
            with torch.no_grad():
                val_errs = []
                for val_ids in calib_ids[-4:]:
                    _ = teacher(val_ids, use_cache=False)
                    X_v = captured["X_b"]
                    Z_v_t = captured["Z_b"]
                    Z_v_s = run_student(X_v, captured["args"],
                                        captured["kwargs"])
                    val_errs.append(rel_err(Z_v_s, Z_v_t))
                val_rel = sum(val_errs) / len(val_errs)
            elapsed = time.time() - t_train_start
            history.append({
                "step": step,
                "loss": float(loss.item()),
                "block_rel_err": val_rel,
                "elapsed_s": elapsed,
            })
            print(f"[brecq]   step {step:5d}  "
                  f"loss={loss.item():.6f}  "
                  f"rel-err={val_rel:.4f}  "
                  f"t={elapsed:.1f}s")

    # ----- Cleanup -----
    for h in handles:
        h.remove()

    # ----- Save -----
    total_time = time.time() - t_train_start
    result = {
        "config": {
            "model": args.model,
            "block": args.block,
            "rank": args.rank,
            "steps": args.steps,
            "lr": args.lr,
            "tau": args.tau,
            "calib_samples": args.calib_samples,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "ablation_scales_only": args.ablation_scales_only,
        },
        "init_block_rel_err": init_rel,
        "final_block_rel_err": history[-1]["block_rel_err"],
        "total_train_seconds": total_time,
        "fisher_sum_at_block": float(f_b.sum().item()),
        "history": history,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[brecq] saved {args.out}")
    print(f"[brecq] SUMMARY: init rel-err {init_rel:.4f} -> "
          f"final {history[-1]['block_rel_err']:.4f} "
          f"in {total_time:.1f}s")


if __name__ == "__main__":
    main()
