"""Phase 2 entrypoint — LB-ADMM init + STE refinement.

Pipeline:
    1. Load FP model + tokenizer.
    2. Build (or reuse) the teacher activation cache.
    3. Collect (or reuse) K-FAC diagonal preconditioners D_in, D_out via
       calibration forward+backward with hooks.
    4. Block-by-block: LB-ADMM init (paper Step 2-2) instead of plain SVD,
       then the same STE refinement loop Phase 1 uses (Step 2-3).
    5. Evaluate with sliding-window PPL.
    6. Append results.json entry with method.name = "phase2-lbadmm-ste".

Scope note: this implements Algorithm 1 Phase 1 (Global Calibration) +
Phase 2 Steps 2-3 only. Step 1 (TuneFP, error propagation mitigation) and
Phase 3 (TuneScalesKD, global KL) are deferred.

Usage:
    python run_phase2.py --model Qwen/Qwen3-4B --rank 2
    python run_phase2.py --model Qwen/Qwen3-4B --rank 2 --steps 50 --n-calib 4 \
        --admm-K 50 --no-log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activations import Cache, build_cache
from admm import lb_admm_init
from data import eval_token_stream
from phase1 import quantize_model_phase1
from phase2 import quantize_model_with_tunefp
from ppl import sliding_window_ppl
from preconditioner import (
    PRECOND_FILE,
    collect_preconditioners,
    load_preconditioners,
)
from results import append_entry
from tune_scales_kd import tune_scales_kd


HERE = Path(__file__).parent
CACHE_ROOT = HERE / "cache"
STATUS_ROOT = HERE / "runs"

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500, help="STE steps/block")
    ap.add_argument("--lr", type=float, default=1e-5)  # paper: 1e-5 for TuneLatentSTE
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    ap.add_argument("--device", default="cuda")
    # LB-ADMM hyperparameters
    ap.add_argument("--admm-K", type=int, default=400, help="ADMM iterations per matrix")
    ap.add_argument("--rho-start", type=float, default=0.1)
    ap.add_argument("--rho-end", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.2, help="Ledoit-Wolf γ")
    ap.add_argument("--percentile", type=float, default=0.99, help="ROBUSTDIAG clip")
    # Reuse toggles
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--rebuild-precond", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument(
        "--verbose-blocks",
        type=int,
        default=0,
        help="emit detailed dtype/grad/delta logs for the first N blocks",
    )
    ap.add_argument(
        "--tune-fp-steps",
        type=int,
        default=0,
        help="TuneFP (Algorithm 1 Step 1) steps per block. 0 = skip TuneFP "
             "and use cached teacher-to-teacher (X, Y) pairs as in phase1.py.",
    )
    ap.add_argument("--tune-fp-lr", type=float, default=1e-4)
    ap.add_argument("--tune-fp-batch", type=int, default=4)
    ap.add_argument(
        "--tune-scales-steps",
        type=int,
        default=0,
        help="TuneScalesKD (Algorithm 1 Phase 3) steps. 0 = skip. "
             "Runs AFTER block-wise quantization. Freezes binary signs, "
             "KL-fine-tunes only s1/s2 end-to-end against a fresh FP teacher.",
    )
    ap.add_argument("--tune-scales-lr", type=float, default=1e-6)
    ap.add_argument("--tune-scales-batch", type=int, default=1)
    ap.add_argument(
        "--tune-scales-temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for KL (Hinton distillation). "
             "T>1 flattens distributions, making low-LR optimization easier.",
    )
    ap.add_argument("--eval-seq-len", type=int, default=2048)
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    torch_dtype = DTYPE_MAP[args.dtype]

    cache_slug = f"n{args.n_calib}_L{args.seq_len}_seed{args.seed}"
    cache_dir = CACHE_ROOT / _slug(args.model) / cache_slug
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    run_slug = (
        f"phase2_{_slug(args.model)}_r{args.rank}_K{args.admm_K}_"
        f"steps{args.steps}_{cache_slug}"
    )
    status_file = STATUS_ROOT / f"{run_slug}.status.json"

    print(f"[load] tokenizer: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f"[load] model: {args.model} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()

    # --- Step 1: teacher activation cache (same as Phase 1) ---
    need_cache = args.rebuild_cache or not (cache_dir / "meta.json").exists()
    if need_cache:
        print(f"[cache] building teacher activations -> {cache_dir}", flush=True)
        t0 = time.time()
        cache = build_cache(
            model,
            tok,
            cache_dir=cache_dir,
            n_samples=args.n_calib,
            seq_len=args.seq_len,
            seed=args.seed,
            chunk_size=args.chunk_size,
            device=args.device,
        )
        print(f"[cache] built in {time.time() - t0:.1f}s", flush=True)
    else:
        print(f"[cache] reusing {cache_dir}", flush=True)
        cache = Cache.load(cache_dir)

    # --- Step 2: K-FAC diagonal preconditioners ---
    need_precond = args.rebuild_precond or not (cache_dir / PRECOND_FILE).exists()
    if need_precond:
        print(
            f"[precond] collecting D_in/D_out (n={args.n_calib}, gamma={args.gamma})",
            flush=True,
        )
        t0 = time.time()
        precond = collect_preconditioners(
            model,
            tok,
            cache_dir=cache_dir,
            n_samples=args.n_calib,
            seq_len=args.seq_len,
            seed=args.seed,
            gamma=args.gamma,
            percentile=args.percentile,
            chunk_size=1,  # backward is memory-heavy
            device=args.device,
            model_hf_id=args.model,
        )
        print(
            f"[precond] collected {len(precond)} linears in {time.time() - t0:.1f}s",
            flush=True,
        )
    else:
        print(f"[precond] reusing {cache_dir / PRECOND_FILE}", flush=True)
        precond = load_preconditioners(cache_dir)
        print(f"[precond] loaded {len(precond)} linears", flush=True)

    # Build the init_fn_factory. Per-block, it resolves the full-model path
    # of each linear and looks up its (D_in, D_out) in `precond`.
    def init_fn_factory(block_idx: int, block):
        full_prefix = f"model.layers.{block_idx}"

        def init_fn(rel_path: str, lin):
            full_path = f"{full_prefix}.{rel_path}"
            if full_path not in precond:
                raise KeyError(
                    f"no preconditioners for {full_path} — collection missed it"
                )
            D_in, D_out = precond[full_path]
            D_in = D_in.to(lin.weight.device)
            D_out = D_out.to(lin.weight.device)
            return lb_admm_init(
                lin.weight.data,
                D_in=D_in,
                D_out=D_out,
                r=args.rank,
                K=args.admm_K,
                rho_start=args.rho_start,
                rho_end=args.rho_end,
                lam=args.lam,
                target_dtype=lin.weight.dtype,
                seed=args.seed + block_idx,  # vary per block for diversity
            )

        return init_fn

    # --- Step 3: LB-ADMM init + STE refinement ---
    print(
        f"[phase2] rank={args.rank} admm_K={args.admm_K} steps={args.steps} "
        f"lr={args.lr} blocks={cache.n_blocks}",
        flush=True,
    )
    t0 = time.time()
    if args.tune_fp_steps > 0:
        print(
            f"[phase2] using TuneFP pipeline: tune_fp_steps={args.tune_fp_steps} "
            f"tune_fp_lr={args.tune_fp_lr} tune_fp_batch={args.tune_fp_batch}",
            flush=True,
        )
        phase2_block_stats = quantize_model_with_tunefp(
            model,
            cache=cache,
            r=args.rank,
            steps_per_block=args.steps,
            lr=args.lr,
            device=args.device,
            init_fn_factory=init_fn_factory,
            tune_fp_steps=args.tune_fp_steps,
            tune_fp_lr=args.tune_fp_lr,
            tune_fp_batch=args.tune_fp_batch,
            status_file=status_file,
            verbose_first_blocks=args.verbose_blocks,
        )
        # Flatten to the same shape quantize_model_phase1 returned for
        # downstream accounting.
        block_stats = [s.ste for s in phase2_block_stats]
        # Also surface mean TuneFP delta for the results log.
        tunefp_finals = [s.tune_fp.final_loss for s in phase2_block_stats if s.tune_fp is not None]
        tunefp_inits = [s.tune_fp.init_loss for s in phase2_block_stats if s.tune_fp is not None]
        mean_tunefp_init = sum(tunefp_inits) / max(len(tunefp_inits), 1) if tunefp_inits else None
        mean_tunefp_final = sum(tunefp_finals) / max(len(tunefp_finals), 1) if tunefp_finals else None
    else:
        block_stats = quantize_model_phase1(
            model,
            cache=cache,
            r=args.rank,
            steps_per_block=args.steps,
            lr=args.lr,
            device=args.device,
            status_file=status_file,
            init_fn_factory=init_fn_factory,
            verbose_first_blocks=args.verbose_blocks,
        )
        mean_tunefp_init = None
        mean_tunefp_final = None
    phase2_secs = time.time() - t0
    print(
        f"[phase2] done in {phase2_secs:.1f}s "
        f"({phase2_secs / max(len(block_stats), 1):.1f}s/block)",
        flush=True,
    )

    # --- Phase 3 (optional): TuneScalesKD — global scale fine-tune ---
    tune_scales_stats = None
    if args.tune_scales_steps > 0:
        print(
            f"[phase3] TuneScalesKD: {args.tune_scales_steps} steps, "
            f"lr={args.tune_scales_lr:g}, batch={args.tune_scales_batch}",
            flush=True,
        )
        # Fresh FP teacher alongside the quantized student. After block-wise
        # factorization the student's VRAM footprint drops (big Linears
        # replaced with small fp32 factor params), so both fit on the 5080.
        print(f"[phase3] loading teacher model: {args.model} ({args.dtype})", flush=True)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch_dtype,
            device_map=args.device,
        )
        teacher_model.eval()

        t0_scales = time.time()
        tune_scales_stats = tune_scales_kd(
            student_model=model,
            teacher_model=teacher_model,
            tokenizer=tok,
            n_samples=args.n_calib,
            seq_len=args.seq_len,
            seed=args.seed,
            steps=args.tune_scales_steps,
            lr=args.tune_scales_lr,
            device=args.device,
            batch_size=args.tune_scales_batch,
            temperature=args.tune_scales_temperature,
        )
        print(
            f"[phase3] done in {time.time() - t0_scales:.1f}s, "
            f"init_kl={tune_scales_stats.init_loss:.5f} -> "
            f"final_kl={tune_scales_stats.final_loss:.5f} "
            f"(bad_steps={tune_scales_stats.bad_steps})",
            flush=True,
        )
        # Free teacher model before eval (student forward needs VRAM).
        del teacher_model
        torch.cuda.empty_cache()

    # --- Eval ---
    print("[eval] tokenizing WikiText-2 test split", flush=True)
    stream = eval_token_stream(tok)
    print(f"[eval] sliding-window PPL (seq_len={args.eval_seq_len})", flush=True)
    result = sliding_window_ppl(
        model,
        stream,
        seq_len=args.eval_seq_len,
        device=args.device,
    )
    print(
        f"[eval] ppl={result.ppl:.4f} nll_mean={result.nll_mean:.4f} "
        f"windows={result.num_windows} tokens={result.num_tokens}",
        flush=True,
    )

    if args.no_log:
        print("[log] --no-log set, skipping results.json append", flush=True)
        return

    mean_init = sum(s.init_mse for s in block_stats) / max(len(block_stats), 1)
    mean_final = sum(s.final_mse for s in block_stats) / max(len(block_stats), 1)
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    revision = getattr(model.config, "_commit_hash", None)

    method_name_parts = ["phase2"]
    if args.tune_fp_steps > 0:
        method_name_parts.append("tunefp")
    method_name_parts.extend(["lbadmm", "ste"])
    if args.tune_scales_steps > 0:
        method_name_parts.append("kd")
    method_name = "-".join(method_name_parts)
    method_params = {
        "r": args.rank,
        "admm_K": args.admm_K,
        "rho_start": args.rho_start,
        "rho_end": args.rho_end,
        "lam": args.lam,
        "gamma": args.gamma,
        "percentile": args.percentile,
        "steps_per_block": args.steps,
        "lr": args.lr,
        "init": "lb-admm",
        "ste": "clipped-identity",
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "n_calib": args.n_calib,
        "calib_seq_len": args.seq_len,
        "calib_seed": args.seed,
        "mean_block_init_mse": mean_init,
        "mean_block_final_mse": mean_final,
        "phase2_wall_seconds": phase2_secs,
    }
    if args.tune_fp_steps > 0:
        method_params.update({
            "tune_fp_steps": args.tune_fp_steps,
            "tune_fp_lr": args.tune_fp_lr,
            "tune_fp_batch": args.tune_fp_batch,
            "mean_tunefp_init_loss": mean_tunefp_init,
            "mean_tunefp_final_loss": mean_tunefp_final,
        })
    if args.tune_scales_steps > 0 and tune_scales_stats is not None:
        method_params.update({
            "tune_scales_steps": args.tune_scales_steps,
            "tune_scales_lr": args.tune_scales_lr,
            "tune_scales_batch": args.tune_scales_batch,
            "tune_scales_temperature": args.tune_scales_temperature,
            "tune_scales_init_kl": tune_scales_stats.init_loss,
            "tune_scales_final_kl": tune_scales_stats.final_loss,
            "tune_scales_steps_taken": tune_scales_stats.steps_taken,
            "tune_scales_bad_steps": tune_scales_stats.bad_steps,
            "n_scale_params": tune_scales_stats.n_scale_params,
        })

    entry = append_entry(
        model_hf_id=args.model,
        model_revision=revision,
        model_dtype=args.dtype,
        method_name=method_name,
        method_params=method_params,
        eval_info={
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
            "seq_len": result.seq_len,
            "stride": result.stride,
            "num_tokens": result.num_tokens,
            "num_windows": result.num_windows,
            "metric": "ppl",
            "value": result.ppl,
            "nll_mean": result.nll_mean,
        },
        hardware={"gpu": gpu_name, "torch": torch.__version__},
        notes=args.notes,
    )
    print(f"[log] appended entry id={entry['id']}", flush=True)

    final_status_file = status_file.with_name(status_file.stem + ".final.json")
    with open(final_status_file, "w") as f:
        json.dump(
            {
                "entry_id": entry["id"],
                "ppl": result.ppl,
                "block_stats": [vars(s) for s in block_stats],
            },
            f,
            indent=2,
        )
    print(f"[log] block stats -> {final_status_file}", flush=True)


if __name__ == "__main__":
    main()
