"""Benchmark harness: compare baseline vs factored model across suites.

Design: load the baseline model once, save its logits on a held-out text (cheap
reference). Then instantiate one or more candidate models (varying rank budgets,
role subsets), compare against baseline, run suite(s), dump JSON.

Usage:
    # Baseline pass (saves reference logits + runs self-suite for ceiling metrics)
    python -m benchmarks.harness \\
        --model Qwen/Qwen2.5-0.5B --baseline --out runs/baseline.json \\
        --suites throughput quality --quality-preset quick

    # Candidate pass at a specific rank
    python -m benchmarks.harness \\
        --model Qwen/Qwen2.5-0.5B --rank 256 --roles all \\
        --baseline-logits runs/baseline-logits.pt \\
        --suites consistency throughput quality \\
        --out runs/rank256.json
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make "import prototype" work when running via `python -m benchmarks.harness`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import prototype as proto  # noqa: E402

from benchmarks.backends.hf import HFBackend  # noqa: E402
from benchmarks.suites import consistency, throughput, quality  # noqa: E402


def apply_factoring(model, tokenizer, rank, roles, calib_samples, seq_len, device,
                    activation_aware=True):
    """Decompose + reconstruct in place. Returns per-role stats."""
    texts = proto.load_calibration_texts(tokenizer, calib_samples)
    xtx_by_name, groups = proto.collect_activation_stats(model, tokenizer, texts,
                                                          device, seq_len)
    report = {}
    for role in roles:
        entries = groups.get(role, [])
        if not entries:
            continue
        _, names, modules = zip(*entries)
        weights = [m.weight.data.cpu() for m in modules]
        xtx_list = [xtx_by_name[n] for n in names]
        t0 = time.time()
        factors = proto.precompute_cross_layer_svd(weights, xtx_list,
                                                   activation_aware=activation_aware)
        _, recons = proto.reconstruct_at_rank(factors, rank)
        proto.swap_weights(modules, recons)
        report[role] = {
            "shape": [factors["d_out"], factors["d_in"]],
            "n_layers": factors["L"],
            "rank_applied": min(rank, factors["sigma"].numel()),
            "svd_seconds": round(time.time() - t0, 2),
        }
    return report


def load_or_record_baseline_logits(baseline, path, n_tokens=4096):
    """Record baseline logits to disk (one-time) or load cached copy."""
    p = Path(path)
    if p.exists():
        saved = torch.load(p, map_location="cpu", weights_only=False)
        return saved["token_ids"], saved["logits"]

    text = consistency._held_out_text()
    token_ids = baseline.tokenize(text)[:n_tokens]
    # Chunk so we don't blow VRAM on very long sequences
    chunks = []
    window = 1024
    for i in range(0, len(token_ids), window):
        chunk = token_ids[i:i + window]
        chunks.append(baseline.logits(chunk).cpu().to(torch.float16))
    logits = torch.cat(chunks, dim=0)

    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"token_ids": token_ids, "logits": logits}, p)
    return token_ids, logits


def consistency_against_cached(candidate, token_ids, baseline_logits, window=1024):
    """Run consistency using pre-saved baseline logits instead of two models in memory."""
    import torch.nn.functional as F
    from benchmarks.suites.consistency import ConsistencyResult

    kl_accum = top1 = top5 = ent_delta = scored = 0.0
    for begin in range(0, len(token_ids), window):
        chunk = token_ids[begin:begin + window]
        if len(chunk) < 8:
            continue
        c_logits = candidate.logits(chunk).float().cpu()
        b_logits = baseline_logits[begin:begin + len(chunk)].float()

        b_logp = F.log_softmax(b_logits, dim=-1)
        c_logp = F.log_softmax(c_logits, dim=-1)
        b_p = b_logp.exp()
        c_p = c_logp.exp()

        kl = (b_p * (b_logp - c_logp)).sum(-1) / torch.log(torch.tensor(2.0))
        kl_accum += kl.sum().item()

        b_top1 = b_logits.argmax(-1)
        c_top1 = c_logits.argmax(-1)
        top1 += (b_top1 == c_top1).sum().item()
        c_top5 = c_logits.topk(5, dim=-1).indices
        top5 += (b_top1.unsqueeze(-1) == c_top5).any(-1).sum().item()

        b_ent = -(b_p * b_logp).sum(-1)
        c_ent = -(c_p * c_logp).sum(-1)
        ent_delta += (c_ent - b_ent).sum().item()
        scored += len(chunk)

    return ConsistencyResult(
        n_tokens=int(scored),
        kl_div_bits=kl_accum / max(scored, 1),
        top1_agree=top1 / max(scored, 1),
        top5_agree=top5 / max(scored, 1),
        entropy_delta=ent_delta / max(scored, 1),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--baseline", action="store_true",
                   help="no factoring; record baseline logits + run self-benchmarks")
    p.add_argument("--rank", type=int, default=None,
                   help="truncation rank for factoring (ignored if --baseline)")
    p.add_argument("--roles", nargs="+", default=["all"],
                   help="roles to factor: 'all' or a subset of ROLES")
    p.add_argument("--no-asvd", action="store_true", help="disable activation-aware SVD")
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--baseline-logits", default="runs/baseline-logits.pt",
                   help="path to record/load baseline reference logits")
    p.add_argument("--consistency-tokens", type=int, default=4096)
    p.add_argument("--suites", nargs="+",
                   default=["consistency", "throughput"],
                   choices=["consistency", "throughput", "quality"])
    p.add_argument("--quality-preset", default="quick", choices=["quick", "full"])
    p.add_argument("--quality-limit", type=int, default=None,
                   help="limit lm-eval tasks to N examples (debug only)")
    p.add_argument("--out", default="runs/result.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} on {args.device} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()

    roles = proto.ROLES if args.roles == ["all"] else args.roles
    factoring_report = None
    if not args.baseline:
        if args.rank is None:
            p.error("--rank required unless --baseline")
        print(f"factoring at rank {args.rank} over roles: {roles}")
        factoring_report = apply_factoring(model, tokenizer, args.rank, roles,
                                           args.calib_samples, args.calib_seq_len,
                                           args.device, activation_aware=not args.no_asvd)

    backend = HFBackend(args.model, device=args.device, dtype=dtype,
                         model=model, tokenizer=tokenizer)

    result = {
        "model": args.model,
        "baseline": args.baseline,
        "rank": args.rank,
        "roles_factored": roles if not args.baseline else [],
        "activation_aware": not args.no_asvd,
        "factoring": factoring_report,
        "suites": {},
    }

    if "consistency" in args.suites:
        if args.baseline:
            print("recording baseline logits (for future candidate runs)")
            load_or_record_baseline_logits(backend, args.baseline_logits,
                                           n_tokens=args.consistency_tokens)
            result["suites"]["consistency"] = {"note": "baseline; logits saved"}
        else:
            print("running consistency suite")
            token_ids, ref_logits = load_or_record_baseline_logits(
                backend, args.baseline_logits, n_tokens=args.consistency_tokens)
            # ^ note: when called on the candidate, the cached baseline is reused.
            cr = consistency_against_cached(backend, token_ids, ref_logits)
            print(f"  kl_div_bits={cr.kl_div_bits:.4f}  top1={cr.top1_agree:.4f}  "
                  f"top5={cr.top5_agree:.4f}  entropy_delta={cr.entropy_delta:+.4f}")
            result["suites"]["consistency"] = cr.to_dict()

    if "throughput" in args.suites:
        print("running throughput suite")
        tr = throughput.run(backend)
        print(f"  median decode {tr.summary['median_decode_tok_per_s']:.1f} tok/s, "
              f"mean TTFT {tr.summary['mean_ttft_ms']:.1f} ms, "
              f"peak VRAM {tr.peak_vram_mb:.0f} MB")
        result["suites"]["throughput"] = tr.to_dict()

    if "quality" in args.suites:
        print(f"running quality suite ({args.quality_preset})")
        try:
            qr = quality.run(backend, preset=args.quality_preset, limit=args.quality_limit)
            print("  " + ", ".join(f"{t}={v:.3f}" for t, v in qr.scores.items()))
            result["suites"]["quality"] = qr.to_dict()
        except ImportError as e:
            print(f"  [skipped] {e}")
            result["suites"]["quality"] = {"error": str(e)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {out}")

    backend.close()


if __name__ == "__main__":
    main()
