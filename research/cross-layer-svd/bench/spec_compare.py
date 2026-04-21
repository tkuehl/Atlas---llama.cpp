"""
spec_compare.py — compare two JSONL result files from spec_bench.py,
emit a markdown comparison table to stdout.

Usage:
    python spec_compare.py results/baseline.jsonl results/ngram_cache.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows):
    slices = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        slices[r["slice"]].append(r)

    agg = {}
    for slice_name, items in slices.items():
        tok_s = [x["timings"].get("predicted_per_second", 0) for x in items]
        ttft_ms = [x["timings"].get("prompt_ms", 0) for x in items]
        draft_n = [x["timings"].get("draft_n", 0) for x in items]
        draft_acc = [x["timings"].get("draft_n_accepted", 0) for x in items]
        cache_n = [x["timings"].get("cache_n", 0) for x in items]

        tot_drafted = sum(draft_n)
        tot_accepted = sum(draft_acc)
        acceptance = (tot_accepted / tot_drafted) if tot_drafted > 0 else None

        agg[slice_name] = {
            "n": len(items),
            "mean_tok_s": sum(tok_s) / len(tok_s) if tok_s else 0,
            "mean_ttft_ms": sum(ttft_ms) / len(ttft_ms) if ttft_ms else 0,
            "total_drafted": tot_drafted,
            "total_accepted": tot_accepted,
            "acceptance_rate": acceptance,
            "total_cache_n": sum(cache_n),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", help="baseline JSONL")
    ap.add_argument("condition", help="experimental condition JSONL")
    args = ap.parse_args()

    base_rows = load(args.baseline)
    cond_rows = load(args.condition)

    base_agg = aggregate(base_rows)
    cond_agg = aggregate(cond_rows)

    base_name = base_rows[0].get("condition", "baseline") if base_rows else "baseline"
    cond_name = cond_rows[0].get("condition", "condition") if cond_rows else "condition"

    slice_order = ["structured_output", "code", "factual_qa", "reasoning", "conversational"]
    all_slices = [s for s in slice_order if s in base_agg or s in cond_agg]
    all_slices += sorted((set(base_agg) | set(cond_agg)) - set(all_slices))

    print(f"# Speculative Decoding Bench — `{cond_name}` vs `{base_name}`\n")
    total_base_prompts = sum(a["n"] for a in base_agg.values())
    total_cond_prompts = sum(a["n"] for a in cond_agg.values())
    print(f"Baseline `{base_name}`: {total_base_prompts} prompts  ")
    print(f"Condition `{cond_name}`: {total_cond_prompts} prompts\n")

    print("## Per-slice summary\n")
    print("| Slice | Base tok/s | Cond tok/s | Speedup | Accept rate | TTFT Δ (ms) |")
    print("|---|---:|---:|---:|---:|---:|")
    for s in all_slices:
        b = base_agg.get(s, {})
        c = cond_agg.get(s, {})
        b_tok = b.get("mean_tok_s", 0)
        c_tok = c.get("mean_tok_s", 0)
        speedup = (c_tok / b_tok) if b_tok > 0 else 0
        accept = c.get("acceptance_rate")
        accept_str = f"{accept*100:.1f}%" if accept is not None else "—"
        ttft_delta = c.get("mean_ttft_ms", 0) - b.get("mean_ttft_ms", 0)
        print(f"| {s} | {b_tok:.1f} | {c_tok:.1f} | {speedup:.2f}× | {accept_str} | {ttft_delta:+.0f} |")

    # Overall (weighted by prompt count)
    tot_b = sum(a["n"] for a in base_agg.values())
    tot_c = sum(a["n"] for a in cond_agg.values())
    total_base_tok = (sum(a["mean_tok_s"] * a["n"] for a in base_agg.values()) / tot_b) if tot_b else 0
    total_cond_tok = (sum(a["mean_tok_s"] * a["n"] for a in cond_agg.values()) / tot_c) if tot_c else 0
    total_speedup = (total_cond_tok / total_base_tok) if total_base_tok > 0 else 0

    tot_drafted = sum(a["total_drafted"] for a in cond_agg.values())
    tot_accepted = sum(a["total_accepted"] for a in cond_agg.values())
    overall_accept = (tot_accepted / tot_drafted) if tot_drafted > 0 else None

    extras = []
    extras.append(f"baseline={total_base_tok:.1f} tok/s")
    extras.append(f"condition={total_cond_tok:.1f} tok/s")
    extras.append(f"speedup={total_speedup:.2f}×")
    if overall_accept is not None:
        extras.append(f"acceptance={overall_accept*100:.1f}%")
    print(f"\n**Overall:** " + ", ".join(extras))

    # Qualitative spot check
    print("\n## Per-prompt response preview (first 80 chars)\n")
    print("| ID | Slice | Baseline | Condition |")
    print("|---|---|---|---|")
    base_by_id = {r["id"]: r for r in base_rows if "error" not in r}
    cond_by_id = {r["id"]: r for r in cond_rows if "error" not in r}
    for rid in sorted(set(base_by_id) | set(cond_by_id)):
        b = base_by_id.get(rid, {})
        c = cond_by_id.get(rid, {})
        bp = (b.get("response_preview") or "").replace("\n", " ").replace("|", "\\|")[:80]
        cp = (c.get("response_preview") or "").replace("\n", " ").replace("|", "\\|")[:80]
        slc = (b or c).get("slice", "?")
        print(f"| {rid} | {slc} | `{bp}` | `{cp}` |")


if __name__ == "__main__":
    main()
