"""Compare two bench_model.py JSON runs — typically base vs factored.

Produces a markdown report covering speed, resources, automated accuracy
(exact-match, greedy token agreement, top-k overlap), and a side-by-side
response dump for qualitative judgment.

Usage:
    python bench_compare.py --base base.json --factored factored.json \
        --out comparison.md
"""

import argparse
import json
import statistics
from pathlib import Path


def _fmt_delta(base_val: float, other_val: float, higher_is_better: bool,
               unit: str = "", precision: int = 1) -> str:
    """Format a 'base -> other' delta with %, sign, and direction marker.

    `higher_is_better` flips the marker so speed-ups show ✓ even when the
    number grew."""
    if base_val == 0:
        return f"{base_val:.{precision}f}{unit} → {other_val:.{precision}f}{unit}"
    pct = (other_val - base_val) / base_val * 100
    sign = "+" if pct >= 0 else ""
    good = (pct >= 0) == higher_is_better
    marker = "✓" if good else "✗" if abs(pct) > 5 else "~"
    return (f"{base_val:.{precision}f}{unit} → {other_val:.{precision}f}{unit} "
            f"({sign}{pct:.1f}% {marker})")


def _greedy_token_agreement(base_ids: list[int],
                             other_ids: list[int]) -> tuple[int, int]:
    """Count matching tokens at the same positions until first mismatch
    (or end of shorter sequence). Returns (matches, compared)."""
    n = min(len(base_ids), len(other_ids))
    matches = 0
    for i in range(n):
        if base_ids[i] == other_ids[i]:
            matches += 1
        else:
            break
    return matches, n


def _topk_overlap(base_ids: list[int],
                  other_topk: list[list[int]]) -> tuple[int, int]:
    """For each position where base picked token X, was X in factored's
    top-k at that step? Returns (in_topk, total)."""
    n = min(len(base_ids), len(other_topk))
    hits = 0
    for i in range(n):
        if base_ids[i] in other_topk[i]:
            hits += 1
    return hits, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="bench_model.py output for base")
    p.add_argument("--factored", required=True, help="bench_model.py output for factored")
    p.add_argument("--out", required=True, help="path for markdown report")
    p.add_argument("--prompt-preview-chars", type=int, default=120,
                   help="Truncate prompt text in the side-by-side section")
    p.add_argument("--response-preview-chars", type=int, default=400,
                   help="Truncate generated text in the side-by-side section")
    args = p.parse_args()

    base = json.loads(Path(args.base).read_text())
    fact = json.loads(Path(args.factored).read_text())

    base_by_id = {r["id"]: r for r in base["prompts"]}
    fact_by_id = {r["id"]: r for r in fact["prompts"]}
    ids = [r["id"] for r in base["prompts"] if r["id"] in fact_by_id]

    lines: list[str] = []
    push = lines.append

    push("# bench_compare: base vs factored")
    push("")
    push(f"- **Base model:** `{base['base_id']}` ({base['model_kind']})")
    push(f"- **Factored dir:** `{fact['factored_dir']}` ({fact['model_kind']})")
    push(f"- **Device / dtype:** {base['device']} / {base['dtype']}")
    push(f"- **max_new_tokens:** {base['max_new_tokens']}, "
         f"**temperature:** {base['temperature']}")
    push(f"- **Prompts compared:** {len(ids)}")
    push("")

    # ── Speed ───────────────────────────────────────────────────────────
    push("## Speed")
    push("")
    push("| metric | base | factored | delta |")
    push("|---|---|---|---|")
    ba = base["aggregate"]
    fa = fact["aggregate"]
    push(f"| Mean tok/s | {ba['tok_per_sec_mean']:.1f} | "
         f"{fa['tok_per_sec_mean']:.1f} | "
         f"{_fmt_delta(ba['tok_per_sec_mean'], fa['tok_per_sec_mean'], True, ' tok/s')} |")
    push(f"| Median tok/s | {ba['tok_per_sec_median']:.1f} | "
         f"{fa['tok_per_sec_median']:.1f} | "
         f"{_fmt_delta(ba['tok_per_sec_median'], fa['tok_per_sec_median'], True, ' tok/s')} |")
    push(f"| Mean TTFT | {ba['ttft_mean_sec']*1000:.0f} ms | "
         f"{fa['ttft_mean_sec']*1000:.0f} ms | "
         f"{_fmt_delta(ba['ttft_mean_sec']*1000, fa['ttft_mean_sec']*1000, False, ' ms', 0)} |")
    push(f"| Total bench time | {ba['total_bench_sec']:.1f} s | "
         f"{fa['total_bench_sec']:.1f} s | "
         f"{_fmt_delta(ba['total_bench_sec'], fa['total_bench_sec'], False, ' s')} |")
    push(f"| Model load | {base['model_load_sec']:.1f} s | "
         f"{fact['model_load_sec']:.1f} s | "
         f"{_fmt_delta(base['model_load_sec'], fact['model_load_sec'], False, ' s')} |")
    push("")

    # ── Resources ──────────────────────────────────────────────────────
    push("## Resource utilization (during bench)")
    push("")
    push("| metric | base | factored | delta |")
    push("|---|---|---|---|")
    br = base.get("resources_bench", {})
    fr = fact.get("resources_bench", {})
    for key, label, unit, better_is_lower in [
        ("vram_torch_peak_mb", "VRAM peak (torch)", " MB", True),
        ("vram_peak_mb", "VRAM peak (sampled)", " MB", True),
        ("vram_mean_mb", "VRAM mean", " MB", True),
        ("rss_peak_mb", "RSS peak", " MB", True),
        ("rss_mean_mb", "RSS mean", " MB", True),
        ("cpu_pct_peak", "CPU peak %", "%", True),
        ("cpu_pct_mean", "CPU mean %", "%", True),
        ("gpu_util_peak", "GPU util peak", "%", False),
        ("gpu_util_mean", "GPU util mean", "%", False),
    ]:
        if key not in br or key not in fr:
            continue
        push(f"| {label} | {br[key]:.1f}{unit} | {fr[key]:.1f}{unit} | "
             f"{_fmt_delta(br[key], fr[key], not better_is_lower, unit)} |")
    push("")

    # ── Automated accuracy ──────────────────────────────────────────────
    push("## Accuracy (automated)")
    push("")

    # Exact-match on factual prompts
    push("### Exact-match (factual prompts)")
    push("")
    push("| id | expected | base answer | match | factored answer | match |")
    push("|---|---|---|---|---|---|")
    exact_base_hits = exact_fact_hits = exact_n = 0
    for pid in ids:
        b = base_by_id[pid]
        f = fact_by_id[pid]
        if b.get("type") != "factual":
            continue
        expected = b.get("expected", "")
        bm = b.get("exact_match", False)
        fm = f.get("exact_match", False)
        exact_n += 1
        exact_base_hits += int(bm)
        exact_fact_hits += int(fm)
        push(f"| `{pid}` | {expected} | "
             f"{b['generated_text'][:40].strip()!r} | {'✓' if bm else '✗'} | "
             f"{f['generated_text'][:40].strip()!r} | {'✓' if fm else '✗'} |")
    if exact_n:
        push("")
        push(f"**Exact-match score:** base **{exact_base_hits}/{exact_n}**, "
             f"factored **{exact_fact_hits}/{exact_n}**")
    push("")

    # Greedy token agreement
    push("### Greedy token agreement (first N tokens identical)")
    push("")
    push("Percent of leading tokens where factored's argmax equals base's, "
         "stopping at first divergence. 100% = the two models are indistinguishable "
         "greedy-decoding on this prompt.")
    push("")
    push("| id | type | base tokens | matches | % |")
    push("|---|---|---|---|---|")
    total_match = total_seen = 0
    per_type_agreement: dict[str, tuple[int, int]] = {}
    for pid in ids:
        b = base_by_id[pid]
        f = fact_by_id[pid]
        if "generated_ids" not in b or "generated_ids" not in f:
            continue
        matches, compared = _greedy_token_agreement(b["generated_ids"],
                                                     f["generated_ids"])
        total_match += matches
        total_seen += compared
        pct = 100 * matches / compared if compared else 0
        push(f"| `{pid}` | {b['type']} | {compared} | {matches} | {pct:.0f}% |")
        t = b["type"]
        prev = per_type_agreement.get(t, (0, 0))
        per_type_agreement[t] = (prev[0] + matches, prev[1] + compared)
    if total_seen:
        push("")
        push(f"**Overall greedy agreement:** "
             f"{total_match}/{total_seen} = **{100*total_match/total_seen:.1f}%**")
        push("")
        push("By prompt type:")
        push("")
        for t, (m, s) in sorted(per_type_agreement.items()):
            push(f"- **{t}**: {m}/{s} = {100*m/s:.1f}%")
    push("")

    # Top-k overlap: base's chosen token in factored's top-k?
    push("### Top-5 overlap (base's next token present in factored's top-5)")
    push("")
    push("Looser than greedy match — measures whether the factored model "
         "considers base's choice a high-probability option even when its "
         "own argmax differs.")
    push("")
    push("| id | type | positions | in top-5 | % |")
    push("|---|---|---|---|---|")
    total_hits = total_positions = 0
    for pid in ids:
        b = base_by_id[pid]
        f = fact_by_id[pid]
        if "generated_ids" not in b or "topk_ids_per_step" not in f:
            continue
        hits, total = _topk_overlap(b["generated_ids"], f["topk_ids_per_step"])
        total_hits += hits
        total_positions += total
        pct = 100 * hits / total if total else 0
        push(f"| `{pid}` | {b['type']} | {total} | {hits} | {pct:.0f}% |")
    if total_positions:
        push("")
        push(f"**Overall top-5 overlap:** "
             f"{total_hits}/{total_positions} = "
             f"**{100*total_hits/total_positions:.1f}%**")
    push("")

    # ── Side-by-side response dump (for qualitative judging) ────────────
    push("## Side-by-side responses (qualitative)")
    push("")
    push("Paste any of these into the chat for qualitative judgment.")
    push("")
    for pid in ids:
        b = base_by_id[pid]
        f = fact_by_id[pid]
        push(f"### `{pid}` ({b['type']})")
        push("")
        preview_prompt = b["prompt"]
        if len(preview_prompt) > args.prompt_preview_chars:
            preview_prompt = preview_prompt[:args.prompt_preview_chars] + "…"
        push(f"**Prompt:** {preview_prompt}")
        if "expected" in b:
            push(f"  *(expected: {b['expected']})*")
        push("")
        b_txt = b["generated_text"][:args.response_preview_chars]
        f_txt = f["generated_text"][:args.response_preview_chars]
        push(f"**Base:** {b_txt!r}")
        push("")
        push(f"**Factored:** {f_txt!r}")
        push("")

    # UTF-8 explicit: the report contains → and ✓/✗ which cp1252 can't encode.
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    # Summary line for the terminal
    if exact_n:
        print(f"  exact-match base {exact_base_hits}/{exact_n} "
              f"factored {exact_fact_hits}/{exact_n}")
    if total_seen:
        print(f"  greedy agreement {100*total_match/total_seen:.1f}%")
    if total_positions:
        print(f"  top-5 overlap {100*total_hits/total_positions:.1f}%")


if __name__ == "__main__":
    main()
