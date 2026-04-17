"""Compare two or more harness runs and emit a markdown diff table."""

import argparse
import json
from pathlib import Path


def _fmt(x, prec=4):
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def load(path):
    return json.loads(Path(path).read_text())


def diff(baseline_run, candidate_runs):
    lines = []
    lines.append(f"# Benchmark diff — baseline: `{baseline_run['model']}`")
    lines.append("")

    # Consistency
    cons_rows = [("run", "rank", "kl_div_bits", "top1_agree", "top5_agree", "entropy_delta")]
    for c in candidate_runs:
        s = c.get("suites", {}).get("consistency", {})
        if "kl_div_bits" not in s:
            continue
        cons_rows.append((
            Path(c.get("_source", "?")).stem,
            c.get("rank", "-"),
            _fmt(s["kl_div_bits"]),
            _fmt(s["top1_agree"]),
            _fmt(s["top5_agree"]),
            _fmt(s["entropy_delta"]),
        ))
    if len(cons_rows) > 1:
        lines.append("## Consistency (vs baseline logits)")
        lines.append("| " + " | ".join(cons_rows[0]) + " |")
        lines.append("|" + "|".join("---" for _ in cons_rows[0]) + "|")
        for row in cons_rows[1:]:
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

    # Throughput
    thr_rows = [("run", "rank", "median_tok_s", "mean_ttft_ms", "peak_vram_mb")]
    for run in [baseline_run] + candidate_runs:
        s = run.get("suites", {}).get("throughput")
        if not s:
            continue
        thr_rows.append((
            Path(run.get("_source", "baseline")).stem,
            run.get("rank", "-"),
            _fmt(s["summary"]["median_decode_tok_per_s"], 1),
            _fmt(s["summary"]["mean_ttft_ms"], 1),
            _fmt(s["peak_vram_mb"], 0),
        ))
    if len(thr_rows) > 1:
        lines.append("## Throughput")
        lines.append("| " + " | ".join(thr_rows[0]) + " |")
        lines.append("|" + "|".join("---" for _ in thr_rows[0]) + "|")
        for row in thr_rows[1:]:
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

    # Quality
    all_tasks = set()
    for run in [baseline_run] + candidate_runs:
        q = run.get("suites", {}).get("quality", {})
        all_tasks.update(q.get("scores", {}).keys())
    if all_tasks:
        lines.append("## Quality (lm-evaluation-harness)")
        header = ["run", "rank"] + sorted(all_tasks)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for run in [baseline_run] + candidate_runs:
            q = run.get("suites", {}).get("quality", {}).get("scores", {})
            row = [Path(run.get("_source", "baseline")).stem, str(run.get("rank", "-"))]
            for t in sorted(all_tasks):
                row.append(_fmt(q.get(t, "-"), 3))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="baseline run JSON")
    p.add_argument("--candidates", nargs="+", required=True, help="candidate run JSONs")
    p.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    args = p.parse_args()

    base = load(args.baseline)
    base["_source"] = args.baseline
    cands = []
    for c in args.candidates:
        obj = load(c)
        obj["_source"] = c
        cands.append(obj)

    md = diff(base, cands)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
