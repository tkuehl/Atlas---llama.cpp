"""Parse llama-perplexity logs into a JSON summary.

llama-perplexity emits a tail line of the form:
    Final estimate: PPL = 8.1234 +/- 0.04567

Plus intermediate per-chunk values. We just extract the final number
and confidence interval, plus the model filename and token count.
"""

import argparse
import json
import re
from pathlib import Path


FINAL = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")
NCHUNKS = re.compile(r"perplexity:\s*calculating perplexity over (\d+) chunks, n_ctx=(\d+)")
MODEL_PATH = re.compile(r"llama_model_loader:.*\bfrom\s+'([^']+)'")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    ppl_match = FINAL.search(text)
    if not ppl_match:
        return {"log": str(path), "error": "no Final estimate line found"}
    nchunks_match = NCHUNKS.search(text)
    model_match = MODEL_PATH.search(text)
    return {
        "log": str(path),
        "model": Path(model_match.group(1)).name if model_match else None,
        "ppl": float(ppl_match.group(1)),
        "ppl_stderr": float(ppl_match.group(2)),
        "n_chunks": int(nchunks_match.group(1)) if nchunks_match else None,
        "n_ctx": int(nchunks_match.group(2)) if nchunks_match else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()

    results = [parse_log(Path(p)) for p in args.logs]

    # Pretty print table
    print(f"{'config':30s}  {'ppl':>10s}  {'+/-':>8s}  chunks")
    print("-" * 65)
    for r in results:
        if "error" in r:
            print(f"{Path(r['log']).stem:30s}  ERROR: {r['error']}")
            continue
        print(f"{Path(r['log']).stem:30s}  {r['ppl']:>10.4f}  {r['ppl_stderr']:>8.4f}  {r['n_chunks']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
