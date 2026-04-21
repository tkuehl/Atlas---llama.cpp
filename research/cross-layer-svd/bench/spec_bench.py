"""
spec_bench.py — POST each prompt in a fixture to a running llama-server's
OpenAI-compat endpoint, record the `timings` from each response as JSONL.

The server is assumed to be already running and reachable at --server.
For the start-server-then-bench-then-stop flow use run_condition.py,
which wraps this.
"""

import argparse
import json
import sys
import time

import httpx


def run(args):
    with open(args.fixture, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    prompts = fixture["prompts"]

    client = httpx.Client(timeout=args.timeout)
    out = open(args.out, "w", encoding="utf-8")

    print(f"[bench] {len(prompts)} prompts → {args.server}", file=sys.stderr)

    for i, p in enumerate(prompts, 1):
        body = {
            "messages": [{"role": "user", "content": p["prompt"]}],
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": p.get("max_tokens", 256),
            "stream": False,
        }
        if args.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
        t0 = time.time()
        try:
            r = client.post(f"{args.server}/v1/chat/completions", json=body)
            r.raise_for_status()
        except Exception as e:
            print(f"[bench] prompt {p['id']} failed: {e}", file=sys.stderr)
            row = {
                "condition": args.condition_name,
                "id": p["id"],
                "slice": p["slice"],
                "error": str(e),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            continue
        wall = time.time() - t0
        resp = r.json()

        timings = resp.get("timings", {})
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        row = {
            "condition": args.condition_name,
            "id": p["id"],
            "slice": p["slice"],
            "wall_seconds": wall,
            "response_preview": content[:160],
            "response_length_chars": len(content),
            "timings": timings,
        }
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(
            f"[bench] {i:>2}/{len(prompts)} {p['id']:<12} "
            f"tok/s={timings.get('predicted_per_second', 0):.1f} "
            f"draft_n={timings.get('draft_n', 0)} "
            f"accepted={timings.get('draft_n_accepted', 0)} "
            f"cache_n={timings.get('cache_n', 0)}",
            file=sys.stderr,
        )

    out.close()
    print(f"[bench] done → {args.out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:11502")
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition-name", default="unnamed")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--enable-thinking", type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                    default=False,
                    help="Qwen3-style enable_thinking flag via chat_template_kwargs. Default false: "
                         "measures output-shape generation, not reasoning. Pass --enable-thinking=true "
                         "to keep model-default behavior.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
