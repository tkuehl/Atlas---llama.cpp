"""
run_condition.py — Launch a llama-server with given flags, wait until it
is ready, run spec_bench.py against it, then terminate the server.

One invocation = one condition's JSONL file. Re-run with different
--extra-args / --name to produce comparable JSONLs.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

import httpx

COMMON_ARGS = [
    "--host", "127.0.0.1",
    "--port", "11502",
    "-ngl", "999",
    "--flash-attn", "on",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "--cache-reuse", "256",
    "-c", "8192",
    "--jinja",
]


def start_server(binary, model, extra_args, log_path):
    cmd = [binary, "-m", model] + COMMON_ARGS + extra_args
    print(f"[run] starting llama-server", file=sys.stderr)
    print(f"[run]   cmd: {' '.join(cmd)}", file=sys.stderr)
    print(f"[run]   log: {log_path}", file=sys.stderr)
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def wait_ready(base_url, proc, timeout_s=300):
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            print(f"[run] server exited early with code {proc.returncode}", file=sys.stderr)
            return False
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"[run] server ready after {elapsed:.1f}s", file=sys.stderr)
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="path to llama-server.exe")
    ap.add_argument("--model", required=True, help="path to the GGUF")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="condition name, e.g. 'baseline' or 'ngram_cache'")
    ap.add_argument("--extra-args", default="", help='extra flags appended to the server command, e.g. "--spec-type ngram-cache --draft-max 8"')
    ap.add_argument("--port", type=int, default=11502)
    ap.add_argument("--ready-timeout", type=int, default=300)
    ap.add_argument("--enable-thinking", type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                    default=False,
                    help="Qwen3 enable_thinking. Default false for workload-shape bench.")
    args = ap.parse_args()

    for p in (args.binary, args.model, args.fixture):
        if not os.path.exists(p):
            print(f"[run] missing: {p}", file=sys.stderr)
            sys.exit(2)

    extra = shlex.split(args.extra_args) if args.extra_args else []
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir or ".", f"{args.name}.server.log")

    proc, log_f = start_server(args.binary, args.model, extra, log_path)
    try:
        base = f"http://127.0.0.1:{args.port}"
        if not wait_ready(base, proc, timeout_s=args.ready_timeout):
            print("[run] server never became ready — see log", file=sys.stderr)
            sys.exit(3)

        # Invoke spec_bench programmatically
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import spec_bench

        bench_args = argparse.Namespace(
            fixture=args.fixture,
            server=base,
            out=args.out,
            condition_name=args.name,
            timeout=600.0,
            enable_thinking=args.enable_thinking,
        )
        spec_bench.run(bench_args)

    finally:
        print("[run] terminating server", file=sys.stderr)
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("[run] server didn't exit cleanly, killing", file=sys.stderr)
            proc.kill()
            proc.wait(timeout=5)
        log_f.close()
        print("[run] done", file=sys.stderr)


if __name__ == "__main__":
    main()
