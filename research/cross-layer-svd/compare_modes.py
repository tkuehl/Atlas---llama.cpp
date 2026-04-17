"""Run prototype.py in all three decomposition modes and print a diff table.

Usage:
    python compare_modes.py --ranks 64 128 256 512 --calib-samples 32 --eval-tokens 2048
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = HERE / ".venv" / "Scripts" / "python.exe"


def run_mode(mode, args):
    out = HERE / f"results_{mode}.json"
    cmd = [str(PYTHON), str(HERE / "prototype.py"),
           "--mode", mode,
           "--calib-samples", str(args.calib_samples),
           "--eval-tokens", str(args.eval_tokens),
           "--ranks", *[str(r) for r in args.ranks],
           "--out", str(out)]
    if args.no_asvd:
        cmd.append("--no-asvd")
    print(f"\n==== mode: {mode} ====\n  {' '.join(cmd)}")
    env = {"PYTHONIOENCODING": "utf-8"}
    import os
    env = {**os.environ, **env}
    subprocess.run(cmd, check=True, env=env)
    return json.loads(out.read_text())


def print_table(runs, baseline_ppl):
    modes = list(runs.keys())
    ranks = sorted(set(s["rank"] for r in runs.values() for s in r["sweeps"]))
    header = ["rank"] + modes
    print("\n\n=== PPL vs rank, by mode ===")
    print(f"baseline PPL = {baseline_ppl:.3f}")
    print("  " + " | ".join(f"{h:>18}" for h in header))
    for rk in ranks:
        row = [f"{rk:>18}"]
        for m in modes:
            match = next((s for s in runs[m]["sweeps"] if s["rank"] == rk), None)
            if match is None:
                row.append(f"{'-':>18}")
            else:
                ppl = match["ppl"]
                dp = match["delta_pct"]
                row.append(f"{ppl:>10.2f} ({dp:+.0f}%)")
        print("  " + " | ".join(row))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranks", nargs="+", type=int, default=[64, 128, 256, 512])
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--eval-tokens", type=int, default=2048)
    p.add_argument("--no-asvd", action="store_true")
    p.add_argument("--modes", nargs="+",
                   default=["per-matrix", "cross-layer-h", "cross-layer-v"],
                   choices=["per-matrix", "cross-layer-h", "cross-layer-v"])
    args = p.parse_args()

    runs = {}
    for mode in args.modes:
        runs[mode] = run_mode(mode, args)

    baseline_ppl = next(iter(runs.values()))["baseline_ppl"]
    print_table(runs, baseline_ppl)


if __name__ == "__main__":
    main()
