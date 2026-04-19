"""Entropy characterization of a GGUF model.

Measures Shannon entropy of weight-storage bytes to estimate the lossless
compression ceiling for techniques like Cloudflare Unweight (exponent
Huffman on BF16) and any hypothetical Huffman/ANS layer on Q4_K_M codes.

Reports per-tensor and aggregate:
  * Raw-byte entropy (universal, works on any quant format)
  * For F16 / BF16: sign / exponent / mantissa entropy separately
    — compression ceiling = (exp_width - exp_entropy) / stored_width

Usage:
  python entropy_char.py --gguf path/to/model.gguf [--out report.json]
                         [--max-tensors N] [--filter regex]

The script is intentionally dependency-light (numpy + gguf-py only) and
reuses the same gguf-py path trick as caldera.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402


def shannon_bits(arr: np.ndarray) -> float:
    """Shannon entropy in bits. Input is a flat integer array."""
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def analyze_float16(raw: bytes) -> dict:
    """F16 (IEEE half): 1 sign + 5 exponent + 10 mantissa."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    sign = (u16 >> 15) & 0x1
    exp = (u16 >> 10) & 0x1F  # 5 bits
    mant = u16 & 0x3FF  # 10 bits
    return {
        "elements": int(u16.size),
        "sign_bits_entropy": shannon_bits(sign),
        "exp_bits_entropy": shannon_bits(exp),
        "exp_bits_stored": 5,
        "exp_ceiling_pct": (5 - shannon_bits(exp)) / 5 * 100,
        "mant_bits_entropy": shannon_bits(mant),
        "mant_bits_stored": 10,
    }


def analyze_bfloat16(raw: bytes) -> dict:
    """BF16: 1 sign + 8 exponent + 7 mantissa. Cloudflare's target."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    sign = (u16 >> 15) & 0x1
    exp = (u16 >> 7) & 0xFF  # 8 bits
    mant = u16 & 0x7F  # 7 bits
    return {
        "elements": int(u16.size),
        "sign_bits_entropy": shannon_bits(sign),
        "exp_bits_entropy": shannon_bits(exp),
        "exp_bits_stored": 8,
        "exp_ceiling_pct": (8 - shannon_bits(exp)) / 8 * 100,
        "mant_bits_entropy": shannon_bits(mant),
        "mant_bits_stored": 7,
    }


def analyze_raw(raw: bytes) -> dict:
    """Universal fallback — raw byte entropy. Best we can do for Q*_K
    without parsing block structure. Gives a lower bound on the lossless
    compression headroom of the stored bytes."""
    u8 = np.frombuffer(raw, dtype=np.uint8)
    H = shannon_bits(u8)
    return {
        "bytes": int(u8.size),
        "raw_byte_entropy_bits": H,
        "raw_byte_ceiling_pct": (8 - H) / 8 * 100,
    }


def analyze_tensor(t) -> dict:
    qtype = t.tensor_type
    # GGUFReader yields tensor.data as a numpy view over the packed blob.
    raw = bytes(t.data.tobytes()) if hasattr(t.data, "tobytes") else bytes(t.data)
    out = {
        "name": t.name,
        "shape": list(t.shape),
        "qtype": qtype.name,
        "stored_bytes": len(raw),
    }
    if qtype == GGMLQuantizationType.F16:
        out.update(analyze_float16(raw))
    elif qtype == GGMLQuantizationType.BF16:
        out.update(analyze_bfloat16(raw))
    out.update(analyze_raw(raw))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--max-tensors", type=int, default=None)
    p.add_argument("--filter", default=None,
                   help="regex; only analyze tensors whose name matches")
    args = p.parse_args()

    reader = GGUFReader(args.gguf)
    name_re = re.compile(args.filter) if args.filter else None

    results = []
    for i, t in enumerate(reader.tensors):
        if name_re and not name_re.search(t.name):
            continue
        if args.max_tensors and len(results) >= args.max_tensors:
            break
        r = analyze_tensor(t)
        results.append(r)

        # Per-tensor stdout line
        msg = f"{r['name'][:58]:<58} {r['qtype']:<10} "
        msg += f"raw_H={r['raw_byte_entropy_bits']:.3f}b  "
        msg += f"ceil={r['raw_byte_ceiling_pct']:.1f}%  "
        if "exp_bits_entropy" in r:
            msg += f"expH={r['exp_bits_entropy']:.2f}/"
            msg += f"{r['exp_bits_stored']}b ({r['exp_ceiling_pct']:.1f}%)"
        print(msg)

    # Aggregate by qtype
    by_qtype: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_qtype[r["qtype"]].append(r)

    agg = {}
    total_bytes = 0
    total_savings_raw = 0.0
    total_savings_exp = 0.0
    for qt, rs in by_qtype.items():
        b = sum(r["stored_bytes"] for r in rs)
        total_bytes += b
        # Weighted raw-byte ceiling
        raw_saving = sum(r["stored_bytes"] * r["raw_byte_ceiling_pct"] / 100
                         for r in rs)
        total_savings_raw += raw_saving
        exp_saving = sum(r["stored_bytes"] * r.get("exp_ceiling_pct", 0) / 100
                         * (r.get("exp_bits_stored", 0) / 16) for r in rs
                         if "exp_bits_entropy" in r)
        total_savings_exp += exp_saving
        agg[qt] = {
            "count": len(rs),
            "total_stored_bytes": b,
            "mean_raw_byte_entropy": float(np.mean([r["raw_byte_entropy_bits"]
                                                    for r in rs])),
            "weighted_raw_byte_ceiling_pct": raw_saving / b * 100 if b else 0,
        }
        if any("exp_bits_entropy" in r for r in rs):
            agg[qt]["mean_exp_bits_entropy"] = float(np.mean(
                [r["exp_bits_entropy"] for r in rs if "exp_bits_entropy" in r]
            ))
            agg[qt]["mean_exp_ceiling_pct"] = float(np.mean(
                [r["exp_ceiling_pct"] for r in rs if "exp_ceiling_pct" in r]
            ))

    summary = {
        "gguf": str(args.gguf),
        "n_tensors_analyzed": len(results),
        "total_bytes": total_bytes,
        "raw_byte_ceiling_pct_overall": (total_savings_raw / total_bytes * 100
                                         if total_bytes else 0),
        "exponent_only_ceiling_pct_overall": (total_savings_exp / total_bytes * 100
                                              if total_bytes else 0),
        "by_qtype": agg,
    }

    print("\n=== Summary ===")
    print(f"Tensors analyzed:   {summary['n_tensors_analyzed']}")
    print(f"Total stored bytes: {summary['total_bytes']:,}")
    print(f"Raw-byte lossless ceiling:    "
          f"{summary['raw_byte_ceiling_pct_overall']:.1f}%")
    if summary["exponent_only_ceiling_pct_overall"]:
        print(f"Exponent-only (BF16/F16 only) ceiling: "
              f"{summary['exponent_only_ceiling_pct_overall']:.2f}% "
              f"of total bytes")
    for qt, a in agg.items():
        print(f"  {qt:<8} n={a['count']:<4}  "
              f"{a['total_stored_bytes']/1e9:.2f} GB  "
              f"raw_H={a['mean_raw_byte_entropy']:.3f}b/B  "
              f"ceil={a['weighted_raw_byte_ceiling_pct']:.1f}%", end="")
        if "mean_exp_ceiling_pct" in a:
            print(f"  exp_ceil={a['mean_exp_ceiling_pct']:.1f}%")
        else:
            print()

    out_path = args.out or (Path(args.gguf).stem + ".entropy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "tensors": results}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
