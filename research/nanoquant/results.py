"""Append-only results log.

All NanoQuant runs (baselines and quantized) write one entry to `results.json`
so we have a single, long-lived PPL / zero-shot history tagged by git commit,
model, method, hyperparameters, and hardware.

Schema (v1):

  {
    "schema_version": 1,
    "entries": [
      {
        "id":            str,   # "<utc>-<model-slug>-<method>"
        "timestamp_utc": str,   # ISO-8601
        "git": {"commit": str, "dirty": bool, "branch": str | None},
        "model": {"hf_id": str, "revision": str | None, "dtype": str},
        "method": {"name": str, "params": dict},
        "eval":   {"dataset": str, "split": str, "seq_len": int,
                   "stride": int, "num_tokens": int, "num_windows": int,
                   "metric": str, "value": float, "nll_mean": float},
        "hardware": {"gpu": str, "torch": str},
        "notes": str | None
      },
      ...
    ]
  }

Writes are atomic (temp file + replace) so an interrupted run can't corrupt
the log.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.json"
SCHEMA_VERSION = 1


def _run(cmd: list[str]) -> str:
    out = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def git_state() -> dict:
    commit = _run(["git", "rev-parse", "HEAD"]) or None
    dirty_out = _run(["git", "status", "--porcelain"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or None
    return {"commit": commit, "dirty": bool(dirty_out), "branch": branch}


def _load() -> dict:
    if not RESULTS_PATH.exists():
        return {"schema_version": SCHEMA_VERSION, "entries": []}
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"results.json schema_version={data.get('schema_version')} "
            f"!= expected {SCHEMA_VERSION}"
        )
    return data


def _atomic_write(data: dict) -> None:
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, RESULTS_PATH)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def append_entry(
    *,
    model_hf_id: str,
    model_dtype: str,
    method_name: str,
    method_params: dict,
    eval_info: dict,
    hardware: dict,
    model_revision: str | None = None,
    notes: str | None = None,
) -> dict:
    """Append a run to results.json. Returns the stored entry."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {
        "id": f"{ts}-{_slug(model_hf_id)}-{_slug(method_name)}",
        "timestamp_utc": ts,
        "git": git_state(),
        "model": {
            "hf_id": model_hf_id,
            "revision": model_revision,
            "dtype": model_dtype,
        },
        "method": {"name": method_name, "params": method_params},
        "eval": eval_info,
        "hardware": hardware,
        "notes": notes,
    }
    data = _load()
    data["entries"].append(entry)
    _atomic_write(data)
    return entry
