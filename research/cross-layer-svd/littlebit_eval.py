"""Load a trained LittleBit checkpoint and characterize the model.

Runs against the state_dict saved by littlebit_qat_model.py's
--checkpoint path.  Does NOT retrain; just loads and evaluates.

Measurements:
  1. Full wikitext-2 test PPL (all ~250k tokens, not the truncated
     25k-token estimate used during training).
  2. Per-layer activation drift vs. teacher on a calibration batch
     (the empirical compounding story §13 predicted locally).
  3. Generation samples from a small prompt set.

Usage:
  python littlebit_eval.py \\
      --checkpoint littlebit_qat_checkpoint_r512.pt \\
      --out littlebit_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from littlebit_qat_model import (
    LittleBitLinearHF,
    wrap_model_littlebit_shapes,
    wikitext_ppl,
)


def load_student_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Instantiate a LittleBit-wrapped student and load state_dict."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading checkpoint {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"  checkpoint config: model={cfg['model']}  "
          f"rank={cfg['rank']}  steps={cfg['steps']}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32,
    )
    wrapped = wrap_model_littlebit_shapes(
        model, r=cfg["rank"], tau=cfg.get("tau", 100.0),
    )
    print(f"  wrapped {wrapped} Linear layers (shape-only, no SVD)",
          flush=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, cfg, ckpt.get("summary", {})


def generation_samples(model, tokenizer, device: torch.device,
                       teacher=None, prompts=None,
                       max_new_tokens: int = 60) -> list[dict]:
    if prompts is None:
        prompts = [
            "The capital of France is",
            "To compress neural network weights to sub-1 bit per weight,",
            "Once upon a time,",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    ",
            "Q: What's 7 times 8?\nA:",
        ]
    out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        rec = {"prompt": p}
        with torch.no_grad():
            s = model.generate(ids, max_new_tokens=max_new_tokens,
                               do_sample=False)
            rec["student"] = tokenizer.decode(s[0, ids.shape[1]:],
                                              skip_special_tokens=True)
            if teacher is not None:
                t = teacher.generate(ids, max_new_tokens=max_new_tokens,
                                     do_sample=False)
                rec["teacher"] = tokenizer.decode(t[0, ids.shape[1]:],
                                                   skip_special_tokens=True)
        out.append(rec)
    return out


def per_layer_activation_drift(model, teacher, tokenizer,
                               device: torch.device,
                               n_tokens: int = 4096,
                               seq_len: int = 512) -> list[dict]:
    """Forward a calibration batch through both models, hook every
    layer's hidden-state output, measure rel Frobenius err per layer.

    Relies on Qwen2Model.layers being iterable; works the same for
    teacher and student (architectures are identical)."""
    from datasets import load_dataset
    print("  collecting calibration tokens...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in ds)
    tokens = tokenizer(text, return_tensors="pt").input_ids[0][:n_tokens]
    n_windows = max(1, n_tokens // seq_len)
    print(f"  {n_tokens} tokens = {n_windows} windows of {seq_len}",
          flush=True)

    # Hook every decoder layer on both models.
    t_hidden: list[list[torch.Tensor]] = [[] for _ in range(100)]
    s_hidden: list[list[torch.Tensor]] = [[] for _ in range(100)]

    def make_hook(store, idx):
        def hook(_mod, _inp, out):
            # decoder layer outputs a tuple; first element is hidden
            h = out[0] if isinstance(out, tuple) else out
            store[idx].append(h.detach().cpu())
        return hook

    t_handles = []
    s_handles = []
    for i, layer in enumerate(teacher.model.layers):
        t_handles.append(layer.register_forward_hook(make_hook(t_hidden, i)))
    for i, layer in enumerate(model.model.layers):
        s_handles.append(layer.register_forward_hook(make_hook(s_hidden, i)))

    with torch.no_grad():
        for w in range(n_windows):
            chunk = tokens[w * seq_len:(w + 1) * seq_len].to(device).unsqueeze(0)
            _ = teacher(chunk)
            _ = model(chunk)

    for h in t_handles:
        h.remove()
    for h in s_handles:
        h.remove()

    layers_info = []
    for i in range(len(teacher.model.layers)):
        t_cat = torch.cat(t_hidden[i], dim=1).float()
        s_cat = torch.cat(s_hidden[i], dim=1).float()
        t_norm = torch.linalg.norm(t_cat).item()
        rel = (torch.linalg.norm(t_cat - s_cat) / max(1e-30, t_norm)).item()
        layers_info.append({
            "layer": i, "rel_err": rel,
            "captured": max(0.0, 1 - rel ** 2),
        })
    return layers_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--teacher-model", default=None,
                   help="Override teacher model id; defaults to "
                        "the one in the checkpoint config.")
    p.add_argument("--eval-max-tokens", type=int, default=250_000,
                   help="Full wikitext-2 test is ~250k tokens.")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--drift-tokens", type=int, default=4096)
    p.add_argument("--out", default="littlebit_eval.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)

    student, tok, cfg, training_summary = load_student_from_checkpoint(
        ckpt_path, device,
    )

    print("loading teacher...", flush=True)
    from transformers import AutoModelForCausalLM
    teacher_id = args.teacher_model or cfg["model"]
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for pp in teacher.parameters():
        pp.requires_grad_(False)

    out = {
        "checkpoint": str(ckpt_path),
        "config": cfg,
        "training_summary": training_summary,
    }

    print("\n=== 1. Full wikitext-2 test PPL ===", flush=True)
    t0 = time.time()
    teacher_ppl = wikitext_ppl(teacher, tok, split="test",
                               seq_len=args.seq_len,
                               max_tokens=args.eval_max_tokens,
                               device=device)
    print(f"teacher PPL (full test, {args.eval_max_tokens} tok cap): "
          f"{teacher_ppl:.3f}  ({time.time() - t0:.0f}s)",
          flush=True)

    t0 = time.time()
    student_ppl = wikitext_ppl(student, tok, split="test",
                               seq_len=args.seq_len,
                               max_tokens=args.eval_max_tokens,
                               device=device)
    print(f"student PPL (full test, {args.eval_max_tokens} tok cap): "
          f"{student_ppl:.3f}  ({time.time() - t0:.0f}s)",
          flush=True)
    out["wikitext2_full"] = {
        "teacher_ppl": teacher_ppl,
        "student_ppl": student_ppl,
        "seq_len": args.seq_len,
        "max_tokens": args.eval_max_tokens,
    }

    print("\n=== 2. Per-layer activation drift ===", flush=True)
    drift = per_layer_activation_drift(
        student, teacher, tok, device,
        n_tokens=args.drift_tokens, seq_len=args.seq_len,
    )
    mean_rel = sum(d["rel_err"] for d in drift) / max(1, len(drift))
    mean_cap = sum(d["captured"] for d in drift) / max(1, len(drift))
    print(f"  per-layer mean rel-err: {mean_rel:.4f}  "
          f"captured: {mean_cap:.4f}",
          flush=True)
    print(f"  worst layer: rel-err={max(d['rel_err'] for d in drift):.4f}",
          flush=True)
    out["activation_drift"] = {
        "per_layer": drift,
        "mean_rel_err": mean_rel,
        "mean_captured": mean_cap,
    }

    print("\n=== 3. Generation samples ===", flush=True)
    samples = generation_samples(student, tok, device, teacher=teacher)
    for s in samples:
        print(f"\nprompt> {s['prompt']!r}", flush=True)
        print(f"teacher> {s['teacher']!r}", flush=True)
        print(f"student> {s['student']!r}", flush=True)
    out["generation_samples"] = samples

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
