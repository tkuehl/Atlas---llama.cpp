"""Benchmark a single model — base or factored — through a fixed prompt set.

Records per-prompt latency, tok/s, generated text + token ids + log-probs,
and peak/mean resource utilization (GPU VRAM, GPU util %, CPU %, RSS)
sampled on a background thread. Emits a JSON file consumable by
`bench_compare.py` for side-by-side analysis.

Usage:
    # Base HF model:
    python bench_model.py --base Qwen/Qwen2.5-3B --out base.json

    # Factored model from our pipeline (reconstructed dense in-memory):
    python bench_model.py --base Qwen/Qwen2.5-3B \
        --factored factored_out_qwen25_3b_r100 --out factored.json

    # Use a custom prompt fixture:
    python bench_model.py --base Qwen/Qwen2.5-3B \
        --prompts my_prompts.json --out base.json
"""

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from basis_sharing import load_factored_reconstruct  # noqa: E402


class ResourceSampler:
    """Polls process + GPU resource usage on a background thread.

    100 ms cadence is a reasonable floor — tight enough to catch
    transient peaks during a decode step, loose enough that the
    sampler itself doesn't perturb timings. All numbers are in MB for
    memory and percent for utilization; stored raw so the caller can
    compute peak/mean/percentile after the run.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[tuple] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.process = psutil.Process()
        self._gpu_ok = torch.cuda.is_available()
        # Prime cpu_percent so the first real sample isn't 0
        self.process.cpu_percent()

    def start(self) -> None:
        if self._gpu_ok:
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.summary()

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            rss = self.process.memory_info().rss / 1e6
            cpu = self.process.cpu_percent()
            vram = 0.0
            gpu_util = 0
            if self._gpu_ok:
                vram = torch.cuda.memory_allocated() / 1e6
                try:
                    gpu_util = torch.cuda.utilization()
                except Exception:
                    # Older torch or driver — skip utilization
                    gpu_util = -1
            self.samples.append((t, rss, vram, gpu_util, cpu))
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        rss = [s[1] for s in self.samples]
        vram = [s[2] for s in self.samples]
        gpu_util = [s[3] for s in self.samples if s[3] >= 0]
        cpu = [s[4] for s in self.samples]
        out = {
            "n_samples": len(self.samples),
            "duration_sec": self.samples[-1][0] - self.samples[0][0],
            "rss_peak_mb": max(rss),
            "rss_mean_mb": statistics.mean(rss),
            "cpu_pct_peak": max(cpu),
            "cpu_pct_mean": statistics.mean(cpu),
        }
        if vram and any(v > 0 for v in vram):
            out["vram_peak_mb"] = max(vram)
            out["vram_mean_mb"] = statistics.mean(vram)
            # torch's internal peak tracker is more reliable than our
            # 100ms polling for sub-sample transient spikes.
            if self._gpu_ok:
                out["vram_torch_peak_mb"] = (
                    torch.cuda.max_memory_allocated() / 1e6)
        if gpu_util:
            out["gpu_util_peak"] = max(gpu_util)
            out["gpu_util_mean"] = statistics.mean(gpu_util)
        return out


def load_factored_model(base_id: str, factored_dir: str,
                        device: str, dtype: torch.dtype) -> tuple:
    """Load base tokenizer/model config, overlay reconstructed factored
    weights. Mirrors chat_factored.py but skips the baseline comparison."""
    print(f"loading base {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    model.eval()
    print(f"reconstructing factored weights from {factored_dir}")
    state, _ = load_factored_reconstruct(factored_dir)
    existing = {k: v.dtype for k, v in model.state_dict().items()}
    casted = {k: v.to(existing[k]) for k, v in state.items() if k in existing}
    missing, unexpected = model.load_state_dict(casted, strict=False)
    print(f"  loaded {len(casted)} tensors, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    return model, tokenizer


def load_base_model(base_id: str, device: str, dtype: torch.dtype) -> tuple:
    print(f"loading base {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int,
                 temperature: float, device: str, topk_k: int = 5) -> dict:
    """Generate for a single prompt. Records timing (TTFT, total),
    generated tokens + their log-probs, and the top-k token ids at each
    step so bench_compare.py can compute agreement metrics without
    re-running the model.

    We do manual stepping instead of `model.generate()` because generate
    hides TTFT and per-token timing and doesn't expose the top-k tensor."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Prefill
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :]
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_sec = time.perf_counter() - t0

    generated_ids: list[int] = []
    generated_logprobs: list[float] = []
    generated_topk: list[list[int]] = []
    token_times: list[float] = []
    t_first_token: float | None = None

    t_last = time.perf_counter()
    for step in range(max_new_tokens):
        # Log-probs (softmax-then-log, numerically stable via log_softmax)
        logprobs = torch.log_softmax(next_logits, dim=-1)
        # Top-k for agreement scoring
        topk = torch.topk(logprobs, k=topk_k, dim=-1)
        topk_ids = topk.indices[0].tolist()

        if temperature <= 0:
            next_id = int(next_logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated_ids.append(next_id)
        generated_logprobs.append(float(logprobs[0, next_id].item()))
        generated_topk.append(topk_ids)

        if device == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        token_times.append(now - t_last)
        t_last = now
        if t_first_token is None:
            t_first_token = now - t0 - prefill_sec  # elapsed after prefill finished

        # Stop on EOS
        if next_id == tokenizer.eos_token_id:
            break

        # Next step
        next_input = torch.tensor([[next_id]], device=device)
        out = model(next_input, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

    if device == "cuda":
        torch.cuda.synchronize()
    total_sec = time.perf_counter() - t0

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_new = len(generated_ids)
    decode_sec = total_sec - prefill_sec
    tok_per_sec = n_new / decode_sec if decode_sec > 0 else 0.0

    return {
        "prompt_len_tokens": prompt_len,
        "prefill_sec": prefill_sec,
        "ttft_sec": (t_first_token if t_first_token is not None else prefill_sec),
        "total_sec": total_sec,
        "decode_sec": decode_sec,
        "n_new_tokens": n_new,
        "tok_per_sec": tok_per_sec,
        "generated_text": text,
        "generated_ids": generated_ids,
        "generated_logprobs": generated_logprobs,
        "topk_ids_per_step": generated_topk,
        "per_token_sec": token_times,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Base model id or local path (HF-compatible). When "
                        "--factored is set, this is used for the tokenizer "
                        "and the model shell; factored weights overlay it.")
    p.add_argument("--factored", default=None,
                   help="Path to factored_out_* directory. Omit to benchmark "
                        "the base model as-is.")
    p.add_argument("--prompts", default=str(HERE / "bench_prompts.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 for deterministic greedy decode (recommended "
                        "for base-vs-factored comparison).")
    p.add_argument("--topk-k", type=int, default=5,
                   help="Record top-k token ids at each decode step for "
                        "downstream agreement scoring.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"  (note: CUDA not available, falling back to {device})")
    dtype = getattr(torch, args.dtype)

    prompts_doc = json.loads(Path(args.prompts).read_text())
    prompts = prompts_doc["prompts"]

    # Model load
    t_load = time.perf_counter()
    sampler = ResourceSampler(interval=0.1)
    sampler.start()
    if args.factored:
        model, tokenizer = load_factored_model(args.base, args.factored,
                                               device, dtype)
        model_kind = "factored"
    else:
        model, tokenizer = load_base_model(args.base, device, dtype)
        model_kind = "base"
    load_sec = time.perf_counter() - t_load
    load_resources = sampler.stop()
    print(f"  model loaded in {load_sec:.1f}s")

    # Benchmark run
    sampler = ResourceSampler(interval=0.1)
    sampler.start()
    t_bench = time.perf_counter()

    per_prompt = []
    for prompt in prompts:
        print(f"  [{prompt['id']}] {prompt['text'][:60]}"
              + ("..." if len(prompt["text"]) > 60 else ""))
        r = generate_one(model, tokenizer, prompt["text"],
                         args.max_new_tokens, args.temperature,
                         device, args.topk_k)
        r["id"] = prompt["id"]
        r["type"] = prompt["type"]
        r["prompt"] = prompt["text"]
        if "expected" in prompt:
            r["expected"] = prompt["expected"]
            # Naive exact-match: "Berlin" in "Berlin, Germany" -> True.
            # Case-insensitive. Caller can refine downstream.
            r["exact_match"] = prompt["expected"].lower() in r["generated_text"].lower()
        per_prompt.append(r)
        print(f"    -> {r['tok_per_sec']:.1f} tok/s ({r['n_new_tokens']} "
              f"tokens, TTFT {r['ttft_sec']*1000:.0f}ms)")

    bench_sec = time.perf_counter() - t_bench
    bench_resources = sampler.stop()

    # Aggregates over prompts
    ttfts = [r["ttft_sec"] for r in per_prompt]
    tokps = [r["tok_per_sec"] for r in per_prompt if r["tok_per_sec"] > 0]
    aggregate = {
        "n_prompts": len(per_prompt),
        "total_new_tokens": sum(r["n_new_tokens"] for r in per_prompt),
        "total_bench_sec": bench_sec,
        "ttft_mean_sec": statistics.mean(ttfts) if ttfts else 0,
        "ttft_median_sec": statistics.median(ttfts) if ttfts else 0,
        "tok_per_sec_mean": statistics.mean(tokps) if tokps else 0,
        "tok_per_sec_median": statistics.median(tokps) if tokps else 0,
    }

    report = {
        "model_kind": model_kind,
        "base_id": args.base,
        "factored_dir": args.factored,
        "device": device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "model_load_sec": load_sec,
        "resources_load": load_resources,
        "resources_bench": bench_resources,
        "aggregate": aggregate,
        "prompts": per_prompt,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print(f"  agg: {aggregate['tok_per_sec_mean']:.1f} tok/s mean, "
          f"TTFT {aggregate['ttft_mean_sec']*1000:.0f}ms mean, "
          f"peak VRAM {bench_resources.get('vram_peak_mb', 0):.0f} MB, "
          f"peak RSS {bench_resources.get('rss_peak_mb', 0):.0f} MB")


if __name__ == "__main__":
    main()
