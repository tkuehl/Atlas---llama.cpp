"""Interactive chat with a factored model — no GGUF conversion, no C++ runtime.

Loads the HuggingFace model, applies the factored weights we computed via
basis_sharing.py (reconstructing dense in-memory), and lets you prompt it.
This exists to gut-check that factored quality is actually usable before we
invest in Phase 2c/3 C++ work.

Usage:
    # Single-shot:
    python chat_factored.py --factored factored_out_qwen3b --prompt "What is 2+2?"

    # Also compare against the unfactored baseline:
    python chat_factored.py --factored factored_out_qwen3b --prompt "..." --compare

    # Interactive REPL:
    python chat_factored.py --factored factored_out_qwen3b --repl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from basis_sharing import load_factored_reconstruct  # noqa


def load_factored_model(factored_dir, device, dtype):
    manifest = json.loads((Path(factored_dir) / "manifest.json").read_text())
    model_id = manifest["model_id"]
    print(f"loading base model {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    print(f"reconstructing factored tensors from {factored_dir}")
    state, _ = load_factored_reconstruct(factored_dir)
    # state dict keys are HF-style (model.layers.{i}.{role}.weight); strict=False
    # because not every model tensor is in `state` (we only emit ones we factored
    # plus the untouched ones we saved, which should cover everything).
    existing = {k: v.dtype for k, v in model.state_dict().items()}
    casted = {}
    for k, v in state.items():
        if k in existing:
            casted[k] = v.to(existing[k])
    missing, unexpected = model.load_state_dict(casted, strict=False)
    print(f"  loaded {len(casted)} tensors, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"  [warn] model is missing weights not present in factored dir: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    return model, tokenizer, manifest


def load_baseline_model(model_id, device, dtype):
    print(f"loading baseline {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7, device="cuda"):
    """Run chat completion and return text + tokens/sec."""
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    else:
        text = prompt
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    t0 = time.time()
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    dt = time.time() - t0
    new_tokens = out.shape[1] - input_ids.shape[1]
    response = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return response, new_tokens / max(dt, 1e-6), dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factored", required=True, help="path to factored_out dir")
    p.add_argument("--prompt", default=None, help="single-shot prompt")
    p.add_argument("--repl", action="store_true", help="interactive REPL")
    p.add_argument("--compare", action="store_true",
                   help="also generate from the unfactored baseline model")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    device = args.device if torch.cuda.is_available() else "cpu"

    factored_model, tokenizer, manifest = load_factored_model(args.factored, device, dtype)
    baseline_model = None
    if args.compare:
        # Load baseline on CPU first to save VRAM, or swap sequentially
        # For simplicity we'll load both; 3B + 3B = 12 GB at bf16, may be tight
        print("(loading baseline for comparison — if this OOMs, run without --compare)")
        baseline_model, _ = load_baseline_model(manifest["model_id"], device, dtype)

    def run_one(prompt):
        print(f"\n>>> {prompt}")
        resp, tps, dt = generate(factored_model, tokenizer, prompt,
                                 args.max_new_tokens, args.temperature, device)
        print(f"\n--- factored ({tps:.1f} tok/s, {dt:.1f}s) ---")
        print(resp)
        if baseline_model is not None:
            resp_b, tps_b, dt_b = generate(baseline_model, tokenizer, prompt,
                                           args.max_new_tokens, args.temperature, device)
            print(f"\n--- baseline ({tps_b:.1f} tok/s, {dt_b:.1f}s) ---")
            print(resp_b)

    if args.prompt:
        run_one(args.prompt)
    elif args.repl:
        print("\n[Ctrl-C to exit]")
        try:
            while True:
                line = input("\nprompt> ").strip()
                if not line:
                    continue
                run_one(line)
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
    else:
        # Default demo prompts
        demos = [
            "What is 2 + 2?",
            "Write a haiku about the moon.",
            "List three Python list methods.",
        ]
        for prompt in demos:
            run_one(prompt)


if __name__ == "__main__":
    main()
