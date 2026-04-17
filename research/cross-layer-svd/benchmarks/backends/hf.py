"""HuggingFace backend. Wraps an AutoModelForCausalLM for the consistency + quality suites."""

import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Backend


class HFBackend(Backend):
    def __init__(self, model_id: str, device: str = "cuda", dtype=torch.float16,
                 model=None, tokenizer=None):
        self.name = f"hf:{model_id}"
        self.device = device
        if model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def logits(self, token_ids):
        ids = torch.tensor([token_ids], device=self.device)
        out = self.model(ids)
        return out.logits[0]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=0.0):
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                          temperature=max(temperature, 1e-6), pad_token_id=self.tokenizer.eos_token_id)
        t0 = time.perf_counter()
        # TTFT: run a single forward to measure first-token latency
        first = self.model.generate(ids, max_new_tokens=1, do_sample=False,
                                    pad_token_id=self.tokenizer.eos_token_id)
        ttft = time.perf_counter() - t0
        out = self.model.generate(ids, **gen_kwargs)
        total = time.perf_counter() - t0
        text = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        n = out.shape[1] - ids.shape[1]
        return {"text": text, "ttft_s": ttft, "total_s": total, "n_tokens": int(n)}

    def tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def close(self):
        del self.model
        torch.cuda.empty_cache()
