"""llama-server HTTP backend (OpenAI-compat). Thin stub; finalized when C++ runtime lands."""

import time
from typing import List

import httpx

from .base import Backend


class LlamaCppBackend(Backend):
    def __init__(self, base_url: str = "http://127.0.0.1:11500", model_name: str = "local"):
        self.name = f"llamacpp:{base_url}"
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.client = httpx.Client(timeout=120.0)

    def logits(self, token_ids):
        raise NotImplementedError(
            "llama-server exposes logits only via the /completion endpoint with 'n_probs' + 'tokens'. "
            "Hook this up when we need quality/consistency evals against the C++ runtime."
        )

    def generate(self, prompt, max_new_tokens, temperature=0.0):
        t0 = time.perf_counter()
        with self.client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                 json={"model": self.model_name,
                                       "messages": [{"role": "user", "content": prompt}],
                                       "max_tokens": max_new_tokens,
                                       "temperature": temperature,
                                       "stream": True}) as r:
            r.raise_for_status()
            ttft = None
            parts = []
            n = 0
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                import json
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                delta = obj["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    parts.append(delta)
                    n += 1
        total = time.perf_counter() - t0
        return {"text": "".join(parts), "ttft_s": ttft or total, "total_s": total, "n_tokens": n}

    def tokenize(self, text):
        r = self.client.post(f"{self.base_url}/tokenize", json={"content": text})
        r.raise_for_status()
        return r.json()["tokens"]

    def close(self):
        self.client.close()
