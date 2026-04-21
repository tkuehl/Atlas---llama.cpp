"""Download wikitext-2-raw test split to data/wiki.test.raw.

This is the standard llama.cpp perplexity input — documented in
examples/perplexity/README.md. Tests run at 512-token ctx on the full
test set (~280K tokens), producing ~540 perplexity windows.
"""

from pathlib import Path
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "wiki.test.raw"
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"[fetch] wikitext-2-raw-v1 test split")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n".join(ds["text"])

print(f"[write] {OUT}  ({len(text):,} chars)")
OUT.write_text(text, encoding="utf-8")
print("[done]")
