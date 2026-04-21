# Extreme Quantization — Journal

Running log of decisions, findings, and course corrections for the
extreme-quantization research vein. Append-only; don't rewrite past
entries — supersede them with new entries instead.

---

## 2026-04-20 — Vein opened

Split off from the broader "efficient LLM architecture" exploration.
The conversation surveyed four directions: implicit weight
representations, codebook/VQ quantization, retrieval-as-memory, and
state-space models. Codebook/quantization won on both research volume
and practical maturity — it's the only direction shipping 70B on
consumer GPUs today.

Narrowed further from "quantization in general" to "quantization of
information-dense modern models" (Llama-3+, Qwen3+). The Llama-2-era
results don't transfer cleanly; there's a real open problem here that
stacking published techniques should be able to close.

Initial DESIGN.md committed. Vision doc only — no code, no benchmarks.

Pipeline positioning: this vein adds stages (rotation, QAT) that
compose with CALDERA rather than replacing it. The active CALDERA
work is not blocked by this vein and vice versa.

**Next decision point:** run the §6 open-question #1 ablation —
Hadamard rotation alone on Qwen3-8B at 3-bit. If it closes the gap
outright, the pipeline plan simplifies and T1 becomes nearly free.
If it doesn't, we get a concrete sense of how much the downstream
stages need to contribute.
