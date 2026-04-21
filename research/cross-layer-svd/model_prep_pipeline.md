# Model Prep Pipeline

Planning doc for a pluggable pipeline that takes a HuggingFace model and
emits an optimized GGUF ready to serve from this llama.cpp fork.
Stages are swappable; new research wins slot in as additional stages
without reworking the interface.

Status: **planning**, no runner yet. Components listed as "shipped"
already exist as scripts in `research/cross-layer-svd/`; the pipeline
runner is what binds them.

## Motivation

Research so far (see [JOURNAL.md](JOURNAL.md)) has produced several
candidate optimizations — CALDERA (`W ≈ Q + L·R`), windowed Basis
Sharing + balanced truncation (deprecated), standard `llama-quantize`
quants, and a queued cross-matrix-codebook bet — plus shared
infrastructure: calibration gramian collection, `bench_model.py` /
`bench_compare.py`, and GGUF emission helpers.

Each experiment so far has been a bespoke script path. The next step is
a single entry point: "given model X from HF, run it through pipeline Y,
get a validated GGUF." Every new research direction becomes *a new
stage or a new pipeline recipe*, not a new script tree.

## Pipeline shape

```
HF model ─►[Stage 1]─►[Stage 2]─►[Stage 3]─► ... ─► GGUF ─► bench vs baseline
              │          │          │
              └──────────┴── shared calibration data ─┘
```

A pipeline is an ordered list of stages plus a config. Stages
communicate through a typed intermediate state:

- `HFHandle` — a loaded HuggingFace model + tokenizer (`hf_load` output)
- `DenseWeights` — in-memory or on-disk fp16 / bf16 tensor dict
- `CalibrationStats` — `xtx` (input-activation gramians) and/or `ggt`
  (output-gradient gramians) per role × layer
- `FactoredWeights` — stage-specific decomposition (e.g. CALDERA
  `{Q, L, R}` triples) keyed by canonical tensor name
- `GGUFPath` — emitted file on disk

## Stage contract

Each stage declares four things:

| Field | Meaning |
|---|---|
| `requires` | set of state types the stage consumes |
| `produces` | set of state types the stage emits (additive; prior state is preserved) |
| `calibration` | `none` / `xtx` / `ggt` / `both` |
| `params` | stage-specific knobs (rank, qtype, target_ratio, sample count, dtype) |

The runner composes a pipeline by checking `requires` / `produces`
match across adjacent stages and running **one** calibration pass that
feeds whichever stages need it. Re-running with different stage params
reuses cached calibration (the gramian cache in
[JOURNAL.md](JOURNAL.md) 2026-04-18 backlog lands here).

## Stage inventory (initial)

| Stage | Status | Source |
|---|---|---|
| `hf_load` | shipped (wraps upstream) | `convert_hf_to_gguf.py` + tokenizer loader |
| `calibrate` | shipped | `basis_sharing.py::collect_stats` + `StreamingCollector` |
| `quantize_standard` | shipped (upstream) | `llama-quantize` — Q4_K_M / Q3_K / Q2_K |
| `caldera` | in flight | `caldera.py` — single-matrix prototype; full-model integration queued (JOURNAL 2026-04-19) |
| `basis_sharing_factored` | deprecated | factored-SVD track shelved 2026-04-19; kept in inventory for reference only |
| `emit_gguf_factored` | shipped | `convert_factored_gguf.py` — reusable for CALDERA L/R companion tensors |
| `bench` | shipped | `bench_model.py` + `bench_compare.py` against `bench_prompts.json` |

Deprecated stages stay in the inventory so we can rerun old experiments
reproducibly. Default pipelines won't include them.

## Sample pipelines

- **baseline-q4**: `hf_load → emit_gguf → quantize_standard(Q4_K_M) → bench`
- **caldera-q4**: `hf_load → calibrate(xtx) → caldera(qtype=Q4_K, rank=64) → emit_gguf → bench`
- **caldera-q3-aggressive**: `hf_load → calibrate(xtx) → caldera(qtype=Q3_K, rank=128) → emit_gguf → bench`
- **research-factored**: `hf_load → calibrate(xtx,ggt) → basis_sharing_factored(target_ratio=1.5) → emit_gguf_factored → bench`
  (kept for reproducibility; not a shipping default)

Each run produces a GGUF + a markdown bench report that can be diffed
against the `baseline-q4` report on the same model.

## Acceptance criteria

A pipeline graduates from "research" to "default for model X" when,
compared to **mainline `llama-quantize` Q4_K_M on the same base model**
using `bench_compare.py`:

- greedy agreement ≥ **90%** (per JOURNAL 2026-04-18 — PPL alone is
  unreliable; argmax agreement + top-5 overlap are the real quality
  signals)
- WikiText-2 PPL within ±5% of Q4_K_M's
- decode tok/s within ±10% of Q4_K_M's
- GGUF size ≤ Q4_K_M's

Failing any one: research-only, not promoted.

## Target model list (initial)

Pipelines should be runnable against at least:

- **Qwen 3 8B** (Atlas prod: `Qwen3-8B-Q4_K_M.gguf`, see
  `docker-compose.yml` llama-chat)
- **Qwen 2.5 3B** (existing 3B quality reference, JOURNAL 2026-04-17)
- **Qwen 2.5 7B** (existing 7B reference, JOURNAL 2026-04-18)

Additions (Llama 3 8B, Gemma 2 9B, Mistral 7B) come once CALDERA proves
on the Qwen family.

## Open design questions

- **Intermediate state format.** On-disk per-layer safetensors
  (resumable, mirrors current `compute_factors` streaming checkpoints,
  tolerates 30B+) vs in-memory (fast but OOM-prone). Leaning on-disk
  with pinned-host cache; revisit if calibration dominates wall time
  once the gramian cache lands.
- **Config schema.** Probably one YAML per `(model × pipeline)` under
  `research/cross-layer-svd/configs/`, e.g.
  `qwen3-8b.caldera-q4.yaml`. Not yet committed.
- **CI / regression.** Cheap smoke (≤ 2 min, one model × baseline-q4) on
  every commit; nightly full sweep across all `(model × pipeline)`
  pairs with bench reports archived. Not yet wired.
- **Gated models.** Llama 3 and friends need HF tokens. Document once
  in the pipeline README and reference a `.env` pattern rather than
  per-config.
- **Serving handoff.** When a pipeline wins for a model, how does Atlas
  Nova.Api consume it? Current hypothesis: drop the GGUF in `./models/`,
  point `LlamaCpp:Models` at it (same as any other model). No API
  changes.

## Not in scope (yet)

- **Custom inference kernels.** Out of scope until a pipeline wins at
  end-to-end bench against mainline Q4_K_M. The lesson from JOURNAL
  2026-04-19: don't build a kernel for a decomposition that hasn't
  proved value first.
- **Streaming / PCIe-offloaded runtime.** Shelved in JOURNAL 2026-04-19.
  The `-ngl` partial-offload path and speculative decoding (upstream
  `--draft-model`) cover the bigger-model-on-small-card case without
  custom runtime work.
- **Atlas Nova.Api integration.** A winning pipeline drops in as a GGUF
  file; no `LlamaCppOptions` changes. If a pipeline *requires* loader
  support (companion tensors for CALDERA L/R, say), that integration
  gets its own section here once the first one lands.

## Next actions

1. Formalize the `Stage` interface (Python dataclass + protocol) in
   `research/cross-layer-svd/pipeline.py`.
2. Wrap existing scripts (`basis_sharing.collect_stats`, `caldera.py`,
   `convert_factored_gguf.py`, `bench_model.py`) as stages without
   changing their internals.
3. Implement the runner: validate stage graph, run calibration once,
   dispatch stages in order, emit final GGUF + bench report.
4. First working pipeline: `baseline-q4` on Qwen 2.5 3B — proves the
   runner end-to-end against an artifact we already know.
5. Second: `caldera-q4` on Qwen 2.5 3B — proves the CALDERA path against
   the same model, scored by the acceptance criteria above.
