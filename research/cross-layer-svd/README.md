# LittleBit QAT research — entry point

**Fresh sessions start here.**  This directory holds the full research
line for reproducing and extending Samsung's LittleBit paper
(arXiv [2506.13771](https://arxiv.org/abs/2506.13771)) in our Windows-
native research fork.

## Read these first

1. **[JOURNAL.md](JOURNAL.md)** — chronological log of every run,
   finding, and decision.  Latest entry is the freshest state.
2. **[consolidated_implementation_roadmap.md](consolidated_implementation_roadmap.md)** —
   master plan.  Sprint sequence, priorities, decisions locked in.

After those two, branch to whichever concern you're picking up.

## Plan docs by axis

| Axis | Doc |
|---|---|
| Paper walkthrough + sanity experiments | [littlebit_math.md](littlebit_math.md) |
| Per-technique enhancement catalog | [littlebit_enhancements.md](littlebit_enhancements.md) |
| Memory-first ablations (10 runs) | [savings_exploration_plan.md](savings_exploration_plan.md) |
| Quality-combination opportunities (19 items) | [unexplored_efficiency_gains.md](unexplored_efficiency_gains.md) |
| Scale to 30B+ local (NVMe + teacher cache) | [scale_to_30b_architecture.md](scale_to_30b_architecture.md) |
| Wall-time reduction (12 techniques) | [wall_time_reduction_plan.md](wall_time_reduction_plan.md) |
| Online draft adaptation during deployment | [online_draft_adaptation.md](online_draft_adaptation.md) |
| External-techniques survey (Liger, COAT, etc.) | [memory_efficient_training_research.md](memory_efficient_training_research.md) |
| Inference runtime — 4 layers from export to kernel | [inference_runtime.md](inference_runtime.md) |

## Code + artifacts

- **Training script**: [`littlebit_qat_model.py`](littlebit_qat_model.py)
- **Sanity checks**: [`littlebit_sanity.py`](littlebit_sanity.py)
- **Single-matrix QAT experiments**:
  [`littlebit_qat_single.py`](littlebit_qat_single.py),
  [`littlebit_qat_activation.py`](littlebit_qat_activation.py)
- **Deployment export** (Layer 1): [`littlebit_export.py`](littlebit_export.py)
- **Evaluation harness**: [`littlebit_eval.py`](littlebit_eval.py)
- **Archived SVD + CALDERA research**: see [DESIGN.md](DESIGN.md)
  and [JOURNAL.md](JOURNAL.md) entries from 2026-04-10 through
  2026-04-19

## Current status (as of last JOURNAL entry)

- Phase B training complete on Qwen 2.5 0.5B: **PPL 76.8 full-test,
  88.5% hidden-state capture, sentence-level loop generation**
- Sprint 0 shipped (TF32, bf16 saves, forward hooks, compile
  fallback, plateau early-stop)
- Layer 1 deployment export shipped (292 MB, 7× compression)
- **Next planned**: Sprint 2 — teacher cache infrastructure

## Decisions locked in

- **Goal**: standalone inference (not just speculation draft)
- **No cloud**: fully local
- **Windows-native CUDA**: raw `torch.utils.cpp_extension` for
  kernel work (no Triton, no WSL2, no Warp)
- **Scale ladder**: 0.5B (done) → 1.5B → 3B → 7B
- **Method validated at 0.5B**; quality gap expected to close at 7B+

## Paper and prior-art references

See [`memory_efficient_training_research.md §Sources`](memory_efficient_training_research.md)
for full citation list.  Quick links:

- LittleBit paper: [arXiv 2506.13771](https://arxiv.org/abs/2506.13771)
- Samsung's code repo: [`SamsungLabs/LittleBit`](https://github.com/SamsungLabs/LittleBit)
- Liger Kernel: [GitHub](https://github.com/linkedin/Liger-Kernel) /
  [arXiv 2410.10989](https://arxiv.org/pdf/2410.10989)
- COAT FP8 training: [arXiv 2410.19313](https://arxiv.org/html/2410.19313v1)
- GaLore: [paper 2403.03507](https://huggingface.co/papers/2403.03507)
- DeepSpeed ZeRO-Infinity: [blog + repo](https://www.deepspeed.ai/tutorials/zero/)
