# LittleBit inference runtime

High-level view of what's needed to turn a trained LittleBit
checkpoint into a fast standalone-deployable model.  Linked to
detail docs where the implementation specifics live.

Status: **planning**. No layer shipped yet.  Phase B checkpoint
([JOURNAL.md](JOURNAL.md) 2026-04-21 entry) is the first real
artifact available to consume once Layer 1 is built.

## 1. Target

Turn a trained student (currently a 1.5 GB fp32 `.pt` file) into a
**~300 MB deployable artifact** that runs as a standalone language
model with inference speed comparable to, or better than, the
fp16 base.

Paper claims 11.6× inference speedup at 0.1 BPW on Llama2-70B MLP
via their custom kernel.  Matching that on our hardware is the goal.

## 2. Runtime stack (four layers)

| Layer | Purpose | Effort | Status |
|---|---|---|---|
| **1. Deployment format export** | Convert training `.pt` to bit-packed deployment binary | ~3 days | **Open** |
| **2. Python inference wrapper** | Load packed format, run forward, validate correctness | ~1 week | Open |
| **3. Custom CUDA kernel (native)** | Fast fused binary GEMM via `torch.utils.cpp_extension` | 2-4 weeks | Open |
| **4. llama.cpp GGUF integration** | Upstream-deployable tensor type + CUDA kernel | 2-3 months | Open |

### Layer 1 — deployment format

Converts fp32 training checkpoint to packed deployment file:
- `sign(U_fp)` → bit-packed uint32 (32 entries per word)
- `sign(V_fp)` → bit-packed uint32
- `h`, `g`, `ell` → fp16
- Embeddings + lm_head + norms → fp16 (or nf4 optional)
- Header with model config + layer shapes + rank

Expected output size for Qwen 2.5 0.5B r=512: **~305 MB** (see
[JOURNAL.md](JOURNAL.md) 2026-04-21 "final size" analysis for
breakdown).

No kernel work required.  Pure Python + NumPy bit manipulation.

### Layer 2 — Python inference

Load packed format, reconstruct student at runtime, forward:

1. Unpack bit-packed signs to ±1 fp16 tensors (one-time on load, or
   on-demand per layer)
2. Apply scales via Prop. 1 efficient form (Eq. 5):
   `y = ((((x ⊙ g) · V_sign) ⊙ ℓ) · U_signᵀ) ⊙ h`
3. Use standard PyTorch for everything else (attention, norms, etc.)

**No speedup** at this stage — in fact slightly slower than fp16
baseline due to the unpack overhead.  Purpose: **validate
correctness**.  Deployed model's logits should match training-
forward logits to within FP rounding.

### Layer 3 — Custom CUDA kernel (Windows-native)

**Decided toolchain: `torch.utils.cpp_extension` with raw CUDA C++.**
Not Triton (Windows compatibility).  Not Warp (less mature).
Rationale in [memory_efficient_training_research.md](memory_efficient_training_research.md)
§"For our specific use cases" and confirmed with user 2026-04-21.

Kernel responsibilities:
- Fused binary GEMM using XNOR-popcount primitives for signed matmul
- Eq. 5 form fused: `y = ((((x ⊙ g) · V_sign) ⊙ ℓ) · U_signᵀ) ⊙ h`
  executed as a single kernel launch
- SmoothSignEfficient-compatible backward for any training-time uses
- Integration with PyTorch's dispatcher (so `model(x)` calls our kernel
  transparently after load)

Prerequisites on the dev machine:
- CUDA Toolkit (nvcc)
- Visual Studio Build Tools (MSVC for Windows)
- Python dev headers (bundled with venv)

Expected per-step speedup: paper's 11.6× is for 70B MLP layers; for
0.5B-7B matrices expect **3-6× over fp16 matmul**.  Concrete numbers
gate on kernel tuning.

### Layer 4 — llama.cpp upstream integration

Once Layer 3 validates and produces a fast Windows-native runtime,
port the kernel concept to `llama.cpp`:
- Define new GGUF quantization type (e.g., `LBIT_Q512`)
- Add CPU dequantize path to ggml (fallback)
- Add CUDA matmul kernel (reuse Layer 3 work)
- Wire into `llama-server`
- Submit upstream PR

Enables Atlas deployment per the upstream-only invariant (see
[speculative_decoding.md](speculative_decoding.md)).

## 3. Linked detail docs

Implementation specifics for each concern live in dedicated docs:

| Concern | Doc |
|---|---|
| Teacher-forward elimination during training | [scale_to_30b_architecture.md §2](scale_to_30b_architecture.md) |
| Storage / VRAM budgets at all scales | [scale_to_30b_architecture.md §11](scale_to_30b_architecture.md) (NVMe tier) |
| Memory optimizations applicable to inference | [memory_efficient_training_research.md](memory_efficient_training_research.md) |
| Paper's Prop. 1 math | [littlebit_math.md §4](littlebit_math.md) |
| Paper's kernel claims and their caveats | [littlebit_math.md §4.2](littlebit_math.md) and §12 analysis |
| Why Windows-native CUDA over Triton | [memory_efficient_training_research.md](memory_efficient_training_research.md) "toolchain" |

## 4. Ordering in the full plan

From [consolidated_implementation_roadmap.md](consolidated_implementation_roadmap.md):

| Sprint | Content |
|---|---|
| Sprint 0 (shipped) | Wall-time + disk optimizations |
| Sprint 2 (next) | Teacher cache |
| Sprint 5 | Scale ladder up to 7B |
| **Sprint 5.5** | **Layer 1 export** (this doc) — can happen any time, independent |
| **Sprint 5.6** | **Layer 2 Python inference** |
| **Sprint 7** | **Layer 3 CUDA kernel** (after 7B result validates the method) |
| **Sprint 8** | **Layer 4 llama.cpp integration** (after Layer 3 establishes performance baseline) |

Layer 1 + 2 can happen now against Phase B's checkpoint — no new
training required.  Useful for validating the deployment path early.

## 5. Windows-native toolchain commitments

Per user decision 2026-04-21:
- **CUDA kernels**: raw C++ via `torch.utils.cpp_extension.load()`
- **No Triton, no WSL2, no Warp**
- Knowledge transfers directly to Layer 4 (`llama.cpp` CUDA integration)
- Requires CUDA Toolkit + MSVC installed on dev machine

Tradeoffs accepted: more verbose than Triton (2-3× lines per kernel),
no pre-built optimized libraries (Liger/Unsloth), slower initial
velocity.  Gain: unified toolchain for research fork and eventual
upstream contribution.

## 6. What's not yet decided

- **Rank packing strategy**: bit-packed per-row, per-column, or
  interleaved?  Affects kernel access pattern.  Decide during
  Layer 1 design.
- **Embedding quantization**: keep fp16 (272 MB at 0.5B) or
  quantize to nf4 via bitsandbytes (68 MB)?  Quality-side decision.
- **Layer 3 backward path**: do we need training-time CUDA kernel
  or just inference?  If training-time, Sprint 7 doubles in scope.
  Probably not needed if teacher cache reuses existing PyTorch
  backward path.
- **Deployment audience**: just Atlas (upstream `llama.cpp` path)
  or also other runtimes (HF transformers, vLLM, etc.)?  If
  multi-runtime, add conversion scripts per target.

## 7. Bottom line

Four layers, three decisions already made:
- Windows-native CUDA via cpp_extension (Layer 3 toolchain)
- Layer 1+2 can start now against Phase B checkpoint
- Layer 3 gated on validated 7B result (don't optimize a broken method)

Layer 1 export is the **most actionable immediate deployment work**
since it uses the Phase B artifact we already have.
