"""Apply QuaRot-style R1 Hadamard rotation to a Qwen3 HF checkpoint.

R1 is the "residual stream rotation" — a single orthogonal Q applied
to the hidden-state axis throughout the network. When Q is orthogonal
(Q @ Q.T = I), the network is mathematically equivalent to the
original at FP precision. The point is that the rotated weight
matrices have flatter per-channel magnitude distributions, which
makes low-bit quantization less lossy.

Rationale for each step
-----------------------
In a pre-norm transformer with RMSNorm(x) * gamma feeding a linear
layer, we cannot slip an orthogonal Q through the learned per-channel
gamma because diag(gamma) does not commute with Q. QuaRot's fix:
absorb gamma into the following linear layer, set gamma := 1, then
apply Q. The RMSNorm without scale is itself Q-invariant (since Q
preserves norms), so the transformation becomes a valid identity at
FP precision.

Qwen3 architecture points of application
-----------------------------------------
- embed_tokens.weight           [vocab, hidden]   -> W @ Q.T
- For each layer:
    input_layernorm.weight      (absorbed into q/k/v, then set to 1)
    self_attn.q_proj.weight     [h_q*head_dim, hidden]     -> W @ Q.T (after gamma fusion)
    self_attn.k_proj.weight     [h_kv*head_dim, hidden]    -> W @ Q.T
    self_attn.v_proj.weight     [h_kv*head_dim, hidden]    -> W @ Q.T
    self_attn.o_proj.weight     [hidden, h_q*head_dim]     -> Q @ W  (output side)
    post_attention_layernorm.weight  (absorbed into gate/up, set to 1)
    mlp.gate_proj.weight        [intermediate, hidden]     -> W @ Q.T
    mlp.up_proj.weight          [intermediate, hidden]     -> W @ Q.T
    mlp.down_proj.weight        [hidden, intermediate]     -> Q @ W
- model.norm.weight             (absorbed into lm_head, set to 1)
- lm_head.weight                [vocab, hidden]   -> W @ Q.T

Note: Qwen3 additionally has q_norm and k_norm (per-head RMSNorm on Q/K
before RoPE). These operate on the head_dim axis, not the hidden axis,
so R1 passes through them unchanged. They are left untouched.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def hadamard_matrix(n: int, dtype=torch.float64) -> torch.Tensor:
    """Sylvester's construction. n must be a power of 2."""
    assert n > 0 and (n & (n - 1)) == 0, f"{n} must be a power of 2"
    H = torch.ones((1, 1), dtype=dtype)
    size = 1
    while size < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
        size *= 2
    return H / np.sqrt(n)  # orthogonal: Q @ Q.T = I


def fuse_norm_into_next(
    norm_weight: torch.Tensor, linear_weights: list[torch.Tensor]
) -> None:
    """In-place: multiply each linear weight's input dim by norm_weight,
    then set norm_weight to ones.

    Linear weights are [out_features, in_features]. Multiplying by
    gamma [in_features] broadcasts across rows.
    """
    gamma = norm_weight.data.to(torch.float64)
    for w in linear_weights:
        w.data = (w.data.to(torch.float64) * gamma).to(w.data.dtype)
    norm_weight.data.fill_(1.0)


def apply_input_rotation(weight: torch.Tensor, Q_T: torch.Tensor) -> None:
    """Rotate a linear layer's input dim: W_new = W @ Q.T. In-place."""
    # Compute in float64 for precision, write back in original dtype
    orig_dtype = weight.data.dtype
    weight.data = (weight.data.to(torch.float64) @ Q_T).to(orig_dtype)


def apply_output_rotation(weight: torch.Tensor, Q: torch.Tensor) -> None:
    """Rotate a linear layer's output dim: W_new = Q @ W. In-place."""
    orig_dtype = weight.data.dtype
    weight.data = (Q @ weight.data.to(torch.float64)).to(orig_dtype)


def rotate_qwen3(model, Q: torch.Tensor) -> None:
    """Apply R1 rotation to a Qwen3 model in place. Q is [hidden, hidden], orthogonal."""
    Q_T = Q.T.contiguous()

    # 1. Token embedding: residual stream enters here.
    apply_input_rotation(model.model.embed_tokens.weight, Q_T)

    # 2. Per-layer
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        t0 = time.time()
        # Fuse input_layernorm gamma into q/k/v
        fuse_norm_into_next(
            layer.input_layernorm.weight,
            [
                layer.self_attn.q_proj.weight,
                layer.self_attn.k_proj.weight,
                layer.self_attn.v_proj.weight,
            ],
        )
        # Rotate attention input side (q/k/v inputs are the residual stream)
        apply_input_rotation(layer.self_attn.q_proj.weight, Q_T)
        apply_input_rotation(layer.self_attn.k_proj.weight, Q_T)
        apply_input_rotation(layer.self_attn.v_proj.weight, Q_T)
        # Rotate attention output side (o_proj output goes back into residual stream)
        apply_output_rotation(layer.self_attn.o_proj.weight, Q)

        # Fuse post_attention_layernorm gamma into gate/up
        fuse_norm_into_next(
            layer.post_attention_layernorm.weight,
            [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight],
        )
        apply_input_rotation(layer.mlp.gate_proj.weight, Q_T)
        apply_input_rotation(layer.mlp.up_proj.weight, Q_T)
        apply_output_rotation(layer.mlp.down_proj.weight, Q)

        print(f"  layer {i+1}/{n_layers} rotated ({time.time() - t0:.1f}s)", flush=True)

    # 3. Final norm → lm_head
    fuse_norm_into_next(model.model.norm.weight, [model.lm_head.weight])
    apply_input_rotation(model.lm_head.weight, Q_T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="HF model directory")
    ap.add_argument("--out", dest="dst", required=True, help="Output HF directory")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[load] {src} -> dtype={args.dtype} on CPU")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype=dtype,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(src))
    print(f"[load] done in {time.time() - t0:.1f}s")

    hidden = model.config.hidden_size
    print(f"[hadamard] building Q for hidden={hidden}")
    Q = hadamard_matrix(hidden, dtype=torch.float64)

    # Sanity: Q @ Q.T == I
    err = (Q @ Q.T - torch.eye(hidden, dtype=torch.float64)).abs().max().item()
    assert err < 1e-10, f"Hadamard orthogonality error: {err}"
    print(f"[hadamard] orthogonality error: {err:.2e}")

    print(f"[rotate] applying R1 rotation to {model.config.num_hidden_layers} layers")
    t0 = time.time()
    rotate_qwen3(model, Q)
    print(f"[rotate] done in {time.time() - t0:.1f}s")

    print(f"[save] writing rotated checkpoint to {dst}")
    t0 = time.time()
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    print(f"[save] done in {time.time() - t0:.1f}s")

    # Drop a marker for provenance
    (dst / "ROTATION_APPLIED.json").write_text(
        json.dumps(
            {
                "scheme": "QuaRot R1 (residual stream Hadamard)",
                "hidden_size": hidden,
                "hadamard_size": hidden,
                "norm_fusion": True,
                "source": str(src.resolve()),
                "dtype": args.dtype,
            },
            indent=2,
        )
    )
    print("[done]")


if __name__ == "__main__":
    main()
