"""Export a trained LittleBit checkpoint to deployment format.

Reads a training .pt file produced by littlebit_qat_model.py.
Produces an output directory with:

  config.json       — model/architecture metadata, LittleBit layer
                      list, quantization params
  model.safetensors — all tensors in deployment format:
                        LittleBit U_sign, V_sign: uint8 bit-packed
                        (1 bit per sign, 8 signs per byte)
                        h, g, ell:                fp16
                        Non-wrapped (embed, norms, lm_head): fp16

Round-trip validation mode verifies that loading the packed format
and reconstructing matches the training-forward output to within
FP rounding.

Usage:
  # export
  python littlebit_export.py \\
      --checkpoint littlebit_qat_checkpoint_r512_phaseB.pt \\
      --out phaseB_deployment/

  # verify round-trip
  python littlebit_export.py \\
      --checkpoint littlebit_qat_checkpoint_r512_phaseB.pt \\
      --out phaseB_deployment/ --verify
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save, load_file as st_load


# -- Sign bit-packing ----------------------------------------------

def pack_signs(t: torch.Tensor) -> torch.Tensor:
    """Pack a {-1,+1}-valued tensor into uint8 along the last axis.

    Encoding: bit_i = (tensor[..., i] >= 0), LSB-first per byte.
    Shape:    (..., n) -> (..., ceil(n/8)) uint8
    """
    bits = (t >= 0).to(torch.uint8)
    n = bits.shape[-1]
    pad = (8 - n % 8) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    bits = bits.reshape(*bits.shape[:-1], -1, 8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    return (bits * weights).sum(-1).to(torch.uint8)


def unpack_signs(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Inverse: unpack uint8 to a ±1 float tensor of length orig_last_dim."""
    n_words = packed.shape[-1]
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    # Broadcast shift: (..., n_words, 8)
    bits = ((packed.unsqueeze(-1) & weights) > 0).to(torch.float32)
    out = bits.reshape(*packed.shape[:-1], -1)[..., :orig_last_dim]
    return out * 2 - 1  # {0,1} -> {-1,+1}


# -- Checkpoint parsing --------------------------------------------

def is_littlebit_key(k: str) -> tuple[bool, str]:
    """Return (is_littlebit_param, suffix).

    LittleBitLinearHF has parameters named U_fp, V_fp, h, g, ell, bias.
    Signs (U_fp / V_fp) get packed; scales (h, g, ell) get fp16.
    """
    for suf in ("U_fp", "V_fp", "h", "g", "ell", "bias"):
        if k.endswith("." + suf):
            return True, suf
    return False, ""


def classify_state_dict(sd: dict) -> dict:
    """Group keys by LittleBit-layer-name or native status.

    Returns:
      {
        "littlebit_layers": {
            "model.layers.0.self_attn.q_proj": {
                "U_fp": tensor, "V_fp": tensor, ...
            },
            ...
        },
        "native": { "model.embed_tokens.weight": tensor, ... },
      }
    """
    littlebit = {}
    native = {}
    for k, v in sd.items():
        lb, suf = is_littlebit_key(k)
        if lb:
            prefix = k[: -(len(suf) + 1)]  # strip ".suf"
            littlebit.setdefault(prefix, {})[suf] = v
        else:
            native[k] = v
    return {"littlebit_layers": littlebit, "native": native}


# -- Export ---------------------------------------------------------

def build_packed_state(classified: dict) -> tuple[dict, dict, list]:
    """Returns (safetensors_state_dict, metadata, tied_pairs).

    safetensors_state_dict: flat name -> tensor (appropriate dtype)
    metadata: per-layer info (shapes, rank, dtype) for reconstruction
    tied_pairs: list of [canonical_name, alias_name] for tensors sharing
                storage (e.g. Qwen 0.5B's tied lm_head/embed_tokens).
                Alias is NOT written to safetensors; loader must
                reconstruct via config.
    """
    out_tensors = {}
    layer_meta = {}
    # Dedup by storage pointer: if two keys share the same storage,
    # save only the first and record the tie.
    seen_storage = {}  # data_ptr -> canonical_name
    tied_pairs = []

    for layer_name, params in classified["littlebit_layers"].items():
        U = params["U_fp"]  # shape (d_out, r)
        V = params["V_fp"]  # shape (d_in, r)
        h = params["h"]     # (d_out,)
        g = params["g"]     # (d_in,)
        ell = params["ell"] # (r,)
        bias = params.get("bias")  # Optional (d_out,)

        d_out, r = U.shape
        d_in = V.shape[0]
        assert V.shape[1] == r

        U_packed = pack_signs(U)  # (d_out, ceil(r/8))
        V_packed = pack_signs(V)  # (d_in,  ceil(r/8))

        out_tensors[f"{layer_name}.U_sign"] = U_packed
        out_tensors[f"{layer_name}.V_sign"] = V_packed
        out_tensors[f"{layer_name}.h"]   = h.to(torch.float16)
        out_tensors[f"{layer_name}.g"]   = g.to(torch.float16)
        out_tensors[f"{layer_name}.ell"] = ell.to(torch.float16)
        if bias is not None:
            out_tensors[f"{layer_name}.bias"] = bias.to(torch.float16)

        layer_meta[layer_name] = {
            "d_out": int(d_out),
            "d_in":  int(d_in),
            "r":     int(r),
            "has_bias": bias is not None,
        }

    # Native (non-LittleBit) tensors → fp16, dedup on storage sharing
    for k, v in classified["native"].items():
        if torch.is_tensor(v) and v.is_floating_point():
            # Check if this tensor shares storage with another already
            # saved — common for tied embeddings (embed_tokens + lm_head)
            ptr = v.storage().data_ptr() if v.numel() else None
            if ptr is not None and ptr in seen_storage:
                canonical = seen_storage[ptr]
                tied_pairs.append([canonical, k])
                continue
            fp16 = v.to(torch.float16)
            out_tensors[k] = fp16
            if ptr is not None:
                seen_storage[ptr] = k
        else:
            out_tensors[k] = v

    return out_tensors, layer_meta, tied_pairs


# -- Reconstruction (for verification) -----------------------------

def reconstruct_linear_fp16(layer_meta: dict, packed_tensors: dict,
                            layer_name: str) -> torch.Tensor:
    """Rebuild a fp16 weight matrix from packed form for verification."""
    meta = layer_meta[layer_name]
    U_packed = packed_tensors[f"{layer_name}.U_sign"]
    V_packed = packed_tensors[f"{layer_name}.V_sign"]
    h   = packed_tensors[f"{layer_name}.h"].to(torch.float32)
    g   = packed_tensors[f"{layer_name}.g"].to(torch.float32)
    ell = packed_tensors[f"{layer_name}.ell"].to(torch.float32)

    r = meta["r"]
    U_sign = unpack_signs(U_packed, r)  # (d_out, r) ±1
    V_sign = unpack_signs(V_packed, r)  # (d_in, r) ±1
    # W = diag(h) @ U_sign @ diag(ell) @ V_sign.T @ diag(g)
    W = (U_sign * ell[None, :]) @ V_sign.T          # (d_out, d_in)
    W = W * g[None, :]
    W = W * h[:, None]
    return W  # fp32 for verification


def verify_roundtrip(training_sd: dict, packed_tensors: dict,
                     layer_meta: dict, tol: float = 1e-2) -> bool:
    """Compare reconstructed LittleBit weights to training state_dict.

    The packed form is lossless for signs (which are 1-bit) and lossy
    for scales (fp32 -> fp16, ~1e-3 relative).  Max Frobenius rel-err
    should be well under `tol`.
    """
    bad = []
    for layer_name, meta in layer_meta.items():
        # Training-format reconstruction
        U_fp = training_sd[f"{layer_name}.U_fp"].to(torch.float32)
        V_fp = training_sd[f"{layer_name}.V_fp"].to(torch.float32)
        h = training_sd[f"{layer_name}.h"].to(torch.float32)
        g = training_sd[f"{layer_name}.g"].to(torch.float32)
        ell = training_sd[f"{layer_name}.ell"].to(torch.float32)
        U_sign = torch.sign(U_fp)
        V_sign = torch.sign(V_fp)
        W_training = (U_sign * ell[None, :]) @ V_sign.T * g[None, :] * h[:, None]

        # Packed-format reconstruction
        W_packed = reconstruct_linear_fp16(layer_meta, packed_tensors,
                                            layer_name)

        diff = torch.linalg.norm(W_training - W_packed)
        norm = torch.linalg.norm(W_training)
        rel_err = (diff / max(1e-30, norm)).item()
        if rel_err > tol:
            bad.append((layer_name, rel_err))

    if bad:
        print(f"  {len(bad)} layers exceeded rel-err tol={tol}:")
        for name, err in bad[:5]:
            print(f"    {name}: {err:.2e}")
        return False
    return True


# -- CLI --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Training .pt file")
    p.add_argument("--out", required=True,
                   help="Output directory")
    p.add_argument("--verify", action="store_true",
                   help="Round-trip compare packed vs training state")
    p.add_argument("--verify-tol", type=float, default=1e-2,
                   help="Max acceptable rel Frobenius err on reconstruction")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {ckpt_path}...", flush=True)
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Two possible formats: rolling ckpt (has 'model') vs end-of-training
    # ckpt (has 'state_dict').
    sd = ckpt.get("state_dict") or ckpt.get("model")
    if sd is None:
        raise RuntimeError("No 'state_dict' or 'model' key in checkpoint")
    print(f"  loaded in {time.time() - t0:.1f}s; "
          f"{len(sd)} tensors", flush=True)

    training_sd_size_bytes = sum(
        v.element_size() * v.numel() if torch.is_tensor(v) else 0
        for v in sd.values()
    )
    print(f"  training state_dict size: "
          f"{training_sd_size_bytes / 1024 / 1024:.1f} MB", flush=True)

    classified = classify_state_dict(sd)
    n_lb = len(classified["littlebit_layers"])
    n_native = len(classified["native"])
    print(f"  littlebit layers: {n_lb}", flush=True)
    print(f"  native tensors:   {n_native}", flush=True)

    packed_tensors, layer_meta, tied_pairs = build_packed_state(classified)
    if tied_pairs:
        print(f"  deduplicated {len(tied_pairs)} tied tensor(s):", flush=True)
        for canonical, alias in tied_pairs:
            print(f"    {alias} -> {canonical}", flush=True)

    # Verify round-trip before writing (catches bugs early)
    if args.verify:
        print("verifying round-trip correctness...", flush=True)
        t0 = time.time()
        # Bring training sd to fp32 for fair compare
        training_sd_fp32 = {
            k: (v.to(torch.float32) if torch.is_tensor(v) and
                v.is_floating_point() else v)
            for k, v in sd.items()
        }
        ok = verify_roundtrip(training_sd_fp32, packed_tensors,
                              layer_meta, tol=args.verify_tol)
        print(f"  {'OK' if ok else 'FAILED'} (tol={args.verify_tol:.0e}) "
              f"in {time.time() - t0:.1f}s",
              flush=True)
        if not ok:
            raise SystemExit(1)

    # Write safetensors + config
    config_path = out_dir / "config.json"
    st_path = out_dir / "model.safetensors"

    cfg = {
        "format": "littlebit-v1",
        "source_checkpoint": str(ckpt_path),
        "model_info": ckpt.get("config", {}),
        "littlebit_layers": layer_meta,
        "tied_pairs": tied_pairs,
        "n_littlebit_layers": n_lb,
        "n_native_tensors": n_native,
        "encoding": {
            "signs": "uint8 bit-packed, LSB-first, padded with 0 to "
                     "next byte boundary along last axis",
            "scales": "fp16",
            "native_tensors": "fp16",
        },
    }
    config_path.write_text(json.dumps(cfg, indent=2))

    st_save(packed_tensors, str(st_path))

    total_bytes = st_path.stat().st_size + config_path.stat().st_size
    print(f"wrote {out_dir}/", flush=True)
    print(f"  config.json:       {config_path.stat().st_size:,} B", flush=True)
    print(f"  model.safetensors: {st_path.stat().st_size:,} B "
          f"({st_path.stat().st_size / 1024 / 1024:.1f} MB)",
          flush=True)
    print(f"total deployment:    {total_bytes:,} B "
          f"({total_bytes / 1024 / 1024:.1f} MB)",
          flush=True)
    print(f"compression ratio:   "
          f"{training_sd_size_bytes / total_bytes:.2f}x",
          flush=True)


if __name__ == "__main__":
    main()
