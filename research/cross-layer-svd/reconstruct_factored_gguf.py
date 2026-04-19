"""Phase 2c MVP: reconstruct a factored GGUF back to a standard dense GGUF.

Lets us load the output of convert_factored_gguf.py with vanilla llama.cpp
without any C++ changes. Useful for end-to-end validation that our factored
pipeline produces a semantically correct model artifact.

Output GGUF has standard tensor names only (blk.{L}.{role}.weight), so
llama-server / llama-cli loads it as a normal model.

Usage:
    python reconstruct_factored_gguf.py --factored model-factored.gguf \
                                        --out model-dense.gguf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, GGUFValueType


ROLE_TAGS = ["attn_q", "attn_k", "attn_v", "attn_output",
             "ffn_gate", "ffn_up", "ffn_down"]

SHARED_PAT = re.compile(
    r"^shared\.(?P<role>[a-z_]+)\.w(?P<win>\d+)\.(?P<kind>basis|coeffs)(?:\.(?P<layer>\d+))?$"
)
PERMATRIX_PAT = re.compile(
    r"^permatrix\.(?P<role>[a-z_]+)\.(?P<layer>\d+)\.(?P<kind>U|V)$"
)


def _gguf_tensor_to_torch(t, dtype=torch.float32):
    """Load a GGUFReader tensor's data into a torch tensor.

    GGUF reports tensor shape in REVERSE order vs numpy (fastest-varying axis
    first). Bytes on disk are laid out according to the as-written numpy shape.
    So if we wrote a (d_out, rank) numpy array, GGUFReader sees shape (rank, d_out)
    but the flat data is still (d_out, rank) row-major. We reshape to the
    as-written shape by reversing the reader's shape tuple.
    """
    arr = np.array(t.data)
    tt = torch.from_numpy(arr).to(dtype)
    shape_reversed = [int(d) for d in reversed(t.shape)]
    tt = tt.reshape(shape_reversed)
    return tt


def _torch_to_np(t, out_dtype=torch.float16):
    """Return (numpy_array, raw_dtype) for a reconstructed dense weight.

    For bf16 we reinterpret the raw 2-byte payload as int16 and tag the write
    with GGMLQuantizationType.BF16 so the writer stores true bf16 (numpy has
    no bf16 dtype). For fp16/fp32 the writer infers from the numpy dtype, so
    raw_dtype is None.
    """
    t = t.to(out_dtype).contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy(), GGMLQuantizationType.BF16
    return t.numpy(), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factored", required=True, help="factored GGUF from convert_factored_gguf.py")
    p.add_argument("--out", required=True, help="output dense GGUF")
    p.add_argument("--out-dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"],
                   help="dtype for reconstructed dense weights (default: float16)")
    args = p.parse_args()

    out_dtype = getattr(torch, args.out_dtype)

    reader = GGUFReader(args.factored)

    factored_enabled = False
    try:
        factored_enabled = bool(reader.get_field("factored.enabled").contents())
    except Exception:
        pass
    if not factored_enabled:
        raise SystemExit("input does not have factored.enabled=True; is this a factored GGUF?")

    arch = reader.get_field("general.architecture").contents()
    print(f"reading factored GGUF: arch={arch}, {len(reader.tensors)} tensors")

    # Index factored tensors
    #   shared_bases[role][window] = GGUFTensorInfo
    #   shared_coeffs[role][window][layer] = GGUFTensorInfo
    #   permatrix[role][layer] = (U_info, V_info)
    shared_bases = {}
    shared_coeffs = {}
    permatrix = {}
    passthrough = []  # (name, GGUFTensorInfo) — non-factored tensors to copy as-is

    for t in reader.tensors:
        m = SHARED_PAT.match(t.name)
        if m:
            role = m.group("role")
            win = int(m.group("win"))
            if m.group("kind") == "basis":
                shared_bases.setdefault(role, {})[win] = t
            else:
                layer = int(m.group("layer"))
                shared_coeffs.setdefault(role, {}).setdefault(win, {})[layer] = t
            continue
        m = PERMATRIX_PAT.match(t.name)
        if m:
            role = m.group("role")
            layer = int(m.group("layer"))
            kind = m.group("kind")
            slot = permatrix.setdefault(role, {}).setdefault(layer, {})
            slot[kind] = t
            continue
        passthrough.append((t.name, t))

    n_shared_windows = sum(len(v) for v in shared_coeffs.values())
    n_permatrix_layers = sum(len(v) for v in permatrix.values())
    print(f"  shared: {len(shared_bases)} roles, {n_shared_windows} window×layer coeff tensors")
    print(f"  permatrix: {len(permatrix)} roles, {n_permatrix_layers} layer pairs")
    print(f"  passthrough: {len(passthrough)} non-factored tensors")

    # Open writer
    writer = GGUFWriter(args.out, arch)

    # Copy KV metadata (skip ones writer synthesizes + our factored markers)
    skip_fields = {
        "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
        "general.architecture",
        "factored.format_version", "factored.enabled",
        "factored.window_size", "factored.target_ratio", "factored.refit",
    }
    for field_name, field in reader.fields.items():
        if field_name in skip_fields:
            continue
        try:
            writer.add_key_value(field_name, field.contents(), field.types[0])
        except Exception as e:
            print(f"  [warn] skipping field {field_name}: {e}")

    # Reconstruct + emit factored tensors as standard blk names
    n_reconstructed = 0
    # shared: for each role, each window, each layer → W = basis @ coeff
    for role, wins in shared_coeffs.items():
        basis_by_win = shared_bases.get(role, {})
        for win_id, layer_coeffs in wins.items():
            basis_info = basis_by_win.get(win_id)
            if basis_info is None:
                raise RuntimeError(f"missing basis for shared.{role}.w{win_id:03d}")
            B = _gguf_tensor_to_torch(basis_info)
            for layer_i, coeff_info in layer_coeffs.items():
                A = _gguf_tensor_to_torch(coeff_info)
                W = B @ A  # [d_out × r] @ [r × d_in] = [d_out × d_in]
                out_name = f"blk.{layer_i}.{role}.weight"
                arr, raw = _torch_to_np(W, out_dtype)
                writer.add_tensor(out_name, arr, raw_dtype=raw)
                n_reconstructed += 1
    # permatrix: W = U @ V per layer
    for role, layers in permatrix.items():
        for layer_i, pair in layers.items():
            if "U" not in pair or "V" not in pair:
                raise RuntimeError(f"missing U or V for permatrix.{role}.{layer_i:03d}")
            U = _gguf_tensor_to_torch(pair["U"])
            V = _gguf_tensor_to_torch(pair["V"])
            W = U @ V
            out_name = f"blk.{layer_i}.{role}.weight"
            arr, raw = _torch_to_np(W, out_dtype)
            writer.add_tensor(out_name, arr, raw_dtype=raw)
            n_reconstructed += 1

    # Copy passthrough tensors verbatim
    for name, t in passthrough:
        writer.add_tensor(name, np.array(t.data), raw_dtype=t.tensor_type)

    print(f"  reconstructed {n_reconstructed} weights, copied {len(passthrough)} passthrough")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
