"""Phase 2b: emit a factored GGUF from a base GGUF + our intermediate format.

Input:
  --base-gguf   path/to/model.gguf       (produced by convert_hf_to_gguf.py)
  --factored    path/to/factored_out/    (produced by basis_sharing.py --save-dir)
  --out         path/to/model-factored.gguf

What we do:
  1. Load KV metadata + vocab from the base GGUF (architecture, tokenizer, etc.)
  2. Add our factored-format KV keys so a factored-aware loader can recognize it
  3. Copy all non-factored tensors verbatim from the base GGUF
  4. For each factored weight in the base GGUF, SKIP the original and emit the
     factored tensors instead, with names per DESIGN.md §4:
        shared.{role_tag}.w{W:03d}.basis
        shared.{role_tag}.w{W:03d}.coeffs.{LAYER:03d}
        permatrix.{role_tag}.{LAYER:03d}.U
        permatrix.{role_tag}.{LAYER:03d}.V

The factored GGUF is backward-incompatible with unmodified llama.cpp (the
original weight tensors are gone), so a factored-aware loader is required.
Phase 2c adds that loader path to the C++ side.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter
from safetensors import safe_open


ROLE_TAG_TO_HF = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

# In a llama.cpp GGUF, a per-layer tensor is named blk.{L}.{role_tag}.weight.
# Strip the trailing ".weight" to recover the tag.
def _parse_blk_tensor(name):
    """Return (layer_idx, role_tag) or None if `name` isn't a per-layer weight."""
    if not name.startswith("blk.") or not name.endswith(".weight"):
        return None
    middle = name[len("blk."):-len(".weight")]  # e.g. "12.ffn_gate"
    parts = middle.split(".", 1)
    if len(parts) != 2:
        return None
    try:
        layer = int(parts[0])
    except ValueError:
        return None
    return layer, parts[1]


def _np_dtype_for(torch_dtype):
    return {
        torch.float16: np.float16,
        torch.bfloat16: np.float32,  # numpy has no bf16; store as fp32
        torch.float32: np.float32,
    }.get(torch_dtype, np.float32)


def _tensor_to_np(t, out_dtype=torch.float16):
    """Convert a torch tensor to a numpy array suitable for gguf writer."""
    t = t.to(out_dtype).contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t.numpy()


def build_factored_tensor_map(manifest, factored_tensors):
    """Map HF-style blk.{L}.{role_tag}.weight names to the list of (new_name, tensor)
    outputs that should replace them in the output GGUF."""
    replaced = {}  # {original_gguf_name: [(new_name, tensor), ...]}

    for role, info in manifest["shared_roles"].items():
        tag = info["tag"]
        for win in info["windows"]:
            basis_name = f"shared.{tag}.w{win['window_id']:03d}.basis"
            basis = factored_tensors[win["basis_key"]]
            # For each layer in this window, replace the original tensor.
            for i_local, layer_i in enumerate(win["layers"]):
                coeff_name = f"shared.{tag}.w{win['window_id']:03d}.coeffs.{layer_i:03d}"
                coeff = factored_tensors[win["coeffs_keys"][i_local]]
                orig = f"blk.{layer_i}.{tag}.weight"
                # First occurrence gets both the shared basis and its own coeffs,
                # second+ occurrence only gets coeffs (basis is shared)
                if orig not in replaced:
                    replaced[orig] = []
                if i_local == 0:
                    replaced[orig].append((basis_name, basis))
                replaced[orig].append((coeff_name, coeff))

    for role, info in manifest["permatrix_roles"].items():
        tag = info["tag"]
        for lf in info["layers"]:
            orig = f"blk.{lf['layer']}.{tag}.weight"
            u_name = f"permatrix.{tag}.{lf['layer']:03d}.U"
            v_name = f"permatrix.{tag}.{lf['layer']:03d}.V"
            replaced[orig] = [
                (u_name, factored_tensors[lf["U_key"]]),
                (v_name, factored_tensors[lf["V_key"]]),
            ]

    return replaced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-gguf", required=True,
                   help="standard GGUF from convert_hf_to_gguf.py (source of metadata + untouched tensors)")
    p.add_argument("--factored", required=True,
                   help="intermediate directory with manifest.json + factored.safetensors")
    p.add_argument("--out", required=True, help="output factored GGUF path")
    p.add_argument("--out-dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    out_dtype = getattr(torch, args.out_dtype)

    # Read manifest + factor tensors
    factored_dir = Path(args.factored)
    manifest = json.loads((factored_dir / "manifest.json").read_text())
    factored_tensors = {}
    with safe_open(str(factored_dir / "factored.safetensors"), framework="pt") as f:
        for k in f.keys():
            factored_tensors[k] = f.get_tensor(k)

    replaced = build_factored_tensor_map(manifest, factored_tensors)
    print(f"manifest references {len(replaced)} factored tensors "
          f"in {len(manifest['shared_roles'])} shared + "
          f"{len(manifest['permatrix_roles'])} permatrix roles")

    # Read base GGUF
    reader = GGUFReader(args.base_gguf)
    print(f"base GGUF: arch={reader.get_field('general.architecture').contents()}, "
          f"{len(reader.tensors)} tensors, {len(reader.fields)} KV fields")

    # Start writer. Preserve architecture so downstream loader is happy.
    arch = reader.get_field("general.architecture").contents()
    writer = GGUFWriter(args.out, arch)

    # Copy KV metadata (skip ones GGUFWriter synthesizes)
    skip_fields = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                   "general.architecture"}
    for field_name, field in reader.fields.items():
        if field_name in skip_fields:
            continue
        try:
            writer.add_key_value(field_name, field.contents(), field.types[0])
        except Exception as e:
            print(f"  [warn] skipping field {field_name}: {e}")

    # Add our factored-format metadata
    writer.add_key_value("factored.format_version", 1, gguf_type_int32())
    writer.add_key_value("factored.enabled", True, gguf_type_bool())
    writer.add_key_value("factored.window_size", manifest["window_size"], gguf_type_int32())
    writer.add_key_value("factored.target_ratio", float(manifest["target_ratio"]),
                         gguf_type_float32())
    writer.add_key_value("factored.refit", bool(manifest["refit"]), gguf_type_bool())

    # Emit tensors: copy base tensors, swap factored ones for their factor pairs
    n_copied = n_replaced = n_extra = 0
    for tensor_info in reader.tensors:
        orig_name = tensor_info.name
        if orig_name in replaced:
            for new_name, t in replaced[orig_name]:
                writer.add_tensor(new_name, _tensor_to_np(t, out_dtype))
                n_extra += 1
            n_replaced += 1
        else:
            # Copy verbatim (in whatever quant/dtype the base used)
            data = np.array(tensor_info.data)
            writer.add_tensor(orig_name, data, raw_dtype=tensor_info.tensor_type)
            n_copied += 1

    print(f"wrote {n_copied} copied tensors + {n_extra} factored tensors "
          f"(replaced {n_replaced} originals)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}")


# Helper shims for gguf's type enums (varies across gguf-py versions).
def gguf_type_int32():
    from gguf import GGUFValueType
    return GGUFValueType.INT32


def gguf_type_bool():
    from gguf import GGUFValueType
    return GGUFValueType.BOOL


def gguf_type_float32():
    from gguf import GGUFValueType
    return GGUFValueType.FLOAT32


if __name__ == "__main__":
    main()
