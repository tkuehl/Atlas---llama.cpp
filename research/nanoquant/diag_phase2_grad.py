"""Focused gradient-flow diagnostic for Phase 2's block 0.

Reproduces the real Phase 2 setup end-to-end for ONE block, then probes
the autograd graph to figure out why U_latent.grad ends up at 0 even
though the loss is nonzero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from activations import Cache
from admm import lb_admm_init
from preconditioner import load_preconditioners
from quant import BinaryFactoredLinear, quant_params, replace_linears_with_quant


sys.stdout.reconfigure(line_buffering=True)

MODEL = "Qwen/Qwen3-4B"
CACHE = Path(__file__).parent / "cache" / "qwen-qwen3-4b" / "n4_L2048_seed0"

print("[diag] loading model (bf16)", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
model.eval()

print("[diag] loading cache + preconditioners", flush=True)
cache = Cache.load(CACHE)
precond = load_preconditioners(CACHE)
aux_kwargs = cache.load_aux_kwargs("cuda")
X = cache.load_boundary(0)
Z = cache.load_boundary(1)

print(f"[diag] X shape {tuple(X.shape)} dtype {X.dtype}; Z shape {tuple(Z.shape)}", flush=True)

block = model.model.layers[0]

# ---- PRE-replace state check -------------------------------------------
print("[diag] PRE-replace: grad_enabled=", torch.is_grad_enabled(), flush=True)
print(
    "[diag] PRE-replace: any param requires_grad?",
    any(p.requires_grad for p in block.parameters()),
    flush=True,
)

# ---- Replace linears with LB-ADMM init --------------------------------
def init_fn(rel_path, lin):
    full_path = f"model.layers.0.{rel_path}"
    D_in, D_out = precond[full_path]
    return lb_admm_init(
        lin.weight.data,
        D_in=D_in.to("cuda"),
        D_out=D_out.to("cuda"),
        r=2,
        K=50,
    )


replaced = replace_linears_with_quant(block, r=2, init_fn=init_fn)
block.to("cuda")
print(f"[diag] replaced: {replaced}", flush=True)

# Inspect first BinaryFactoredLinear
bf = next(m for m in block.modules() if isinstance(m, BinaryFactoredLinear))
print(
    f"[diag] bf.U_latent dtype={bf.U_latent.dtype} shape={tuple(bf.U_latent.shape)} "
    f"|U|mean={bf.U_latent.abs().mean().item():.4g} "
    f"|U|max={bf.U_latent.abs().max().item():.4g}",
    flush=True,
)

# ---- Freeze all + enable quant params ---------------------------------
for p in block.parameters():
    p.requires_grad_(False)
trainable = []
for p in quant_params(block):
    p.requires_grad_(True)
    trainable.append(p)
print(f"[diag] {len(trainable)} trainables; first.requires_grad={trainable[0].requires_grad}", flush=True)

# ---- Build target dtype and forward -----------------------------------
target_dtype = next(p for p in block.parameters()).dtype
print(f"[diag] target_dtype (first block param) = {target_dtype}", flush=True)

x = X[0:1].to(device="cuda", dtype=target_dtype)
z = Z[0:1].to(device="cuda", dtype=target_dtype)
print(f"[diag] x dtype={x.dtype} shape={tuple(x.shape)} requires_grad={x.requires_grad}", flush=True)

block.train()

# Try WITH and WITHOUT inference_mode / grad context
print("\n[diag] --- single forward + backward ---", flush=True)
print("[diag] torch.is_grad_enabled():", torch.is_grad_enabled(), flush=True)

with torch.enable_grad():
    y = block(x, **aux_kwargs)
    if isinstance(y, tuple):
        y = y[0]
    print(
        f"[diag] y dtype={y.dtype} shape={tuple(y.shape)} "
        f"requires_grad={y.requires_grad} grad_fn={y.grad_fn}",
        flush=True,
    )
    loss = F.mse_loss(y.float(), z.float())
    print(f"[diag] loss={loss.item():.6f} requires_grad={loss.requires_grad} grad_fn={loss.grad_fn}", flush=True)

    loss.backward()

# Report grad state
for i, p in enumerate(trainable[:8]):
    g = p.grad
    if g is None:
        print(f"  param[{i}] shape={tuple(p.shape)} grad=None requires_grad={p.requires_grad}", flush=True)
    else:
        print(
            f"  param[{i}] shape={tuple(p.shape)} grad.norm={g.norm().item():.4g} "
            f"grad.max={g.abs().max().item():.4g} grad.dtype={g.dtype}",
            flush=True,
        )

# ---- Trace backward with retain_grad at each BinaryFactoredLinear ----
print("\n[diag] --- direct autograd.grad on just loss -> U_latent ---", flush=True)

# Reset grads and retry with autograd.grad directly
for p in trainable:
    p.grad = None

# Instrument BinaryFactoredLinear forward: retain grad on each intermediate
bf_of_interest = next(m for m in block.modules() if isinstance(m, BinaryFactoredLinear))

# Monkey-patch a retain_grad path
orig_forward = bf_of_interest.forward

retained = {}

def wrapped_forward(x):
    cdtype = x.dtype
    U_cast = bf_of_interest.U_latent.to(cdtype)
    U_cast.retain_grad()
    retained["U_cast"] = U_cast
    V_cast = bf_of_interest.V_latent.to(cdtype)
    V_cast.retain_grad()
    retained["V_cast"] = V_cast
    from quant import sign_ste
    U_sign = sign_ste(U_cast)
    U_sign.retain_grad()
    retained["U_sign"] = U_sign
    V_sign = sign_ste(V_cast)
    z = x * bf_of_interest.s2.to(cdtype)
    z = z @ V_sign
    z.retain_grad()
    retained["z_after_Vsign"] = z
    z = z @ U_sign.T
    z.retain_grad()
    retained["z_after_Usign"] = z
    z = z * bf_of_interest.s1.to(cdtype)
    if bf_of_interest.bias is not None:
        z = z + bf_of_interest.bias.to(cdtype)
    return z

bf_of_interest.forward = wrapped_forward

y2 = block(x, **aux_kwargs)
if isinstance(y2, tuple):
    y2 = y2[0]
loss2 = F.mse_loss(y2.float(), z.float())
print(f"[diag] loss2={loss2.item():.6f}", flush=True)

loss2.backward()

for name, t in retained.items():
    if t.grad is None:
        print(f"  [intermediate] {name}: grad=None", flush=True)
    else:
        print(
            f"  [intermediate] {name}: grad.norm={t.grad.norm().item():.4g} "
            f"grad.max={t.grad.abs().max().item():.4g}",
            flush=True,
        )
print(
    f"  [param] U_latent.grad norm={bf_of_interest.U_latent.grad.norm().item() if bf_of_interest.U_latent.grad is not None else 'None'}",
    flush=True,
)

# Restore
bf_of_interest.forward = orig_forward


# ---- Test: force large-magnitude params and retry -----------------
print("\n[diag] --- forcing |U_latent| to ~1.0 to test if tiny magnitudes are the cause ---", flush=True)

# Rescale U_latent, V_latent to |·| ≈ 1, compensate via s1, s2.
# Preserves sign pattern and W_eff = s1 · sign(U) · sign(V)^T · s2.
# After rescale, clipped-STE gradient will still flow (|U|<1 per row).
for m in block.modules():
    if isinstance(m, BinaryFactoredLinear):
        u_rowmean = m.U_latent.abs().mean(dim=1, keepdim=True).clamp(min=1e-30)
        v_rowmean = m.V_latent.abs().mean(dim=1, keepdim=True).clamp(min=1e-30)
        with torch.no_grad():
            m.U_latent.div_(u_rowmean)
            m.V_latent.div_(v_rowmean)
            # s1, s2 absorb the scale so the effective weight is unchanged
            m.s1.mul_(u_rowmean.squeeze(1))
            m.s2.mul_(v_rowmean.squeeze(1))

bf2 = next(m for m in block.modules() if isinstance(m, BinaryFactoredLinear))
print(
    f"[diag] after rescale: |U|mean={bf2.U_latent.abs().mean().item():.4g} "
    f"|s1|mean={bf2.s1.abs().mean().item():.4g}",
    flush=True,
)

# Diagnostic on the preconditioner + original W for q_proj
D_in_qp, D_out_qp = precond["model.layers.0.self_attn.q_proj"]
W_qp = model.model.layers[0].self_attn.q_proj  # wait — already replaced
# Need the original W_fp. Can we get it? Not easily — it's gone. Let's just
# look at the preconditioner stats.
print(
    f"[diag] q_proj D_in: min={D_in_qp.min().item():.4g} mean={D_in_qp.mean().item():.4g} max={D_in_qp.max().item():.4g}",
    flush=True,
)
print(
    f"[diag] q_proj D_out: min={D_out_qp.min().item():.4g} mean={D_out_qp.mean().item():.4g} max={D_out_qp.max().item():.4g}",
    flush=True,
)

# Show the effective W_eff = s1 · sign(U) · sign(V)^T · s2 magnitude
W_eff = (
    (bf2.s1.unsqueeze(1) * torch.sign(bf2.U_latent))
    @ (torch.sign(bf2.V_latent).T * bf2.s2.unsqueeze(0))
)
print(
    f"[diag] W_eff: min={W_eff.abs().min().item():.4g} "
    f"mean={W_eff.abs().mean().item():.4g} max={W_eff.abs().max().item():.4g}",
    flush=True,
)

# For comparison: a healthy W (from another block that hasn't been replaced)
W_ref = model.model.layers[1].self_attn.q_proj.weight
print(
    f"[diag] layer-1 q_proj W (reference): min={W_ref.abs().min().item():.4g} "
    f"mean={W_ref.abs().mean().item():.4g} max={W_ref.abs().max().item():.4g}",
    flush=True,
)

# Reset grads
for p in trainable:
    p.grad = None

y3 = block(x, **aux_kwargs)
if isinstance(y3, tuple):
    y3 = y3[0]
loss3 = F.mse_loss(y3.float(), z.float())
print(f"[diag] after-rescale loss3={loss3.item():.6f}", flush=True)
loss3.backward()

for i, p in enumerate(trainable[:8]):
    g = p.grad
    gstr = f"{g.norm().item():.4g}" if g is not None else "None"
    print(f"  param[{i}] shape={tuple(p.shape)} grad.norm={gstr}", flush=True)
