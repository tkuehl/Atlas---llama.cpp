"""Compare hook-captured hidden states to output_hidden_states=True.

Same forward pass, same student, both paths active simultaneously.
For each decoder layer, check:
  - Are the tensors the same object? (identity)
  - Are their values byte-identical? (numerical)
  - Do both have requires_grad and a non-None grad_fn?
  - Does backward through MSE(hook) produce the same gradients on
    student params as backward through MSE(output_hidden_states)?

If all four agree, the hook capture is semantically identical to the
HF path and the bug is elsewhere.  If any diverge, we've found it.
"""

from __future__ import annotations

import os as _os
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":
    import subprocess as _subprocess
    _env = dict(_os.environ)
    _env["PYTHONUTF8"] = "1"
    _sys.exit(_subprocess.run([_sys.executable] + _sys.argv, env=_env).returncode)

import torch
from torch import nn
from transformers import AutoModelForCausalLM


from littlebit_qat_model import HiddenCapture  # noqa: E402


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
    ).to(device)
    # Keep liger OUT of this test to isolate hook semantics from
    # kernel-patch interactions.
    # Keep compile OUT of this test to isolate hook semantics from
    # dynamo.
    # Keep grad-checkpoint OUT of this test (same).
    # We want pure-eager HF transformers + hooks.
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    cap = HiddenCapture()
    handles = cap.install(model)

    batch = torch.randint(0, 1000, (1, 128), device=device)

    # Single forward with output_hidden_states=True AND hooks active.
    out = model(batch, output_hidden_states=True)
    hs_list = out.hidden_states                  # N+1 tensors (embed + N layers)
    hook_list = cap.states                       # N tensors (one per layer)

    n_layers = len(model.model.layers)
    print(f"layers: {n_layers}")
    print(f"output_hidden_states count: {len(hs_list)}")
    print(f"hook captures count:        {len(hook_list)}\n")

    hs_post = hs_list[1:]                        # post-layer, N tensors
    print("===== per-layer comparison =====")
    all_identical = True
    for i, (hook_h, hs_h) in enumerate(zip(hook_list, hs_post)):
        same_obj = hook_h is hs_h
        same_vals = torch.equal(hook_h.detach(), hs_h.detach())
        hook_rg = hook_h.requires_grad
        hs_rg = hs_h.requires_grad
        hook_fn = type(hook_h.grad_fn).__name__ if hook_h.grad_fn is not None else None
        hs_fn = type(hs_h.grad_fn).__name__ if hs_h.grad_fn is not None else None
        flag = "OK " if (same_obj and same_vals and hook_rg and hs_rg) else "!!! "
        print(f"  {flag}layer {i:2d}: "
              f"same_obj={same_obj} same_vals={same_vals} "
              f"hook.requires_grad={hook_rg} hs.requires_grad={hs_rg} "
              f"hook.grad_fn={hook_fn} hs.grad_fn={hs_fn}")
        if not (same_obj and same_vals and hook_rg and hs_rg):
            all_identical = False
    print(f"\nALL IDENTICAL: {all_identical}")

    # ---- Gradient propagation test ----
    # Compute MSE to a fixed target through each path, do backward,
    # see if gradients on model parameters are identical.

    # Fixed target = clone of hook-captured (so target is identical
    # for both paths).  Detach so it's not a gradient source.
    target = [h.detach().clone() for h in hook_list]

    # Path 1: MSE through hooks
    model.zero_grad(set_to_none=True)
    cap.clear()
    out1 = model(batch, output_hidden_states=False)
    hook_mse = sum(torch.nn.functional.mse_loss(h, t)
                   for h, t in zip(cap.states, target))
    hook_mse.backward()
    grads_hook = {n: p.grad.detach().clone() if p.grad is not None else None
                  for n, p in model.named_parameters()}

    # Path 2: MSE through output_hidden_states
    model.zero_grad(set_to_none=True)
    cap.clear()
    out2 = model(batch, output_hidden_states=True)
    hs_mse = sum(torch.nn.functional.mse_loss(h, t)
                 for h, t in zip(out2.hidden_states[1:], target))
    hs_mse.backward()
    grads_hs = {n: p.grad.detach().clone() if p.grad is not None else None
                for n, p in model.named_parameters()}

    print(f"\nhook MSE loss:  {hook_mse.item():.6f}")
    print(f"hs   MSE loss:  {hs_mse.item():.6f}")
    print(f"loss values equal: {torch.allclose(hook_mse.detach(), hs_mse.detach())}")

    print(f"\n===== gradient comparison =====")
    grad_differ = []
    params_with_hook_grad = 0
    params_with_hs_grad = 0
    hook_total = 0.0
    hs_total = 0.0
    for name in grads_hook:
        gh = grads_hook[name]
        gs = grads_hs[name]
        if gh is not None:
            params_with_hook_grad += 1
            hook_total += gh.abs().sum().item()
        if gs is not None:
            params_with_hs_grad += 1
            hs_total += gs.abs().sum().item()
        if gh is None and gs is None:
            continue
        if gh is None or gs is None:
            grad_differ.append((name, "one is None"))
            continue
        if not torch.allclose(gh, gs, atol=1e-5, rtol=1e-4):
            delta = (gh - gs).abs().max().item()
            grad_differ.append((name, f"max |diff| = {delta:.3e}"))

    print(f"params with grad (hook path): {params_with_hook_grad}")
    print(f"params with grad (hs path):   {params_with_hs_grad}")
    print(f"sum |grad| (hook path): {hook_total:.4f}")
    print(f"sum |grad| (hs path):   {hs_total:.4f}")
    print(f"ratio hook/hs: {hook_total / max(hs_total, 1e-12):.4f}")
    print(f"params where grads differ: {len(grad_differ)}")
    for name, why in grad_differ[:10]:
        print(f"  {name}: {why}")
    if len(grad_differ) > 10:
        print(f"  ... and {len(grad_differ) - 10} more")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
