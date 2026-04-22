"""Measure the magnitude of hook[23] vs hs[23] and the per-layer MSE
contribution between teacher and student via both paths.

If layer 23's hook-captured (pre-final-norm) tensor is much larger in
magnitude than layer 23's post-final-norm tensor, the hook-MSE sum is
dominated by a single ill-conditioned layer — explaining the quality
regression.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class HiddenCapture:
    def __init__(self):
        self.states = []

    def _hook(self, _m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        self.states.append(h)

    def install(self, model):
        return [layer.register_forward_hook(self._hook)
                for layer in model.model.layers]

    def clear(self):
        self.states.clear()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Teacher-sized model for both roles — this is a magnitude test,
    # not a training-dynamics test, so teacher==student architecture.
    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
    ).to(device)
    student.train()

    t_cap = HiddenCapture()
    s_cap = HiddenCapture()
    t_cap.install(teacher)
    s_cap.install(student)

    batch = torch.randint(0, 1000, (1, 128), device=device)

    with torch.no_grad():
        t_out = teacher(batch, output_hidden_states=True)
    s_out = student(batch, output_hidden_states=True)

    print(f"\n===== per-layer magnitudes =====")
    print(f"{'layer':>5} {'hook_s_norm':>12} {'hs_s_norm':>12} "
          f"{'hook_mse':>12} {'hs_mse':>12}")
    hook_mse_total = 0.0
    hs_mse_total = 0.0
    for i in range(len(student.model.layers)):
        s_hook = s_cap.states[i]
        t_hook = t_cap.states[i]
        s_hs = s_out.hidden_states[i + 1]
        t_hs = t_out.hidden_states[i + 1]
        hook_mse_i = F.mse_loss(s_hook, t_hook).item()
        hs_mse_i = F.mse_loss(s_hs, t_hs).item()
        hook_mse_total += hook_mse_i
        hs_mse_total += hs_mse_i
        print(f"  {i:3d} {s_hook.norm().item():12.4f} "
              f"{s_hs.norm().item():12.4f} "
              f"{hook_mse_i:12.4f} {hs_mse_i:12.4f}")

    print(f"\ntotal hook MSE: {hook_mse_total:.4f}")
    print(f"total hs   MSE: {hs_mse_total:.4f}")
    print(f"ratio (hook / hs): {hook_mse_total / hs_mse_total:.2f}x")

    # Share of total MSE contributed by each layer
    print(f"\n===== per-layer share of MSE sum =====")
    print(f"{'layer':>5} {'hook_share':>12} {'hs_share':>12}")
    for i in range(len(student.model.layers)):
        s_hook = s_cap.states[i]
        t_hook = t_cap.states[i]
        s_hs = s_out.hidden_states[i + 1]
        t_hs = t_out.hidden_states[i + 1]
        hook_mse_i = F.mse_loss(s_hook, t_hook).item()
        hs_mse_i = F.mse_loss(s_hs, t_hs).item()
        print(f"  {i:3d} "
              f"{hook_mse_i / hook_mse_total * 100:11.2f}% "
              f"{hs_mse_i / hs_mse_total * 100:11.2f}%")


if __name__ == "__main__":
    main()
