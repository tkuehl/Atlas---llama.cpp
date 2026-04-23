"""Build a LittleBit student model with activation-weighted ALS scale
refinement for every linear in the transformer stack.

Pipeline:
  1. Load teacher model.
  2. Load per-linear Gramians (from `littlebit_gramians.py`).
  3. For each Linear in model.model.layers:
     a. Compute Frobenius Dual-SVID (signs + initial h, g, ell).
     b. Run ALS on (h, g, ell) under activation-weighted objective:
        min tr((W - W_hat) H (W - W_hat)^T)  with signs frozen.
     c. Build LittleBitLinearHF with refined params.
  4. Save student state_dict + optionally evaluate WikiText PPL.

ALS on GPU (torch) to make down_proj (4864×4864 Gramian) tractable —
numpy would take >1 hr per down_proj; GPU ~1–2 min.

Usage:
    python littlebit_init_refined.py --model Qwen/Qwen2.5-0.5B \\
        --gramians qwen05b_gramians.pt --rank 512 --iters 15 \\
        --out s5_0_refined_init.student.pt --eval
"""

from __future__ import annotations

import os as _os
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":
    import subprocess as _subprocess
    _env = dict(_os.environ)
    _env["PYTHONUTF8"] = "1"
    _sys.exit(_subprocess.run(
        [_sys.executable] + _sys.argv, env=_env
    ).returncode)

try:
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn

from littlebit_qat_model import LittleBitLinearHF
from littlebit_qat_brecq import (
    convert_block_to_littlebit,
    write_status,
    get_or_eval_teacher_ppl,
)
from littlebit_qat_brecq_full import eval_ppl


def dual_svid_fp64(W_np, r):
    """Frobenius Dual-SVID on CPU fp64.  Returns (Up, Vp, h, g, ell)
    as numpy arrays."""
    import numpy as np
    d_out, d_in = W_np.shape
    r_eff = min(r, d_out, d_in)
    U_full, S_full, VT_full = np.linalg.svd(W_np, full_matrices=False)
    Uk = U_full[:, :r_eff]
    Sk = S_full[:r_eff]
    Vk = VT_full[:r_eff, :].T
    sqrt_S = np.sqrt(Sk)
    Up = Uk * sqrt_S[None, :]
    Vp = Vk * sqrt_S[None, :]
    U_abs = np.abs(Up)
    V_abs = np.abs(Vp)
    uU, sU, vtU = np.linalg.svd(U_abs, full_matrices=False)
    uV, sV, vtV = np.linalg.svd(V_abs, full_matrices=False)
    h0 = uU[:, 0] * np.sqrt(sU[0])
    l_u0 = vtU[0, :] * np.sqrt(sU[0])
    g0 = uV[:, 0] * np.sqrt(sV[0])
    l_v0 = vtV[0, :] * np.sqrt(sV[0])
    if h0.sum() < 0:
        h0 = -h0; l_u0 = -l_u0
    if g0.sum() < 0:
        g0 = -g0; l_v0 = -l_v0
    ell0 = l_u0 * l_v0
    return Up, Vp, h0, g0, ell0


def als_scale_refine_gpu(
    W: torch.Tensor,          # (d_out, d_in) fp32 on device
    H: torch.Tensor,          # (d_in, d_in) fp32 on device
    sU: torch.Tensor,         # (d_out, r) fp32 on device (values ±1)
    sV: torch.Tensor,         # (d_in, r) fp32 on device (values ±1)
    h: torch.Tensor,          # (d_out,) fp32 on device
    g: torch.Tensor,          # (d_in,) fp32 on device
    ell: torch.Tensor,        # (r,) fp32 on device
    iters: int = 15,
    tol: float = 1e-4,
    verbose: bool = False,
):
    """Activation-weighted ALS over (h, g, ell) with sU, sV fixed.

    GPU torch version of the numpy implementation in
    `littlebit_init_activation.py`.  All tensors must be on the same
    device in fp32.  Runs iters alternating-least-squares sweeps.

    Returns refined (h, g, ell) tensors on the same device.
    """
    d_out, d_in = W.shape
    r = sU.shape[1]

    def recon():
        return (h[:, None] * sU) @ torch.diag(ell) @ sV.T * g[None, :]

    def act_err():
        D = W - recon()
        return float(torch.sqrt(torch.trace(D @ H @ D.T)).item())

    prev = act_err()
    if verbose:
        print(f"    [als] init act-err: {prev:.4f}")

    for it in range(iters):
        # ----- Solve for ell -----
        hU = h[:, None] * sU      # (d_out, r)
        gV = g[:, None] * sV      # (d_in, r)
        hUhU = hU.T @ hU          # (r, r)
        gVHgV = gV.T @ H @ gV     # (r, r)
        A_ll = hUhU * gVHgV
        WHgV = W @ H @ gV         # (d_out, r)
        b_ll = (hU * WHgV).sum(dim=0)  # (r,)
        A_ll_reg = A_ll + 1e-10 * torch.eye(r, device=W.device) * \
                   torch.trace(A_ll) / r
        ell = torch.linalg.solve(A_ll_reg, b_ll)

        # ----- Solve for h (rows independent) -----
        C = sU @ torch.diag(ell) @ sV.T * g[None, :]  # (d_out, d_in)
        WH = W @ H
        num = (WH * C).sum(dim=1)
        den = (C @ H * C).sum(dim=1)
        h = num / torch.clamp(den, min=1e-20)

        # ----- Solve for g -----
        D_mat = (h[:, None] * sU) @ torch.diag(ell) @ sV.T  # (d_out, d_in)
        M = D_mat.T @ D_mat            # (d_in, d_in)
        A_g = M * H
        b_g = (D_mat * (W @ H)).sum(dim=0)
        A_g_reg = A_g + 1e-10 * torch.eye(d_in, device=W.device) * \
                  torch.trace(A_g) / d_in
        g = torch.linalg.solve(A_g_reg, b_g)

        err = act_err()
        if verbose:
            print(f"    [als] iter {it+1}: act-err={err:.4f}  "
                  f"Δ={prev-err:.4f}")
        if abs(prev - err) < tol * max(prev, 1.0):
            break
        prev = err

    return h, g, ell


def build_refined_lb_linear(
    lin: nn.Linear,
    H_cpu: torch.Tensor,
    r: int,
    iters: int,
    device: torch.device,
    tau: float = 100.0,
    shadow_dtype: torch.dtype = torch.float32,
) -> LittleBitLinearHF:
    """Build a LittleBitLinearHF for `lin` via Dual-SVID + ALS scale
    refinement against Gramian H."""
    # Step 1: Frobenius Dual-SVID (numpy fp64 for stability).
    W_np = lin.weight.data.detach().to(torch.float64).cpu().numpy()
    Up_np, Vp_np, h_np, g_np, ell_np = dual_svid_fp64(W_np, r)

    # Step 2: ALS on GPU (fp32).
    import numpy as np
    W = torch.from_numpy(W_np.astype(np.float32)).to(device)
    H = H_cpu.to(torch.float32).to(device)
    sU = torch.from_numpy(np.sign(Up_np).astype(np.float32)).to(device)
    sV = torch.from_numpy(np.sign(Vp_np).astype(np.float32)).to(device)
    h = torch.from_numpy(h_np.astype(np.float32)).to(device)
    g = torch.from_numpy(g_np.astype(np.float32)).to(device)
    ell = torch.from_numpy(ell_np.astype(np.float32)).to(device)

    h_ref, g_ref, ell_ref = als_scale_refine_gpu(
        W, H, sU, sV, h, g, ell, iters=iters,
    )

    # Step 3: build LittleBitLinearHF.  Use the "magnitudes" from Up/Vp
    # (the original SVD products) but overwrite h, g, ell with refined
    # scales.  Signs come from sign(Up_np) / sign(Vp_np) via the usual
    # LittleBit forward (which calls smooth_sign on U_fp, V_fp).
    d_out, d_in = W_np.shape
    r_eff = min(r, d_out, d_in)
    out = LittleBitLinearHF(
        d_in=d_in, d_out=d_out, r=r_eff,
        bias=lin.bias is not None, tau=tau,
        shadow_dtype=shadow_dtype,
    )
    with torch.no_grad():
        out.U_fp.copy_(torch.from_numpy(Up_np.astype(np.float32))
                       .to(shadow_dtype))
        out.V_fp.copy_(torch.from_numpy(Vp_np.astype(np.float32))
                       .to(shadow_dtype))
        out.h.copy_(h_ref.cpu().to(torch.float32))
        out.g.copy_(g_ref.cpu().to(torch.float32))
        out.ell.copy_(ell_ref.cpu().to(torch.float32))
        if lin.bias is not None:
            out.bias.copy_(lin.bias.data.detach().to(torch.float32).cpu())
    return out


def convert_block_refined(
    block: nn.Module,
    block_full_name: str,
    gramians: dict,
    rank: int,
    iters: int,
    device: torch.device,
    tau: float = 100.0,
) -> nn.Module:
    """Replace every nn.Linear in block with scale-refined LittleBit version.
    Mirrors convert_block_to_littlebit but uses ALS refinement per linear."""
    targets = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                parent = block.get_submodule(parent_name)
            else:
                parent = block
                attr = name
            full_name = f"{block_full_name}.{name}"
            targets.append((parent, attr, module, name, full_name))

    print(f"  refining {len(targets)} linear layers:")
    for parent, attr, lin, local_name, full_name in targets:
        if full_name not in gramians:
            print(f"    WARNING: no Gramian for {full_name}; using identity")
            H = torch.eye(lin.in_features, dtype=torch.float32)
        else:
            H = gramians[full_name]
        t_lin = time.time()
        lb = build_refined_lb_linear(
            lin, H, r=rank, iters=iters, device=device, tau=tau,
        )
        setattr(parent, attr, lb)
        print(f"    {local_name}: ({lin.in_features}, {lin.out_features}) "
              f"r={min(rank, lin.in_features, lin.out_features)}  "
              f"t={time.time()-t_lin:.1f}s")

    # Same dtype-preserving wrapper as convert_block_to_littlebit.
    _orig_forward = block.forward
    def _dtype_preserving_forward(hidden_states, *args, **kwargs):
        in_dtype = hidden_states.dtype
        out = _orig_forward(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            if out[0] is None:
                return out
            return (out[0].to(in_dtype),) + out[1:]
        return out.to(in_dtype)
    block.forward = _dtype_preserving_forward
    return block


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--gramians", default="qwen05b_gramians.pt")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--iters", type=int, default=15,
                   help="ALS iterations per linear")
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eval-max-tokens", type=int, default=25000)
    p.add_argument("--out", default="s5_0_refined_init.student.pt")
    p.add_argument("--eval", action="store_true",
                   help="Evaluate student PPL after build")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load teacher -----
    print(f"[refine] loading teacher {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)
    print(f"[refine]   teacher loaded in {time.time()-t0:.1f}s")

    # ----- Load Gramians -----
    print(f"[refine] loading Gramians from {args.gramians}")
    gram_data = torch.load(args.gramians, weights_only=False)
    gramians = gram_data["gramians"]
    print(f"[refine]   {len(gramians)} Gramians loaded")

    # ----- Clone teacher for student, then refine -----
    print(f"[refine] building student (scale-refined LittleBit)")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    student.eval()
    for pr in student.parameters():
        pr.requires_grad_(False)

    num_blocks = len(student.model.layers)
    t_build = time.time()
    status_path = str(Path(args.out).with_suffix(".status.json"))
    write_status(status_path, {
        "phase": "refining",
        "blocks_total": num_blocks,
        "blocks_done": 0,
    })

    for b in range(num_blocks):
        print(f"[refine] === block {b}/{num_blocks-1} ===")
        t_b = time.time()
        student.model.layers[b] = convert_block_refined(
            student.model.layers[b],
            block_full_name=f"model.layers.{b}",
            gramians=gramians,
            rank=args.rank,
            iters=args.iters,
            device=device,
            tau=args.tau,
        ).to(device)
        print(f"[refine]   block {b} done in {time.time()-t_b:.1f}s")
        write_status(status_path, {
            "phase": "refining",
            "blocks_total": num_blocks,
            "blocks_done": b + 1,
            "elapsed_s": time.time() - t_build,
        })

    total_build = time.time() - t_build
    print(f"[refine] total refinement time: {total_build:.1f}s "
          f"({total_build/num_blocks:.1f}s/block)")

    # ----- Save state dict -----
    ckpt = Path(args.out)
    print(f"[refine] saving state_dict -> {ckpt}")
    torch.save(student.state_dict(), ckpt)

    # ----- Optional eval -----
    if args.eval:
        teacher_ppl = get_or_eval_teacher_ppl(
            teacher, tokenizer, device,
            model_id=args.model,
            seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
        )
        print(f"[refine] eval student PPL...")
        t_e = time.time()
        student_ppl, n_tok = eval_ppl(
            student, tokenizer, device,
            seq_len=args.seq_len, max_tokens=args.eval_max_tokens,
        )
        ratio = student_ppl / teacher_ppl
        print(f"[refine]   student PPL = {student_ppl:.3f} "
              f"({ratio:.2f}x teacher, {time.time()-t_e:.1f}s)")
        result = {
            "config": {
                "model": args.model, "rank": args.rank, "iters": args.iters,
                "gramians": args.gramians,
            },
            "teacher_ppl": teacher_ppl,
            "student_ppl": student_ppl,
            "ppl_ratio": ratio,
            "build_seconds": total_build,
        }
        summary_path = ckpt.with_suffix(".json")
        summary_path.write_text(json.dumps(result, indent=2))
        print(f"[refine] saved {summary_path}")


if __name__ == "__main__":
    main()
