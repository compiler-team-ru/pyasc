# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import math
import subprocess
from pathlib import Path

import torch
try:
    import torch_npu
except ModuleNotFoundError:
    torch_npu = None

ASCENDC_DEMO = Path(__file__).resolve().parent / "ascendc" / "build" / "demo"


# Forward one benchmark request to the PyAsc kernel runner.
# Inputs: Q/K/V tensors, mask, scale, and warmup/iteration counts.
# Output: None; the delegated runner synchronizes after all launches.
def run_pyasc(q, k, v, mask, scale, warmup, iters):
    from fused_infer_attention import run_pyasc as run_kernel

    run_kernel(q, k, v, mask, scale, warmup, iters)


# Run the Ascend C demo for the requested attention shape.
# Inputs: BNSD Q shape and warmup/iteration counts.
# Output: None; raises when the demo process fails.
def run_ascendc(q_shape, warmup, iters):
    batch, heads, seq_len, head_dim = q_shape
    cmd = [
        str(ASCENDC_DEMO),
        "--batch",
        str(batch),
        "--heads",
        str(heads),
        "--seq",
        str(seq_len),
        "--head_dim",
        str(head_dim),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    subprocess.run(cmd, check=True)


# Launch torch_npu fused attention for warmup and measurement.
# Inputs: Q/K/V tensors, causal mask, scale, and repeat counts.
# Output: None; synchronizes the NPU after all requested launches.
def run_torch_npu(q, k, v, mask, scale, warmup, iters):
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for the torch_npu backend")
    head_num = q.shape[1]
    for _ in range(warmup + iters):
        torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=head_num,
            scale=scale,
            input_layout="BNSD",
            atten_mask=mask,
        )
    torch.npu.synchronize()


# Parse benchmark arguments, prepare inputs, and run the selected backend.
# Inputs: command-line backend, BNSD shape, and repeat options.
# Output: None; exits after the selected benchmark finishes.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pyasc", "ascendc", "torch_npu"], default="torch_npu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("-r", type=str, default="NPU")
    parser.add_argument("-v", type=str, default="Ascend910B4")
    args = parser.parse_args()

    shape = (args.batch, args.heads, args.seq, args.head_dim)
    if args.backend == "ascendc":
        run_ascendc(shape, args.warmup, args.iters)
        return

    if args.backend == "pyasc":
        import asc.runtime.config as config
        config.set_platform(config.Backend(args.r), config.Platform(args.v))

    torch.manual_seed(20260706)
    torch.npu.manual_seed(20260706)

    batch, heads, seq_len, head_dim = shape
    device = "npu"
    dtype = torch.float16

    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    # Causal mask: upper triangle = True (masked)
    causal_mask = torch.triu(
        torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )

    if args.backend == "pyasc":
        run_pyasc(q, k, v, causal_mask, scale, args.warmup, args.iters)
    else:
        run_torch_npu(q, k, v, causal_mask, scale, args.warmup, args.iters)

    print(f"DONE op=fused_infer_attention backend={args.backend} "
          f"shape={shape} warmup={args.warmup} iters={args.iters}")


if __name__ == "__main__":
    main()
