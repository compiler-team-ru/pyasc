# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse

import torch

try:
    import torch_npu
except ModuleNotFoundError:
    torch_npu = None


# Allocate Linear inputs and launch the PyAsc kernel for warmup and measurement.
# Inputs: matrix dimensions, warmup/iteration counts, backend, and platform.
# Output: None; synchronizes the NPU after all requested launches.
def run_pyasc(m, k, n, warmup, iters, backend, platform):
    import asc.runtime.config as config
    from linear import linear_launch

    config.set_platform(config.Backend(backend), config.Platform(platform))
    x = torch.empty(m, k, dtype=torch.float16, device="npu")
    weight = torch.empty(n, k, dtype=torch.float16, device="npu")
    for _ in range(warmup + iters):
        linear_launch(x, weight)
    torch.npu.synchronize()


# Allocate Linear inputs and launch torch_npu for warmup and measurement.
# Inputs: matrix dimensions and warmup/iteration counts.
# Output: None; synchronizes the NPU after all requested launches.
def run_torch_npu(m, k, n, warmup, iters):
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for the torch_npu backend")
    x = torch.empty(m, k, dtype=torch.float16, device="npu")
    weight = torch.empty(n, k, dtype=torch.float16, device="npu")
    for _ in range(warmup + iters):
        torch_npu.npu_linear(x, weight)
    torch.npu.synchronize()


# Parse benchmark arguments, run one backend, and print the completed shape.
# Inputs: command-line backend, shape, repeat, and platform options.
# Output: None; exits after the selected benchmark finishes.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pyasc", "torch_npu"], default="pyasc")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("-r", type=str, default="NPU")
    parser.add_argument("-v", type=str, default="Ascend910B4")
    args = parser.parse_args()

    torch.manual_seed(20260621)
    if args.backend == "pyasc":
        run_pyasc(args.m, args.k, args.n, args.warmup, args.iters, args.r, args.v)
    else:
        run_torch_npu(args.m, args.k, args.n, args.warmup, args.iters)
    print(f"DONE op=linear backend={args.backend} m={args.m} k={args.k} n={args.n} "
          f"warmup={args.warmup} iters={args.iters}")


if __name__ == "__main__":
    main()
