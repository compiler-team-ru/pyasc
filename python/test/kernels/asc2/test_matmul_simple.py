# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch

import asc
import asc.runtime.config as config
import asc2


@asc2.jit(always_compile=True, kernel_type=asc.runtime.config.KernelType.AIC_ONLY)
def matmul_kernel(a_ptr: asc.GlobalAddress, b_ptr: asc.GlobalAddress, c_ptr: asc.GlobalAddress, a_shape: asc.ConstExpr,
                  b_shape: asc.ConstExpr, c_shape: asc.ConstExpr):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    a = asc2.load(a_gm, a_shape, offsets=[0, 0], location=asc2.TileLocation.L0A)
    b = asc2.load(b_gm, b_shape, offsets=[0, 0], location=asc2.TileLocation.L0B)
    c = a @ b
    asc2.store(c, c_gm, offsets=[0, 0])


def matmul_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_kernel[1](a, b, c, a.shape, b.shape, c.shape)
    return c


def test_matmul_simple(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    m, k, n = 64, 128, 256
    dtype = torch.float16
    a = torch.rand((m, k), dtype=dtype, device=device)
    b = torch.rand((k, n), dtype=dtype, device=device)
    c = matmul_launch(a, b)
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref)
