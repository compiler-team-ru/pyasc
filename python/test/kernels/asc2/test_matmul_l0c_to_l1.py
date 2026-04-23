# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

import asc
import asc.runtime.config as config
import asc2


@asc2.jit(always_compile=True)
def matmul_kernel(a_ptr: asc.GlobalAddress, b_ptr: asc.GlobalAddress, c_ptr: asc.GlobalAddress, a_shape: asc.ConstExpr,
                  b_shape: asc.ConstExpr, c_shape: asc.ConstExpr, dtype: asc.ConstExpr):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    a = asc2.load(a_gm, a_shape, offsets=[0, 0], location=asc2.TileLocation.L0A)
    b = asc2.load(b_gm, b_shape, offsets=[0, 0], location=asc2.TileLocation.L0B)
    c = a @ b
    c = c.to(dtype)
    c_l1 = asc2.copy(c, c_shape, offsets=[0, 0], location=asc2.TileLocation.L1)
    c_l0a = asc2.copy(c_l1, c_shape, offsets=[0, 0], location=asc2.TileLocation.L0A)
    result = c_l0a @ b
    asc2.store(result, c_gm, offsets=[0, 0])


def matmul_launch(a: torch.Tensor, b: torch.Tensor, dtype) -> torch.Tensor:
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_kernel[1](a, b, c, a.shape, b.shape, c.shape, dtype)
    return c


@pytest.mark.parametrize("m, k, n, torch_dtype, pyasc_dtype", [
    (32, 64, 64, torch.float16, asc.float16),
    (64, 64, 64, torch.bfloat16, asc.bfloat16),
])
def test_matmul_l0c_to_l1(backend: config.Backend, platform: config.Platform, m, k, n, torch_dtype, pyasc_dtype):
    config.set_platform(backend, platform)
    torch.manual_seed(0)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    a = (torch.rand((m, k), dtype=torch_dtype, device=device) - .5) * 10
    b = (torch.rand((k, n), dtype=torch_dtype, device=device) - .5) * 10
    c = matmul_launch(a, b, pyasc_dtype)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch_dtype)
    c_ref = (c_ref.to(torch.float32) @ b.to(torch.float32))
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
