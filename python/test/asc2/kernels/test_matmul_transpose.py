# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import pytest
import torch


@asc2.jit(always_compile=True)
def matmul_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                  a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr, tile_a: asc2.ConstExpr,
                  tile_b: asc2.ConstExpr):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    a = asc2.load(a_gm, tile_a, offsets=[0, 0], location=asc2.TileLocation.L1)
    b = asc2.load(b_gm, tile_b, offsets=[0, 0], location=asc2.TileLocation.L1)
    a_0 = asc2.copy(a, offsets=[0, 0], location=asc2.TileLocation.L0A)
    a_0_transpose = asc2.transpose(a_0)
    b_0 = asc2.copy(b, offsets=[0, 0], location=asc2.TileLocation.L0B).T
    c = a_0_transpose @ b_0
    asc2.store(c, c_gm, offsets=[0, 0])


def matmul_launch(a: torch.Tensor, b: torch.Tensor, tile_a, tile_b) -> torch.Tensor:
    c = torch.zeros((a.shape[1], b.shape[0]), dtype=torch.float32)
    matmul_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_a, tile_b)
    return c


@pytest.mark.parametrize("m, k, n, dtype, tile_a, tile_b", [
    (64, 64, 64, torch.float16, [64, 64], [64, 64]),
    (128, 32, 64, torch.float16, [32, 128], [64, 32]),
    (32, 32, 32, torch.bfloat16, [32, 32], [32, 32]),
    (32, 128, 64, torch.bfloat16, [128, 32], [64, 128]),
    (32, 32, 32, torch.float32, [32, 32], [32, 32]),
    (64, 32, 128, torch.float32, [32, 64], [128, 32]),
])
def test_matmul_transpose(m, k, n, dtype, tile_a, tile_b):
    a = (torch.rand((k, m), dtype=dtype) - .5) * 10
    b = (torch.rand((n, k), dtype=dtype) - .5) * 10
    c = matmul_launch(a, b, tile_a, tile_b)
    c_ref = (a.T.to(torch.float32) @ b.T.to(torch.float32))
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
