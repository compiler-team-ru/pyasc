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
    a = asc2.load(a_gm, tile_a, offsets=[0, 0], location=asc2.TileLocation.L0A)
    b = asc2.load(b_gm, tile_b, offsets=[0, 0], location=asc2.TileLocation.L0B)
    c = asc2.matmul(a, b, hf32=True)
    asc2.store(c, c_gm, offsets=[0, 0])


def matmul_launch(a: torch.Tensor, b: torch.Tensor, tile_a, tile_b) -> torch.Tensor:
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_a, tile_b)
    return c


@pytest.mark.parametrize("m, k, n, dtype, tile_a, tile_b", [
    (64, 128, 128, torch.float32, [64, 128], [128, 128]),
])
def test_matmul_hf32(m, k, n, dtype, tile_a, tile_b):
    a = torch.rand((m, k), dtype=dtype)
    b = torch.rand((k, n), dtype=dtype)
    c = matmul_launch(a, b, tile_a, tile_b)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32))
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
