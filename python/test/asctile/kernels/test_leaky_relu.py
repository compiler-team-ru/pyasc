# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import pytest
import torch


@asctile.jit
def leaky_relu(x, alpha):
    return asctile.where(x >= 0, x, x * alpha)


@asctile.jit(always_compile=True)
def leaky_relu_kernel(x_ptr: asctile.GlobalAddress, alpha: float, out_ptr: asctile.GlobalAddress, size: int,
                      tile_size: asctile.ConstExpr[int], tile_per_block: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    base_offset = asctile.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=2):
        tile_offset = base_offset + i * tile_size
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size])
        out = leaky_relu(x, alpha)
        asctile.copy_out(out, out_gm, [tile_offset])


def leaky_relu_launch(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    assert alpha.dim() == 0, "'alpha' must be a zero-dim tensor, that is, a scalar value"
    out = torch.empty_like(x)
    size = out.numel()
    core_num = 8
    tile_size = 128
    num_tiles = asctile.ceildiv(size, tile_size)
    leaky_relu_kernel[core_num](x, alpha, out, size, tile_size, asctile.ceildiv(num_tiles, core_num))
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_leaky_relu(dtype: torch.dtype):
    size = 2048
    x = torch.rand(size, dtype=dtype) * 10.0 - 5.0
    alpha = torch.tensor(0.1, dtype=dtype)
    out = leaky_relu_launch(x, alpha)
    torch.testing.assert_close(out, torch.where(x >= 0, x, x * alpha))
