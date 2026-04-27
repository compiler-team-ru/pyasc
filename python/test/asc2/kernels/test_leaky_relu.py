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


@asc2.jit(always_compile=True)
def leaky_relu_kernel(x_ptr: asc.GlobalAddress, alpha: float, out_ptr: asc.GlobalAddress, size: int,
                      tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        out = asc2.where(x >= 0, x, x * alpha)
        asc2.store(out, out_gm, offsets=[tile_offset])


def leaky_relu_launch(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    assert alpha.dim() == 0, "'alpha' must be a zero-dim tensor, that is, a scalar value"
    out = torch.empty_like(x)
    size = out.numel()
    core_num = 16
    tile_size = 128
    num_tiles = asc.ceildiv(size, tile_size)
    leaky_relu_kernel[core_num](x, alpha, out, size, tile_size, asc.ceildiv(num_tiles, core_num))
    return out


def test_leaky_relu(backend: config.Backend, platform: config.Platform, device_id: int):
    config.set_platform(backend, platform, device_id)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    size = 8192
    x = torch.rand(size, dtype=torch.bfloat16, device=device) * 10.0 - 5.0
    alpha = torch.tensor(0.1, dtype=torch.bfloat16)
    out = leaky_relu_launch(x, alpha)
    torch.testing.assert_close(out, torch.where(x >= 0, x, x * alpha))
