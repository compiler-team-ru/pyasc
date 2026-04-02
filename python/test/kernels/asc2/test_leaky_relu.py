# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

import asc
import asc.runtime.config as config
import asc2


@asc2.jit(always_compile=True)
def leaky_relu_kernel(x_ptr: asc.GlobalAddress, alpha: float, out_ptr: asc.GlobalAddress, size: int,
                      tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        out = asc2.where(x >= 0, x, x * alpha)
        asc2.store(out, out_gm, offsets=[tile_offset])


def leaky_relu_launch(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    core_num = 16
    tile_size = 128
    num_tiles = asc.ceildiv(size, tile_size)
    leaky_relu_kernel[core_num](x, alpha, out, size, tile_size, asc.ceildiv(num_tiles, core_num))
    return out


def test_leaky_relu(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    rng = np.random.default_rng(seed=2026)
    size = 8192
    x = rng.random(size, dtype=np.float32) * 3
    alpha = 0.1
    out = leaky_relu_launch(x, alpha)
    np.testing.assert_allclose(out, np.where(x >= 0, x, x * alpha))
