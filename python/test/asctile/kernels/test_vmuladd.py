# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import numpy as np


@asctile.jit(always_compile=True, vf_fusion=True)
def vmuladd_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress, z_ptr: asctile.GlobalAddress,
                   out_ptr: asctile.GlobalAddress, size: int, tile_size: asctile.ConstExpr[int],
                   tile_per_block: asctile.ConstExpr[int], buffer_factor: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    z_gm = asctile.global_tensor(z_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    base_offset = asctile.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=buffer_factor):
        tile_offset = base_offset + i * tile_size
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size])
        y = asctile.copy_in(y_gm, [tile_offset], [tile_size])
        z = asctile.copy_in(z_gm, [tile_offset], [tile_size])
        out = x * y + z
        asctile.copy_out(out, out_gm, [tile_offset])


def vmuladd_launch(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    core_num = 16
    tile_size = 32
    num_tiles = asctile.ceildiv(size, tile_size)
    vmuladd_kernel[core_num](x, y, z, out, size, tile_size, asctile.ceildiv(num_tiles, core_num), buffer_factor=2)
    return out


def test_vmuladd(torch_seed: int):
    rng = np.random.default_rng(torch_seed)
    size = 8192
    x = rng.random(size, dtype=np.float32) * 10
    y = rng.random(size, dtype=np.float32) * 10
    z = rng.random(size, dtype=np.float32) * 10
    out = vmuladd_launch(x, y, z)
    np.testing.assert_allclose(out, x * y + z)
