# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import torch


# Available parameters for @asc2.jit decorator can be seen in the documentation:
# https://compiler-team-ru.github.io/pyasc/python-api/rst/runtime/index.html
@asc2.jit
def vadd_kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, size: int,
                tile_size: asc2.ConstExpr[int], tile_per_block: asc2.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    y_gm = asc2.tensor(y_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        y = asc2.load(y_gm, [tile_size], offsets=[tile_offset])
        out = x + y
        asc2.store(out, out_gm, offsets=[tile_offset])


def vadd_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    size = out.numel()
    core_num = 16
    tile_size = 128
    num_tiles = asc2.ceildiv(size, tile_size)
    vadd_kernel[core_num](x, y, out, size, tile_size, asc2.ceildiv(num_tiles, core_num))
    return out


if __name__ == "__main__":
    asc2.set_platform(backend="Model", soc_version="Ascend950PR_9599", device_id=0)
    size = 8192
    x = torch.randn(size, dtype=torch.float32, device="cpu") * 10
    y = torch.randn(size, dtype=torch.float32, device="cpu") * 10
    out = vadd_launch(x, y)
    torch.testing.assert_close(out, x + y, atol=1e-5, rtol=1e-5)
