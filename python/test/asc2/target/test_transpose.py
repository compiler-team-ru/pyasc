# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from typing import List, Literal, Union

import asc
import asc.runtime.config as config
import asc2
import pytest
import torch


@dataclass
class TestCase:
    shape: List[int]
    mode: Literal[1, 2]
    dtype: torch.dtype
    tile_size: List[int]
    block_size: Union[int, List[int]]


# Reads (h,w) sub tile
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_block(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, width: asc.ConstExpr[int],
                    height: asc.ConstExpr[int], tilew: asc.ConstExpr[int], tileh: asc.ConstExpr[int],
                    tilew2: asc.ConstExpr[int], tileh2: asc.ConstExpr[int], repeat: asc.ConstExpr[int]):
    total_tiles_w = asc.ceildiv(width, tilew)

    global_tensor = asc2.tensor(input_ptr, [height, width])
    result_tensor = asc2.tensor(output_ptr, [width, height])
    for i in range(asc2.block_idx(), repeat, asc2.block_num(), parallel=True):
        offset_x = (i % total_tiles_w) * tilew
        offset_y = (i // total_tiles_w) * tileh
        load_width = tilew if tilew < width - offset_x else width - offset_x
        load_height = tileh if tileh < height - offset_y else height - offset_y
        input = asc2.load(global_tensor, [tileh2, tilew2], offsets=[offset_y, offset_x],
                          real_shape=[load_height, load_width])
        transposed = input.transpose()
        asc2.store(transposed, result_tensor, offsets=[offset_x, offset_y], real_shape=[load_width, load_height])


# Reads (height,n) at once
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_column(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, width: asc.ConstExpr[int],
                     height: asc.ConstExpr[int], step: asc.ConstExpr[int], tilew: asc.ConstExpr[int],
                     tileh: asc.ConstExpr[int], total_count: asc.ConstExpr[int]):

    global_tensor = asc2.tensor(input_ptr, [height, width])
    result_tensor = asc2.tensor(output_ptr, [width, height])
    for i in range(asc2.block_idx(), total_count, asc2.block_num()):
        offset = i * step
        load_width = step if step < width - offset else width - offset
        input = asc2.load(global_tensor, [tileh, tilew], offsets=[0, offset], real_shape=[height, load_width])
        transposed = input.transpose()
        asc2.store(transposed, result_tensor, offsets=[offset, 0], real_shape=[load_width, height])


@pytest.mark.parametrize("test_params", [
    TestCase([512, 128], 1, torch.float32, [512, 8], 8),
    TestCase([612, 128], 1, torch.float32, [632, 8], 8),
    TestCase([3200, 256], 1, torch.float32, [3200, 8], 4),
    TestCase([4000, 12592], 2, torch.float32, [512, 512], [356, 334]),
    TestCase([4000, 12592], 2, torch.float16, [256, 256], [251, 248]),
    TestCase([4096, 1024], 2, torch.float32, [256, 256], [251, 164]),
    TestCase([1000, 2048], 2, torch.float32, [160, 256], [146, 252]),
    TestCase([1024, 256], 1, torch.float32, [1024, 8], 8),
    TestCase([128, 6400], 1, torch.float16, [128, 16], 16),
    TestCase([4096, 1024], 2, torch.float32, [256, 176], [251, 164]),
])
def test_transpose(backend: config.Backend, platform: config.Platform, device_id: int, profiler, runs, test_params):
    config.set_platform(backend, platform, device_id)

    input = torch.randn(test_params.shape, dtype=test_params.dtype, device="cpu")
    width = test_params.shape[1]
    height = test_params.shape[0]
    out = torch.zeros_like(input, shape=[width, height])
    if test_params.mode == 1:
        # load n columns
        total_tiles = asc.ceildiv(width, test_params.block_size)
        cores = min(total_tiles, 64)
        with profiler.profile():
            for _ in range(runs):
                transpose_column[cores](input, out, width, height, test_params.block_size, test_params.tile_size[1],
                                        test_params.tile_size[0], total_tiles)
    elif test_params.mode == 2:
        # load sub block (2d)
        total_tiles_w = asc.ceildiv(width, test_params.block_size[1])
        total_tiles_h = asc.ceildiv(height, test_params.block_size[0])
        cores = min(total_tiles_w * total_tiles_h, 64)
        with profiler.profile():
            for _ in range(runs):
                transpose_block[cores](input, out, width, height, test_params.block_size[1], test_params.block_size[0],
                                       test_params.tile_size[1], test_params.tile_size[0],
                                       total_tiles_w * total_tiles_h)
    torch.testing.assert_close(out, input.transpose(0, 1))
