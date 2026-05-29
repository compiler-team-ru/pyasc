# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

import asc2
import pytest
import torch


@asc2.jit(static_alloc=True, reuse_ub=True)
def select(cond_ptr: asc2.GlobalAddress, input_x_ptr: asc2.GlobalAddress, input_y_ptr: asc2.GlobalAddress,
           output_ptr: asc2.GlobalAddress, input_length: asc2.ConstExpr, tile_length: asc2.ConstExpr,
           unroll_factor: asc2.ConstExpr):
    c = asc2.tensor(cond_ptr, [input_length])
    x = asc2.tensor(input_x_ptr, [input_length])
    y = asc2.tensor(input_y_ptr, [input_length])
    z = asc2.tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length
    loop_count = block_loop_num
    tile_length_tail = asc2.number(tile_length, asc2.int_)
    if asc2.block_idx() == asc2.block_num() - 1:
        tail_block_length = input_length - block_length * (asc2.block_num() - 1)
        loop_count = asc2.ceildiv(tail_block_length, tile_length)
        tile_length_tail = tail_block_length - tile_length * (loop_count - 1)

    for i in asc2.range(loop_count, unroll_factor=unroll_factor, parallel=True):
        current_offset = block_offset + i * tile_length
        real_tile_length = tile_length_tail if i == loop_count - 1 and asc2.block_idx(
        ) == asc2.block_num() - 1 else tile_length
        ct = asc2.load(c, [tile_length], real_shape=[real_tile_length], offsets=[current_offset])
        xt = asc2.load(x, [tile_length], real_shape=[real_tile_length], offsets=[current_offset])
        yt = asc2.load(y, [tile_length], real_shape=[real_tile_length], offsets=[current_offset])
        zt = asc2.where(ct != 0, xt, yt)
        asc2.store(zt, z, real_shape=[real_tile_length], offsets=[current_offset])


@pytest.mark.parametrize("block_num, unroll_factor, input_shape, in_out_dtype, tiling_key, tiling_values", [
    (2, 2, [1920], torch.int32, 8, [1920, 0, 512, 2]),
    (15, 2, [1920, 1, 8], torch.float32, 8, [15360, 0, 512, 15]),
    (16, 2, [256, 4, 1, 16], torch.float32, 8, [16384, 0, 512, 16]),
    (13, 2, [256, 50], torch.float32, 8, [12800, 0, 512, 13]),
    (30, 2, [1920, 1, 40], torch.float32, 8, [76800, 0, 1280, 30]),
    (28, 2, [28442], torch.float32, 8, [28442, 0, 512, 28]),
    (29, 2, [327461], torch.float32, 8, [327461, 0, 5760, 29]),
    (50, 2, [2, 1, 256, 256, 16], torch.float32, 8, [2097152, 0, 7040, 50]),
    (50, 1, [2, 1, 256, 256, 16], torch.float16, 8, [2097152, 0, 14080, 50]),
])
def test_select(profiler, runs, block_num, unroll_factor, input_shape, in_out_dtype, tiling_key, tiling_values):
    input_shape_1d = [math.prod(input_shape)]
    _, _, tile_length, block_num = tiling_values

    in_tensor_c = torch.randint(0, 2, input_shape_1d, dtype=torch.int32)
    in_tensor_x = torch.randn(input_shape_1d).to(in_out_dtype)
    in_tensor_y = torch.randn(input_shape_1d).to(in_out_dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=in_out_dtype)

    with profiler.profile():
        for _ in range(runs):
            select[block_num](in_tensor_c, in_tensor_x, in_tensor_y, out_tensor, input_shape_1d[0], tile_length,
                              unroll_factor)

    expected = torch.where(in_tensor_c.bool(), in_tensor_x, in_tensor_y)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
