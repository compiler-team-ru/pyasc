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


@asc2.jit(static_alloc=True, reuse_ub=True, reuse_ub_in_out=True)
def gelu(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length: asc2.ConstExpr,
         tile_length: asc2.ConstExpr, TANH_APPROX_FACTOR: asc2.ConstExpr, NEG_SQRT_EIGHT_OVER_PI: asc2.ConstExpr,
         unroll_factor: asc2.ConstExpr):
    in_gm = asc2.tensor(input_ptr, [input_length])
    out_gm = asc2.tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = block_length * asc2.block_idx()
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
        row = asc2.load(in_gm, [tile_length], real_shape=[real_tile_length], offsets=[current_offset])
        input_sq = row * row
        input_cub = input_sq * row
        input_cub = row + input_cub * TANH_APPROX_FACTOR
        input_cub = input_cub * NEG_SQRT_EIGHT_OVER_PI
        input_cub = asc2.exp(input_cub)
        input_cub = input_cub + 1
        out = row / input_cub
        asc2.store(out, out_gm, real_shape=[real_tile_length], offsets=[current_offset])


def gelu_torch(x: torch.Tensor, TANH_APPROX_FACTOR, NEG_SQRT_EIGHT_OVER_PI):
    input_sq = x * x
    input_cub = input_sq * x
    input_cub = TANH_APPROX_FACTOR * input_cub + x
    input_cub = NEG_SQRT_EIGHT_OVER_PI * input_cub
    input_cub = torch.exp(input_cub)
    input_cub = input_cub + 1
    return x / input_cub


@pytest.mark.parametrize("block_num, unroll_factor, input_shape, in_out_dtype, tiling_key, tiling_values", [
    ## Ascend950PR_9599
    # (72, 2, [24, 512, 1024], torch.float32, 7, [12582912, 72, 15872]),
    # (72, 2, [24, 512, 1024], torch.float16, 3, [12582912, 72, 10496]),
    # (72, 2, [101, 181, 53, 17, 2], torch.float16, 3, [32942362, 72, 10496]),
    # (72, 2, [101, 181, 53, 17, 2], torch.float32, 7, [32942362, 72, 15872]),
    # # (72, 2, [101, 181, 53, 17, 1], torch.bfloat16, 5, [16471181, 72, 10496]),
    # (72, 2, [101, 181, 53, 17, 1], torch.float16, 3, [16471181, 72, 10496]),
    # (72, 2, [101, 181, 53, 17, 1], torch.float32, 7, [16471181, 72, 15872]),

    ## Ascend950PR_957c
    (56, 2, [24, 512, 1024], torch.float32, 7, [12582912, 56, 15872]),
    (56, 2, [24, 512, 1024], torch.float16, 3, [12582912, 56, 10496]),
    (56, 2, [101, 181, 53, 17, 2], torch.float16, 3, [32942362, 56, 10496]),
    (56, 2, [101, 181, 53, 17, 2], torch.float32, 7, [32942362, 56, 15872]),
    # (56, 2, [101, 181, 53, 17, 1], torch.bfloat16, 5, [16471181, 56, 10496]),
    (56, 2, [101, 181, 53, 17, 1], torch.float16, 3, [16471181, 56, 10496]),
    (56, 2, [101, 181, 53, 17, 1], torch.float32, 7, [16471181, 56, 15872]),
])
def test_gelu(profiler, runs, block_num, unroll_factor, input_shape, in_out_dtype, tiling_key, tiling_values):
    TANH_APPROX_FACTOR = 1.0 / 0.044715
    NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715
    input_shape_1d = [math.prod(input_shape)]
    _, _, tile_length = tiling_values

    in_tensor = torch.randn(input_shape_1d, dtype=in_out_dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=in_out_dtype)

    with profiler.profile():
        for _ in range(runs):
            gelu[block_num](in_tensor, out_tensor, input_shape_1d[0], tile_length, TANH_APPROX_FACTOR,
                            NEG_SQRT_EIGHT_OVER_PI, unroll_factor)

    expected = gelu_torch(in_tensor, TANH_APPROX_FACTOR, NEG_SQRT_EIGHT_OVER_PI)
    torch.testing.assert_close(out_tensor, expected, rtol=1e-3, atol=1e-3)
