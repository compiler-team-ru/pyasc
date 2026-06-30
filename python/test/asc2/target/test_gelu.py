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

STATIC = "static"
DYNAMIC = "dynamic"


@asc2.jit(static_alloc=True, reuse_ub=True, reuse_ub_in_out=True)
def gelu(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, tile_length: asc2.ConstExpr,
         TANH_APPROX_FACTOR: asc2.ConstExpr, NEG_SQRT_EIGHT_OVER_PI: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = block_length * asc2.block_idx()

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor, parallel=True):
        current_offset = block_offset + i * tile_length
        row = asc2.load(in_gm, [current_offset], [tile_length])
        input_sq = row * row
        input_cub = input_sq * row
        input_cub = row + input_cub * TANH_APPROX_FACTOR
        input_cub = input_cub * NEG_SQRT_EIGHT_OVER_PI
        input_cub = asc2.exp(input_cub)
        input_cub = input_cub + 1
        out = row / input_cub
        asc2.store(out, out_gm, [current_offset])


def gelu_torch(x: torch.Tensor, TANH_APPROX_FACTOR, NEG_SQRT_EIGHT_OVER_PI):
    input_sq = x * x
    input_cub = input_sq * x
    input_cub = TANH_APPROX_FACTOR * input_cub + x
    input_cub = NEG_SQRT_EIGHT_OVER_PI * input_cub
    input_cub = torch.exp(input_cub)
    input_cub = input_cub + 1
    return x / input_cub


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
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
def test_gelu(profiler, runs, kernel_type, block_num, unroll_factor, input_shape, in_out_dtype, tiling_key,
              tiling_values):
    TANH_APPROX_FACTOR = 1.0 / 0.044715
    NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715
    input_shape_1d = [math.prod(input_shape)]
    _, _, tile_length = tiling_values

    in_tensor = torch.randn(input_shape_1d, dtype=in_out_dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=in_out_dtype)

    params = [in_tensor, out_tensor]
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(input_shape_1d[0]))
    else:
        params.append(input_shape_1d[0])
    params.extend([tile_length, TANH_APPROX_FACTOR, NEG_SQRT_EIGHT_OVER_PI, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            gelu[block_num](*params)

    expected = gelu_torch(in_tensor, TANH_APPROX_FACTOR, NEG_SQRT_EIGHT_OVER_PI)
    torch.testing.assert_close(out_tensor, expected, rtol=1e-3, atol=1e-3)
