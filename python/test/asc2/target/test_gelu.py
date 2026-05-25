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


@asc2.jit(static_alloc=True, reuse_ub=True)
def gelu_kernel(x_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, input_length: asc2.ConstExpr,
                tile_length: asc2.ConstExpr, block_loop_num: asc2.ConstExpr, block_loop_num_tail: asc2.ConstExpr,
                block_length: asc2.ConstExpr, TANH_APPROX_FACTOR: asc2.ConstExpr,
                NEG_SQRT_EIGHT_OVER_PI: asc2.ConstExpr, UNROLL_FACTOR: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [input_length])
    out_gm = asc2.tensor(out_ptr, [input_length])

    offset = block_length * asc2.block_idx()
    loop_count = block_loop_num
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_loop_num_tail
    for i in asc2.range(loop_count, unroll_factor=UNROLL_FACTOR, parallel=True):
        current_offset = offset + i * tile_length
        row = asc2.load(x_gm, [tile_length], offsets=[current_offset])
        input_sq = row * row
        input_cub = input_sq * row
        input_cub = row + input_cub * TANH_APPROX_FACTOR
        input_cub = input_cub * NEG_SQRT_EIGHT_OVER_PI
        input_cub = asc2.exp(input_cub)
        input_cub = input_cub + 1
        out = row / input_cub
        asc2.store(out, out_gm, offsets=[current_offset])


def gelu_torch(x: torch.Tensor, TANH_APPROX_FACTOR, NEG_SQRT_EIGHT_OVER_PI):
    input_sq = x * x
    input_cub = input_sq * x
    input_cub = TANH_APPROX_FACTOR * input_cub + x
    input_cub = NEG_SQRT_EIGHT_OVER_PI * input_cub
    input_cub = torch.exp(input_cub)
    input_cub = input_cub + 1
    return x / input_cub


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, tiling_key, tiling_values", [
        (72, 1, [24, 512, 1024], torch.float32, [24, 512, 1024], torch.float32, 7, [12582912, 72, 15872]),
        (72, 2, [24, 512, 1024], torch.float16, [24, 512, 1024], torch.float16, 3, [12582912, 72, 10496]),
        (72, 2, [101, 181, 53, 17, 2], torch.float16, [101, 181, 53, 17, 2], torch.float16, 3, [32942362, 72, 10496]),
        (72, 1, [101, 181, 53, 17, 2], torch.float32, [101, 181, 53, 17, 2], torch.float32, 7, [32942362, 72, 15872]),
        # (72, 2, [101, 181, 53, 17, 1], torch.bfloat16, [101, 181, 53, 17, 1], torch.bfloat16, 5, [16471181, 72, 10496]),
        (72, 2, [101, 181, 53, 17, 1], torch.float16, [101, 181, 53, 17, 1], torch.float16, 3, [16471181, 72, 10496]),
        (72, 1, [101, 181, 53, 17, 1], torch.float32, [101, 181, 53, 17, 1], torch.float32, 7, [16471181, 72, 15872]),
    ])
def test_gelu(backend: asc2.Backend, platform: asc2.Platform, device_id, profiler, runs, core_num, unroll_factor,
              input_shape, input_dtype, output_shape, output_dtype, tiling_key, tiling_values):
    asc2.set_platform(backend, platform, device_id)
    CACHE_LINE_BYTE_LENGTH = 512
    TANH_APPROX_FACTOR = 1.0 / 0.044715
    NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715
    _, _, ub_former = tiling_values

    # Convert any shape to 1D
    input_shape_1d = [torch.prod(torch.tensor(input_shape[:])).item()]

    tile_length = ub_former
    dim0 = input_shape_1d[0]
    length = input_shape_1d[0]
    block_former = asc2.ceildiv(asc2.ceildiv(dim0, core_num), CACHE_LINE_BYTE_LENGTH) * CACHE_LINE_BYTE_LENGTH
    block_num = asc2.ceildiv(dim0, block_former)
    block_tail = dim0 - (block_num - 1) * block_former
    former_block_loop = asc2.ceildiv(block_former, ub_former)
    tail_block_loop = asc2.ceildiv(block_tail, ub_former)
    block_loop_num = former_block_loop
    if core_num == 1:
        block_loop_num_tail = 1
        tile_length = dim0
    else:
        block_loop_num_tail = tail_block_loop

    # Alignment
    padded_length = tile_length * block_loop_num * max(1, (core_num - 1)) + tile_length * block_loop_num_tail
    padded_input_shape = [padded_length]
    padded_output_shape = padded_input_shape

    in_tensor = torch.full(padded_input_shape, dtype=input_dtype, fill_value=0)
    in_tensor[:length] = torch.randn(input_shape_1d, dtype=input_dtype)
    out_tensor = torch.zeros(padded_output_shape, dtype=output_dtype)

    with profiler.profile():
        for _ in range(runs):
            gelu_kernel[core_num](in_tensor, out_tensor, padded_length, tile_length, block_loop_num,
                                  block_loop_num_tail, tile_length * block_loop_num, TANH_APPROX_FACTOR,
                                  NEG_SQRT_EIGHT_OVER_PI, unroll_factor)

    expected = gelu_torch(in_tensor, TANH_APPROX_FACTOR, NEG_SQRT_EIGHT_OVER_PI)
    torch.testing.assert_close(out_tensor[:length], expected[:length], rtol=1e-3, atol=1e-3)
