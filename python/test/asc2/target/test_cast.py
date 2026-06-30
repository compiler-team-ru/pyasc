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


@asc2.jit(static_alloc=True, reuse_ub=True)
def cast_direct(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, block_loop_num,
                block_loop_num_tail, block_length, tile_length: asc2.ConstExpr, dst_dtype: asc2.ConstExpr,
                unroll_factor: asc2.ConstExpr):
    x_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [input_length])

    # block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    # block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, parallel=True, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        xt = asc2.load(x_gm, [current_offset], [tile_length])
        zt = xt.to(dst_dtype)
        asc2.store(zt, out_gm, [current_offset])


@asc2.jit(static_alloc=True, reuse_ub=True)
def cast_two(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, block_loop_num,
             block_loop_num_tail, block_length, tile_length: asc2.ConstExpr, intermediate_dtype: asc2.ConstExpr,
             dst_dtype: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    x_gm = asc2.global_tensor(input_ptr, [input_length])
    out_gm = asc2.global_tensor(output_ptr, [input_length])

    # block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    # block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, parallel=True, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        xt = asc2.load(x_gm, [current_offset], [tile_length])
        middle_tile = xt.to(intermediate_dtype)
        zt = middle_tile.to(dst_dtype)
        asc2.store(zt, out_gm, [current_offset])


# DYNAMIC [2, 5, 7, 42767] only supports unroll_factor = 1
@pytest.mark.parametrize("kernel_type", [STATIC])
@pytest.mark.parametrize("block_num, unroll_factor, input_shape, input_dtype, output_dtype, tiling_key, tiling_values", [
    # Ascend950PR_9599
    # (1, 2, [1, 4096, 1], torch.int8, torch.bfloat16, 0, [1, 25344, 4096, 1, 1, 4096, 4096, 0, 0, 0, 0, 0]),
    # (1, 2, [1, 4096], torch.int8, torch.float32, 0, [1, 17920, 4096, 1, 1, 4096, 4096, 0, 0, 0, 0, 0]),
    # (4, 2, [1, 4096], torch.int32, torch.int64, 0, [4, 10560, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0]),
    # (8, 2, [8192], torch.float32, torch.float32, 0, [8, 21120, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0]),
    # (64, 2, [1, 256, 512], torch.float16, torch.float32, 0, [64, 21120, 2048, 1, 1, 2048, 2048, 0, 0, 0, 0, 0]),
    # (72, 2, [1, 256, 1024], torch.float16, torch.float32, 0, [72, 21120, 3648, 1, 1, 3648, 3136, 0, 0, 0, 0, 0]),
    # (72, 2, [10, 1024, 1024
    #          ], torch.float16, torch.float32, 0, [72, 21120, 145640, 7, 7, 18920, 18920, 0, 0, 0, 0, 0]),
    # (72, 2, [1, 4096, 4000
    #          ], torch.bfloat16, torch.float32, 0, [72, 21120, 227560, 11, 11, 16360, 16360, 0, 0, 0, 0, 0]),
    # (72, 2, [2, 5, 7, 42767
    #          ], torch.int32, torch.int8, 0, [72, 25344, 41584, 2, 2, 16240, 15882, 64, 16, 396, 254, 249]),
    # (72, 2, [8192, 1024], torch.float32, torch.bfloat16, 0, [72, 21120, 116512, 6, 6, 10912, 10656, 0, 0, 0, 0, 0]),

    # Ascend950PR_957c
    (1, 2, [1, 4096, 1], torch.bool, torch.bfloat16, 0, [1, 25344, 4096, 1, 1, 4096, 4096, 0, 0, 0, 0, 0]),
    (1, 2, [1, 4096], torch.bool, torch.float32, 0, [1, 17920, 4096, 1, 1, 4096, 4096, 0, 0, 0, 0, 0]),
    (4, 2, [1, 4096], torch.int32, torch.int64, 0, [4, 10560, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0]),
    (8, 2, [8192], torch.float32, torch.bfloat16, 0, [8, 21120, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0]),
    (56, 2, [1, 256, 512], torch.float16, torch.float32, 0, [56, 21120, 2344, 1, 1, 2344, 2152, 0, 0, 0, 0, 0]),
    (56, 2, [1, 256, 1024], torch.float16, torch.float32, 0, [56, 21120, 4688, 1, 1, 4688, 4304, 0, 0, 0, 0, 0]),
    (56, 2, [10, 1024, 1024], torch.float16, torch.float32, 0, [56, 21120, 187248, 9, 9, 18288, 18160, 0, 0, 0, 0, 0]),
    (56, 2, [1, 4096, 4000], torch.bfloat16, torch.float32, 0, [56, 21120, 292576, 14, 14, 18016, 17760, 0, 0, 0, 0, 0
                                                                ]),
    (56, 2, [2, 5, 7, 42767], torch.int32, torch.int8, 0, [56, 25344, 53464, 3, 3, 2776, 2482, 64, 16, 396, 44, 39]),
    (56, 2, [8192, 1024], torch.float32, torch.bfloat16, 0, [56, 21120, 149800, 8, 8, 1960, 1768, 0, 0, 0, 0, 0]),
])
def test_cast(profiler, runs, kernel_type, block_num, unroll_factor, input_shape, input_dtype, output_dtype, tiling_key,
              tiling_values):
    # There is no cast for bool in Ascend, use int8 instead
    input_dtype = torch.int8 if input_dtype == torch.bool else input_dtype
    output_dtype = torch.int8 if output_dtype == torch.bool else output_dtype

    input_shape_1d = [math.prod(input_shape)]

    _, tile_length, block_length, block_loop_num, block_loop_num_tail, _, _, _, _, _, _, _ = tiling_values
    if block_loop_num == 1 and block_loop_num_tail == 1:
        tile_length = block_length

    # Fix range to correct test int8 cast
    low, high = -127, 128

    if input_dtype.is_floating_point:
        if input_dtype == torch.bfloat16:
            in_tensor_x = torch.empty(input_shape_1d, dtype=torch.float32).uniform_(float(low),
                                                                                    float(high)).to(input_dtype)
        else:
            in_tensor_x = torch.empty(input_shape_1d, dtype=input_dtype).uniform_(float(low), float(high))
    else:
        in_tensor_x = torch.randint(low, high, input_shape_1d, dtype=input_dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=output_dtype)

    dtype_map = {
        torch.int8: asc2.int8,
        torch.int16: asc2.int16,
        torch.int32: asc2.int32,
        torch.int64: asc2.int64,
        torch.float16: asc2.float16,
        torch.float32: asc2.float32,
        torch.bfloat16: asc2.bfloat16,
    }
    dst_dtype = dtype_map[output_dtype]

    params = [in_tensor_x, out_tensor]
    if kernel_type == STATIC:
        params.extend([
            asc2.ConstExpr(input_shape_1d[0]),
            asc2.ConstExpr(block_loop_num),
            asc2.ConstExpr(block_loop_num_tail),
            asc2.ConstExpr(block_length)
        ])
    else:
        params.extend([input_shape_1d[0], block_loop_num, block_loop_num_tail, block_length])

    if input_dtype == torch.int8 or output_dtype == torch.int8:
        intermediate_dtype = asc2.float16
        params.extend([tile_length, intermediate_dtype, dst_dtype, unroll_factor])

        with profiler.profile():
            for _ in range(runs):
                cast_two[block_num](*params)
    else:
        params.extend([tile_length, dst_dtype, unroll_factor])

        with profiler.profile():
            for _ in range(runs):
                cast_direct[block_num](*params)

    expected = in_tensor_x.to(output_dtype)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
