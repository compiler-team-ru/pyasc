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
def cast_kernel_1D(input_x_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, dst_dtype: asc2.ConstExpr,
                   input_shape: asc2.ConstExpr, output_shape: asc2.ConstExpr, block_loop_num: asc2.ConstExpr,
                   block_loop_num_tail: asc2.ConstExpr, tile_length: asc2.ConstExpr, block_length: asc2.ConstExpr,
                   unroll_factor: asc2.ConstExpr):
    x = asc2.tensor(input_x_ptr, input_shape)
    z = asc2.tensor(output_ptr, output_shape)
    block_offset = block_length * asc2.block_idx()
    loop_count = block_loop_num
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_loop_num_tail
    for i in asc2.range(loop_count, unroll_factor=unroll_factor, parallel=True):
        current_offset = block_offset + i * tile_length
        xt = asc2.load(x, [tile_length], offsets=[current_offset])
        zt = xt.to(dst_dtype)
        asc2.store(zt, z, offsets=[current_offset])


@asc2.jit(static_alloc=True, reuse_ub=True)
def double_cast_kernel_1D(input_x_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress,
                          intermediate_dtype: asc2.ConstExpr, dst_dtype: asc2.ConstExpr, input_shape: asc2.ConstExpr,
                          output_shape: asc2.ConstExpr, block_loop_num: asc2.ConstExpr,
                          block_loop_num_tail: asc2.ConstExpr, tile_length: asc2.ConstExpr,
                          block_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    x = asc2.tensor(input_x_ptr, input_shape)
    z = asc2.tensor(output_ptr, output_shape)
    block_offset = block_length * asc2.block_idx()
    loop_count = block_loop_num
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_loop_num_tail
    for i in asc2.range(loop_count, unroll_factor=unroll_factor, parallel=True):
        current_offset = block_offset + i * tile_length
        xt = asc2.load(x, [tile_length], offsets=[current_offset])
        middle_tile = xt.to(intermediate_dtype)
        zt = middle_tile.to(dst_dtype)
        asc2.store(zt, z, offsets=[current_offset])


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, tiling_key, tiling_values", [
        (1, 2, [1, 4096, 1], torch.int8, [1, 4096, 1
                                          ], torch.bfloat16, 0, [1, 25344, 4096, 1, 1, 4096, 4096, 0, 0, 0, 0, 0]),
        (1, 2, [1, 4096], torch.int8, [1, 4096], torch.float32, 0, [1, 17920, 4096, 1, 1, 4096, 4096, 0, 0, 0, 0, 0]),
        (4, 2, [1, 4096], torch.int32, [1, 4096], torch.int64, 0, [4, 10560, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0]),
        (8, 2, [8192], torch.float32, [8192], torch.float32, 0, [8, 21120, 1024, 1, 1, 1024, 1024, 0, 0, 0, 0, 0]),
        (64, 2, [1, 256, 512], torch.float16, [1, 256, 512
                                               ], torch.float32, 0, [64, 21120, 2048, 1, 1, 2048, 2048, 0, 0, 0, 0, 0]),
        (72, 2, [1, 256, 1024], torch.float16, [1, 256, 1024], torch.float32, 0,
         [72, 21120, 3648, 1, 1, 3648, 3136, 0, 0, 0, 0, 0]),
        (72, 2, [10, 1024, 1024], torch.float16, [10, 1024, 1024], torch.float32, 0,
         [72, 21120, 145640, 7, 7, 18920, 18920, 0, 0, 0, 0, 0]),
        (72, 2, [1, 4096, 4000], torch.bfloat16, [1, 4096, 4000], torch.float32, 0,
         [72, 21120, 227560, 11, 11, 16360, 16360, 0, 0, 0, 0, 0]),
        (72, 2, [2, 5, 7, 42767], torch.int32, [2, 5, 7, 42767], torch.int8, 0,
         [72, 25344, 41584, 2, 2, 16240, 15882, 64, 16, 396, 254, 249]),
        (72, 2, [8192, 1024], torch.float32, [8192, 1024], torch.bfloat16, 0,
         [72, 21120, 116512, 6, 6, 10912, 10656, 0, 0, 0, 0, 0]),
    ])
def test_cast(profiler, runs, core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, tiling_key,
              tiling_values):
    _, ub_former, block_former, ub_loop_former_block, ub_loop_tail_block, _, _, _, _, _, _, _ = tiling_values
    input_shape_1d = [math.prod(input_shape)]
    length = input_shape_1d[0]
    alignment_elements = 32 // input_dtype.itemsize
    if ub_loop_former_block == 1 and ub_loop_tail_block == 1:
        ub_former = block_former
    tile_length = (ub_former + alignment_elements - 1) // alignment_elements * alignment_elements
    block_loop_num = ub_loop_former_block
    block_length = tile_length * block_loop_num
    block_loop_num_tail = ub_loop_tail_block
    padded_length = block_length * (core_num - 1) + tile_length * block_loop_num_tail
    padded_input_shape = [padded_length]
    padded_output_shape = padded_input_shape
    in_tensor_x = torch.full(padded_input_shape, dtype=input_dtype, fill_value=0)
    low, high = -127, 128
    if input_dtype.is_floating_point:
        if input_dtype == torch.bfloat16:
            in_tensor_x[:length] = torch.empty(length, dtype=torch.float32).uniform_(float(low),
                                                                                     float(high)).to(input_dtype)
        else:
            in_tensor_x[:length] = torch.empty(length, dtype=input_dtype).uniform_(float(low), float(high))
    else:
        in_tensor_x[:length] = torch.randint(low, high, input_shape_1d, dtype=input_dtype)
    out_tensor = torch.zeros(padded_output_shape, dtype=output_dtype)
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
    with profiler.profile():
        for _ in range(runs):
            if input_dtype == torch.int8 or output_dtype == torch.int8:
                intermediate_dtype = asc2.float16
                double_cast_kernel_1D[core_num](in_tensor_x, out_tensor, intermediate_dtype, dst_dtype,
                                                padded_input_shape, padded_output_shape, block_loop_num,
                                                block_loop_num_tail, tile_length, block_length, unroll_factor)
            else:
                cast_kernel_1D[core_num](in_tensor_x, out_tensor, dst_dtype, padded_input_shape, padded_output_shape,
                                         block_loop_num, block_loop_num_tail, tile_length, block_length, unroll_factor)
    expected = in_tensor_x.to(output_dtype)
    torch.testing.assert_close(out_tensor[:length], expected[:length], atol=1e-3, rtol=1e-3)
