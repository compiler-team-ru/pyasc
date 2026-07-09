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


@asc2.jit(static_alloc=True, reuse_alloc=1)
def reduce_sum_rows(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows, input_num_cols,
                    output_length, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
    out_gm = asc2.global_tensor(output_ptr, [output_length])

    rows_per_block = asc2.ceildiv(input_num_rows, asc2.block_num())
    start_offset = asc2.block_idx() * rows_per_block
    row_iters = asc2.ceildiv(rows_per_block, tile_shape[0])
    column_iters = asc2.ceildiv(input_num_cols, tile_shape[1])

    for i in asc2.range(row_iters, parallel=True, unroll_factor=unroll_factor):
        row_start_offset = start_offset + i * tile_shape[0]
        cache = asc2.zeros([tile_shape[0]], dtype=asc2.float32)
        for j in asc2.range(column_iters, parallel=False):
            tensor_part = asc2.copy_in(in_gm, [row_start_offset, j * tile_shape[1]], tile_shape, pad_value=0)
            output = asc2.reduce_sum(tensor_part, 1)
            cache = output + cache
        asc2.copy_out(cache, out_gm, [row_start_offset])


@asc2.jit(static_alloc=True, reuse_alloc=1)
def reduce_sum_cols(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows, input_num_cols,
                    output_length, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
    out_gm = asc2.global_tensor(output_ptr, [output_length])

    cols_per_block = asc2.ceildiv(input_num_cols, asc2.block_num())
    start_offset = asc2.block_idx() * cols_per_block
    column_iters = asc2.ceildiv(cols_per_block, tile_shape[1])
    row_iters = asc2.ceildiv(input_num_rows, tile_shape[0])

    for j in asc2.range(column_iters, parallel=True, unroll_factor=unroll_factor):
        col_start_offset = start_offset + j * tile_shape[1]
        cache = asc2.zeros([tile_shape[1]], dtype=asc2.float32)
        for i in asc2.range(row_iters, parallel=False):
            tensor_part = asc2.copy_in(in_gm, [i * tile_shape[0], col_start_offset], tile_shape)
            output = asc2.reduce_sum(tensor_part, 0)
            cache = output + cache
        asc2.copy_out(cache, out_gm, [col_start_offset])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize(
    "block_num, unroll_factor, input_shape, in_out_dtype, output_shape, axis, tiling_key, tiling_values", [
        # reduce by row
        (1, 2, [1, 160], torch.float32, [
            1,
        ], 1, 5143, [
            1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 56, 0, 0.0062500000931322575, [1, 160, 0, 0, 0, 0, 0, 0, 0],
            [160, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 160, 0, 0, 0, 0, 0, 0, 0], [160, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (54, 2, [40960, 32], torch.float32, [40960], 1, 5143, [
            2, 107, 386, 1, 1, 1, 1, 40960, 58368, 1792, 56, 0, 0.03125, [40960, 32, 0, 0, 0, 0, 0, 0, 0],
            [32, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [40960, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (3, 1, [30, 1000], torch.float32, [30, 1], 1, 5143, [
            1, 3, 14, 1, 1, 1, 1, 30, 59136, 512, 56, 0, 0.0010000000474974513, [30, 1000, 0, 0, 0, 0, 0, 0, 0],
            [1000, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [30, 1000, 0, 0, 0, 0, 0, 0, 0], [1000, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (8, 2, [128, 904], torch.float32, [128, 1], 1, 5143, [
            1, 8, 16, 1, 1, 1, 1, 128, 59136, 512, 56, 0, 0.0011061946861445904, [128, 904, 0, 0, 0, 0, 0, 0, 0],
            [904, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [128, 904, 0, 0, 0, 0, 0, 0, 0], [904, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (4, 1, [32, 1627], torch.float32, [32, 1], 1, 5143, [
            1, 4, 9, 1, 1, 1, 1, 32, 59136, 512, 56, 0, 0.0006146281375549734, [32, 1627, 0, 0, 0, 0, 0, 0, 0],
            [1627, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [32, 1627, 0, 0, 0, 0, 0, 0, 0], [1627, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (9, 2, [408, 312], torch.float32, [
            408,
        ], 1, 5143, [
            2, 9, 47, 1, 1, 1, 1, 408, 59136, 512, 56, 0, 0.0032051282469183207, [408, 312, 0, 0, 0, 0, 0, 0, 0],
            [312, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [408, 312, 0, 0, 0, 0, 0, 0, 0], [312, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (8, 2, [512, 10], torch.float32, [512, 1], 1, 5143, [
            1, 8, 64, 1, 1, 1, 1, 512, 57600, 3584, 56, 0, 0.10000000149011612, [512, 10, 0, 0, 0, 0, 0, 0, 0],
            [10, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [512, 10, 0, 0, 0, 0, 0, 0, 0], [10, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (3, 1, [64, 512], torch.float32, [64, 1], 1, 5143, [
            1, 3, 28, 1, 1, 1, 1, 64, 59136, 512, 56, 0, 0.001953125, [64, 512, 0, 0, 0, 0, 0, 0, 0],
            [512, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [64, 512, 0, 0, 0, 0, 0, 0, 0], [512, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [
            1024,
        ], torch.float32, [
            1,
        ], 0, 5143, [
            1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 56, 0, 0.0009765625, [1, 1024, 0, 0, 0, 0, 0, 0, 0],
            [1024, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1024, 0, 0, 0, 0, 0, 0, 0], [1024, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        # RuntimeError: UB overflow: 253952 is available, 256064 is used
        # (1, 1, [4000,], torch.float32, [1,], 0, 5143, [1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 56, 0, 0.0002500000118743628, [1, 4000, 0, 0, 0, 0, 0, 0, 0], [4000, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 4000, 0, 0, 0, 0, 0, 0, 0], [4000, 1, 0, 0, 0, 0, 0, 0, 0]]),
        (56, 2, [1576, 768], torch.float32, [
            1576,
        ], 1, 6167, [
            1, 28, 57, 2, 4, 248, 2, 1576, 59136, 512, 56, 0, 0.0013020833721384406, [1576, 768, 0, 0, 0, 0, 0, 0, 0],
            [768, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1576, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        # reduce by column
        (54, 2, [32, 40960], torch.float32, [
            40960,
        ], 0, 10281, [
            2, 107, 384, 1, 1, 1, 1, 40960, 58368, 1792, 56, 0, 0.03125, [1, 32, 40960, 0, 0, 0, 0, 0, 0],
            [1310720, 40960, 1, 0, 0, 0, 0, 0, 0], [40960, 40960, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 32, 40960, 0, 0, 0, 0, 0, 0], [1310720, 40960, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (43, 2, [96, 16384], torch.float32, [
            16384,
        ], 0, 10281, [
            3, 128, 128, 1, 1, 1, 1, 16384, 59136, 512, 56, 0, 0.010416666977107525, [1, 96, 16384, 0, 0, 0, 0, 0, 0],
            [1572864, 16384, 1, 0, 0, 0, 0, 0, 0], [16384, 16384, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 96, 16384, 0, 0, 0, 0, 0, 0], [1572864, 16384, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (47, 2, [256, 17808], torch.float32, [
            17808,
        ], 0, 141353, [
            3, 140, 128, 4, 4, 85, 1, 17808, 59136, 512, 56, 0, 0.00390625, [1, 256, 17808, 0, 0, 0, 0, 0, 0],
            [4558848, 17808, 1, 0, 0, 0, 0, 0, 0], [17808, 17808, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 256, 17808, 0, 0, 0, 0, 0, 0], [4558848, 17808, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [24, 256], torch.float32, [
            256,
        ], 0, 5161, [
            1, 1, 1, 1, 1, 1, 1, 256, 58112, 2304, 56, 0, 0.0416666679084301, [1, 24, 256, 0, 0, 0, 0, 0, 0],
            [6144, 256, 1, 0, 0, 0, 0, 0, 0], [256, 256, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 24, 256, 0, 0, 0, 0, 0, 0], [6144, 256, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (9, 2, [80, 1025], torch.float32, [1, 1025], 0, 10281, [
            1, 9, 128, 1, 1, 1, 1, 1025, 59136, 512, 56, 0, 0.012500000186264515, [1, 80, 1025, 0, 0, 0, 0, 0, 0],
            [82000, 1025, 1, 0, 0, 0, 0, 0, 0], [1025, 1025, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 80, 1025, 0, 0, 0, 0, 0, 0], [82000, 1025, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (54, 2, [2072, 20], torch.float32, [
            20,
        ], 0, 6185, [
            1, 1, 1, 1, 54, 39, 54, 20, 59136, 512, 56, 0, 0.0004826254735235125, [1, 2072, 20, 0, 0, 0, 0, 0, 0],
            [41440, 20, 1, 0, 0, 0, 0, 0, 0], [20, 20, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 2072, 20, 0, 0, 0, 0, 0, 0], [41440, 20, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (14, 2, [128, 1502], torch.float32, [1, 1502], 0, 10281, [
            1, 14, 112, 1, 1, 1, 1, 1502, 59136, 512, 56, 0, 0.0078125, [1, 128, 1502, 0, 0, 0, 0, 0, 0],
            [192256, 1502, 1, 0, 0, 0, 0, 0, 0], [1502, 1502, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 128, 1502, 0, 0, 0, 0, 0, 0], [192256, 1502, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (2, 2, [128, 128], torch.float32, [
            128,
        ], 0, 10281, [
            1, 2, 64, 1, 1, 1, 1, 128, 59136, 512, 56, 0, 0.0078125, [1, 128, 128, 0, 0, 0, 0, 0, 0],
            [16384, 128, 1, 0, 0, 0, 0, 0, 0], [128, 128, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 128, 128, 0, 0, 0, 0, 0, 0], [16384, 128, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [96, 96], torch.float32, [
            96,
        ], 0, 5161, [
            1, 1, 1, 1, 1, 1, 1, 96, 59136, 512, 56, 0, 0.010416666977107525, [1, 96, 96, 0, 0, 0, 0, 0, 0],
            [9216, 96, 1, 0, 0, 0, 0, 0, 0], [96, 96, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 96, 96, 0, 0, 0, 0, 0, 0], [9216, 96, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [256, 17], torch.float32, [
            17,
        ], 0, 5161, [
            1, 1, 1, 1, 1, 1, 1, 17, 59136, 512, 56, 0, 0.00390625, [1, 256, 17, 0, 0, 0, 0, 0, 0],
            [4352, 17, 1, 0, 0, 0, 0, 0, 0], [17, 17, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 256, 17, 0, 0, 0, 0, 0, 0], [4352, 17, 1, 0, 0, 0, 0, 0, 0]
        ]),
    ])
def test_reduce_sum(profiler, runs, kernel_type, block_num, unroll_factor, input_shape, in_out_dtype, output_shape,
                    axis, tiling_key, tiling_values):
    keep_dims = (len(input_shape) == len(output_shape))
    input_shape_2d = input_shape

    if len(input_shape) == 1:
        axis = 1
        input_shape_2d = [1, input_shape[0]]
    elif axis == 0:
        num_cols = math.prod(input_shape[1:])
        input_shape_2d = [input_shape[0], num_cols]
    elif axis == len(input_shape) - 1:
        axis = 1
        num_rows = math.prod(input_shape[:-1])
        input_shape_2d = [num_rows, input_shape[-1]]
    else:
        raise NotImplementedError("ReduceSum for middle dimension(s) is not implemented yet")

    _, _, ub_factor_a, _, _, ub_factor_r, _, _, _, _, _, _, _, _, _, _, _, _, _ = tiling_values
    length_a = input_shape_2d[1 - axis] if ub_factor_a == 1 else ub_factor_a
    length_r = input_shape_2d[axis] if ub_factor_r == 1 else ub_factor_r
    tile_shape = [length_a, length_r] if axis == 1 else [length_r, length_a]
    # Alignment for tile_shape
    ALIGNMENT_ELEMENTS = 32 // in_out_dtype.itemsize
    tile_shape = tile_shape[0], asc2.ceildiv(tile_shape[1], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS
    if axis == 1:
        tile_shape = asc2.ceildiv(tile_shape[0], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS, tile_shape[1]

    num_rows, num_cols = input_shape_2d
    output_shape_1d = [num_rows] if axis == 1 else [num_cols]

    in_tensor = torch.randn(input_shape_2d, dtype=in_out_dtype)
    out_tensor = torch.zeros(output_shape_1d, dtype=in_out_dtype)
    kernel_impl = reduce_sum_rows if axis == 1 else reduce_sum_cols

    params = [in_tensor, out_tensor]
    if kernel_type == STATIC:
        params.extend(
            [asc2.ConstExpr(input_shape_2d[0]),
             asc2.ConstExpr(input_shape_2d[1]),
             asc2.ConstExpr(output_shape_1d[0])])
    else:
        params.extend([input_shape_2d[0], input_shape_2d[1], output_shape_1d[0]])
    params.extend([tile_shape, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            kernel_impl[block_num](*params)
    if keep_dims:
        target_shape = [output_shape_1d[0], 1] if axis == 1 else [1, output_shape_1d[0]]
        out_tensor = out_tensor.reshape(target_shape)

    expected = torch.sum(in_tensor, axis, keepdim=keep_dims)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
