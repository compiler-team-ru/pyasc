# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

import asc
import asc2
from asc.runtime import config


@asc2.jit(static_alloc=True, reuse_ub=True)
def reduce_sum_rows(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr,
                    output_shape: asc.ConstExpr, tile_shape: asc.ConstExpr, ROWS_PER_BLOCK: asc.ConstExpr,
                    keep_dims: asc.ConstExpr, UNROLL_FACTOR: asc.ConstExpr):
    x_gm = asc2.tensor(input_ptr, input_shape)
    out_gm = asc2.tensor(output_ptr, output_shape)

    start_offset = asc2.block_idx() * ROWS_PER_BLOCK
    ROW_ITERS_PER_BLOCK = asc.ceildiv(ROWS_PER_BLOCK, tile_shape[0])
    COLUMN_ITERS_PER_BLOCK = asc.ceildiv(input_shape[1], tile_shape[1])
    for i in asc2.range(ROW_ITERS_PER_BLOCK, parallel=True, unroll_factor=UNROLL_FACTOR):
        row_start_offset = start_offset + i * tile_shape[0]
        cache = asc2.zeros([tile_shape[0]], dtype=asc.float32)
        for j in asc2.range(COLUMN_ITERS_PER_BLOCK, parallel=False):
            tensor_part = asc2.load(x_gm, tile_shape, offsets=[row_start_offset, j * tile_shape[1]])
            output = asc2.reduce_sum(tensor_part, 1)
            cache = output + cache
        if keep_dims:
            # TODO Use asc2.store for real shape:
            asc2.store(cache.reshape(tile_shape[0], 1), out_gm, offsets=[row_start_offset, 0])
        else:
            asc2.store(cache, out_gm, offsets=[row_start_offset])


@asc2.jit(static_alloc=True, reuse_ub=True)
def reduce_sum_cols(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr,
                    output_shape: asc.ConstExpr, tile_shape: asc.ConstExpr, COLS_PER_BLOCK: asc.ConstExpr,
                    keep_dims: asc.ConstExpr, UNROLL_FACTOR: asc.ConstExpr):
    x_gm = asc2.tensor(input_ptr, input_shape)
    out_gm = asc2.tensor(output_ptr, output_shape)

    start_offset = asc2.block_idx() * COLS_PER_BLOCK
    ROW_ITERS_PER_BLOCK = asc.ceildiv(input_shape[0], tile_shape[0])
    COLUMN_ITERS_PER_BLOCK = asc.ceildiv(COLS_PER_BLOCK, tile_shape[1])
    for j in asc2.range(COLUMN_ITERS_PER_BLOCK, parallel=True, unroll_factor=UNROLL_FACTOR):
        col_start_offset = start_offset + j * tile_shape[1]
        cache = asc2.zeros([tile_shape[1]], dtype=asc.float32)
        for i in asc2.range(ROW_ITERS_PER_BLOCK, parallel=False):
            tensor_part = asc2.load(x_gm, tile_shape, offsets=[i * tile_shape[0], col_start_offset])
            output = asc2.reduce_sum(tensor_part, 0)
            cache = output + cache
        if keep_dims:
            # TODO Use store for real shape:
            asc2.store(cache.reshape(1, tile_shape[1]), out_gm, offsets=[0, col_start_offset])
        else:
            asc2.store(cache, out_gm, offsets=[col_start_offset])


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, axis, tiling_key, tiling_values", [
        # reduce by row
        (1, 2, [1, 160], torch.float32, [
            1,
        ], torch.float32, 1, 5143, [
            1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 56, 0, 0.0062500000931322575, [1, 160, 0, 0, 0, 0, 0, 0, 0],
            [160, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 160, 0, 0, 0, 0, 0, 0, 0], [160, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (54, 2, [40960, 32], torch.float32, [40960], torch.float32, 1, 5143, [
            2, 107, 386, 1, 1, 1, 1, 40960, 58368, 1792, 56, 0, 0.03125, [40960, 32, 0, 0, 0, 0, 0, 0, 0],
            [32, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [40960, 32, 0, 0, 0, 0, 0, 0, 0], [32, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (3, 2, [30, 1000], torch.float32, [30, 1], torch.float32, 1, 5143, [
            1, 3, 14, 1, 1, 1, 1, 30, 59136, 512, 56, 0, 0.0010000000474974513, [30, 1000, 0, 0, 0, 0, 0, 0, 0],
            [1000, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [30, 1000, 0, 0, 0, 0, 0, 0, 0], [1000, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (8, 2, [128, 904], torch.float32, [128, 1], torch.float32, 1, 5143, [
            1, 8, 16, 1, 1, 1, 1, 128, 59136, 512, 56, 0, 0.0011061946861445904, [128, 904, 0, 0, 0, 0, 0, 0, 0],
            [904, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [128, 904, 0, 0, 0, 0, 0, 0, 0], [904, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (4, 2, [32, 1627], torch.float32, [32, 1], torch.float32, 1, 5143, [
            1, 4, 9, 1, 1, 1, 1, 32, 59136, 512, 56, 0, 0.0006146281375549734, [32, 1627, 0, 0, 0, 0, 0, 0, 0],
            [1627, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [32, 1627, 0, 0, 0, 0, 0, 0, 0], [1627, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (9, 2, [408, 312], torch.float32, [
            408,
        ], torch.float32, 1, 5143, [
            2, 9, 47, 1, 1, 1, 1, 408, 59136, 512, 56, 0, 0.0032051282469183207, [408, 312, 0, 0, 0, 0, 0, 0, 0],
            [312, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [408, 312, 0, 0, 0, 0, 0, 0, 0], [312, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (8, 2, [512, 10], torch.float32, [512, 1], torch.float32, 1, 5143, [
            1, 8, 64, 1, 1, 1, 1, 512, 57600, 3584, 56, 0, 0.10000000149011612, [512, 10, 0, 0, 0, 0, 0, 0, 0],
            [10, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [512, 10, 0, 0, 0, 0, 0, 0, 0], [10, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (3, 2, [64, 512], torch.float32, [64, 1], torch.float32, 1, 5143, [
            1, 3, 28, 1, 1, 1, 1, 64, 59136, 512, 56, 0, 0.001953125, [64, 512, 0, 0, 0, 0, 0, 0, 0],
            [512, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [64, 512, 0, 0, 0, 0, 0, 0, 0], [512, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [
            1024,
        ], torch.float32, [
            1,
        ], torch.float32, 0, 5143, [
            1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 56, 0, 0.0009765625, [1, 1024, 0, 0, 0, 0, 0, 0, 0],
            [1024, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1024, 0, 0, 0, 0, 0, 0, 0], [1024, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        # RuntimeError: UB overflow: 253952 is available, 256064 is used
        # (1, 1, [4000,], torch.float32, [1,], torch.float32, 0, 5143, [1, 1, 1, 1, 1, 1, 1, 1, 59136, 512, 56, 0, 0.0002500000118743628, [1, 4000, 0, 0, 0, 0, 0, 0, 0], [4000, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 4000, 0, 0, 0, 0, 0, 0, 0], [4000, 1, 0, 0, 0, 0, 0, 0, 0]]),
        (56, 2, [1576, 768], torch.float32, [
            1576,
        ], torch.float32, 1, 6167, [
            1, 28, 57, 2, 4, 248, 2, 1576, 59136, 512, 56, 0, 0.0013020833721384406, [1576, 768, 0, 0, 0, 0, 0, 0, 0],
            [768, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1576, 768, 0, 0, 0, 0, 0, 0, 0], [768, 1, 0, 0, 0, 0, 0, 0, 0]
        ]),
        # reduce by column
        (54, 2, [32, 40960], torch.float32, [
            40960,
        ], torch.float32, 0, 10281, [
            2, 107, 384, 1, 1, 1, 1, 40960, 58368, 1792, 56, 0, 0.03125, [1, 32, 40960, 0, 0, 0, 0, 0, 0],
            [1310720, 40960, 1, 0, 0, 0, 0, 0, 0], [40960, 40960, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 32, 40960, 0, 0, 0, 0, 0, 0], [1310720, 40960, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (43, 2, [96, 16384], torch.float32, [
            16384,
        ], torch.float32, 0, 10281, [
            3, 128, 128, 1, 1, 1, 1, 16384, 59136, 512, 56, 0, 0.010416666977107525, [1, 96, 16384, 0, 0, 0, 0, 0, 0],
            [1572864, 16384, 1, 0, 0, 0, 0, 0, 0], [16384, 16384, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 96, 16384, 0, 0, 0, 0, 0, 0], [1572864, 16384, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (47, 2, [256, 17808], torch.float32, [
            17808,
        ], torch.float32, 0, 141353, [
            3, 140, 128, 4, 4, 85, 1, 17808, 59136, 512, 56, 0, 0.00390625, [1, 256, 17808, 0, 0, 0, 0, 0, 0],
            [4558848, 17808, 1, 0, 0, 0, 0, 0, 0], [17808, 17808, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 256, 17808, 0, 0, 0, 0, 0, 0], [4558848, 17808, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [24, 256], torch.float32, [
            256,
        ], torch.float32, 0, 5161, [
            1, 1, 1, 1, 1, 1, 1, 256, 58112, 2304, 56, 0, 0.0416666679084301, [1, 24, 256, 0, 0, 0, 0, 0, 0],
            [6144, 256, 1, 0, 0, 0, 0, 0, 0], [256, 256, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 24, 256, 0, 0, 0, 0, 0, 0], [6144, 256, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (9, 2, [80, 1025], torch.float32, [1, 1025], torch.float32, 0, 10281, [
            1, 9, 128, 1, 1, 1, 1, 1025, 59136, 512, 56, 0, 0.012500000186264515, [1, 80, 1025, 0, 0, 0, 0, 0, 0],
            [82000, 1025, 1, 0, 0, 0, 0, 0, 0], [1025, 1025, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 80, 1025, 0, 0, 0, 0, 0, 0], [82000, 1025, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (54, 2, [2072, 20], torch.float32, [
            20,
        ], torch.float32, 0, 6185, [
            1, 1, 1, 1, 54, 39, 54, 20, 59136, 512, 56, 0, 0.0004826254735235125, [1, 2072, 20, 0, 0, 0, 0, 0, 0],
            [41440, 20, 1, 0, 0, 0, 0, 0, 0], [20, 20, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 2072, 20, 0, 0, 0, 0, 0, 0], [41440, 20, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (14, 2, [128, 1502], torch.float32, [1, 1502], torch.float32, 0, 10281, [
            1, 14, 112, 1, 1, 1, 1, 1502, 59136, 512, 56, 0, 0.0078125, [1, 128, 1502, 0, 0, 0, 0, 0, 0],
            [192256, 1502, 1, 0, 0, 0, 0, 0, 0], [1502, 1502, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 128, 1502, 0, 0, 0, 0, 0, 0], [192256, 1502, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (2, 2, [128, 128], torch.float32, [
            128,
        ], torch.float32, 0, 10281, [
            1, 2, 64, 1, 1, 1, 1, 128, 59136, 512, 56, 0, 0.0078125, [1, 128, 128, 0, 0, 0, 0, 0, 0],
            [16384, 128, 1, 0, 0, 0, 0, 0, 0], [128, 128, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 128, 128, 0, 0, 0, 0, 0, 0], [16384, 128, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [96, 96], torch.float32, [
            96,
        ], torch.float32, 0, 5161, [
            1, 1, 1, 1, 1, 1, 1, 96, 59136, 512, 56, 0, 0.010416666977107525, [1, 96, 96, 0, 0, 0, 0, 0, 0],
            [9216, 96, 1, 0, 0, 0, 0, 0, 0], [96, 96, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 96, 96, 0, 0, 0, 0, 0, 0], [9216, 96, 1, 0, 0, 0, 0, 0, 0]
        ]),
        (1, 2, [256, 17], torch.float32, [
            17,
        ], torch.float32, 0, 5161, [
            1, 1, 1, 1, 1, 1, 1, 17, 59136, 512, 56, 0, 0.00390625, [1, 256, 17, 0, 0, 0, 0, 0, 0],
            [4352, 17, 1, 0, 0, 0, 0, 0, 0], [17, 17, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 256, 17, 0, 0, 0, 0, 0, 0], [4352, 17, 1, 0, 0, 0, 0, 0, 0]
        ]),
    ])
def test_reduce_sum(backend, platform, device_id, profiler, runs, core_num, unroll_factor, input_shape, input_dtype,
                    output_shape, output_dtype, axis, tiling_key, tiling_values):
    config.set_platform(backend, platform, device_id)

    keep_dims = (len(input_shape) == len(output_shape))
    if keep_dims:
        print("keep_dims=True is not implemented yet, set keep_dims=False")
        keep_dims = False

    # Convert any shape to 2D
    input_shape_2d = input_shape
    if len(input_shape) == 1:
        axis = 1
        input_shape_2d = [1, input_shape[0]]
    elif axis == 0:
        num_cols = torch.prod(torch.tensor(input_shape[1:])).item()
        input_shape_2d = [input_shape[0], num_cols]
    elif axis == len(input_shape) - 1:
        axis = 1
        num_rows = torch.prod(torch.tensor(input_shape[:-1])).item()
        input_shape_2d = [num_rows, input_shape[-1]]
    else:
        assert False, "ReduceSum for middle dimension(s) is not implemented yet"

    _, _, ub_factor_a, _, _, ub_factor_r, _, _, _, _, _, _, _, _, _, _, _, _, _ = tiling_values
    length_a = input_shape_2d[1 - axis] if ub_factor_a == 1 else ub_factor_a
    length_r = input_shape_2d[axis] if ub_factor_r == 1 else ub_factor_r
    tile_shape = [length_a, length_r] if axis == 1 else [length_r, length_a]

    # Alignment
    num_rows, num_cols = input_shape_2d
    ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
    tile_shape = tile_shape[0], asc.ceildiv(tile_shape[1], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS
    if axis == 1:
        tile_shape = asc.ceildiv(tile_shape[0], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS, tile_shape[1]

    if axis == 1:
        ROWS_PER_BLOCK = asc.ceildiv(num_rows, core_num)
        num_rows_padded = ROWS_PER_BLOCK * core_num
        num_rows_padded = asc.ceildiv(num_rows_padded, tile_shape[0]) * tile_shape[0]
        num_cols_padded = asc.ceildiv(num_cols, tile_shape[1]) * tile_shape[1]
        WORK_PER_BLOCK = ROWS_PER_BLOCK
    else:
        COLS_PER_BLOCK = asc.ceildiv(num_cols, core_num)
        num_cols_padded = COLS_PER_BLOCK * core_num
        num_cols_padded = asc.ceildiv(num_cols_padded, tile_shape[1]) * tile_shape[1]
        num_rows_padded = asc.ceildiv(num_rows, tile_shape[0]) * tile_shape[0]
        WORK_PER_BLOCK = COLS_PER_BLOCK

    padded_input_shape = [num_rows_padded, num_cols_padded]
    if keep_dims:
        padded_output_shape = [num_rows_padded, 1] if axis == 1 else [1, num_cols_padded]
    else:
        padded_output_shape = [num_rows_padded] if axis == 1 else [num_cols_padded]

    in_tensor = torch.zeros(padded_input_shape, dtype=input_dtype)
    in_tensor[:num_rows, :num_cols] = torch.randn(input_shape_2d, dtype=input_dtype)
    out_tensor = torch.zeros(padded_output_shape, dtype=output_dtype)

    kernel_impl = reduce_sum_rows if axis == 1 else reduce_sum_cols
    with profiler.profile():
        for run in range(runs):
            kernel_impl[core_num](in_tensor, out_tensor, padded_input_shape, padded_output_shape, tile_shape,
                                  WORK_PER_BLOCK, keep_dims, unroll_factor)

    expected = torch.sum(in_tensor, axis, keepdim=keep_dims)
    if keep_dims:
        out_tensor = out_tensor[:num_rows, :] if axis == 1 else out_tensor[:, :num_cols]
        expected = expected[:num_rows, :] if axis == 1 else expected[:, :num_cols]
    else:
        out_tensor = out_tensor[:num_rows] if axis == 1 else out_tensor[:num_cols]
        expected = expected[:num_rows] if axis == 1 else expected[:num_cols]
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
