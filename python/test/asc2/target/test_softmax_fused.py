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


# The current implementation works for columns as long as the shape specified in asc2.load fits in UB.
@asc2.jit(static_alloc=True, reuse_ub=True)
def softmax_fused(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_num_rows, input_num_cols,
                  rows_per_core, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_num_rows, input_num_cols])
    out_gm = asc2.global_tensor(output_ptr, [input_num_rows, input_num_cols])

    # rows_per_block = asc2.ceildiv(input_num_rows, asc2.block_num())
    rows_per_block = rows_per_core
    start_offset = asc2.block_idx() * rows_per_block
    block_loop_num = asc2.number(asc2.ceildiv(rows_per_block, tile_shape[0]), asc2.int_)

    # TODO: remove redundant tail handling when the accuracy issue is resolved
    if asc2.block_idx() == asc2.block_num() - 1:
        tail_rows_per_block = input_num_rows - rows_per_block * (asc2.block_num() - 1)
        block_loop_num = asc2.ceildiv(tail_rows_per_block, tile_shape[0])

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor, parallel=True):
        row_start_offset = start_offset + i * tile_shape[0]
        rows = asc2.load(in_gm, [row_start_offset, 0], [tile_shape[0], tile_shape[1]], pad_value=float('-inf'))
        out = asc2.softmax(rows)
        asc2.store(out, out_gm, [row_start_offset, 0])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("block_num, unroll_factor, input_shape, in_out_dtype, tiling_key, tiling_values", [
    (56, 2, [668, 3], torch.float32, 1000, [668, 3, 8, 12, 12, 1]),
    (52, 2, [512, 4], torch.float32, 1000, [512, 4, 8, 10, 10, 1]),
    (4, 2, [4, 122], torch.float32, 1000, [4, 122, 128, 1, 1, 2]),
    (52, 2, [256, 98], torch.float32, 1000, [256, 98, 104, 5, 5, 2]),
    (1, 2, [64, 10], torch.float32, 500, [64, 10, 1, 1, 64, 64, 8, 16]),
    (1, 2, [64, 1, 8], torch.float32, 500, [64, 8, 1, 1, 64, 64, 8, 8]),
    (56, 1, [4, 16, 128, 128], torch.float32, 1000, [8192, 128, 128, 98, 147, 2]),
    (56, 2, [32, 400, 30], torch.float32, 1000, [12800, 30, 32, 229, 229, 1]),
    (56, 1, [1, 12, 256, 256], torch.float32, 1000, [3072, 256, 256, 49, 55, 4]),
    (12, 1, [2048, 7, 7], torch.float32, 500, [14336, 7, 12, 1, 1280, 256, 8, 8]),
    (56, 1, [8, 12, 197, 197], torch.float32, 1000, [18912, 197, 200, 63, 338, 4]),
    (56, 1, [2, 12, 512, 512], torch.float32, 1000, [12288, 512, 512, 24, 220, 8]),
])
def test_softmax_fused(profiler, runs, kernel_type, block_num, unroll_factor, input_shape, in_out_dtype, tiling_key,
                       tiling_values):
    input_shape_2d = [1, input_shape[0]] if len(input_shape) == 1 else [math.prod(input_shape[:-1]), input_shape[-1]]

    rows_per_iter = rows_per_core = None
    if tiling_key == 500:
        _, _, _, rows_per_core, rows_per_iter, _, _, _ = tiling_values
    elif tiling_key == 1000:
        _, _, _, rows_per_iter, rows_per_core, _ = tiling_values
    elif tiling_key == 10000:
        _, rows_per_iter, rows_per_core = tiling_values

    # Alignment for tile_shape
    num_rows, num_cols = input_shape_2d
    ALIGNMENT_ELEMENTS = 32 // in_out_dtype.itemsize
    tile_shape = [rows_per_iter, asc2.ceildiv(num_cols, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS]

    in_tensor = torch.randn(input_shape_2d, dtype=in_out_dtype)
    out_tensor = torch.zeros(input_shape_2d, dtype=in_out_dtype)

    params = [in_tensor, out_tensor]
    if kernel_type == STATIC:
        params.extend(
            [asc2.ConstExpr(input_shape_2d[0]),
             asc2.ConstExpr(input_shape_2d[1]),
             asc2.ConstExpr(rows_per_core)])
    else:
        params.extend([input_shape_2d[0], input_shape_2d[1], rows_per_core])
    params.extend([tile_shape, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            softmax_fused[block_num](*params)

    expected = torch.softmax(in_tensor, dim=1)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
