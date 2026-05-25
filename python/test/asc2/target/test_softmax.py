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


# The current implementation works for columns as long as the shape specified in asc2.load fits in UB.
@asc2.jit(static_alloc=True, reuse_ub=True)
def softmax(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_shape: asc2.ConstExpr,
            output_shape: asc2.ConstExpr, total_rows: asc2.ConstExpr, UNROLL_FACTOR: asc2.ConstExpr):
    x_gm = asc2.tensor(input_ptr, input_shape)
    out_gm = asc2.tensor(output_ptr, output_shape)

    for i in range(asc2.block_idx(), total_rows, asc2.block_num(), unroll_factor=UNROLL_FACTOR, parallel=True):
        row = asc2.load(x_gm, [1, input_shape[-1]], offsets=[i, 0])
        row_minus_max = row - asc2.reduce_max(row)
        numerator = asc2.exp(row_minus_max)
        denominator = asc2.reduce_sum(numerator)
        out = numerator / denominator
        asc2.store(out, out_gm, offsets=[i, 0])


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, axis, tiling_key, tiling_values", [
        (56, 2, [668, 3], torch.float32, [668, 3], torch.float32, -1, 1000, [668, 3, 8, 12, 12, 1]),
        (52, 2, [512, 4], torch.float32, [512, 4], torch.float32, -1, 1000, [512, 4, 8, 10, 10, 1]),
        (4, 2, [4, 122], torch.float32, [4, 122], torch.float32, -1, 1000, [4, 122, 128, 1, 1, 2]),
        (52, 2, [256, 98], torch.float32, [256, 98], torch.float32, -1, 1000, [256, 98, 104, 5, 5, 2]),
        (1, 2, [64, 10], torch.float32, [64, 10], torch.float32, -1, 500, [64, 10, 1, 1, 64, 64, 8, 16]),
        (1, 2, [64, 1, 8], torch.float32, [64, 1, 8], torch.float32, -1, 500, [64, 8, 1, 1, 64, 64, 8, 8]),
        (56, 2, [4, 16, 128, 128], torch.float32, [4, 16, 128, 128
                                                   ], torch.float32, -1, 1000, [8192, 128, 128, 98, 147, 2]),
        (56, 2, [32, 400, 30], torch.float32, [32, 400, 30], torch.float32, -1, 1000, [12800, 30, 32, 229, 229, 1]),
        (56, 2, [1, 12, 256, 256], torch.float32, [1, 12, 256, 256
                                                   ], torch.float32, -1, 1000, [3072, 256, 256, 49, 55, 4]),
        (12, 2, [2048, 7, 7], torch.float32, [2048, 7, 7], torch.float32, -1, 500, [14336, 7, 12, 1, 1280, 256, 8, 8]),
        (56, 2, [8, 12, 197, 197], torch.float32, [8, 12, 197, 197
                                                   ], torch.float32, -1, 1000, [18912, 197, 200, 63, 338, 4]),
        (56, 2, [2, 12, 512, 512], torch.float32, [2, 12, 512, 512
                                                   ], torch.float32, -1, 1000, [12288, 512, 512, 24, 220, 8]),
    ])
def test_softmax(backend, platform, device_id, profiler, runs, core_num, unroll_factor, input_shape, input_dtype,
                 output_shape, output_dtype, axis, tiling_key, tiling_values):
    asc2.set_platform(backend, platform, device_id)

    total_rows, rows_per_iter, rows_per_core = None, None, None
    if tiling_key == 500:
        total_rows, _, _, rows_per_core, rows_per_iter, _, _, _ = tiling_values
    if tiling_key == 1000:
        total_rows, _, _, rows_per_iter, rows_per_core, _ = tiling_values
    if tiling_key == 10000:
        total_rows, rows_per_iter, rows_per_core = tiling_values

    # Convert any shape to 2D
    if len(input_shape) == 1:
        input_shape_2d = [1, input_shape[0]]
    else:
        input_shape_2d = [torch.prod(torch.tensor(input_shape[:-1])).item(), input_shape[-1]]

    # Alignment
    num_rows, num_cols = input_shape_2d
    ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
    num_cols_padded = asc2.ceildiv(num_cols, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS
    num_rows_padded = asc2.ceildiv(num_rows, rows_per_iter) * rows_per_iter
    padded_input_shape = [num_rows_padded, num_cols_padded]
    padded_output_shape = padded_input_shape

    in_tensor = torch.full(padded_input_shape, dtype=input_dtype, fill_value=float('-inf'))
    in_tensor[:num_rows, :num_cols] = torch.randn(input_shape_2d, dtype=input_dtype)
    out_tensor = torch.zeros(padded_output_shape, dtype=output_dtype)

    with profiler.profile():
        for _ in range(runs):
            softmax[core_num](in_tensor, out_tensor, padded_input_shape, padded_output_shape, total_rows, unroll_factor)

    expected = torch.softmax(in_tensor, dim=1)
    torch.testing.assert_close(out_tensor[:num_rows, :num_cols], expected[:num_rows, :num_cols], atol=1e-3, rtol=1e-3)
