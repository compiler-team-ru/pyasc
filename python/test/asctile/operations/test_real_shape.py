# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import pytest
import torch

STATIC = "static"
DYNAMIC = "dynamic"


@asctile.jit(always_compile=True)
def load_real_shape_1d_kernel(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, size: int,
                              tile_size: asctile.ConstExpr, real_size: int, offset: asctile.ConstExpr):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [1])
    tile = asctile.copy_in(x_gm, [offset], [tile_size], real_shape=[real_size], pad_value=float('-inf'))
    max_val = asctile.reduce_max(tile)
    result = asctile.full([1], max_val, dtype=tile.dtype)
    asctile.copy_out(result, out_gm, [0])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("size, tile_size, real_size, offset", [
    (16, 128, 15, 14),
    (16, 16, 8, 0),
    (16, 16, 5, 3),
    (32, 16, 10, 5),
    (42, 8, 5, 17),
])
def test_load_real_shape_1d(kernel_type, size, tile_size, real_size, offset):
    x = torch.arange(1, size + 1, dtype=torch.float32)
    valid_end = offset + real_size
    tile_end = offset + tile_size
    if tile_end <= size:
        x[valid_end:tile_end] = 1000.0
    out = torch.empty(1, dtype=torch.float32)
    real_size_arg = asctile.ConstExpr(real_size) if kernel_type == STATIC else real_size
    load_real_shape_1d_kernel[1](x, out, size, tile_size, real_size_arg, offset)
    expected_region = x[offset:offset + real_size]
    expected = torch.amax(expected_region).unsqueeze(0)
    torch.testing.assert_close(out, expected)


@asctile.jit(always_compile=True)
def load_real_shape_2d_kernel(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, rows: int, cols: int,
                              real_rows: int, real_cols: int, tile_shape: asctile.ConstExpr,
                              offsets: asctile.ConstExpr):
    x_gm = asctile.global_tensor(x_ptr, [rows, cols])
    out_gm = asctile.global_tensor(out_ptr, [1])
    tile = asctile.copy_in(x_gm, offsets, tile_shape, real_shape=[real_rows, real_cols], pad_value=float('-inf'))
    max_val = asctile.reduce_max(tile)
    result = asctile.full([1], max_val, dtype=tile.dtype)
    asctile.copy_out(result, out_gm, [0])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("shape, tile_shape, real_shape, offsets", [
    ([2, 16], [2, 128], [2, 15], [0, 14]),
    ([2, 16], [2, 16], [2, 3], [0, 0]),
    ([2, 42], [2, 8], [2, 8], [0, 0]),
    ([2, 42], [2, 8], [2, 5], [0, 0]),
    ([2, 42], [2, 8], [2, 7], [0, 17]),
    ([2, 42], [2, 8], [2, 7], [0, 35]),
    ([2, 42], [2, 16], [2, 9], [0, 0]),
    ([4, 42], [4, 8], [3, 7], [0, 0]),
    ([4, 42], [3, 8], [2, 5], [1, 0]),
    ([4, 42], [2, 8], [2, 5], [1, 17]),
    ([32, 32], [32, 40], [32, 1], [3, 5]),
])
def test_load_real_shape_2d(require_c310, kernel_type, shape, tile_shape, real_shape, offsets):
    require_c310()
    rows, cols = shape
    offset_row, offset_col = offsets
    real_rows, real_cols = real_shape
    x = torch.rand((rows, cols), dtype=torch.float32) * 10.0
    valid_end_col = offset_col + real_cols
    tile_end_col = offset_col + tile_shape[1]
    if tile_end_col <= cols:
        x[:, valid_end_col:tile_end_col] = 1000.0
    out = torch.empty(1, dtype=torch.float32)
    real_rows_arg = asctile.ConstExpr(real_rows) if kernel_type == STATIC else real_rows
    real_cols_arg = asctile.ConstExpr(real_cols) if kernel_type == STATIC else real_cols
    load_real_shape_2d_kernel[1](x, out, rows, cols, real_rows_arg, real_cols_arg, tile_shape, offsets)
    expected_region = x[offset_row:offset_row + real_rows, offset_col:offset_col + real_cols]
    expected = torch.amax(expected_region).unsqueeze(0)
    torch.testing.assert_close(out, expected)


@asctile.jit(always_compile=True)
def store_real_shape_2d_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
                               out_ptr: asctile.GlobalAddress, in_rows: int, in_cols: int, out_rows: int, out_cols: int,
                               real_rows: int, real_cols: int, tile_shape: asctile.ConstExpr,
                               offsets: asctile.ConstExpr):
    x_gm = asctile.global_tensor(x_ptr, [in_rows, in_cols])
    y_gm = asctile.global_tensor(y_ptr, [in_rows, in_cols])
    out_gm = asctile.global_tensor(out_ptr, [out_rows, out_cols])
    x_tile = asctile.copy_in(x_gm, offsets, tile_shape)
    y_tile = asctile.copy_in(y_gm, offsets, tile_shape)
    result = x_tile + y_tile
    asctile.copy_out(result, out_gm, [0, 0], real_shape=[real_rows, real_cols])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("input_shape, output_shape, tile_shape, real_shape, offsets", [
    ([2, 8], [2, 8], [2, 8], [2, 8], [0, 0]),
    ([2, 8], [2, 8], [2, 8], [2, 5], [0, 0]),
    ([2, 42], [2, 5], [2, 8], [2, 5], [0, 0]),
    ([4, 32], [4, 32], [2, 8], [2, 5], [0, 0]),
    ([4, 32], [4, 32], [2, 8], [2, 5], [2, 8]),
    ([4, 16], [4, 16], [4, 16], [3, 9], [0, 0]),
    ([5, 16], [5, 16], [2, 8], [2, 5], [2, 3]),
])
def test_store_real_shape_2d(require_c310, kernel_type, input_shape, output_shape, tile_shape, real_shape, offsets):
    require_c310()
    in_rows, in_cols = input_shape
    out_rows, out_cols = output_shape
    real_rows, real_cols = real_shape
    offset_row, offset_col = offsets
    x = torch.rand((in_rows, in_cols), dtype=torch.float32) * 1.0
    y = torch.rand((in_rows, in_cols), dtype=torch.float32) * 2.0
    out = torch.full((out_rows, out_cols), 1000.0, dtype=torch.float32)
    real_rows_arg = asctile.ConstExpr(real_rows) if kernel_type == STATIC else real_rows
    real_cols_arg = asctile.ConstExpr(real_cols) if kernel_type == STATIC else real_cols
    store_real_shape_2d_kernel[1](x, y, out, in_rows, in_cols, out_rows, out_cols, real_rows_arg, real_cols_arg,
                                  tile_shape, offsets)
    x_region = x[offset_row:offset_row + tile_shape[0], offset_col:offset_col + tile_shape[1]]
    y_region = y[offset_row:offset_row + tile_shape[0], offset_col:offset_col + tile_shape[1]]
    expected = torch.full((out_rows, out_cols), 1000.0, dtype=torch.float32)
    computed = x_region[:real_rows, :real_cols] + y_region[:real_rows, :real_cols]
    expected[:real_rows, :real_cols] = computed
    torch.testing.assert_close(out, expected)
