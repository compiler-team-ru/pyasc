# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import pytest
import torch

STATIC = "static"
DYNAMIC = "dynamic"


@pytest.fixture(autouse=True)
def set_platform(backend: asc2.Backend, platform: asc2.Platform, device_id: int):
    asc2.set_platform(backend, platform, device_id, check=False)


@asc2.jit(always_compile=True)
def load_real_shape_1d_kernel(x_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, size: int,
                              tile_shape: asc2.ConstExpr, real_shape: asc2.ConstExpr, offset: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [1])
    tile = asc2.load(x_gm, tile_shape, real_shape=real_shape, offsets=offset, pad_value=float('-inf'))
    max_val = asc2.reduce_max(tile)
    result = asc2.full([1], max_val, dtype=tile.dtype)
    asc2.store(result, out_gm, offsets=[0])


@asc2.jit(always_compile=True)
def load_real_shape_1d_dynamic_kernel(x_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, size: int, real_size: int,
                                      tile_shape: asc2.ConstExpr, offset: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [1])
    tile = asc2.load(x_gm, tile_shape, real_shape=[real_size], offsets=offset, pad_value=float('-inf'))
    max_val = asc2.reduce_max(tile)
    result = asc2.full([1], max_val, dtype=tile.dtype)
    asc2.store(result, out_gm, offsets=[0])


load_1d_test_cases = [
    (16, 128, 15, 14),
    (16, 16, 8, 0),
    (16, 16, 5, 3),
    (32, 16, 10, 5),
    (42, 8, 5, 17),
]


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("shape, tile_shape, real_shape, offset", load_1d_test_cases)
def test_load_real_shape_1d(kernel_type, shape, tile_shape, real_shape, offset):
    torch.manual_seed(42)

    x = torch.arange(1, shape + 1, dtype=torch.float32)

    valid_end = offset + real_shape
    tile_end = offset + tile_shape
    if tile_end <= shape:
        x[valid_end:tile_end] = 1000.0

    out = torch.empty(1, dtype=torch.float32)

    if kernel_type == STATIC:
        load_real_shape_1d_kernel[1](x, out, size=shape, tile_shape=[tile_shape], real_shape=[real_shape],
                                     offset=[offset])
    else:
        load_real_shape_1d_dynamic_kernel[1](x, out, size=shape, real_size=real_shape, tile_shape=[tile_shape],
                                             offset=[offset])

    expected_region = x[offset:offset + real_shape]
    expected = torch.amax(expected_region).unsqueeze(0)
    torch.testing.assert_close(out, expected)


@asc2.jit(always_compile=True)
def load_real_shape_kernel(x_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, rows: int, cols: int,
                           tile_shape: asc2.ConstExpr, real_shape: asc2.ConstExpr, offsets: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [rows, cols])
    out_gm = asc2.tensor(out_ptr, [1])
    tile = asc2.load(x_gm, tile_shape, real_shape=real_shape, offsets=offsets, pad_value=float('-inf'))
    max_val = asc2.reduce_max(tile)
    result = asc2.full([1], max_val, dtype=tile.dtype)
    asc2.store(result, out_gm, offsets=[0])


@asc2.jit(always_compile=True)
def load_real_shape_dynamic_kernel(x_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, rows: int, cols: int,
                                   real_rows: int, real_cols: int, tile_shape: asc2.ConstExpr, offsets: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [rows, cols])
    out_gm = asc2.tensor(out_ptr, [1])
    tile = asc2.load(x_gm, tile_shape, real_shape=[real_rows, real_cols], offsets=offsets, pad_value=float('-inf'))
    max_val = asc2.reduce_max(tile)
    result = asc2.full([1], max_val, dtype=tile.dtype)
    asc2.store(result, out_gm, offsets=[0])


load_test_cases = [
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
]


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("shape, tile_shape, real_shape, offsets", load_test_cases)
def test_load_real_shape(platform, require_c310, kernel_type, shape, tile_shape, real_shape, offsets):
    require_c310(platform)
    torch.manual_seed(42)

    rows, cols = shape
    offset_row, offset_col = offsets
    real_rows, real_cols = real_shape

    x = torch.rand((rows, cols), dtype=torch.float32) * 10.0

    valid_end_col = offset_col + real_cols
    tile_end_col = offset_col + tile_shape[1]
    if tile_end_col <= cols:
        x[:, valid_end_col:tile_end_col] = 1000.0

    out = torch.empty(1, dtype=torch.float32)

    if kernel_type == STATIC:
        load_real_shape_kernel[1](x, out, rows=rows, cols=cols, tile_shape=tile_shape, real_shape=real_shape,
                                  offsets=offsets)
    else:
        load_real_shape_dynamic_kernel[1](x, out, rows=rows, cols=cols, real_rows=real_rows, real_cols=real_cols,
                                          tile_shape=tile_shape, offsets=offsets)

    expected_region = x[offset_row:offset_row + real_rows, offset_col:offset_col + real_cols]
    expected = torch.amax(expected_region).unsqueeze(0)
    torch.testing.assert_close(out, expected)


@asc2.jit(always_compile=True)
def store_real_shape_kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress,
                            in_rows: int, in_cols: int, out_rows: int, out_cols: int, tile_shape: asc2.ConstExpr,
                            real_shape: asc2.ConstExpr, offsets: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [in_rows, in_cols])
    y_gm = asc2.tensor(y_ptr, [in_rows, in_cols])
    out_gm = asc2.tensor(out_ptr, [out_rows, out_cols])
    x_tile = asc2.load(x_gm, tile_shape, offsets=offsets)
    y_tile = asc2.load(y_gm, tile_shape, offsets=offsets)
    result = x_tile + y_tile
    asc2.store(result, out_gm, real_shape=real_shape, offsets=[0, 0])


@asc2.jit(always_compile=True)
def store_real_shape_dynamic_kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress,
                                    in_rows: int, in_cols: int, out_rows: int, out_cols: int, real_rows: int,
                                    real_cols: int, tile_shape: asc2.ConstExpr, offsets: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [in_rows, in_cols])
    y_gm = asc2.tensor(y_ptr, [in_rows, in_cols])
    out_gm = asc2.tensor(out_ptr, [out_rows, out_cols])
    x_tile = asc2.load(x_gm, tile_shape, offsets=offsets)
    y_tile = asc2.load(y_gm, tile_shape, offsets=offsets)
    result = x_tile + y_tile
    asc2.store(result, out_gm, real_shape=[real_rows, real_cols], offsets=[0, 0])


store_test_cases = [
    ([2, 8], [2, 8], [2, 8], [2, 8], [0, 0]),
    ([2, 8], [2, 8], [2, 8], [2, 5], [0, 0]),
    ([2, 42], [2, 5], [2, 8], [2, 5], [0, 0]),
    ([4, 32], [4, 32], [2, 8], [2, 5], [0, 0]),
    ([4, 32], [4, 32], [2, 8], [2, 5], [2, 8]),
    ([4, 16], [4, 16], [4, 16], [3, 9], [0, 0]),
    ([5, 16], [5, 16], [2, 8], [2, 5], [2, 3]),
]


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("input_shape, output_shape, tile_shape, real_shape, offsets", store_test_cases)
def test_store_real_shape(platform, require_c310, kernel_type, input_shape, output_shape, tile_shape, real_shape,
                          offsets):
    require_c310(platform)
    torch.manual_seed(42)

    in_rows, in_cols = input_shape
    out_rows, out_cols = output_shape
    real_rows, real_cols = real_shape
    offset_row, offset_col = offsets

    x = torch.rand((in_rows, in_cols), dtype=torch.float32) * 1.0
    y = torch.rand((in_rows, in_cols), dtype=torch.float32) * 2.0
    out = torch.full((out_rows, out_cols), 1000.0, dtype=torch.float32)

    if kernel_type == STATIC:
        store_real_shape_kernel[1](x, y, out, in_rows=in_rows, in_cols=in_cols, out_rows=out_rows, out_cols=out_cols,
                                   tile_shape=tile_shape, real_shape=real_shape, offsets=offsets)
    else:
        store_real_shape_dynamic_kernel[1](x, y, out, in_rows=in_rows, in_cols=in_cols, out_rows=out_rows,
                                           out_cols=out_cols, real_rows=real_rows, real_cols=real_cols,
                                           tile_shape=tile_shape, offsets=offsets)

    x_region = x[offset_row:offset_row + tile_shape[0], offset_col:offset_col + tile_shape[1]]
    y_region = y[offset_row:offset_row + tile_shape[0], offset_col:offset_col + tile_shape[1]]
    expected = torch.full((out_rows, out_cols), 1000.0, dtype=torch.float32)
    computed = x_region[:real_rows, :real_cols] + y_region[:real_rows, :real_cols]
    expected[:real_rows, :real_cols] = computed
    torch.testing.assert_close(out, expected)
