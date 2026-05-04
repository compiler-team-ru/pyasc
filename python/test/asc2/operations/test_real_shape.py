# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

import asc
import asc2
from asc.runtime import config


@asc2.jit(always_compile=True)
def real_shape_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, rows: int, cols: int,
                      tile_shape: asc.ConstExpr, real_shape: asc.ConstExpr, offsets: asc.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [rows, cols])
    out_gm = asc2.tensor(out_ptr, [1])
    tile = asc2.load(x_gm, tile_shape, real_shape=real_shape, offsets=offsets, pad_value=float('-inf'))
    max_val = asc2.reduce_max(tile)
    result = asc2.full([1], max_val, dtype=tile.dtype)
    asc2.store(result, out_gm, offsets=[0])


test_cases = [
    ([2, 42], [2, 8], [2, 8], [0, 0]),
    ([2, 42], [2, 8], [2, 5], [0, 0]),
    ([2, 42], [2, 8], [2, 7], [0, 17]),
    ([2, 42], [2, 8], [2, 7], [0, 35]),
    ([2, 42], [2, 16], [2, 9], [0, 0]),
    ([4, 42], [4, 8], [3, 7], [0, 0]),
    ([4, 42], [3, 8], [2, 5], [1, 0]),
    ([4, 42], [2, 8], [2, 5], [1, 17]),
]


@pytest.mark.parametrize("shape, tile_shape, real_shape, offsets", test_cases)
def test_real_shape(backend, platform, device_id, shape, tile_shape, real_shape, offsets):
    torch.manual_seed(42)
    config.set_platform(backend, platform, device_id, check=False)

    rows, cols = shape
    offset_row, offset_col = offsets
    real_rows, real_cols = real_shape

    x = torch.rand((rows, cols), dtype=torch.float32) * 10.0

    valid_end_col = offset_col + real_cols
    tile_end_col = offset_col + tile_shape[1]
    if tile_end_col <= cols:
        x[:, valid_end_col:tile_end_col] = 1000.0

    out = torch.empty(1, dtype=torch.float32)
    real_shape_kernel[1](x, out, rows=rows, cols=cols, tile_shape=tile_shape, real_shape=real_shape, offsets=offsets)

    expected_region = x[offset_row:offset_row + real_rows, offset_col:offset_col + real_cols]
    expected = torch.amax(expected_region).unsqueeze(0)
    torch.testing.assert_close(out, expected)
