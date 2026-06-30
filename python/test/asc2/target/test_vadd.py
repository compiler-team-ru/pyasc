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
def add(input_x_ptr: asc2.GlobalAddress, input_y_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length,
        tile_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    x = asc2.global_tensor(input_x_ptr, [input_length])
    y = asc2.global_tensor(input_y_ptr, [input_length])
    z = asc2.global_tensor(output_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor, parallel=True):
        current_offset = block_offset + i * tile_length
        xt = asc2.load(x, [current_offset], [tile_length])
        yt = asc2.load(y, [current_offset], [tile_length])
        zt = xt + yt
        asc2.store(zt, z, [current_offset])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("block_num, unroll_factor, input_shape, in_out_dtype, tiling_key, tiling_values", [
    (36, 2, [9216], torch.float32, 8, [9216, 0, 128, 36]),
    (35, 2, [8732], torch.float32, 8, [8732, 0, 128, 35]),
    (32, 2, [8192], torch.float32, 8, [8192, 0, 128, 32]),
    (47, 2, [979139], torch.float32, 8, [979139, 0, 10496, 47]),
    (29, 2, [87768], torch.float32, 8, [87768, 0, 1536, 29]),
    (29, 2, [395520], torch.float32, 8, [395520, 0, 7040, 29]),
    (54, 1, [6691304], torch.float32, 8, [6691304, 0, 10496, 54]),
    (56, 1, [5224328], torch.float32, 8, [5224328, 0, 10496, 56]),
])
def test_add(profiler, runs, kernel_type, block_num, unroll_factor, input_shape, in_out_dtype, tiling_key,
             tiling_values):
    input_shape_1d = [math.prod(input_shape)]
    _, _, tile_length, block_num = tiling_values

    in_tensor_x = torch.randn(input_shape_1d, dtype=in_out_dtype)
    in_tensor_y = torch.randn(input_shape_1d, dtype=in_out_dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=in_out_dtype)

    params = [in_tensor_x, in_tensor_y, out_tensor]
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(input_shape_1d[0]))
    else:
        params.append(input_shape_1d[0])
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            add[block_num](*params)

    expected = torch.add(in_tensor_x, in_tensor_y)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
