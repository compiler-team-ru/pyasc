# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import pytest
import torch

import asc
import asc2
from asc.runtime import config


@asc2.jit(static_alloc=True, reuse_ub=True)
def select_kernel_1D(cond_ptr: asc.GlobalAddress, input_x_ptr: asc.GlobalAddress, input_y_ptr: asc.GlobalAddress,
                     output_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr, output_shape: asc.ConstExpr,
                     block_loop_num: asc.ConstExpr, block_loop_num_tail: asc.ConstExpr, tile_length: asc.ConstExpr,
                     block_length: asc.ConstExpr, unroll_factor: asc.ConstExpr):
    c = asc2.tensor(cond_ptr, input_shape)
    x = asc2.tensor(input_x_ptr, input_shape)
    y = asc2.tensor(input_y_ptr, input_shape)
    z = asc2.tensor(output_ptr, output_shape)

    block_offset = asc2.block_idx() * block_length
    loop_count = block_loop_num
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_loop_num_tail

    for i in asc2.range(loop_count, unroll_factor=unroll_factor, parallel=True):
        current_offset = block_offset + i * tile_length
        ct = asc2.load(c, [tile_length], offsets=[current_offset])
        xt = asc2.load(x, [tile_length], offsets=[current_offset])
        yt = asc2.load(y, [tile_length], offsets=[current_offset])
        zt = asc2.where(ct != 0, xt, yt)
        asc2.store(zt, z, offsets=[current_offset])


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, tiling_key, tiling_values", [
        (2, 2, [1920], torch.int32, [1920], torch.int32, 8, [1920, 0, 512, 2]),
        (15, 2, [1920, 1, 8], torch.float32, [1920, 1, 8], torch.float32, 8, [15360, 0, 512, 15]),
        (16, 2, [256, 4, 1, 16], torch.float32, [256, 4, 1, 16], torch.float32, 8, [16384, 0, 512, 16]),
        (13, 2, [256, 50], torch.float32, [256, 50], torch.float32, 8, [12800, 0, 512, 13]),
        (30, 2, [1920, 1, 40], torch.float32, [1920, 1, 40], torch.float32, 8, [76800, 0, 1280, 30]),
        (28, 2, [28442], torch.float32, [28442], torch.float32, 8, [28442, 0, 512, 28]),
        (29, 2, [327461], torch.float32, [327461], torch.float32, 8, [327461, 0, 5760, 29]),
        (50, 2, [2, 1, 256, 256, 16], torch.float32, [2, 1, 256, 256, 16], torch.float32, 8, [2097152, 0, 7040, 50]),
        (50, 1, [2, 1, 256, 256, 16], torch.float16, [2, 1, 256, 256, 16], torch.float16, 8, [2097152, 0, 14080, 50]),
    ])
def test_select(backend, platform, device_id, profiler, runs, core_num, unroll_factor, input_shape, input_dtype,
                output_shape, output_dtype, tiling_key, tiling_values):
    config.set_platform(backend, platform, device_id)
    _, _, tile_length, core_num = tiling_values
    input_shape_1d = [math.prod(input_shape)]
    length = input_shape_1d[0]

    ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
    tile_length = asc.ceildiv(tile_length, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS

    block_loop_num = asc.ceildiv(asc.ceildiv(length, core_num), tile_length)
    block_length = tile_length * block_loop_num
    block_loop_num_tail = asc.ceildiv(length - block_length * (core_num - 1), tile_length)
    padded_length = block_length * (core_num - 1) + tile_length * block_loop_num_tail
    padded_input_shape = [padded_length]
    padded_output_shape = padded_input_shape

    in_tensor_c = torch.zeros(padded_input_shape, dtype=torch.int32)
    in_tensor_c[:length] = torch.randint(0, 2, input_shape_1d, dtype=torch.int32)
    in_tensor_x = torch.zeros(padded_input_shape, dtype=input_dtype)
    in_tensor_x[:length] = torch.randn(input_shape_1d).to(input_dtype)
    in_tensor_y = torch.zeros(padded_input_shape, dtype=input_dtype)
    in_tensor_y[:length] = torch.randn(input_shape_1d).to(input_dtype)
    out_tensor = torch.zeros(padded_output_shape, dtype=output_dtype)

    with profiler.profile():
        for run in range(runs):
            select_kernel_1D[core_num](in_tensor_c, in_tensor_x, in_tensor_y, out_tensor, padded_input_shape,
                                       padded_output_shape, block_loop_num, block_loop_num_tail, tile_length,
                                       block_length, unroll_factor)

    expected = torch.where(in_tensor_c.bool(), in_tensor_x, in_tensor_y)
    torch.testing.assert_close(out_tensor[:length], expected[:length], atol=1e-3, rtol=1e-3)
