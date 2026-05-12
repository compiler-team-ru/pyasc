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
def add_kernel_1D(input_x_ptr: asc.GlobalAddress, input_y_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress,
                  input_shape: asc.ConstExpr, output_shape: asc.ConstExpr, block_loop_num: asc.ConstExpr,
                  block_loop_num_tail: asc.ConstExpr, tile_length: asc.ConstExpr, block_length: asc.ConstExpr,
                  UNROLL_FACTOR: asc.ConstExpr):
    x = asc2.tensor(input_x_ptr, input_shape)
    y = asc2.tensor(input_y_ptr, input_shape)
    z = asc2.tensor(output_ptr, output_shape)

    block_offset = asc2.block_idx() * block_length
    loop_count = block_loop_num
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_loop_num_tail

    for i in asc2.range(loop_count, unroll_factor=UNROLL_FACTOR, parallel=True):
        current_offset = block_offset + i * tile_length
        xt = asc2.load(x, [tile_length], offsets=[current_offset])
        yt = asc2.load(y, [tile_length], offsets=[current_offset])
        zt = xt + yt
        asc2.store(zt, z, offsets=[current_offset])


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, tiling_key, tiling_values", [
        (36, 2, [9216], torch.float32, [9216], torch.float32, 8, [9216, 0, 128, 36]),
        (35, 2, [8732], torch.float32, [8732], torch.float32, 8, [8732, 0, 128, 35]),
        (32, 2, [8192], torch.float32, [8192], torch.float32, 8, [8192, 0, 128, 32]),
        (47, 2, [979139], torch.float32, [979139], torch.float32, 8, [979139, 0, 10496, 47]),
        (29, 2, [87768], torch.float32, [87768], torch.float32, 8, [87768, 0, 1536, 29]),
        (29, 2, [395520], torch.float32, [395520], torch.float32, 8, [395520, 0, 7040, 29]),
        (54, 1, [6691304], torch.float32, [6691304], torch.float32, 8, [6691304, 0, 10496, 54]),
        (56, 1, [5224328], torch.float32, [5224328], torch.float32, 8, [5224328, 0, 10496, 56]),
    ])
def test_add(backend, platform, device_id, profiler, runs, core_num, unroll_factor, input_shape, input_dtype,
             output_shape, output_dtype, tiling_key, tiling_values):
    config.set_platform(backend, platform, device_id)
    _, _, tile_length, core_num = tiling_values

    # Convert any shape to 1D
    input_shape_1d = [torch.prod(torch.tensor(input_shape[:])).item()]

    # Alignment
    length = input_shape_1d[0]

    ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
    tile_length = asc.ceildiv(tile_length, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS

    block_loop_num = asc.ceildiv(asc.ceildiv(length, core_num), tile_length)
    block_length = tile_length * block_loop_num
    block_loop_num_tail = asc.ceildiv(length - block_length * (core_num - 1), tile_length)
    padded_length = block_length * (core_num - 1) + tile_length * block_loop_num_tail
    padded_input_shape = [padded_length]
    padded_output_shape = padded_input_shape

    in_tensor_x = torch.full(padded_input_shape, dtype=input_dtype, fill_value=0)
    in_tensor_x[:length] = torch.randn(input_shape_1d, dtype=input_dtype)
    in_tensor_y = torch.full(padded_input_shape, dtype=input_dtype, fill_value=0)
    in_tensor_y[:length] = torch.randn(input_shape_1d, dtype=input_dtype)
    out_tensor = torch.zeros(padded_output_shape, dtype=output_dtype)

    with profiler.profile():
        for run in range(runs):
            add_kernel_1D[core_num](in_tensor_x, in_tensor_y, out_tensor, padded_input_shape, padded_output_shape,
                                    block_loop_num, block_loop_num_tail, tile_length, block_length, unroll_factor)

    expected = torch.add(in_tensor_x, in_tensor_y)
    torch.testing.assert_close(out_tensor[:length], expected[:length], atol=1e-3, rtol=1e-3)
