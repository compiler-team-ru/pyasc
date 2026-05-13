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


# static_alloc=False due to bug
@asc2.jit(static_alloc=False, reuse_ub=True)
def bcast_last_dim(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr,
                   output_shape: asc.ConstExpr, tile_shape: asc.ConstExpr, ROW_PER_BLOCK: asc.ConstExpr,
                   UNROLL_FACTOR: asc.ConstExpr):
    input_gm = asc2.tensor(input_ptr, input_shape)
    output_gm = asc2.tensor(output_ptr, output_shape)

    start_offset = asc2.block_idx() * ROW_PER_BLOCK
    ROW_ITERS_PER_BLOCK = asc.ceildiv(ROW_PER_BLOCK, tile_shape[0])
    COL_ITERS_PER_BLOCK = asc.ceildiv(output_shape[1], tile_shape[1])
    for i in asc2.range(ROW_ITERS_PER_BLOCK, parallel=True, unroll_factor=UNROLL_FACTOR):
        row_start_offset = start_offset + i * tile_shape[0]
        tensor_part = asc2.load(input_gm, [tile_shape[0]], offsets=[row_start_offset]).reshape(tile_shape[0], 1)
        # TODO: Use it after padding for asc2.load is fixed:
        # tensor_part = asc2.load(input_gm, [tile_shape[0], 1], offsets=[row_start_offset, 0])
        res = tensor_part.broadcast_to(tile_shape[0], tile_shape[1])
        for j in asc2.range(COL_ITERS_PER_BLOCK, parallel=False):
            asc2.store(res, output_gm, offsets=[row_start_offset, j * tile_shape[1]])


@asc2.jit(static_alloc=False, reuse_ub=True)
def bcast_scalar(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr,
                 output_shape: asc.ConstExpr, tile_shape: asc.ConstExpr, COL_PER_BLOCK: asc.ConstExpr,
                 UNROLL_FACTOR: asc.ConstExpr):
    input_gm = asc2.tensor(input_ptr, input_shape)
    output_gm = asc2.tensor(output_ptr, output_shape)

    start_offset = asc2.block_idx() * COL_PER_BLOCK
    COL_ITERS_PER_BLOCK = asc.ceildiv(COL_PER_BLOCK, tile_shape[1])
    for i in asc2.range(COL_ITERS_PER_BLOCK, parallel=True, unroll_factor=UNROLL_FACTOR):
        scalar = asc2.load(input_gm, offsets=[0] * len(input_shape))
        res = asc2.full(tile_shape, scalar)
        asc2.store(res, output_gm, offsets=[0, start_offset + i * tile_shape[1]])


@asc2.jit(static_alloc=False, reuse_ub=True)
def bcast_first(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr,
                output_shape: asc.ConstExpr, tile_shape: asc.ConstExpr, ROW_PER_BLOCK: asc.ConstExpr,
                UNROLL_FACTOR: asc.ConstExpr):
    input_gm = asc2.tensor(input_ptr, input_shape)
    output_gm = asc2.tensor(output_ptr, output_shape)

    start_offset = asc2.block_idx() * ROW_PER_BLOCK
    ROW_ITERS_PER_BLOCK = asc.ceildiv(ROW_PER_BLOCK, tile_shape[0])
    COL_ITERS_PER_BLOCK = asc.ceildiv(output_shape[1], tile_shape[1])
    for j in asc2.range(COL_ITERS_PER_BLOCK, parallel=True, unroll_factor=UNROLL_FACTOR):
        col_start_offset = j * tile_shape[1]
        tensor_part = asc2.load(input_gm, [tile_shape[1]], offsets=[col_start_offset])
        # TODO: Should this work or not?
        # tensor_part = asc2.load(input_gm, [1, tile_shape[1]], offsets=[0, col_start_offset])
        res = tensor_part.broadcast_to(tile_shape[0], tile_shape[1])
        for i in asc2.range(ROW_ITERS_PER_BLOCK, parallel=True):
            asc2.store(res, output_gm, offsets=[start_offset + i * tile_shape[0], col_start_offset])


def get_broadcast_axes(input_shape, output_shape):
    rank_diff = len(output_shape) - len(input_shape)
    assert rank_diff >= 0, f"Input rank={len(input_shape)} cannot exceed output rank={len(output_shape)}"
    # Add new dimensions to the left
    padded_input = [1] * rank_diff + input_shape
    broadcast_axis = []
    for i, (in_dim, out_dim) in enumerate(zip(padded_input, output_shape)):
        if in_dim != out_dim:
            assert in_dim == 1, f"Incompatible: {in_dim} cannot broadcast to {out_dim}"
            broadcast_axis.append(i)
    return broadcast_axis, padded_input


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, tiling_key, tiling_values",
    [(3, 2, [1, 1], torch.float32, [1, 160], torch.float32, 11003, [
        11003, 1, 0, 1, 1, 2, 64, 3, 1, 1, 1, 1, 64, 32, 1, 64, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],
        [64, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),  # bcast scalar
     (1, 2, [1, 1], torch.float32, [1, 8], torch.float32, 11001, [
         11001, 1, 0, 1, 1, 0, 64, 1, 1, 1, 1, 1, 8, 8, 1, 8, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],
         [8, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (43, 2, [1], torch.float32, [8192], torch.float32, 11003, [
         11003, 1, 0, 1, 1, 2, 192, 43, 1, 1, 1, 1, 192, 128, 1, 192, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [192, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (32, 2, [1], torch.float32, [4096], torch.float32, 11003, [
         11003, 1, 0, 1, 1, 2, 128, 32, 1, 1, 1, 1, 128, 128, 1, 128, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [128, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (48, 2, [1], torch.bfloat16, [8192, 3], torch.bfloat16, 11003, [
         11003, 1, 0, 1, 1, 2, 512, 48, 1, 1, 1, 1, 512, 512, 1, 512, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [512, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (2, 2, [1, 8, 1], torch.bfloat16, [1, 8, 20], torch.bfloat16, 11001, [
         11001, 1, 0, 2, 1, 2, 128, 2, 1, 1, 1, 1, 4, 4, 1, 4, 1, 20, 1, 1, 0, 0, [1, 0, 0, 0, 0], [20, 1, 0, 0, 0],
         [4, 20, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     (32, 2, [1, 1280], torch.float32, [760, 1280], torch.float32, 11001, [
         11001, 1, 0, 2, 1, 2, 15360, 32, 1, 1, 1, 1, 24, 16, 1, 12, 1280, 1280, 0, 0, 0, 0, [0, 1, 0, 0, 0],
         [1280, 1, 0, 0, 0], [12, 1280, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by first dim row per block
     (56, 2, [1623, 1], torch.float32, [1623, 512], torch.float32, 11001, [
         11001, 1, 0, 2, 1, 2, 14848, 56, 1, 1, 1, 1, 29, 28, 1, 29, 1, 512, 1, 1, 0, 0, [1, 0, 0, 0, 0],
         [512, 1, 0, 0, 0], [29, 512, 1, 1, 1], [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     (56, 2, [1, 1], torch.float32, [1, 1048576], torch.float32, 11003, [
         11003, 1, 0, 1, 1, 2, 9376, 56, 1, 1, 1, 1, 18752, 17216, 1, 9376, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [9376, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (56, 2, [1048576, 1], torch.float32, [1048576, 4], torch.float32, 11000, [
         11000, 1, 0, 2, 2, 2, 37472, 56, 1, 1, 1, 1, 18736, 18096, 1, 9368, 1, 4, 1, 1, 0, 0, [1, 0, 0, 0, 0],
         [4, 1, 0, 0, 0], [9368, 4, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     (56, 2, [1, 336], torch.float32, [3360, 336], torch.float32, 11001, [
         11001, 1, 0, 2, 1, 2, 10112, 56, 1, 1, 1, 1, 60, 60, 1, 30, 336, 336, 0, 0, 0, 0, [0, 1, 0, 0, 0],
         [336, 1, 0, 0, 0], [30, 336, 1, 1, 1], [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by first dim row per block
     (48, 2, [1427, 1], torch.float32, [1427, 1427], torch.float32, 11001, [
         11001, 1, 0, 2, 2, 2, 15744, 48, 1, 1, 1, 1, 30, 17, 1, 10, 1, 1427, 1, 1, 0, 0, [1, 0, 0, 0, 0],
         [1427, 1, 0, 0, 0], [10, 1427, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     ])
def test_reduce(backend, platform, device_id, profiler, runs, core_num, unroll_factor, input_shape, input_dtype,
                output_shape, output_dtype, tiling_key, tiling_values):
    config.set_platform(backend, platform, device_id)
    is_scalar_input = torch.prod(torch.tensor(input_shape)).item() == 1
    tilingKey, _, _, _, bufferCnt, _, _, _, _, _, _, _, _, _, _, uLpUnit, _, uOutOffset, _, _, _, _, _, _, _, _, _ = tiling_values
    tile_shape = [1, uLpUnit] if is_scalar_input else [uLpUnit, uOutOffset]

    # Convert any shape to 2D
    if is_scalar_input:
        input_shape = [1, 1]
        output_shape = [1, torch.prod(torch.tensor(output_shape)).item()]
    else:
        broadcast_axis, padded_input = get_broadcast_axes(input_shape, output_shape)
        if broadcast_axis[0] == 0:
            num_cols = torch.prod(torch.tensor(output_shape[1:])).item()
            output_shape = [output_shape[0], num_cols]
            num_cols = torch.prod(torch.tensor(padded_input[1:])).item()
            input_shape = [padded_input[0], num_cols]
        elif broadcast_axis[0] == len(input_shape) - 1:
            num_rows = torch.prod(torch.tensor(output_shape[:-1])).item()
            output_shape = [num_rows, output_shape[-1]]
            num_rows = torch.prod(torch.tensor(padded_input[:-1])).item()
            input_shape = [num_rows, padded_input[-1]]
        else:
            assert False, "broadcast for middle dimension(s) is not implemented yet"
    broadcast_axis, padded_input = get_broadcast_axes(input_shape, output_shape)
    assert len(broadcast_axis) == 1, "broadcast for several dimensions is not implemented yet"
    axis = broadcast_axis[0]

    output_shape_2d = output_shape
    num_rows, num_cols = output_shape_2d

    if is_scalar_input:
        # Alignment
        COL_PER_BLOCK = asc.ceildiv(num_cols, core_num)
        num_cols_padded = COL_PER_BLOCK * core_num
        padded_output_shape = [num_rows, num_cols_padded]

        in_tensor = torch.arange(1, dtype=input_dtype) + 1
        out_tensor = torch.ones(padded_output_shape, dtype=output_dtype)
        with profiler.profile():
            for run in range(runs):
                bcast_scalar[core_num](in_tensor, out_tensor, input_shape, padded_output_shape, tile_shape,
                                       COL_PER_BLOCK, unroll_factor)
        expected = torch.broadcast_to(in_tensor, padded_output_shape)
    else:
        # Alignment
        ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
        ROW_PER_BLOCK = asc.ceildiv(num_rows, core_num)

        if input_shape == [1048576, 1]:
            # TODO: Why the original tile_shape doesn't work? no space left on UB after padding?
            tile_shape = [tile_shape[0] // 3, tile_shape[1]]

        num_rows_padded = max(ROW_PER_BLOCK, tile_shape[0]) * core_num
        tile_shape = [tile_shape[0], asc.ceildiv(tile_shape[1], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS]
        COL_ITERS = asc.ceildiv(num_cols, tile_shape[1])
        num_cols_padded = COL_ITERS * tile_shape[1]

        input_shape_padded = num_cols_padded if axis == 0 else num_rows_padded
        padded_input_shape = [input_shape_padded]
        padded_output_shape = [num_rows_padded, num_cols_padded]

        in_tensor = torch.ones(input_shape_padded, dtype=input_dtype) * (-1)
        in_tensor[:input_shape[0]] = torch.arange(input_shape[0], dtype=output_dtype) + 1
        out_tensor = torch.ones(padded_output_shape, dtype=output_dtype)

        kernel_impl = bcast_first if axis == 0 else bcast_last_dim
        with profiler.profile():
            for run in range(runs):
                kernel_impl[core_num](in_tensor, out_tensor, padded_input_shape, padded_output_shape, tile_shape,
                                      ROW_PER_BLOCK, unroll_factor)
        to_reshape = [1, padded_input_shape[0]] if axis == 0 else [padded_input_shape[0], 1]
        reshaped_in_tensor = in_tensor.reshape(to_reshape)
        expected = torch.broadcast_to(reshaped_in_tensor, padded_output_shape)
    torch.testing.assert_close(out_tensor[:num_rows, :num_cols], expected[:num_rows, :num_cols], atol=1e-3, rtol=1e-3)
