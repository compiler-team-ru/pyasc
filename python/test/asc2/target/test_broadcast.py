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


@asc2.jit(static_alloc=False, reuse_ub=True)
def broadcast_scalar(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, output_num_rows,
                     output_num_cols, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.tensor(input_ptr, [input_length])
    out_gm = asc2.tensor(output_ptr, [output_num_rows, output_num_cols])

    cols_per_block = asc2.ceildiv(output_num_cols, asc2.block_num())
    start_offset = asc2.block_idx() * cols_per_block
    column_iters = asc2.ceildiv(cols_per_block, tile_shape[1])

    for i in asc2.range(column_iters, parallel=True, unroll_factor=unroll_factor):
        scalar = asc2.load(in_gm, offsets=[0])
        res = asc2.full(tile_shape, scalar)
        asc2.store(res, out_gm, offsets=[0, start_offset + i * tile_shape[1]])


@asc2.jit(static_alloc=False, reuse_ub=True)
def broadcast_first_dim(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, output_num_rows,
                        output_num_cols, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.tensor(input_ptr, [input_length])
    out_gm = asc2.tensor(output_ptr, [output_num_rows, output_num_cols])

    rows_per_block = asc2.ceildiv(output_num_rows, asc2.block_num())
    start_offset = asc2.block_idx() * rows_per_block
    row_iters = asc2.ceildiv(rows_per_block, tile_shape[0])

    column_iters = asc2.ceildiv(output_num_cols, tile_shape[1])

    for j in asc2.range(column_iters, parallel=True, unroll_factor=unroll_factor):
        col_start_offset = j * tile_shape[1]
        tensor_part = asc2.load(in_gm, [tile_shape[1]], offsets=[col_start_offset])
        res = tensor_part.broadcast_to(tile_shape[0], tile_shape[1])
        for i in asc2.range(row_iters, parallel=True):
            asc2.store(res, out_gm, offsets=[start_offset + i * tile_shape[0], col_start_offset])


# static_alloc=False due to bug
@asc2.jit(static_alloc=False, reuse_ub=True)
def broadcast_last_dim(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_length, output_num_rows,
                       output_num_cols, tile_shape: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.tensor(input_ptr, [input_length])
    out_gm = asc2.tensor(output_ptr, [output_num_rows, output_num_cols])

    rows_per_block = asc2.ceildiv(output_num_rows, asc2.block_num())
    start_offset = asc2.block_idx() * rows_per_block
    row_iters = asc2.ceildiv(rows_per_block, tile_shape[0])
    column_iters = asc2.ceildiv(output_num_cols, tile_shape[1])

    for i in asc2.range(row_iters, parallel=True, unroll_factor=unroll_factor):
        row_start_offset = start_offset + i * tile_shape[0]
        tensor_part = asc2.load(in_gm, [tile_shape[0]], offsets=[row_start_offset]).reshape(tile_shape[0], 1)
        res = tensor_part.broadcast_to(tile_shape[0], tile_shape[1])
        for j in asc2.range(column_iters, parallel=False):
            asc2.store(res, out_gm, offsets=[row_start_offset, j * tile_shape[1]])


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


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize(
    "block_num, unroll_factor, input_shape, in_out_dtype, output_shape, tiling_key, tiling_values",
    [(3, 2, [1, 1], torch.float32, [1, 160], 11003, [
        11003, 1, 0, 1, 1, 2, 64, 3, 1, 1, 1, 1, 64, 32, 1, 64, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],
        [64, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),  # bcast scalar
     (1, 2, [1, 1], torch.float32, [1, 8], 11001, [
         11001, 1, 0, 1, 1, 0, 64, 1, 1, 1, 1, 1, 8, 8, 1, 8, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],
         [8, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (43, 2, [1], torch.float32, [8192], 11003, [
         11003, 1, 0, 1, 1, 2, 192, 43, 1, 1, 1, 1, 192, 128, 1, 192, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [192, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (32, 2, [1], torch.float32, [4096], 11003, [
         11003, 1, 0, 1, 1, 2, 128, 32, 1, 1, 1, 1, 128, 128, 1, 128, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [128, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (48, 2, [1], torch.bfloat16, [8192, 3], 11003, [
         11003, 1, 0, 1, 1, 2, 512, 48, 1, 1, 1, 1, 512, 512, 1, 512, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [512, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (2, 2, [1, 8, 1], torch.bfloat16, [1, 8, 20], 11001, [
         11001, 1, 0, 2, 1, 2, 128, 2, 1, 1, 1, 1, 4, 4, 1, 4, 1, 20, 1, 1, 0, 0, [1, 0, 0, 0, 0], [20, 1, 0, 0, 0],
         [4, 20, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     (32, 2, [1, 1280], torch.float32, [760, 1280], 11001, [
         11001, 1, 0, 2, 1, 2, 15360, 32, 1, 1, 1, 1, 24, 16, 1, 12, 1280, 1280, 0, 0, 0, 0, [0, 1, 0, 0, 0],
         [1280, 1, 0, 0, 0], [12, 1280, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by first dim row per block
     (56, 2, [1623, 1], torch.float32, [1623, 512], 11001, [
         11001, 1, 0, 2, 1, 2, 14848, 56, 1, 1, 1, 1, 29, 28, 1, 29, 1, 512, 1, 1, 0, 0, [1, 0, 0, 0, 0],
         [512, 1, 0, 0, 0], [29, 512, 1, 1, 1], [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     (56, 2, [1, 1], torch.float32, [1, 1048576], 11003, [
         11003, 1, 0, 1, 1, 2, 9376, 56, 1, 1, 1, 1, 18752, 17216, 1, 9376, 1, 1, 0, 1, 0, 0, [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0], [9376, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast scalar
     (56, 2, [1048576, 1], torch.float32, [1048576, 4], 11000, [
         11000, 1, 0, 2, 2, 2, 37472, 56, 1, 1, 1, 1, 18736, 18096, 1, 9368, 1, 4, 1, 1, 0, 0, [1, 0, 0, 0, 0],
         [4, 1, 0, 0, 0], [9368, 4, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     (56, 2, [1, 336], torch.float32, [3360, 336], 11001, [
         11001, 1, 0, 2, 1, 2, 10112, 56, 1, 1, 1, 1, 60, 60, 1, 30, 336, 336, 0, 0, 0, 0, [0, 1, 0, 0, 0],
         [336, 1, 0, 0, 0], [30, 336, 1, 1, 1], [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by first dim row per block
     (48, 2, [1427, 1], torch.float32, [1427, 1427], 11001, [
         11001, 1, 0, 2, 2, 2, 15744, 48, 1, 1, 1, 1, 30, 17, 1, 10, 1, 1427, 1, 1, 0, 0, [1, 0, 0, 0, 0],
         [1427, 1, 0, 0, 0], [10, 1427, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]),  # bcast by last dim
     ])
def test_broadcast(profiler, runs, kernel_type, block_num, unroll_factor, input_shape, in_out_dtype, output_shape,
                   tiling_key, tiling_values):
    is_scalar_input = math.prod(input_shape) == 1
    tilingKey, _, _, _, bufferCnt, _, _, _, _, _, _, _, _, _, _, uLpUnit, _, uOutOffset, _, _, _, _, _, _, _, _, _ = tiling_values
    tile_shape = [1, uLpUnit] if is_scalar_input else [uLpUnit, uOutOffset]
    if tiling_key in {11000, 11004}:
        unroll_factor = 1
    elif tiling_key == 11002:
        unroll_factor = 2
    else:
        unroll_factor = bufferCnt

    # Alignment for tile_shape
    ALIGNMENT_ELEMENTS = 32 // in_out_dtype.itemsize
    tile_shape = [tile_shape[0], asc2.ceildiv(tile_shape[1], ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS]

    if is_scalar_input:
        input_shape_2d = [1, 1]
        output_shape_2d = [1, math.prod(output_shape)]
    else:
        broadcast_axis, padded_input = get_broadcast_axes(input_shape, output_shape)
        if broadcast_axis[0] == 0:
            num_cols = math.prod(output_shape[1:])
            output_shape_2d = [output_shape[0], num_cols]
            num_cols = math.prod(padded_input[1:])
            input_shape_2d = [padded_input[0], num_cols]
        elif broadcast_axis[0] == len(input_shape) - 1:
            num_rows = math.prod(output_shape[:-1])
            output_shape_2d = [num_rows, output_shape[-1]]
            num_rows = math.prod(padded_input[:-1])
            input_shape_2d = [num_rows, padded_input[-1]]
        else:
            raise NotImplementedError("Broadcast for middle dimension(s) is not implemented yet")
    broadcast_axis, padded_input = get_broadcast_axes(input_shape_2d, output_shape_2d)
    if len(broadcast_axis) != 1:
        raise NotImplementedError("Broadcast for several dimensions is not implemented yet")
    axis = broadcast_axis[0]

    if is_scalar_input:
        in_tensor = torch.arange(1, dtype=in_out_dtype) + 1
        out_tensor = torch.ones(output_shape_2d, dtype=in_out_dtype)

        params = [in_tensor, out_tensor]
        if kernel_type == STATIC:
            params.extend([asc2.ConstExpr(1), asc2.ConstExpr(output_shape_2d[0]), asc2.ConstExpr(output_shape_2d[1])])
        else:
            params.extend([1, output_shape_2d[0], output_shape_2d[1]])
        params.extend([tile_shape, unroll_factor])

        with profiler.profile():
            for _ in range(runs):
                broadcast_scalar[block_num](*params)
        expected = torch.broadcast_to(in_tensor, output_shape_2d)
    else:
        if input_shape == [1048576, 1]:
            tile_shape = [tile_shape[0] // 3, tile_shape[1]]

        num_rows, num_cols = output_shape_2d
        input_length = num_cols if axis == 0 else num_rows
        in_tensor = torch.arange(input_length, dtype=in_out_dtype) + 1
        out_tensor = torch.ones(output_shape_2d, dtype=in_out_dtype)

        params = [in_tensor, out_tensor]
        if kernel_type == STATIC:
            params.extend(
                [asc2.ConstExpr(input_length),
                 asc2.ConstExpr(output_shape_2d[0]),
                 asc2.ConstExpr(output_shape_2d[1])])
        else:
            params.extend([input_length, output_shape_2d[0], output_shape_2d[1]])
        params.extend([tile_shape, unroll_factor])

        kernel_impl = broadcast_first_dim if axis == 0 else broadcast_last_dim
        with profiler.profile():
            for _ in range(runs):
                kernel_impl[block_num](*params)
        to_reshape = [1, input_length] if axis == 0 else [input_length, 1]
        reshaped_in_tensor = in_tensor.reshape(to_reshape)
        expected = torch.broadcast_to(reshaped_in_tensor, output_shape_2d)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
