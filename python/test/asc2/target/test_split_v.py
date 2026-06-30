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

STATIC = "static"
DYNAMIC = "dynamic"


@asc2.jit(static_alloc=True, reuse_ub=True)
def split_v(input_ptr: asc2.GlobalAddress, output0_ptr: asc2.GlobalAddress, output1_ptr: asc2.GlobalAddress,
            input_length, split_boundary, tile_length, ub_chunk_size: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    in_gm = asc2.global_tensor(input_ptr, [input_length])
    out0_gm = asc2.global_tensor(output0_ptr, [split_boundary])
    out1_gm = asc2.global_tensor(output1_ptr, [input_length - split_boundary])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = block_length * asc2.block_idx()
    chunks_per_tile = asc2.ceildiv(tile_length, ub_chunk_size)

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor, parallel=True):
        tile_offset = block_offset + i * tile_length
        for j in asc2.range(chunks_per_tile, parallel=False):
            current_offset = tile_offset + j * ub_chunk_size
            remaining = tile_length - j * ub_chunk_size
            chunk_size = ub_chunk_size if remaining >= ub_chunk_size else remaining
            if current_offset + chunk_size <= split_boundary:
                tile = asc2.copy_in(in_gm, [current_offset], [ub_chunk_size], real_shape=[chunk_size])
                asc2.copy_out(tile, out0_gm, [current_offset], real_shape=[chunk_size])
            elif current_offset >= split_boundary:
                tile = asc2.copy_in(in_gm, [current_offset], [ub_chunk_size], real_shape=[chunk_size])
                asc2.copy_out(tile, out1_gm, [current_offset - split_boundary], real_shape=[chunk_size])
            else:
                part1 = split_boundary - current_offset
                tile1 = asc2.copy_in(in_gm, [current_offset], [ub_chunk_size], real_shape=[part1])
                asc2.copy_out(tile1, out0_gm, [current_offset], real_shape=[part1])
                part2 = chunk_size - part1
                tile2 = asc2.copy_in(in_gm, [split_boundary], [ub_chunk_size], real_shape=[part2])
                asc2.copy_out(tile2, out1_gm, [0], real_shape=[part2])


@pytest.mark.parametrize("kernel_type", [STATIC, DYNAMIC])
@pytest.mark.parametrize("block_num, unroll_factor, input_shape, split_sizes, in_out_dtype, tiling_values", [
    (9, 2, [1034, 16], [1024, 10], torch.float32, [0, 1839]),
    (72, 2, [22717, 8], [2048, 20669], torch.float32, [0, 2525]),
    (72, 1, [270610, 32], [512, 270098], torch.float32, [0, 120272]),
    (72, 2, [33473, 16], [1024, 32449], torch.float32, [0, 7439]),
    (72, 2, [36480, 8], [2048, 34432], torch.float32, [0, 4054]),
    (9, 2, [4105, 4], [4096, 9], torch.float32, [0, 1825]),
    (72, 1, [743658, 32], [512, 743146], torch.float32, [0, 330515]),
    (72, 1, [106055, 8], [2048, 104007], torch.float32, [0, 11784]),
    (72, 1, [166912, 8], [164864, 2048], torch.float32, [0, 18546]),
    (2, 1, [2048, 8], [1024, 1024], torch.float32, [0, 8192]),
    (72, 2, [32470, 8], [2048, 30422], torch.float32, [0, 3608]),
    (72, 1, [292435, 32], [512, 291923], torch.float32, [0, 129972]),
    (72, 2, [32507, 8], [2048, 30459], torch.float32, [0, 3612]),
    (72, 1, [301556, 32], [512, 301044], torch.float32, [0, 134025]),
    (72, 2, [32185, 8], [2048, 30137], torch.float32, [0, 3577]),
    (72, 1, [295979, 32], [512, 295467], torch.float32, [0, 131547]),
    (72, 1, [391200, 8], [386400, 4800], torch.float32, [0, 43467]),
    (2, 1, [4800, 8], [2400, 2400], torch.float32, [0, 19200]),
    (72, 2, [58401, 8], [2048, 56353], torch.float32, [0, 6489]),
    (72, 1, [634566, 32], [512, 634054], torch.float32, [0, 282030]),
    (72, 2, [24740, 8], [2048, 22692], torch.float32, [0, 2749]),
    (72, 1, [4237147, 16], [3798370, 438777], torch.float32, [0, 941589]),
    (72, 1, [3582880, 64], [1971728, 1611152], torch.float32, [0, 3184783]),
    (27, 2, [6683, 8], [2048, 4635], torch.float32, [0, 1981]),
    (9, 2, [513, 32], [512, 1], torch.float32, [0, 1824]),
    (72, 1, [934964, 64], [256, 934708], torch.float32, [0, 831080]),
    (72, 2, [10177, 16], [1024, 9153], torch.float32, [0, 2262]),
])
def test_split_2d(profiler, runs, kernel_type, block_num, unroll_factor, input_shape, split_sizes, in_out_dtype,
                  tiling_values):
    axis, tile_length = tiling_values
    UB_CHUNK_SIZE = 248 * 1024 // 4 // in_out_dtype.itemsize  # 248KB / 4 buffers / size_of(in_out_dtype)
    ub_chunk_size = tile_length if tile_length <= UB_CHUNK_SIZE else UB_CHUNK_SIZE
    dim0, dim1 = input_shape
    input_length = dim0 * dim1
    split_boundary = split_sizes[0] * dim1

    input_tensor = torch.randn(input_shape, dtype=in_out_dtype).contiguous()
    output0 = torch.empty(split_sizes[0], dim1, dtype=in_out_dtype)
    output1 = torch.empty(split_sizes[1], dim1, dtype=in_out_dtype)

    params = [input_tensor, output0, output1]
    if kernel_type == STATIC:
        params.extend([asc2.ConstExpr(input_length), asc2.ConstExpr(split_boundary), asc2.ConstExpr(tile_length)])
    else:
        params.extend([input_length, split_boundary, tile_length])
    params.extend([ub_chunk_size, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            split_v[block_num](*params)

    expected = torch.split(input_tensor, split_sizes, dim=axis)
    assert output0.shape == expected[0].shape
    assert output1.shape == expected[1].shape
    torch.testing.assert_close(output0, expected[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(output1, expected[1], rtol=1e-3, atol=1e-3)
