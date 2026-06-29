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


# Reads (h,w) sub tile
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_block(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, width: asc2.ConstExpr[int],
                    height: asc2.ConstExpr[int], block_width: asc2.ConstExpr[int], block_height: asc2.ConstExpr[int],
                    tile_width: asc2.ConstExpr[int], tile_height: asc2.ConstExpr[int], repeat: asc2.ConstExpr[int],
                    unroll_factor: asc2.ConstExpr[int]):
    total_tiles_w = asc2.ceildiv(width, block_width)

    global_tensor = asc2.tensor(input_ptr, [height, width])
    result_tensor = asc2.tensor(output_ptr, [width, height])
    for i in asc2.range(asc2.block_idx(), repeat, asc2.block_num(), parallel=True, unroll_factor=unroll_factor):
        offset_x = (i % total_tiles_w) * block_width
        offset_y = (i // total_tiles_w) * block_height
        load_width = block_width if block_width < width - offset_x else width - offset_x
        load_height = block_height if block_height < height - offset_y else height - offset_y
        input = asc2.load(global_tensor, [offset_y, offset_x], [tile_height, tile_width],
                          real_shape=[load_height, load_width])
        transposed = input.transpose()
        asc2.store(transposed, result_tensor, [offset_x, offset_y], real_shape=[load_width, load_height])


# Reads (height,n) at once
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_column(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, width: asc2.ConstExpr[int],
                     height: asc2.ConstExpr[int], block_size: asc2.ConstExpr[int], tile_width: asc2.ConstExpr[int],
                     tile_height: asc2.ConstExpr[int], total_count: asc2.ConstExpr[int],
                     unroll_factor: asc2.ConstExpr[int]):

    global_tensor = asc2.tensor(input_ptr, [height, width])
    result_tensor = asc2.tensor(output_ptr, [width, height])
    for i in asc2.range(asc2.block_idx(), total_count, asc2.block_num(), parallel=True, unroll_factor=unroll_factor):
        offset = i * block_size
        load_width = block_size if block_size < width - offset else width - offset
        input = asc2.load(global_tensor, [0, offset], [tile_height, tile_width], real_shape=[height, load_width])
        transposed = input.transpose()
        asc2.store(transposed, result_tensor, [offset, 0], real_shape=[load_width, height])


# Reads (n,width) at once
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_line(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, width: asc2.ConstExpr[int],
                   height: asc2.ConstExpr[int], block_size: asc2.ConstExpr[int], tile_width: asc2.ConstExpr[int],
                   tile_height: asc2.ConstExpr[int], total_count: asc2.ConstExpr[int],
                   unroll_factor: asc2.ConstExpr[int]):
    global_tensor = asc2.tensor(input_ptr, [height, width])
    result_tensor = asc2.tensor(output_ptr, [width, height])
    for i in asc2.range(asc2.block_idx(), total_count, asc2.block_num(), parallel=True, unroll_factor=unroll_factor):
        offset = i * block_size
        load_height = block_size if block_size < height - offset else height - offset
        input = asc2.load(global_tensor, [offset, 0], [tile_height, tile_width], real_shape=[load_height, width])
        transposed = input.transpose()
        asc2.store(transposed, result_tensor, [0, offset], real_shape=[width, load_height])


# Iteration step is [1..1,axis_step,remaining_transposed_shape] for transposed shape.
# Step for high dimension is 1. Lower dimensions loads as is.
# Selected dimension loads in axis_step:
# axis_step = 10, store_shape_axis = 1, *output_shape*=[1024,256,32] ub shape is [1,10,32]
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_nlast_axis(
        input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress, input_shape: asc2.ConstExpr,  # Read tensor shape
        axis_step: asc2.ConstExpr[int],  # How many compute per step
        load_axes: asc2.
    ConstExpr,  # Axis which iterated reverse order: 0 -> store_shape_axis, 1 -> store_shape_axis-1, ...
        store_shape_axis: asc2.ConstExpr[int],  # 0,...store_shape_axis are iterated in output_shape
        ub_load_shape: asc2.ConstExpr,  # Buffer size in ub
        load_shape: asc2.ConstExpr,  # Used to fill real_shape on load
        permute: asc2.ConstExpr,  # Dimensions to permute
        block_count: asc2.ConstExpr[int],  # How many steps do total
        unroll_factor: asc2.ConstExpr[int]):

    load_shape_axis = load_axes[0]
    output_shape = []
    ub_store_shape = []
    store_shape = []
    for dim in asc2.static_range(0, len(input_shape)):
        output_shape += [input_shape[permute[dim]]]
        ub_store_shape += [ub_load_shape[permute[dim]]]
        store_shape += [load_shape[permute[dim]]]
    inner_total = asc2.ceildiv(input_shape[load_shape_axis], axis_step)

    input_tensor = asc2.tensor(input_ptr, input_shape)
    output_tensor = asc2.tensor(output_ptr, output_shape)

    for i in asc2.range(asc2.block_idx(), block_count, asc2.block_num(), parallel=True, unroll_factor=unroll_factor):
        id0 = i % inner_total  # this one walk by step
        offset0 = id0 * axis_step
        real_count = axis_step if offset0 + axis_step <= input_shape[
            load_shape_axis] else input_shape[load_shape_axis] - offset0

        store_real_shape = store_shape
        store_real_shape[store_shape_axis] = real_count
        store_offsets = [0] * len(input_shape)
        store_offsets[store_shape_axis] = offset0

        load_real_shape = load_shape
        load_real_shape[load_shape_axis] = real_count
        load_offsets = [0] * len(input_shape)
        load_offsets[load_shape_axis] = offset0

        if store_shape_axis > 0:
            id1 = i // inner_total % output_shape[store_shape_axis - 1]
            load_offsets[load_axes[1]] = id1
            store_offsets[store_shape_axis - 1] = id1
        if store_shape_axis > 1:
            id2 = i // inner_total // output_shape[store_shape_axis - 1] % output_shape[store_shape_axis - 2]
            load_offsets[load_axes[2]] = id2
            store_offsets[store_shape_axis - 2] = id2

        tile = asc2.load(input_tensor, load_offsets, ub_load_shape, real_shape=load_real_shape)
        tile2 = tile.transpose(*permute)

        asc2.store(tile2, output_tensor, store_offsets, real_shape=store_real_shape)


def launch_nlast_axis(input, permute, axis, axis_step, cores, dtype, runs, profiler):
    input_shape = list(input.shape)
    output_shape = [input_shape[permute[i]] for i in range(0, len(input_shape))]
    output = torch.zeros(output_shape, dtype=dtype)
    items_in_block = 32 // input.element_size()

    store_shape_axis = -1  # Where load_shape_axis goes in target shape
    load_shape_axis = permute[axis]
    load_axes = [load_shape_axis]
    if axis > 0:
        load_axes += [permute[axis - 1]]
    if axis > 1:
        load_axes += [permute[axis - 2]]
    for i in range(0, len(input_shape)):
        if permute[i] == load_shape_axis:
            store_shape_axis = i

    count_main = asc2.ceildiv(output_shape[axis], axis_step)
    count_axis0 = 1 if axis < 1 else output_shape[axis - 1]
    count_axis1 = 1 if axis < 2 else output_shape[axis - 2]
    count_axis2 = 1 if axis < 3 else output_shape[axis - 3]

    block_count = count_main * count_axis0 * count_axis1 * count_axis2
    load_shape = [i for i in input_shape]
    load_shape[load_shape_axis] = axis_step
    for i in range(0, axis):
        load_shape[permute[i]] = 1

    ub_shape = [i for i in load_shape]
    ub_shape[-1] = asc2.ceildiv(ub_shape[-1], items_in_block) * items_in_block
    ub_shape[permute[-1]] = asc2.ceildiv(ub_shape[permute[-1]], items_in_block) * items_in_block
    ub_load_shape = ub_shape

    with profiler.profile():
        for _ in range(runs):
            transpose_nlast_axis[cores](input, output, input_shape, axis_step, load_axes, store_shape_axis,
                                        ub_load_shape, load_shape, permute, block_count, 1)
    return output


# Split input data for one axis loads at once slice of size axis_step on that dimension
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_one_axis(
    input_ptr: asc2.GlobalAddress,
    output_ptr: asc2.GlobalAddress,
    input_shape: asc2.ConstExpr,  # Read tensor shape
    axis_step: asc2.ConstExpr,  # How many compute per step
    load_shape_axis: asc2.ConstExpr,  # Dim num in input_shape
    store_shape_axis: asc2.ConstExpr,  # Dim num in transposed shape
    ub_load_shape: asc2.ConstExpr,  # Buffer size in ub
    load_shape: asc2.ConstExpr,  # Used to fill real_shape on load
    permute: asc2.ConstExpr,  # Dimensions to permute
    block_count: asc2.ConstExpr,  # How many steps do total
    unroll_factor: asc2.ConstExpr,
):

    output_shape = []
    ub_store_shape = []
    store_shape = []
    for dim in asc2.static_range(0, len(input_shape)):
        output_shape += [input_shape[permute[dim]]]
        ub_store_shape += [ub_load_shape[permute[dim]]]
        store_shape += [load_shape[permute[dim]]]
    input_tensor = asc2.tensor(input_ptr, input_shape)
    output_tensor = asc2.tensor(output_ptr, output_shape)
    for i in asc2.range(asc2.block_idx(), block_count, asc2.block_num(), parallel=True, unroll_factor=unroll_factor):
        offset = i * axis_step
        read_count = axis_step if offset + axis_step < input_shape[
            load_shape_axis] else input_shape[load_shape_axis] - offset
        read_offsets = [0] * (load_shape_axis) + [offset] + [0] * (len(load_shape) - 1 - load_shape_axis)
        read_shape = load_shape[:load_shape_axis] + [read_count] + load_shape[load_shape_axis + 1:]
        load_tensor = asc2.load(input_tensor, read_offsets, ub_load_shape, real_shape=read_shape)
        transposed_tensor = load_tensor.transpose(*permute)
        write_offsets = [0] * (store_shape_axis) + [offset] + [0] * (len(store_shape) - 1 - store_shape_axis)
        store_real_shape = store_shape[:store_shape_axis] + [read_count] + store_shape[store_shape_axis + 1:]
        asc2.store(transposed_tensor, output_tensor, write_offsets, real_shape=store_real_shape)


def launch_one_axis(input, permute, axis, axis_step, cores, dtype, runs, profiler):
    input_shape = list(input.shape)
    output_shape = [input_shape[permute[i]] for i in range(0, len(input_shape))]
    items_in_block = 32 // input.element_size()
    output = torch.zeros(output_shape, dtype=dtype)

    load_shape_axis = axis
    store_shape_axis = 0  # Where load_shape_axis goes in target shape
    for i in range(0, len(input_shape)):
        if permute[i] == load_shape_axis:
            store_shape_axis = i

    block_count = asc2.ceildiv(input_shape[load_shape_axis], axis_step)
    load_shape = [input_shape[i] if i != load_shape_axis else axis_step for i in range(0, len(input_shape))]

    ub_shape = [i for i in load_shape]
    ub_shape[load_shape_axis] = axis_step
    ub_shape[-1] = asc2.ceildiv(ub_shape[-1], items_in_block) * items_in_block
    ub_shape[permute[-1]] = asc2.ceildiv(ub_shape[permute[-1]], items_in_block) * items_in_block
    ub_load_shape = ub_shape

    with profiler.profile():
        for _ in range(runs):
            transpose_one_axis[cores](input, output, input_shape, axis_step, load_shape_axis, store_shape_axis,
                                      ub_load_shape, load_shape, permute, block_count, 1)
    return output


# Split tensor by 2 dimensions (store_axis), loads subpart
# Example: output_shape = [1024,2048,512], store_axis=[0,1] iterate by [STEP1,STEP2,512]
@asc2.jit(static_alloc=True, reuse_ub=True)
def transpose_2_axis(
    input_ptr: asc2.GlobalAddress,
    output_ptr: asc2.GlobalAddress,
    input_shape: asc2.ConstExpr,  # Read tensor shape
    axis_step: asc2.ConstExpr,  # How many compute per step x2
    store_axis: asc2.ConstExpr,  # Dim num in *output_shape*
    ub_load_shape: asc2.ConstExpr,  # Buffer size in ub
    load_shape: asc2.ConstExpr,  # Used to fill real_shape on load
    permute: asc2.ConstExpr,  # Dimensions to permute
    block_count: asc2.ConstExpr,  # How many steps do total
    block_width: asc2.ConstExpr,
    unroll_factor: asc2.ConstExpr,
):

    output_shape = []
    ub_store_shape = []
    store_shape = []
    for dim in asc2.static_range(0, len(input_shape)):
        output_shape += [input_shape[permute[dim]]]
        ub_store_shape += [ub_load_shape[permute[dim]]]
        store_shape += [load_shape[permute[dim]]]
    load_shape_axis0 = permute[store_axis[0]]
    load_shape_axis1 = permute[store_axis[1]]

    input_tensor = asc2.tensor(input_ptr, input_shape)
    output_tensor = asc2.tensor(output_ptr, output_shape)
    for i in asc2.range(asc2.block_idx(), block_count, asc2.block_num(), parallel=True, unroll_factor=unroll_factor):
        offset0 = i % block_width * axis_step[0]
        offset1 = i // block_width * axis_step[1]
        count0 = axis_step[0] if offset0 + axis_step[0] < input_shape[
            load_shape_axis0] else input_shape[load_shape_axis0] - offset0
        count1 = axis_step[1] if offset1 + axis_step[1] < input_shape[
            load_shape_axis1] else input_shape[load_shape_axis1] - offset1

        read_offsets = [0] * len(load_shape)
        read_offsets[load_shape_axis0] = offset0
        read_offsets[load_shape_axis1] = offset1
        read_count = load_shape
        read_count[load_shape_axis0] = count0
        read_count[load_shape_axis1] = count1

        load_tensor = asc2.load(input_tensor, read_offsets, ub_load_shape, real_shape=read_count)
        transposed_tensor = load_tensor.transpose(*permute)

        write_offsets = [0] * len(load_shape)
        write_offsets[store_axis[0]] = offset0
        write_offsets[store_axis[1]] = offset1
        write_count = store_shape
        write_count[store_axis[0]] = count0
        write_count[store_axis[1]] = count1

        asc2.store(transposed_tensor, output_tensor, write_offsets, real_shape=write_count)


def launch_2axis(input, permute, axis, step, dtype, cores, unroll_factor, runs, profiler):
    output_size = [input.shape[i] for i in permute]
    output = torch.zeros(output_size, dtype=dtype)
    items_in_block = 32 // input.element_size()
    axis0 = permute[axis[0]]
    axis1 = permute[axis[1]]

    ub_size = list(input.shape)
    ub_size[axis0] = step[0]
    ub_size[axis1] = step[1]
    ub_size[-1] = asc2.ceildiv(ub_size[-1], items_in_block) * items_in_block
    ub_size[permute[-1]] = asc2.ceildiv(ub_size[permute[-1]], items_in_block) * items_in_block

    load_size = list(input.shape)
    load_size[axis0] = step[0]
    load_size[axis1] = step[1]

    blocks_axis0 = asc2.ceildiv(input.shape[axis0], step[0])
    blocks_axis1 = asc2.ceildiv(input.shape[axis1], step[1])
    total_blocks = blocks_axis0 * blocks_axis1

    with profiler.profile():
        for _ in range(runs):
            transpose_2_axis[cores](input, output, list(input.shape), step, axis, ub_size,\
                                    load_size, permute, total_blocks, blocks_axis0, unroll_factor)
    return output


tests = [
    [
        71, 1, 'testcase02', 10001, [512, 128], [1, 0], torch.int32,
        [
            2, 0, 0, 0, 0, 0, 0, 71, 928, 576, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [512, 128, 0, 0, 0, 0, 0, 0],
            [128, 512, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 1, 512, 128], [1, 1, 1, 128, 512], [1, 1, 1, 512, 0],
            [1, 1, 1, 0, 512], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 512, 0],
            [1, 1, 1, 0, 512]
        ]
    ],
    [
        68, 1, 'testcase10', 10003, [1000, 2048], [1, 0], torch.float32,
        [
            2, 1, 1, 127, 252, 16, 244, 68, 1, 0, 253952, 1, 47, 48, 50, 51, 66, 67, 67, [1000, 2048, 0, 0, 0, 0, 0, 0],
            [2048, 1000, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 1, 1000, 2048], [1, 1, 1, 2048, 1000], [1, 1, 1, 252, 127],
            [1, 1, 1, 127, 252], [1, 1, 1, 252, 16], [1, 1, 1, 16, 252], [1, 1, 1, 244, 127], [1, 1, 1, 127, 244],
            [1, 1, 1, 244, 16], [1, 1, 1, 16, 244]
        ]
    ],
    [
        64, 1, "testcase14", 10002, [12, 64, 44, 80], [1, 2, 3, 0], torch.float16,
        [
            2, 1, 0, 356, 3519, 288, 64, 65, 1, 0, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [12, 225280, 0, 0, 0, 0, 0, 0],
            [225280, 12, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 1, 12, 225280], [1, 1, 1, 225280, 12], [1, 1, 1, 12, 3519],
            [1, 1, 1, 3519, 12], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 12, 64],
            [1, 1, 1, 64, 12]
        ]
    ],
    [
        72, 1, "testcase19", 10002, [3, 61, 144, 149], [2, 3, 0, 1], torch.float32,
        [
            3, 1, 0, 251, 251, 121, 121, 72, 1, 14, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [3, 61, 21456, 0, 0, 0, 0, 0],
            [21456, 61, 3, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0], [0, 1, 4, 3, 2], [1, 1, 3, 61, 21456], [1, 1, 21456, 61, 3], [1, 1, 3, 61, 251],
            [1, 1, 251, 61, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 3, 61, 121],
            [1, 1, 121, 61, 3]
        ]
    ],
    [
        72, 1, 'testcase39', 10004, [4, 64, 128, 40], [2, 0, 1, 3], torch.float32,
        [
            3, 0, 0, 3, 0, 1, 0, 72, 1, 14, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [256, 128, 40, 0, 0, 0, 0, 0],
            [128, 256, 40, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1, 256, 128, 40], [1, 1, 128, 256, 40], [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    ],
    [
        72, 1, "testcase30", 10004, [1136, 128, 42], [1, 0, 2], torch.float32,
        [
            3, 0, 0, 4, 0, 0, 0, 72, 3, 68, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [1136, 128, 42, 0, 0, 0, 0, 0],
            [128, 1136, 42, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1, 1136, 128, 42], [1, 1, 128, 1136, 42], [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    ],
    [
        69, 1, 'testcase60', 10001, [2048, 10, 3], [1, 0, 2], torch.bfloat16,
        [
            3, 0, 0, 0, 0, 0, 0, 69, 896, 512, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [2048, 10, 3, 0, 0, 0, 0, 0],
            [10, 2048, 3, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1, 2048, 10, 3], [1, 1, 10, 2048, 3], [1, 1, 2048, 0, 3],
            [1, 1, 0, 2048, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 2048, 0, 3],
            [1, 1, 0, 2048, 3]
        ]
    ],
]


@pytest.mark.parametrize(
    "block_dim, unroll_factor, testcase_name, tiling_key, input_shape, permute, input_dtype, tiling_values", tests)
def test_transpose(profiler, runs, block_dim, unroll_factor, testcase_name, tiling_key, input_shape, permute,
                   input_dtype, tiling_values):
    in_cut_index = tiling_values[1]
    out_cut_index = tiling_values[2]
    in_ub_factor = tiling_values[3]
    out_ub_factor = tiling_values[4]
    ub_size = tiling_values[10]

    # workaround for shapes
    if permute == [2, 3, 0, 1]:
        input_shape = [input_shape[0] * input_shape[1], input_shape[2] * input_shape[3]]
        permute = [1, 0]
    elif permute == [3, 0, 1, 2]:
        input_shape = [input_shape[0] * input_shape[1] * input_shape[2], input_shape[3]]
        permute = [1, 0]
    elif permute == [1, 2, 3, 0]:
        input_shape = [input_shape[0], input_shape[1] * input_shape[2] * input_shape[3]]
        permute = [1, 0]
    elif permute == [2, 0, 1, 3]:
        input_shape = [input_shape[0] * input_shape[1], input_shape[2], input_shape[3]]
        permute = [1, 0, 2]

    if input_shape == [383, 383, 100] and permute == [0, 2, 1]:
        in_cut_index = 0
        in_ub_factor = 1
    elif input_shape == [826, 512, 64] and permute == [0, 2, 1]:
        in_cut_index = 0
        in_ub_factor = 1
    elif input_shape == [20, 200, 200] and permute == [1, 0, 2]:
        tiling_key = 10002
        in_cut_index = 1
        in_ub_factor = 3
    elif input_shape == [1024, 107, 31, 3] and permute == [2, 1, 0, 3]:
        out_ub_factor = 128
    elif input_shape == [13, 64, 256, 28] and permute == [2, 1, 3, 0]:
        out_ub_factor = 8
    elif input_shape == [1484, 32, 1484] and permute == [2, 1, 0]:
        in_ub_factor = 64
        out_ub_factor = 16
    elif input_shape == [644, 32, 644] and permute == [2, 1, 0]:
        in_ub_factor = 64
        out_ub_factor = 16
    elif input_shape == [12, 64 * 44 * 80] and permute == [1, 0]:
        tiling_key = 10002

    input = torch.randn(input_shape).to(input_dtype)
    out_shape = [input_shape[permute[i]] for i in range(0, len(input_shape))]
    out = torch.zeros(out_shape, dtype=input_dtype)
    items_in_block = 32 // input_dtype.itemsize
    if tiling_key == 10001 and len(permute) == 2:
        width = input_shape[1]
        height = input_shape[0]
        load_sizes = tiling_values[29]
        ub_size = [load_sizes[4], load_sizes[3]]

        if width > height:
            # load n columns
            block_size = tiling_values[4]
            if block_size is None or block_size == 0:
                block_size = asc2.ceildiv(width, block_dim)
            tile_size = [
                asc2.ceildiv(height, items_in_block) * items_in_block,
                asc2.ceildiv(block_size, items_in_block) * items_in_block,
            ]
            total_tiles = asc2.ceildiv(width, block_size)
            with profiler.profile():
                for _ in range(runs):
                    transpose_column[block_dim](input, out, width, height, block_size, tile_size[1], tile_size[0],
                                                total_tiles, unroll_factor)
        else:
            # load n rows
            block_size = tiling_values[4]
            if block_size is None or block_size == 0:
                block_size = asc2.ceildiv(height, block_dim)
            tile_size = [
                asc2.ceildiv(block_size, items_in_block) * items_in_block,
                asc2.ceildiv(width, items_in_block) * items_in_block,
            ]
            total_tiles = asc2.ceildiv(height, block_size)
            with profiler.profile():
                for _ in range(runs):
                    transpose_line[block_dim](input, out, width, height, block_size, tile_size[1], tile_size[0],
                                              total_tiles, unroll_factor)
    elif tiling_key == 10001:
        input_shape = list(input.shape)
        longest_axis = max(input.shape)
        axis = list(input.shape).index(longest_axis)

        axis_step = asc2.ceildiv(input.shape[axis], block_dim)
        out = launch_one_axis(input, permute, axis, axis_step, block_dim, input_dtype, runs, profiler)
    elif tiling_key == 10004:  #Split on N-last dims
        assert (len(input_shape) == 3 or len(input_shape) == 4)
        iterate_axis = 0
        # Select iteration axis which fit in UB
        for axis in range(0, 3, 1):
            subtensor_size = input.element_size() * math.prod(out_shape[axis + 1:]) * unroll_factor
            if subtensor_size < ub_size:
                iterate_axis = axis
                step = ub_size // subtensor_size
                break
        out = launch_nlast_axis(input, permute, iterate_axis, step, block_dim, input_dtype, runs, profiler)
    elif tiling_key == 10002:  #Split on single axis
        longest_axis = max(input.shape)
        axis = list(input.shape).index(longest_axis)

        axis_step = asc2.ceildiv(input.shape[axis], block_dim)
        ub_fit = ub_size * input.shape[axis] // math.prod(input.shape) // input.element_size() // unroll_factor
        assert (ub_fit > 0)
        axis_step = min(axis_step, ub_fit)

        out = launch_one_axis(input, permute, axis, axis_step, block_dim, input_dtype, runs, profiler)
    elif tiling_key == 10003:  #Split on two axes
        input_shape = list(input.shape)
        step = [in_ub_factor, out_ub_factor]
        axis = [0, out_cut_index]  # number in out_shape!
        for i in range(0, len(permute)):
            if permute[i] == in_cut_index:
                axis[0] = i
        assert (out_ub_factor != 0 and in_ub_factor != 0 and out_shape[axis[0]] > in_ub_factor
                and out_shape[axis[1]] > out_ub_factor)
        out = launch_2axis(input, permute, axis, step, input_dtype, block_dim, unroll_factor, runs, profiler)
    else:
        raise RuntimeError(f"Wrong tilekey: {tiling_key}")

    golden = input.permute(*permute)
    torch.testing.assert_close(out, golden)
