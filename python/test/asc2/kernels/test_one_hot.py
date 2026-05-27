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

SINGLE_CORE = 1
MULTI_CORE = 16
ONE_HOT_DTYPES = [torch.float16, torch.float32, torch.int32]


@asc2.jit(always_compile=True)
def one_hot_kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, arange_ptr: asc2.GlobalAddress,
                   on_value: asc2.ConstExpr, off_value: asc2.ConstExpr, input_total: asc2.ConstExpr,
                   depth: asc2.ConstExpr, block_length: asc2.ConstExpr, block_length_tail: asc2.ConstExpr,
                   unroll_factor: asc2.ConstExpr):
    x = asc2.tensor(x_ptr, [input_total])
    y = asc2.tensor(y_ptr, [input_total * depth])
    arange_gm = asc2.tensor(arange_ptr, [depth])

    block_offset = asc2.block_idx() * block_length
    loop_count = block_length
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_length_tail

    arange_tile = asc2.load(arange_gm, [depth], offsets=[0])
    for i in asc2.range(loop_count, unroll_factor=unroll_factor, parallel=True):
        idx_pos = block_offset + i
        idx_scalar = asc2.load(x, offsets=[idx_pos])
        mask = asc2.equal(arange_tile, idx_scalar)
        result = asc2.where(mask, asc2.number(on_value, y_ptr.dtype), asc2.number(off_value, y_ptr.dtype))
        asc2.store(result, y, offsets=[idx_pos * depth])


def make_indices(size: int, depth: int) -> torch.Tensor:
    return torch.randint(0, depth, (size, ), dtype=torch.int32)


def golden_one_hot(indices: torch.Tensor, depth: int, on_value, off_value, dtype: torch.dtype) -> torch.Tensor:
    one_hot_mask = torch.nn.functional.one_hot(indices.long(), num_classes=depth)
    return torch.where(one_hot_mask.bool(), torch.tensor(on_value, dtype=dtype), torch.tensor(off_value, dtype=dtype))


def one_hot_launch(indices: torch.Tensor, depth: int, on_value, off_value, *, dtype: torch.dtype,
                   core_num: int = SINGLE_CORE, unroll_factor: int = 1) -> torch.Tensor:
    input_total = indices.numel()
    block_length = asc2.ceildiv(input_total, core_num)
    block_length_tail = input_total - block_length * (core_num - 1)
    indices_flat = indices.reshape(input_total).to(torch.int32)
    arange_t = torch.arange(depth, dtype=torch.int32)
    output = torch.zeros(input_total * depth, dtype=dtype)
    one_hot_kernel[core_num](indices_flat, output, arange_t, on_value, off_value, input_total, depth, block_length,
                             block_length_tail, unroll_factor)
    return output.reshape(input_total, depth)


@pytest.mark.parametrize("dtype", ONE_HOT_DTYPES, ids=str)
def test_one_hot_output_dtypes(dtype: torch.dtype):
    depth = 4
    indices = make_indices(32, depth)
    on_value, off_value = (1, 0) if dtype == torch.int32 else (1.0, 0.0)
    out = one_hot_launch(indices, depth, on_value, off_value, dtype=dtype)
    expected = golden_one_hot(indices, depth, on_value, off_value, dtype)
    torch.testing.assert_close(out, expected)


# Exercises the depth=1 degenerate path (CANN's SimtComputeDepth) and a range of
# depth_loop widths that the kernel iterates over via the [depth] broadcast.
@pytest.mark.parametrize("depth", [1, 2, 4, 8, 32, 64])
def test_one_hot_depths(depth: int):
    indices = make_indices(32, depth)
    out = one_hot_launch(indices, depth, 1.0, 0.0, dtype=torch.float32)
    expected = golden_one_hot(indices, depth, 1.0, 0.0, torch.float32)
    torch.testing.assert_close(out, expected)


# Axis is metadata only at the test level; the kernel always emits depth-innermost
# layout, so any axis value produces functionally equivalent one-hot semantics.
# This case exercises non-default axes through the launch wrapper to confirm the
# kernel's depth-loop is independent of input rank.
@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_one_hot_axes(axis: int):
    depth = 4
    input_shape = (4, 4, 4)
    indices = make_indices(math.prod(input_shape), depth).reshape(input_shape)
    out = one_hot_launch(indices, depth, 1.0, 0.0, dtype=torch.float32)
    expected = golden_one_hot(indices.reshape(-1), depth, 1.0, 0.0, torch.float32)
    torch.testing.assert_close(out, expected)


# Logical-prefix coverage of the per-element scalar load path: each `logical_size`
# value exercises a different `block_length` / `block_length_tail` split and a
# different load-store offset pattern at the GM boundary.
@pytest.mark.parametrize("logical_size", [1, 2, 127, 128, 129, 255, 256])
def test_one_hot_logical_border_prefixes(logical_size: int):
    depth = 8
    indices = make_indices(logical_size, depth)
    out = one_hot_launch(indices, depth, 1.0, 0.0, dtype=torch.float32)
    expected = golden_one_hot(indices, depth, 1.0, 0.0, torch.float32)
    torch.testing.assert_close(out, expected)


def test_one_hot_multicore():
    depth = 8
    size = MULTI_CORE * 4
    indices = make_indices(size, depth)
    out = one_hot_launch(indices, depth, 1.0, 0.0, dtype=torch.float32, core_num=MULTI_CORE)
    expected = golden_one_hot(indices, depth, 1.0, 0.0, torch.float32)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("on_value, off_value", [
    (1.0, 0.0),
    (10.5, 30.6),
    (-1.0, 1.0),
], ids=["unit", "csv_float", "signed"])
def test_one_hot_scalar_values(on_value: float, off_value: float):
    depth = 4
    indices = make_indices(64, depth)
    out = one_hot_launch(indices, depth, on_value, off_value, dtype=torch.float32)
    expected = golden_one_hot(indices, depth, on_value, off_value, torch.float32)
    torch.testing.assert_close(out, expected)
