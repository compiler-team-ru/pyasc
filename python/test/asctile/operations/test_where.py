# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Tuple

import asctile
import pytest
import torch

SIZE = 32
TILE_SIZE = 128
SINGLE_CORE = 1
MULTI_CORE = 16

# Data types supported for where src0/src1 (int8 only supported for condition)
where_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32]


def create_tensor(dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_floating_point:
        return torch.rand(SIZE, dtype=dtype)
    if dtype.is_signed:
        return torch.randint(-100, 100, (SIZE, ), dtype=dtype)


@asctile.jit(always_compile=True)
def where_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress, z_ptr: asctile.GlobalAddress,
                 op: asctile.ConstExpr):
    x = asctile.global_tensor(x_ptr, [SIZE])
    y = asctile.global_tensor(y_ptr, [SIZE])
    z = asctile.global_tensor(z_ptr, [SIZE])
    xt = asctile.copy_in(x, [0], [SIZE])
    yt = asctile.copy_in(y, [0], [SIZE])
    zt = asctile.where(op(xt, yt), xt, yt)
    asctile.copy_out(zt, z, [0])


@pytest.mark.parametrize("dtype", where_dtypes)
@pytest.mark.parametrize("asc_op, torch_op", [
    (asctile.equal, torch.eq),
    (asctile.not_equal, torch.ne),
    (asctile.greater, torch.gt),
    (asctile.greater_equal, torch.ge),
    (asctile.less, torch.lt),
    (asctile.less_equal, torch.le),
])
def test_where_ops(require_c310, asc_op, torch_op, dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()
    x = create_tensor(dtype)
    y = create_tensor(dtype)
    result = torch.zeros_like(x)
    where_kernel[1](x, y, result, asc_op)
    expected = torch.where(torch_op(x, y), x, y)
    torch.testing.assert_close(result, expected)


@asctile.jit(always_compile=True)
def where_scalar_kernel(x_ptr: asctile.GlobalAddress, scalar, z_ptr: asctile.GlobalAddress, op: asctile.ConstExpr):
    x = asctile.global_tensor(x_ptr, [SIZE])
    z = asctile.global_tensor(z_ptr, [SIZE])
    xt = asctile.copy_in(x, [0], [SIZE])
    zt = asctile.where(op(xt, scalar), asctile.cast(0.0, x_ptr.dtype), asctile.cast(1.0, x_ptr.dtype))
    asctile.copy_out(zt, z, [0])


@pytest.mark.parametrize("dtype", where_dtypes)
@pytest.mark.parametrize("asc_op, torch_op", [
    (asctile.equal, torch.eq),
    (asctile.not_equal, torch.ne),
    (asctile.greater, torch.gt),
    (asctile.greater_equal, torch.ge),
    (asctile.less, torch.lt),
    (asctile.less_equal, torch.le),
])
def test_where_scalar_ops(require_c310, asc_op, torch_op, dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()
    x = create_tensor(dtype)
    y = torch.tensor(0 if dtype.is_signed else 0.5, dtype=dtype)
    result = torch.zeros_like(x)
    where_scalar_kernel[1](x, y, result, asc_op)
    expected = torch.where(torch_op(x, y), torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
    torch.testing.assert_close(result, expected)


@asctile.jit(always_compile=True)
def where_with_cond_kernel(cond_ptr: asctile.GlobalAddress, x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
                           out_ptr: asctile.GlobalAddress, size: asctile.ConstExpr[int],
                           tile_size: asctile.ConstExpr[int], tile_per_block: asctile.ConstExpr[int]):
    cond_gm = asctile.global_tensor(cond_ptr, [size])
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    base_offset = asctile.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=2):
        tile_offset = base_offset + i * tile_size
        c = asctile.copy_in(cond_gm, [tile_offset], [tile_size])
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size])
        y = asctile.copy_in(y_gm, [tile_offset], [tile_size])
        out = asctile.where(c != 0, x, y)
        asctile.copy_out(out, out_gm, [tile_offset])


@asctile.jit(always_compile=True)
def where_scalar_source_kernel(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress,
                               size: asctile.ConstExpr[int], tile_size: asctile.ConstExpr[int],
                               tile_per_block: asctile.ConstExpr[int], scalar_value: asctile.ConstExpr,
                               scalar_on_true: asctile.ConstExpr):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    base_offset = asctile.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=2):
        tile_offset = base_offset + i * tile_size
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size])
        scalar = asctile.cast(scalar_value, x_ptr.dtype)
        if scalar_on_true:
            out = asctile.where(x > 0, scalar, x)
        else:
            out = asctile.where(x > 0, x, scalar)
        asctile.copy_out(out, out_gm, [tile_offset])


def check_dtype(dtype: torch.dtype, require_c310):
    if dtype not in (torch.float16, torch.float32):
        require_c310()


def make_data(size: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    x_i = torch.randint(-8, 9, (size, ), dtype=torch.int32)
    y_i = torch.randint(-9, 10, (size, ), dtype=torch.int32)
    # Force x == y at every 7th position so the test also exercises the case
    # where both arms agree: the kernel output must equal that shared value
    # regardless of the condition tile.
    overlap = torch.arange(size, dtype=torch.int32) % 7 == 0
    x_i = torch.where(overlap, y_i, x_i)
    if dtype.is_floating_point:
        return x_i.to(torch.float32).to(dtype), y_i.to(torch.float32).to(dtype)
    return x_i.to(dtype), y_i.to(dtype)


def make_condition(size: int, pattern: str, dtype: torch.dtype) -> torch.Tensor:
    if pattern == "all_true":
        cond = torch.ones(size, dtype=torch.int32)
    elif pattern == "all_false":
        cond = torch.zeros(size, dtype=torch.int32)
    elif pattern == "alternating":
        cond = torch.arange(size, dtype=torch.int32) % 2
    elif pattern == "first_true":
        cond = torch.zeros(size, dtype=torch.int32)
        cond[0] = 1
    elif pattern == "last_true":
        cond = torch.zeros(size, dtype=torch.int32)
        cond[size - 1] = 1
    else:
        raise ValueError(f"Unknown condition pattern: {pattern}")
    if dtype.is_floating_point:
        return cond.to(torch.float32).to(dtype)
    return cond.to(dtype)


def where_with_cond_launch(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, core_num: int = SINGLE_CORE,
                           tile_size: int = TILE_SIZE) -> torch.Tensor:
    out = torch.empty_like(x)
    size = out.numel()
    num_tiles = asctile.ceildiv(size, tile_size)
    where_with_cond_kernel[core_num](cond, x, y, out, size, tile_size, asctile.ceildiv(num_tiles, core_num))
    return out


def where_scalar_source_launch(x: torch.Tensor, *, scalar_value: int, scalar_on_true: bool, core_num: int = SINGLE_CORE,
                               tile_size: int = TILE_SIZE) -> torch.Tensor:
    out = torch.empty_like(x)
    size = out.numel()
    num_tiles = asctile.ceildiv(size, tile_size)
    where_scalar_source_kernel[core_num](x, out, size, tile_size, asctile.ceildiv(num_tiles, core_num), scalar_value,
                                         scalar_on_true)
    return out


@pytest.mark.parametrize("dtype", where_dtypes)
def test_where_condition_dtypes(require_c310, dtype: torch.dtype):
    check_dtype(dtype, require_c310)
    size = TILE_SIZE
    x, y = make_data(size, dtype)
    cond = make_condition(size, "alternating", dtype)
    out = where_with_cond_launch(cond, x, y)
    expected = torch.where(cond.bool(), x, y)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("data_dtype", where_dtypes)
def test_where_int8_condition(require_c310, data_dtype: torch.dtype):
    require_c310()
    size = TILE_SIZE
    x, y = make_data(size, data_dtype)
    cond = make_condition(size, "alternating", torch.int8)
    out = where_with_cond_launch(cond, x, y)
    expected = torch.where(cond.bool(), x, y)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("pattern", ["all_true", "all_false", "alternating", "first_true", "last_true"])
def test_where_condition_patterns(pattern: str):
    size = TILE_SIZE
    x, y = make_data(size, torch.float32)
    cond = make_condition(size, pattern, torch.float32)
    out = where_with_cond_launch(cond, x, y)
    expected = torch.where(cond.bool(), x, y)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("logical_size", [1, 2, 127, 128, 129, 255, 256])
def test_where_logical_border_prefixes(logical_size: int):
    num_tiles = asctile.ceildiv(logical_size, TILE_SIZE)
    physical_size = TILE_SIZE * asctile.ceildiv(num_tiles, SINGLE_CORE) * SINGLE_CORE
    x, y = make_data(physical_size, torch.float32)
    cond = make_condition(physical_size, "alternating", torch.float32)
    out = where_with_cond_launch(cond, x, y)
    expected = torch.where(cond[:logical_size].bool(), x[:logical_size], y[:logical_size])
    torch.testing.assert_close(out[:logical_size], expected)


def test_where_multicore_unrolled():
    size = TILE_SIZE * MULTI_CORE * 2
    x, y = make_data(size, torch.float32)
    cond = make_condition(size, "alternating", torch.float32)
    out = where_with_cond_launch(cond, x, y, core_num=MULTI_CORE)
    expected = torch.where(cond.bool(), x, y)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scalar_on_true, scalar_value", [(False, -7), (True, 7)],
                         ids=["tensor_then_scalar", "scalar_then_tensor"])
def test_where_scalar_source_layouts(require_c310, scalar_on_true: bool, scalar_value: int, dtype: torch.dtype):
    check_dtype(dtype, require_c310)
    x, _ = make_data(TILE_SIZE, dtype)
    out = where_scalar_source_launch(x, scalar_value=scalar_value, scalar_on_true=scalar_on_true)
    scalar_tensor = torch.full_like(x, scalar_value)
    if scalar_on_true:
        expected = torch.where(x > 0, scalar_tensor, x)
    else:
        expected = torch.where(x > 0, x, scalar_tensor)
    torch.testing.assert_close(out, expected)
