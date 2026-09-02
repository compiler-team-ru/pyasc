# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from typing import Callable, Tuple

import asctile
import pytest
import torch


@asctile.jit(always_compile=True)
def kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, reduce_dim: asctile.ConstExpr,
           input_shape: asctile.ConstExpr, input_offsets: asctile.ConstExpr, output_shape: asctile.ConstExpr,
           output_offsets: asctile.ConstExpr, op: asctile.ConstExpr) -> None:
    g_input = asctile.global_tensor(input_ptr, input_shape)
    g_output = asctile.global_tensor(output_ptr, output_shape)
    input = asctile.copy_in(g_input, input_offsets, input_shape)
    output = op(input, reduce_dim)
    if output.size == 1:
        output = asctile.broadcast_to(output, *output_shape)
    asctile.copy_out(output, g_output, output_offsets)


@asctile.jit(always_compile=True)
def kernel_all(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, input_shape: asctile.ConstExpr,
               input_offsets: asctile.ConstExpr, output_shape: asctile.ConstExpr, output_offsets: asctile.ConstExpr,
               op: asctile.ConstExpr) -> None:
    g_input = asctile.global_tensor(input_ptr, input_shape)
    g_output = asctile.global_tensor(output_ptr, output_shape)
    input = asctile.copy_in(g_input, input_offsets, input_shape)
    scalar = op(input)
    output = asctile.full(output_shape, scalar, dtype=input.dtype)
    asctile.copy_out(output, g_output, output_offsets)


@dataclass
class Op:
    asc_op: Callable
    torch_op: Callable
    basic_dtypes: Tuple[torch.dtype]
    adv_dtypes: Tuple[torch.dtype]


ops = (
    Op(asctile.reduce_sum, torch.sum, basic_dtypes=(torch.int64, torch.float16, torch.float32),
       adv_dtypes=(torch.int32, torch.int64, torch.float32)),
    Op(asctile.reduce_max, torch.amax,
       basic_dtypes=(torch.int16, torch.int32, torch.int64, torch.float16, torch.float32),
       adv_dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32)),
    Op(asctile.reduce_min, torch.amin,
       basic_dtypes=(torch.int16, torch.int32, torch.int64, torch.float16, torch.float32),
       adv_dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32)),
    Op(asctile.reduce_prod, torch.prod, basic_dtypes=(), adv_dtypes=(torch.float32, )),
)


@pytest.mark.parametrize("asc_op, torch_op, dtype, shape, dim", (
    *((op.asc_op, op.torch_op, dtype, shape, None)
      for op in ops
      for dtype in op.basic_dtypes
      for shape in ([32], [64, 32])),
    *((op.asc_op, op.torch_op, dtype, shape, dim)
      for op in ops
      for dtype in op.adv_dtypes
      for shape in ([64, 32], )
      for dim in range(len(shape))),
))
def test_reduce(require_c310, asc_op, torch_op, dtype: torch.dtype, shape, dim):
    if dim is not None or not dtype.is_floating_point:
        require_c310()
    if asc_op is asctile.reduce_sum and dtype == torch.float16 and dim is None:
        pytest.skip("accuracy mismatch on reduce_sum float16 to scalar")
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype) * 2.0
    else:
        input = torch.randint(-5, 5, shape, dtype=dtype)
    if dim is None or len(shape) == 1:
        output_shape = [16]
    else:
        output_shape = list(shape)
        del output_shape[dim]
    output = torch.zeros(output_shape, dtype=dtype)
    input_offsets = [0] * len(shape)
    output_offsets = [0] * len(output_shape)
    if dim is None:
        kernel_all[1](input, output, shape, input_offsets, output_shape, output_offsets, asc_op)
        expected = torch.ones(output_shape, dtype=dtype) * torch_op(input).item()
    else:
        kernel[1](input, output, dim, shape, input_offsets, output_shape, output_offsets, asc_op)
        expected = torch_op(input, dim).to(dtype)
    torch.testing.assert_close(output, expected)


@asctile.jit(always_compile=True)
def reduce_tile_kernel(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, size: int,
                       tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [1])
    tile = asctile.copy_in(x_gm, [0], [tile_size])
    max_val = asctile.reduce_max(tile)
    result = asctile.full([1], max_val, dtype=tile.dtype)
    asctile.copy_out(result, out_gm, [0])


@pytest.mark.parametrize("tile_size", [1, 7, 17])
def test_reduce_partial_tile(tile_size):
    tensor_size = 32
    x = torch.rand(tensor_size, dtype=torch.float32) * -10.0
    x[tile_size:] = 1000.0
    out = torch.empty(1, dtype=torch.float32)
    reduce_tile_kernel[1](x, out, tensor_size, tile_size=tile_size)
    expected = torch.amax(x[:tile_size]).unsqueeze(0)
    torch.testing.assert_close(out, expected)
