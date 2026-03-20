# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

tests = [
    [asc2.reduce_prod, torch.prod, [64, 32], torch.float32, False, 0],
    [asc2.reduce_prod, torch.prod, [64, 32], torch.float32, False, 1],
    # [asc2.reduce_prod, torch.prod, [16, 16], torch.float32, None], -  no L2 api
    [asc2.reduce_min, torch.amin, [64, 32], torch.float32, False, 0],
    [asc2.reduce_min, torch.amin, [64, 32], torch.float32, False, 1],
    # [asc2.reduce_min, torch.amin, [32, 32], torch.float32, None], - no sync
    [asc2.reduce_sum, torch.sum, [64, 32], torch.float32, False, 0],
    [asc2.reduce_sum, torch.sum, [64, 32], torch.float32, False, 1],
    # [asc2.reduce_sum, torch.sum, [32, 32], torch.float32, None], - no sync
    [asc2.reduce_max, torch.amax, [64, 32], torch.float32, False, 0],
    [asc2.reduce_max, torch.amax, [64, 32], torch.float32, False, 1],
    # [asc2.reduce_max, torch.amax, [32, 32], torch.float32, None], - no sync
    [asc2.reduce_prod, torch.prod, [64, 32], torch.float32, True, 0],
]


@asc2.jit(always_compile=True)
def kernel(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, reduce_dim: asc.ConstExpr,
           input_shape: asc.ConstExpr, input_offsets: asc.ConstExpr, output_shape: asc.ConstExpr,
           output_offsets: asc.ConstExpr, op: asc.ConstExpr, keep_dims: asc.ConstExpr) -> None:
    g_input = asc2.tensor(input_ptr, input_shape)
    g_output = asc2.tensor(output_ptr, output_shape)
    input = asc2.load(g_input, input_shape, offsets=input_offsets)
    output = op(input, reduce_dim, keep_dims=keep_dims)
    asc2.store(output, g_output, offsets=output_offsets)


@asc2.jit(always_compile=True)
def kernel_all(input_ptr: asc.GlobalAddress, output_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr,
               input_offsets: asc.ConstExpr, output_shape: asc.ConstExpr, output_offsets: asc.ConstExpr,
               op: asc.ConstExpr) -> None:
    g_input = asc2.tensor(input_ptr, input_shape)
    g_output = asc2.tensor(output_ptr, output_shape)
    input = asc2.load(g_input, input_shape, offsets=input_offsets)
    scalar = op(input)
    output = asc2.full(output_shape, scalar, dtype=input.dtype)
    asc2.store(output, g_output, offsets=output_offsets)


@pytest.mark.parametrize("op, torch_op, shape, dtype, keep_dims, dim", tests)
def test_reduce(backend: config.Backend, platform: config.Platform, op, torch_op, shape, dtype, keep_dims, dim):
    if dim and platform != config.Platform.Ascend910_9599:
        pytest.skip("platform is not supported")
    config.set_platform(backend, platform, check=False)

    input = torch.randn(shape, dtype=dtype) * 2.0

    if dim is None:
        output_shape = [32]
    else:
        output_shape = list(shape)
        if keep_dims:
            output_shape[dim] = 1
        else:
            del output_shape[dim]

    output = torch.zeros(output_shape, dtype=dtype)
    input_offsets = [0] * len(shape)
    output_offsets = [0] * len(output_shape)

    if dim is None:
        kernel_all[1](input, output, shape, input_offsets, output_shape, output_offsets, op)
        expected = torch.ones(output_shape, dtype=dtype) * torch_op(input).item()
    else:
        kernel[1](input, output, dim, shape, input_offsets, output_shape, output_offsets, op, keep_dims)
        expected = torch_op(input, dim, keepdim=keep_dims)
    torch.testing.assert_close(output, expected)
