# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

import asctile
import pytest
import torch

USE_CORE_NUM = 4


@asctile.jit(always_compile=True)
def kernel(x_ptr: asctile.GlobalAddress, z_ptr: asctile.GlobalAddress, tensor_shape: asctile.ConstExpr,
           tile_length: asctile.ConstExpr, op: asctile.ConstExpr):
    offset_x = asctile.block_idx() * tile_length
    xt = asctile.copy_in(asctile.global_tensor(x_ptr, tensor_shape), [offset_x], [tile_length])
    xt += 10  # temporary tile to keep TQue synchronization valid
    op(xt, asctile.global_tensor(z_ptr, [tile_length]), offsets=[0])


@pytest.mark.parametrize("dtype", (torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("asc_op, torch_op", (
    (asctile.atomic_add, torch.add),
    (asctile.atomic_max, torch.maximum),
    (asctile.atomic_min, torch.minimum),
))
def test_atomic_op(require_c310, asc_op, torch_op, dtype):
    if dtype == torch.bfloat16:
        require_c310()  # due to use of addition in test kernel

    def create_input(shape):
        if dtype == torch.float32:
            res = torch.randn(tuple(shape), dtype=dtype)
            res = torch.clamp(res, 1, 100)
        else:
            res = torch.randint(1, 100, tuple(shape), dtype=dtype)
        return res

    tensor_shape = [128]
    size = math.prod(tensor_shape)
    tile_length = size // USE_CORE_NUM
    x = create_input(tensor_shape)
    z = create_input([tile_length])
    torch_z = z.clone()
    kernel[USE_CORE_NUM](x, z, tensor_shape, tile_length, asc_op)
    expected_z = torch_z
    for i in range(USE_CORE_NUM):
        x_block = x[i * tile_length:(i + 1) * tile_length] + 10
        expected_z = torch_op(expected_z, x_block)
    torch.testing.assert_close(z, expected_z)
