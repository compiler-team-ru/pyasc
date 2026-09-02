# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import pytest
import torch

unary_ops = [
    (asctile.abs, torch.abs, [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]),
    (asctile.bitwise_not, torch.bitwise_not, [torch.int8, torch.int16, torch.int32, torch.int64]),
    (asctile.ceil, torch.ceil, [torch.float16, torch.float32]),
    (asctile.cos, torch.cos, [torch.float16, torch.float32]),
    (asctile.cosh, torch.cosh, [torch.float16, torch.float32]),
    (asctile.erf, torch.erf, [torch.float16, torch.float32]),
    (asctile.exp, torch.exp, [torch.float16, torch.float32]),
    (asctile.exp2, torch.exp2, [torch.float16, torch.float32]),
    (asctile.floor, torch.floor, [torch.float16, torch.float32]),
    (asctile.log, torch.log, [torch.float16, torch.float32]),
    (asctile.log2, torch.log2, [torch.float16, torch.float32]),
    (asctile.negative, torch.neg, [torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16,
                                   torch.float32]),
    (asctile.relu, torch.relu, [torch.float16, torch.float32]),
    (asctile.rsqrt, torch.rsqrt, [torch.float16, torch.float32]),
    (asctile.sin, torch.sin, [torch.float16, torch.float32]),
    (asctile.sinh, torch.sinh, [torch.float16, torch.float32]),
    (asctile.sqrt, torch.sqrt, [torch.float16, torch.float32]),
    (asctile.tan, torch.tan, [torch.float16, torch.float32]),
    (asctile.tanh, torch.tanh, [torch.float16, torch.float32]),
]


@asctile.jit(always_compile=True)
def kernel(x_ptr: asctile.GlobalAddress, z_ptr: asctile.GlobalAddress, block_length: asctile.ConstExpr,
           op: asctile.ConstExpr) -> None:
    xt = asctile.copy_in(asctile.global_tensor(x_ptr, [32]), [0], [block_length])
    zt = op(xt)
    asctile.copy_out(zt, asctile.global_tensor(z_ptr, [32]), [0])


@pytest.mark.parametrize("asc_op, torch_op, dtype",
                         [(asc_op, torch_op, d) for asc_op, torch_op, dtypes in unary_ops for d in dtypes])
def test_unary_operations(require_c310, asc_op, torch_op, dtype):
    non_c310_dtypes = (torch.int16, ) if asc_op is asctile.bitwise_not else (torch.float16, torch.float32)
    if dtype not in non_c310_dtypes:
        require_c310()

    def create_input(dtype: torch.dtype):
        if dtype.is_floating_point:
            return torch.randn((size, ), dtype=dtype).clamp(1, 100)
        elif dtype.is_signed:
            return torch.randint(1, 100, (size, ), dtype=dtype)

    size = 32
    x = create_input(dtype)
    z = torch.zeros(size, dtype=dtype)
    kernel[1](x, z, size, asc_op)
    torch.testing.assert_close(z, torch_op(x))
