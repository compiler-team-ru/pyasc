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


@pytest.fixture(autouse=True)
def require_c310_auto(require_c310):
    require_c310()


@pytest.mark.parametrize(
    "dtype",
    (torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("permute, input_shape, real_shape", [
    # 2d cases
    [[1, 0], [16, 32], [16, 32]],
    # 3d cases
    [[0, 2, 1], [2, 16, 32], [2, 16, 32]],
    [[1, 2, 0], [8, 2, 32], [8, 2, 32]],
    [[1, 0, 2], [2, 4, 32], [2, 4, 32]],
    [[2, 0, 1], [2, 16, 32], [2, 16, 32]],
    [[2, 1, 0], [8, 2, 32], [8, 2, 32]],
    # 4d cases
    [[0, 1, 3, 2], [2, 3, 16, 16], [2, 3, 16, 16]],
    [[3, 2, 1, 0], [16, 2, 3, 16], [16, 2, 3, 16]],
    [[3, 1, 2, 0], [32, 2, 3, 16], [32, 2, 3, 16]],
    [[2, 0, 1, 3], [2, 3, 4, 32], [2, 3, 4, 32]],
    [[0, 2, 1, 3], [2, 3, 4, 16], [2, 3, 4, 16]],
    # default
    [[], [64, 32], [64, 32]],
    [[], [2, 32, 16], [2, 32, 16]],
    [[], [3, 2, 32, 16], [3, 2, 32, 16]],
    # with real_shape
    [[2, 0, 1], [3, 32, 64], [3, 30, 50]],
    [[0, 2, 1], [2, 64, 32], [1, 50, 10]],
    [[3, 2, 0, 1], [16, 2, 16, 32], [10, 2, 10, 30]],
    [[1, 0], [32, 64], [30, 60]],
])
def test_transpose(permute, input_shape, real_shape, dtype):
    dim_order = permute
    if len(permute) == 0:
        dim_order = list(range(0, len(input_shape)))
        dim_order[-1], dim_order[-2] = dim_order[-2], dim_order[-1]

    output_shape = [input_shape[i] for i in dim_order]
    write_real_shape = [real_shape[i] for i in dim_order]
    input = torch.rand(input_shape).mul(100).to(dtype=dtype)
    result = torch.zeros(output_shape, dtype=dtype)
    items_align = 32 // input.element_size()
    if output_shape[-1] % items_align != 0 or input_shape[-1] % items_align != 0:
        pytest.skip("data is not 32 byte aligned")

    golden = input
    for i in range(0, len(real_shape)):
        golden = golden.narrow(i, 0, real_shape[i])
    golden = golden.permute(dim_order)

    @asc2.jit(always_compile=True)
    def kernel(input_ptr, result_ptr, input_shape: asc2.ConstExpr, real_shape: asc2.ConstExpr,
               output_shape: asc2.ConstExpr, write_real_shape: asc2.ConstExpr, permute: asc2.ConstExpr):
        g_input = asc2.tensor(input_ptr, input_shape)
        tile = asc2.load(g_input, [0] * len(input_shape), input_shape, real_shape=real_shape)
        g_output = asc2.tensor(result_ptr, output_shape)
        asc2.store(tile.transpose(*permute), g_output, [0] * len(output_shape), real_shape=write_real_shape)

    kernel[1](input, result, input_shape, real_shape, output_shape, write_real_shape, permute)
    for i in range(0, len(write_real_shape)):
        result = result.narrow(i, 0, write_real_shape[i])
    torch.testing.assert_close(result, golden)
