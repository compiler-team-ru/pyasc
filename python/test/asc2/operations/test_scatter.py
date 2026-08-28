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


@asc2.jit(always_compile=True)
def scatter_kernel(result_ptr: asc2.GlobalAddress, changes_ptr: asc2.GlobalAddress, index_ptr: asc2.GlobalAddress,
                   write_count: int, index_count: int, read_count: asc2.ConstExpr, offset: int,
                   check_bounds: asc2.ConstExpr, row_len: asc2.ConstExpr) -> None:

    result_tensor = asc2.global_tensor(result_ptr, [write_count, row_len])
    index_tensor = asc2.global_tensor(index_ptr, [index_count])
    changes_tensor = asc2.global_tensor(changes_ptr, [index_count, row_len])

    index = asc2.copy_in(index_tensor, [0], [read_count])
    data = asc2.copy_in(changes_tensor, [0, 0], [read_count, row_len])
    asc2.scatter(index, data, result_tensor, [offset], 0, check_bounds=check_bounds)


@asc2.jit(always_compile=True)
def scatter_kernel_realshape(result_ptr: asc2.GlobalAddress, changes_ptr: asc2.GlobalAddress,
                             index_ptr: asc2.GlobalAddress, write_count: int, index_count: int,
                             read_count: asc2.ConstExpr, offset: int, check_bounds: asc2.ConstExpr,
                             row_len: asc2.ConstExpr, real_shape: int) -> None:

    result_tensor = asc2.global_tensor(result_ptr, [write_count, row_len])
    index_tensor = asc2.global_tensor(index_ptr, [index_count])
    changes_tensor = asc2.global_tensor(changes_ptr, [index_count, row_len])

    index = asc2.copy_in(index_tensor, [0], [read_count])
    data = asc2.copy_in(changes_tensor, [0, 0], [read_count, row_len])
    asc2.scatter(index, data, result_tensor, [offset], 0, real_shape=real_shape, check_bounds=check_bounds)


@pytest.mark.parametrize("index_dtype", (torch.int8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("data_dtype",
                         (torch.int8, torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("data_count, index_count, index_range, read_count, row_len, offset, check_bounds, real_shape",
                         [(32, 32, 32, 32, 32, 0, False, None),  # simple case
                          (32, 32, 32, 32, 32, 0, False, 16),  # runtime real_shape
                          (32, 32, 64, 32, 32, 0, True, None),  # out-of-bounds indexes
                          (32, 32, 32, 32, 32, 16, True, None),  # read with offset
                          ])
def test_scatter(data_count, index_count, index_range, read_count, row_len, offset, index_dtype, check_bounds,
                 data_dtype, real_shape, require_c310):
    require_c310()
    input = torch.arange(data_count * row_len).to(data_dtype).reshape([data_count, row_len])
    torch.manual_seed(0)
    index = torch.randint(0, index_range, size=(index_count, ), dtype=index_dtype)
    data = torch.arange(index_count * row_len).to(data_dtype).reshape([index_count, row_len])

    golden = input.clone().detach()
    if real_shape is None:
        scatter_kernel[1](input, data, index, data_count, index_count, read_count, offset, check_bounds, row_len)
    else:
        scatter_kernel_realshape[1](input, data, index, data_count, index_count, read_count, offset, check_bounds,
                                    row_len, real_shape)

    real_shape = real_shape or read_count
    for row in range(0, real_shape):
        for col in range(0, row_len):
            if index[row] + offset < data_count:
                golden[index[row] + offset][col] = data[row][col]
    torch.testing.assert_close(golden[:real_shape], input[:real_shape])
