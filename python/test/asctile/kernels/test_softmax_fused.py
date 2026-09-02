# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import torch


@asctile.jit(always_compile=True)
def softmax_kernel(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, num_rows: int,
                   num_cols: asctile.ConstExpr[int], block_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [num_rows, num_cols])
    out_gm = asctile.global_tensor(out_ptr, [num_rows, num_cols])
    start_rows = asctile.block_idx() * block_size
    rows = asctile.copy_in(x_gm, [start_rows, 0], [block_size, num_cols])
    out = asctile.softmax(rows)
    asctile.copy_out(out, out_gm, [start_rows, 0])


def softmax_launch(x: torch.Tensor) -> torch.Tensor:
    core_num = 16
    out = torch.empty_like(x)
    num_rows, num_cols = x.shape
    block_size = (num_rows + core_num - 1) // core_num
    softmax_kernel[core_num](x, out, num_rows, num_cols, block_size)
    return out


def test_softmax():
    x = torch.rand((64, 1024), dtype=torch.float32)
    out = softmax_launch(x)
    torch.testing.assert_close(out, torch.softmax(x, dim=1))
