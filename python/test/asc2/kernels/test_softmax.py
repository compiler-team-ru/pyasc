# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch

import asc
import asc.runtime.config as config
import asc2


@asc2.jit(always_compile=True, vf_fusion=True)
def softmax_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, num_rows: int, num_cols: int,
                   tile_size: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, num_cols])
    for i in range(asc2.block_idx(), num_rows, asc2.block_num(), parallel=True):
        row = asc2.load(x_gm, [1, tile_size], offsets=[i, 0])
        row_minus_max = row - asc2.reduce_max(row)
        numerator = asc2.exp(row_minus_max)
        denominator = asc2.reduce_sum(numerator)
        out = numerator / denominator
        asc2.store(out, out_gm, offsets=[i, 0])


def softmax_launch(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    num_rows, num_cols = x.shape
    core_num = 16
    softmax_kernel[core_num](x, out, num_rows, num_cols, tile_size=1024)
    return out


def test_softmax(backend: config.Backend, platform: config.Platform, device_id: int):
    config.set_platform(backend, platform, device_id)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    x = torch.rand((64, 1024), dtype=torch.float32, device=device)
    out = softmax_launch(x)
    torch.testing.assert_close(out, torch.softmax(x, dim=1))
