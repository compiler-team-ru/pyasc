# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

import pytest
import torch

import asc
import asc.runtime.config as config
import asc2


@asc2.jit(always_compile=True)
def gelu_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, num_rows: asc.ConstExpr,
                num_columns: asc.ConstExpr, tile_size: asc.ConstExpr, approximate: asc.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [num_rows, num_columns])
    out_gm = asc2.tensor(out_ptr, [num_rows, num_columns])
    for i in range(asc2.block_idx(), num_rows, asc2.block_num(), parallel=True):
        row = asc2.load(x_gm, [1, tile_size], offsets=[i, 0])
        if approximate:
            pi = 3.141592653589793238462643383279502884
            k1 = 2 * asc2.sqrt(2 / pi)
            k2 = 2 * asc2.sqrt(2 / pi) * 0.044715
            param = row * (row * row * k2 + k1)
            out = row - row / (asc2.exp(param) + 1)
        else:
            k = asc2.sqrt(0.5)
            out = row * (asc2.erf(row * k) + 1) * 0.5
        asc2.store(out, out_gm, offsets=[i, 0])


def gelu_launch(x: torch.Tensor, approximate: bool):
    output = torch.empty_like(x)
    num_rows, num_columns = x.shape
    core_num = 16
    tile_size = 1024
    gelu_kernel[core_num](x, output, num_rows, num_columns, tile_size, approximate)
    return output


def gelu_torch(x: torch.Tensor, approximate: bool):
    if approximate:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))


@pytest.mark.parametrize("approximate", [True, False])
def test_gelu(backend: config.Backend, platform: config.Platform, approximate: bool):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    x = torch.rand((30, 1024), dtype=torch.float32, device=device)
    out = gelu_launch(x, approximate)
    torch.testing.assert_close(out, gelu_torch(x, approximate), rtol=1e-3, atol=1e-5)
