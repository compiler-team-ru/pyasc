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


@asc2.jit(always_compile=True, reuse_ub=True)
def rms_norm_kernel(input_ptr: asc2.GlobalAddress, gamma_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress,
                    eps: asc2.ConstExpr, size: int, tile_size: asc2.ConstExpr, total_blocks: int,
                    num_blocks: asc2.ConstExpr, norm_shape: asc2.ConstExpr):
    input_gm = asc2.tensor(input_ptr, [size])
    gamma_gm = asc2.tensor(gamma_ptr, [norm_shape])
    out_gm = asc2.tensor(out_ptr, [size])
    gamma = asc2.load(gamma_gm, [0], [norm_shape])
    loop_num = asc2.ceildiv(asc2.ceildiv(total_blocks, asc2.block_num()), num_blocks)
    block_offset = asc2.block_idx() * tile_size * loop_num
    for i in asc2.range(loop_num, unroll_factor=2, parallel=True):
        offset = block_offset + i * tile_size
        input_tensor = asc2.load(input_gm, [offset], [tile_size])
        tensor = input_tensor.reshape([num_blocks, norm_shape])
        out = asc2.rms_norm(tensor, gamma, eps)
        out = out.reshape(tile_size)
        asc2.store(out, out_gm, [offset])


def rms_norm_launch(x: torch.Tensor, gamma, eps=1e-6):
    output = torch.empty_like(x)
    input_size = torch.numel(x)
    norm_shape = torch.numel(gamma)
    total_blocks = input_size // norm_shape
    num_blocks = 1
    tile_size = num_blocks * norm_shape
    rms_norm_kernel[16](x, gamma, output, eps, input_size, tile_size, total_blocks, num_blocks, norm_shape)
    return output


def rms_norm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * gamma


@pytest.mark.parametrize("input_type", [torch.float16, torch.float32])
def test_rms_norm(input_type):
    x = torch.rand((128, 128), dtype=input_type)
    gamma = torch.rand(128, dtype=input_type)
    out = rms_norm_launch(x, gamma)
    torch.testing.assert_close(out, rms_norm_torch(x, gamma), rtol=1e-3, atol=1e-3)
