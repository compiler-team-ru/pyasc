# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


@asc2.jit(always_compile=True)
def matmul_relu_quant_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                             a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr,
                             tile_a: asc2.ConstExpr, tile_b: asc2.ConstExpr, quant_type: asc2.ConstExpr):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    a = asc2.load(a_gm, tile_a, offsets=[0, 0], location=asc2.TileLocation.L0A)
    b = asc2.load(b_gm, tile_b, offsets=[0, 0], location=asc2.TileLocation.L0B)
    c = a @ b
    c = asc2.relu(c).to(quant_type)
    asc2.store(c, c_gm, offsets=[0, 0])


@pytest.mark.parametrize("m, k, n, dtype, tile_a, tile_b, quant_type, quant_type_torch", [
    (64, 128, 128, torch.float32, [64, 128], [128, 128], asc2.float16, torch.float16),
    (64, 128, 256, torch.float16, [64, 128], [128, 256], asc2.float16, torch.float16),
    (64, 128, 256, torch.bfloat16, [64, 128], [128, 256], asc2.float16, torch.float16),
    (47, 21, 35, torch.float16, [47, 21], [21, 35], asc2.bfloat16, torch.bfloat16),
    (1, 32, 11, torch.float16, [1, 32], [32, 11], asc2.bfloat16, torch.bfloat16),
    (11, 19, 41, torch.float32, [11, 19], [19, 41], asc2.bfloat16, torch.bfloat16),
    (15, 67, 27, torch.bfloat16, [15, 67], [67, 27], asc2.float16, torch.float16),
    (19, 1, 19, torch.bfloat16, [19, 1], [1, 19], asc2.float16, torch.float16),
])
def test_matmul_relu_quant(m, k, n, dtype, tile_a, tile_b, quant_type, quant_type_torch):
    torch.manual_seed(0)
    a = (torch.rand((m, k), dtype=dtype) - .5) * 10
    b = (torch.rand((k, n), dtype=dtype) - .5) * 10
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=quant_type_torch)
    matmul_relu_quant_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_a, tile_b, quant_type)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).relu().to(quant_type_torch)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asc2.jit(always_compile=True)
def matmul_hf32_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                       a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr,
                       tile_a: asc2.ConstExpr, tile_b: asc2.ConstExpr):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    a = asc2.load(a_gm, tile_a, offsets=[0, 0], location=asc2.TileLocation.L0A)
    b = asc2.load(b_gm, tile_b, offsets=[0, 0], location=asc2.TileLocation.L0B)
    c = asc2.matmul(a, b, hf32=True)
    asc2.store(c, c_gm, offsets=[0, 0])


def test_matmul_hf32():
    torch.manual_seed(0)
    a = torch.rand((32, 64), dtype=torch.float32)
    b = torch.rand((64, 64), dtype=torch.float32)
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_hf32_kernel[1](a, b, c, a.shape, b.shape, c.shape, [32, 64], [64, 64])
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asc2.jit(always_compile=True)
def matmul_l0c_l1_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                         a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr,
                         dtype: asc2.ConstExpr):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    a = asc2.load(a_gm, a_shape, offsets=[0, 0], location=asc2.TileLocation.L0A)
    b = asc2.load(b_gm, b_shape, offsets=[0, 0], location=asc2.TileLocation.L0B)
    c = a @ b
    c = c.to(dtype)
    c_l1 = asc2.copy(c, c_shape, offsets=[0, 0], location=asc2.TileLocation.L1)
    c_l0a = asc2.copy(c_l1, c_shape, offsets=[0, 0], location=asc2.TileLocation.L0A)
    result = c_l0a @ b
    asc2.store(result, c_gm, offsets=[0, 0])


@pytest.mark.parametrize("m, k, n, torch_dtype, pyasc_dtype", [
    (32, 64, 64, torch.float16, asc2.float16),
    (64, 64, 64, torch.bfloat16, asc2.bfloat16),
])
def test_matmul_l0c_l1(m, k, n, torch_dtype, pyasc_dtype):
    torch.manual_seed(0)
    a = (torch.rand((m, k), dtype=torch_dtype) - .5) * 10
    b = (torch.rand((k, n), dtype=torch_dtype) - .5) * 10
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_l0c_l1_kernel[1](a, b, c, a.shape, b.shape, c.shape, pyasc_dtype)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch_dtype)
    c_ref = c_ref.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asc2.jit(always_compile=True)
def matmul_transpose_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                            a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr,
                            tile_a: asc2.ConstExpr, tile_b: asc2.ConstExpr):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    a = asc2.load(a_gm, tile_a, offsets=[0, 0], location=asc2.TileLocation.L1)
    b = asc2.load(b_gm, tile_b, offsets=[0, 0], location=asc2.TileLocation.L1)
    a_0 = asc2.copy(a, offsets=[0, 0], location=asc2.TileLocation.L0A)
    a_0_transpose = asc2.transpose(a_0)
    b_0 = asc2.copy(b, offsets=[0, 0], location=asc2.TileLocation.L0B).T
    c = a_0_transpose @ b_0
    asc2.store(c, c_gm, offsets=[0, 0])


@pytest.mark.parametrize("m, k, n, dtype, tile_a, tile_b", [
    (64, 64, 64, torch.float16, [64, 64], [64, 64]),
    (128, 32, 64, torch.float16, [32, 128], [64, 32]),
    (32, 32, 32, torch.bfloat16, [32, 32], [32, 32]),
    (32, 128, 64, torch.bfloat16, [128, 32], [64, 128]),
    (32, 32, 32, torch.float32, [32, 32], [32, 32]),
    (64, 32, 128, torch.float32, [32, 64], [128, 32]),
])
def test_matmul_transpose(m, k, n, dtype, tile_a, tile_b):
    torch.manual_seed(0)
    a = (torch.rand((k, m), dtype=dtype) - .5) * 10
    b = (torch.rand((n, k), dtype=dtype) - .5) * 10
    c = torch.zeros((a.shape[1], b.shape[0]), dtype=torch.float32)
    matmul_transpose_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_a, tile_b)
    c_ref = a.T.to(torch.float32) @ b.T.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
