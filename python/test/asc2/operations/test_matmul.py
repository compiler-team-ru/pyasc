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


@asc2.jit(always_compile=True)
def matmul_relu_quant_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                             a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr,
                             tile_a: asc2.ConstExpr, tile_b: asc2.ConstExpr, quant_type: asc2.ConstExpr):
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    a = asc2.copy_in(a_gm, [0, 0], tile_a, asc2.TensorLocation.L0A)
    b = asc2.copy_in(b_gm, [0, 0], tile_b, asc2.TensorLocation.L0B)
    c = a @ b
    c = asc2.relu(c).to(quant_type)
    asc2.copy_out(c, c_gm, [0, 0])


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
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    a = asc2.copy_in(a_gm, [0, 0], tile_a, asc2.TensorLocation.L0A)
    b = asc2.copy_in(b_gm, [0, 0], tile_b, asc2.TensorLocation.L0B)
    c = asc2.matmul(a, b, hf32=True)
    asc2.copy_out(c, c_gm, [0, 0])


def test_matmul_hf32():
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
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    a = asc2.copy_in(a_gm, [0, 0], a_shape, asc2.TensorLocation.L0A)
    b = asc2.copy_in(b_gm, [0, 0], b_shape, asc2.TensorLocation.L0B)
    c = a @ b
    c = c.to(dtype)
    c_l1 = asc2.copy(c, [0, 0], c_shape, asc2.TensorLocation.L1)
    c_l0a = asc2.copy(c_l1, [0, 0], c_shape, asc2.TensorLocation.L0A)
    result = c_l0a @ b
    asc2.copy_out(result, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, torch_dtype, pyasc_dtype", [
    (32, 64, 64, torch.float16, asc2.float16),
    (64, 64, 64, torch.bfloat16, asc2.bfloat16),
])
def test_matmul_l0c_l1(m, k, n, torch_dtype, pyasc_dtype):
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
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    a = asc2.copy_in(a_gm, [0, 0], tile_a, asc2.TensorLocation.L1)
    b = asc2.copy_in(b_gm, [0, 0], tile_b, asc2.TensorLocation.L1)
    a_0 = asc2.copy(a, [0, 0], location=asc2.TensorLocation.L0A)
    a_0_transpose = asc2.transpose(a_0)
    b_0 = asc2.copy(b, [0, 0], location=asc2.TensorLocation.L0B).T
    c = a_0_transpose @ b_0
    asc2.copy_out(c, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, dtype, tile_a, tile_b", [
    (64, 64, 64, torch.float16, [64, 64], [64, 64]),
    (128, 32, 64, torch.float16, [32, 128], [64, 32]),
    (32, 32, 32, torch.bfloat16, [32, 32], [32, 32]),
    (32, 128, 64, torch.bfloat16, [128, 32], [64, 128]),
    (32, 32, 32, torch.float32, [32, 32], [32, 32]),
    (64, 32, 128, torch.float32, [32, 64], [128, 32]),
])
def test_matmul_transpose(m, k, n, dtype, tile_a, tile_b):
    a = (torch.rand((k, m), dtype=dtype) - .5) * 10
    b = (torch.rand((n, k), dtype=dtype) - .5) * 10
    c = torch.zeros((a.shape[1], b.shape[0]), dtype=torch.float32)
    matmul_transpose_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_a, tile_b)
    c_ref = a.T.to(torch.float32) @ b.T.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asc2.jit(always_compile=True)
def matmul_bias_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, bias_ptr: asc2.GlobalAddress,
                       c_ptr: asc2.GlobalAddress, a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr,
                       bias_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr, n_tiles: asc2.ConstExpr):
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    bias_gm = asc2.global_tensor(bias_ptr, bias_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    bias_l1 = asc2.copy_in(bias_gm, [0], bias_shape, asc2.TensorLocation.L1)
    tile_n = c_shape[1] // n_tiles
    for j in asc2.range(n_tiles, unroll_factor=n_tiles):
        bias_bt = asc2.copy(bias_l1, [j * tile_n], [tile_n], asc2.TensorLocation.BT)
        a_l0a = asc2.copy_in(a_gm, [0, 0], [c_shape[0], a_shape[1]], asc2.TensorLocation.L0A)
        b_l0b = asc2.copy_in(b_gm, [0, j * tile_n], [b_shape[0], tile_n], asc2.TensorLocation.L0B)
        c = asc2.matmul(a_l0a, b_l0b, bias_bt)
        asc2.copy_out(c, c_gm, [0, j * tile_n])


@pytest.mark.parametrize("m, k, n, dtype, bias_dtype, n_tiles", [
    (64, 64, 64, torch.float16, torch.float16, 2),
    (64, 128, 64, torch.bfloat16, torch.bfloat16, 4),
    (32, 32, 32, torch.float32, torch.float32, 1),
    (64, 64, 64, torch.float16, torch.float16, 1),
])
def test_matmul_with_bias(m, k, n, dtype, bias_dtype, n_tiles):
    a = (torch.rand((m, k), dtype=dtype) - .5) * 10
    b = (torch.rand((k, n), dtype=dtype) - .5) * 10
    bias = (torch.rand((n, ), dtype=bias_dtype) - .5) * 10
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_bias_kernel[1](a, b, bias, c, a.shape, b.shape, bias.shape, c.shape, n_tiles)
    c_ref = a.to(torch.float32) @ b.to(torch.float32) + bias
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asc2.jit(always_compile=True)
def matmul_acc_bias_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, bias_ptr: asc2.GlobalAddress,
                           c_ptr: asc2.GlobalAddress, a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr,
                           bias_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr, k_tiles: asc2.ConstExpr):
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    bias_gm = asc2.global_tensor(bias_ptr, bias_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    bias_c1 = asc2.copy_in(bias_gm, [0], bias_shape, asc2.TensorLocation.L1)
    bias = asc2.copy(bias_c1, [0], bias_shape, asc2.TensorLocation.BT)
    acc = asc2.zeros_acc(c_shape, dtype=asc2.float32, bias=bias)
    k_offset = a_shape[1] // k_tiles
    a_l1 = asc2.copy_in(a_gm, [0, 0], a_shape, asc2.TensorLocation.L1)
    b_l1 = asc2.copy_in(b_gm, [0, 0], b_shape, asc2.TensorLocation.L1)
    for i in range(k_tiles, unroll_factor=1):
        a_i = asc2.copy(a_l1, [0, i * k_offset], [a_shape[0], k_offset], asc2.TensorLocation.L0A)
        b_i = asc2.copy(b_l1, [i * k_offset, 0], [k_offset, b_shape[1]], asc2.TensorLocation.L0B)
        asc2.matmul_acc(acc, a_i, b_i)
    asc2.copy_out(acc, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, dtype, bias_dtype, k_tiles", [
    (64, 64, 64, torch.float16, torch.float16, 2),
    (64, 128, 64, torch.bfloat16, torch.bfloat16, 4),
    (32, 32, 32, torch.float32, torch.float32, 1),
    (64, 64, 64, torch.float16, torch.float16, 1),
])
def test_matmul_acc_with_bias(m, k, n, dtype, bias_dtype, k_tiles):
    a = (torch.rand((m, k), dtype=dtype) - .5) * 10
    b = (torch.rand((k, n), dtype=dtype) - .5) * 10
    bias = (torch.rand((n, ), dtype=bias_dtype) - .5) * 10
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_acc_bias_kernel[1](a, b, bias, c, a.shape, b.shape, bias.shape, c.shape, k_tiles)
    c_ref = a.to(torch.float32) @ b.to(torch.float32) + bias
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asc2.jit(always_compile=True)
def matmul_add_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                      a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr):
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    a = asc2.copy_in(a_gm, [0, 0], a_shape, "L0A")
    b = asc2.copy_in(b_gm, [0, 0], b_shape, "L0B")
    c = a @ b
    c_ub = c.to(asc2.TensorLocation.UB)
    res = c_ub + c_ub
    asc2.copy_out(res, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n", [
    (16, 16, 16),
    (32, 64, 64),
    (64, 64, 64),
])
def test_matmul_add(m, k, n):
    a = (torch.rand((m, k), dtype=torch.float16) - .5) * 10
    b = (torch.rand((k, n), dtype=torch.float16) - .5) * 10
    c = torch.zeros((m, n), dtype=torch.float32)
    matmul_add_kernel[1](a, b, c, a.shape, b.shape, c.shape)
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    res_ref = c_ref + c_ref
    torch.testing.assert_close(c, res_ref, atol=1e-3, rtol=1e-3)


@asc2.jit(always_compile=True)
def matmul_ub_l1_kernel(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, c_ptr: asc2.GlobalAddress,
                        a_shape: asc2.ConstExpr, b_shape: asc2.ConstExpr, c_shape: asc2.ConstExpr,
                        tile_k: asc2.ConstExpr):
    a_gm = asc2.global_tensor(a_ptr, a_shape)
    b_gm = asc2.global_tensor(b_ptr, b_shape)
    c_gm = asc2.global_tensor(c_ptr, c_shape)
    a_ub = asc2.copy_in(a_gm, [0, 0], a_shape, asc2.TensorLocation.UB)
    b_ub = asc2.copy_in(b_gm, [0, 0], b_shape, asc2.TensorLocation.UB)
    acc = asc2.zeros_acc(c_shape, dtype=asc2.float32)
    k_tiles = a_shape[1] // tile_k
    for i in asc2.range(k_tiles, unroll_factor=2):
        a_l1 = asc2.copy(a_ub, [0, i * tile_k], [a_shape[0], tile_k], asc2.TensorLocation.L1)
        b_l1 = asc2.copy(b_ub, [i * tile_k, 0], [tile_k, b_shape[1]], asc2.TensorLocation.L1)
        a_l0a = asc2.copy(a_l1, [0, 0], [a_shape[0], tile_k], asc2.TensorLocation.L0A)
        b_l0b = asc2.copy(b_l1, [0, 0], [tile_k, b_shape[1]], asc2.TensorLocation.L0B)
        asc2.matmul_acc(acc, a_l0a, b_l0b)
    c_ub = asc2.copy(acc, location=asc2.TensorLocation.UB)
    asc2.copy_out(c_ub, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, dtype, tile_k", [
    (16, 16, 16, torch.float16, 16),
    (16, 32, 16, torch.float16, 16),
    (16, 128, 16, torch.float16, 32),
    (128, 32, 64, torch.float16, 16),
    (16, 16, 16, torch.float32, 16),
    (16, 32, 16, torch.float32, 16),
])
def test_matmul_ub_l1(m, k, n, dtype, tile_k):
    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.zeros((m, n), dtype=torch.float32)
    matmul_ub_l1_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_k)
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)
