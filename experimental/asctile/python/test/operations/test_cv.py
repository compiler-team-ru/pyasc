# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.experimental import asctile
import pytest
import torch


@pytest.fixture(autouse=True)
def require_c310_auto(require_c310):
    require_c310()


@asctile.jit(always_compile=True)
def matmul_add_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                      a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    a = asctile.copy_in(a_gm, [0, 0], a_shape)  # test "auto" location
    b = asctile.copy_in(b_gm, [0, 0], b_shape, "L0B")
    c = a @ b
    res = c + c  # test implicit "copy"
    asctile.copy_out(res, c_gm, [0, 0])


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


@asctile.jit(always_compile=True)
def matmul_ub_l1_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                        a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr,
                        tile_k: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    a_ub = asctile.copy_in(a_gm, [0, 0], a_shape, asctile.TensorLocation.UB)
    b_ub = asctile.copy_in(b_gm, [0, 0], b_shape, asctile.TensorLocation.UB)
    acc = asctile.zeros_acc(c_shape, dtype=asctile.float32)
    k_tiles = a_shape[1] // tile_k
    for i in asctile.range(k_tiles, unroll_factor=2):
        a_l1 = asctile.copy(a_ub, [0, i * tile_k], [a_shape[0], tile_k], asctile.TensorLocation.L1)
        b_l1 = asctile.copy(b_ub, [i * tile_k, 0], [tile_k, b_shape[1]], asctile.TensorLocation.L1)
        a_l0a = asctile.copy(a_l1, [0, 0], [a_shape[0], tile_k], asctile.TensorLocation.L0A)
        b_l0b = asctile.copy(b_l1, [0, 0], [tile_k, b_shape[1]], asctile.TensorLocation.L0B)
        asctile.matmul_acc(acc, a_l0a, b_l0b)
    c_ub = asctile.copy(acc, location=asctile.TensorLocation.UB)
    asctile.copy_out(c_ub, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, dtype, tile_k", [
    (16, 16, 16, torch.float16, 16),
    (16, 32, 16, torch.float16, 16),
    (16, 128, 16, torch.float16, 32),
    (128, 32, 64, torch.float16, 16),
    (16, 16, 16, torch.float32, 16),
    (16, 32, 16, torch.float32, 16),
    (1, 32, 16, torch.float32, 16),
    (1, 32, 16, torch.float16, 16),
    (8, 16, 16, torch.float32, 16),
    (8, 32, 16, torch.float32, 16),
    (16, 16, 8, torch.float32, 16),
    (16, 32, 8, torch.float32, 16),
    (16, 8, 16, torch.float32, 8),
    (16, 16, 16, torch.float32, 8),
])
def test_matmul_ub_l1(m, k, n, dtype, tile_k):
    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.zeros((m, n), dtype=torch.float32)
    matmul_ub_l1_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_k)
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)


@asctile.jit(always_compile=True)
def ub_to_gm_sync_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                         workspace_ptr: asctile.GlobalAddress, M: asctile.ConstExpr[int], K: asctile.ConstExpr[int],
                         N: asctile.ConstExpr[int]):
    a_gm = asctile.global_tensor(a_ptr, [M, K])
    workspace_gm = asctile.global_tensor(workspace_ptr, [M, K])
    b_gm = asctile.global_tensor(b_ptr, [K, N])
    c_gm = asctile.global_tensor(c_ptr, [M, N])
    a_ub = asctile.copy_in(a_gm, [0, 0], [M, K])
    a_doubled = a_ub + a_ub
    asctile.copy_out(a_doubled, workspace_gm, [0, 0])
    asc.sync_all()
    a_l1 = asctile.copy_in(workspace_gm, [0, 0], [M, K], asctile.TensorLocation.L1)
    b_l1 = asctile.copy_in(b_gm, [0, 0], [K, N], asctile.TensorLocation.L1)
    result = asctile.matmul(a_l1, b_l1)
    asctile.copy_out(result, c_gm, [0, 0])


@pytest.mark.parametrize("M, K, N", [(32, 32, 32)])
def test_ub_to_gm_sync(M, K, N):
    a = torch.rand(M, K, dtype=torch.float16)
    b = torch.rand(K, N, dtype=torch.float16)
    workspace = torch.zeros(M, K, dtype=torch.float16)
    c = torch.zeros(M, N, dtype=torch.float32)
    ub_to_gm_sync_kernel[1](a, b, c, workspace, M, K, N)
    c_ref = (a + a).to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asctile.jit(always_compile=True)
def cube_to_gm_sync_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                           workspace_ptr: asctile.GlobalAddress, M: asctile.ConstExpr[int], K: asctile.ConstExpr[int],
                           N: asctile.ConstExpr[int]):
    a_gm = asctile.global_tensor(a_ptr, [M, K])
    workspace_gm = asctile.global_tensor(workspace_ptr, [M, N])
    b_gm = asctile.global_tensor(b_ptr, [K, N])
    c_gm = asctile.global_tensor(c_ptr, [M, N])
    a_l1 = asctile.copy_in(a_gm, [0, 0], [M, K])
    b_l1 = asctile.copy_in(b_gm, [0, 0], [K, N])
    result = asctile.matmul(a_l1, b_l1)
    asctile.copy_out(result, workspace_gm, [0, 0])
    asc.sync_all()
    a_ub = asctile.copy_in(workspace_gm, [0, 0], [M, N])
    a_doubled = a_ub + a_ub
    asctile.copy_out(a_doubled, c_gm, [0, 0])


@pytest.mark.parametrize("M, K, N", [(32, 32, 32)])
def test_cube_to_gm_sync(M, K, N):
    a = torch.rand(M, K, dtype=torch.float16)
    b = torch.rand(K, N, dtype=torch.float16)
    workspace = torch.zeros(M, N, dtype=torch.float32)
    c = torch.zeros(M, N, dtype=torch.float32)
    cube_to_gm_sync_kernel[1](a, b, c, workspace, M, K, N)
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    c_ref = c_ref + c_ref
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
