# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import itertools
import pytest
import torch


@pytest.fixture(autouse=True)
def require_c310_auto(require_c310):
    require_c310()


@asctile.jit(always_compile=True)
def matmul_relu_quant_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                             a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr,
                             tile_a: asctile.ConstExpr, tile_b: asctile.ConstExpr, quant_type: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    a = asctile.copy_in(a_gm, [0, 0], tile_a, asctile.TensorLocation.L0A)
    b = asctile.copy_in(b_gm, [0, 0], tile_b, asctile.TensorLocation.L0B)
    c = a @ b
    c = asctile.relu(c).to(quant_type)
    asctile.copy_out(c, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, dtype, tile_a, tile_b, quant_type, quant_type_torch", [
    (64, 128, 128, torch.float32, [64, 128], [128, 128], asctile.float16, torch.float16),
    (64, 128, 256, torch.float16, [64, 128], [128, 256], asctile.float16, torch.float16),
    (64, 128, 256, torch.bfloat16, [64, 128], [128, 256], asctile.float16, torch.float16),
    (47, 21, 35, torch.float16, [47, 21], [21, 35], asctile.bfloat16, torch.bfloat16),
    (1, 32, 11, torch.float16, [1, 32], [32, 11], asctile.bfloat16, torch.bfloat16),
    (11, 19, 41, torch.float32, [11, 19], [19, 41], asctile.bfloat16, torch.bfloat16),
    (15, 67, 27, torch.bfloat16, [15, 67], [67, 27], asctile.float16, torch.float16),
    (19, 1, 19, torch.bfloat16, [19, 1], [1, 19], asctile.float16, torch.float16),
    (24, 24, 24, torch.float8_e5m2, [32, 32], [32, 32], asctile.float16, torch.float16),
    (24, 24, 24, torch.float8_e4m3fn, [32, 32], [32, 32], asctile.bfloat16, torch.bfloat16),
])
def test_matmul_relu_quant(m, k, n, dtype, tile_a, tile_b, quant_type, quant_type_torch):
    a = ((torch.rand(m, k) - .5) * 10).to(dtype)
    b = ((torch.rand(k, n) - .5) * 10).to(dtype)
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=quant_type_torch)
    matmul_relu_quant_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_a, tile_b, quant_type)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).relu().to(quant_type_torch)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asctile.jit(always_compile=True)
def matmul_hf32_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                       a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    a = asctile.copy_in(a_gm, [0, 0], a_shape, asctile.TensorLocation.L0A)
    b = asctile.copy_in(b_gm, [0, 0], b_shape, asctile.TensorLocation.L0B)
    c = asctile.matmul(a, b, hf32=True)
    asctile.copy_out(c, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n", [
    (32, 64, 64),
    (8, 16, 16),
    (16, 8, 16),
    (16, 16, 8),
])
def test_matmul_hf32(m, k, n):
    dtype = torch.float32
    a = torch.rand((m, k), dtype=dtype)
    b = torch.rand((k, n), dtype=dtype)
    c = torch.zeros((m, n), dtype=dtype)
    matmul_hf32_kernel[1](a, b, c, a.shape, b.shape, c.shape)
    c_ref = a @ b
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asctile.jit(always_compile=True)
def matmul_l0c_l1_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                         a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr,
                         dtype: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    a = asctile.copy_in(a_gm, [0, 0], a_shape, asctile.TensorLocation.L0A)
    b = asctile.copy_in(b_gm, [0, 0], b_shape, asctile.TensorLocation.L0B)
    c = a @ b
    c = c.to(dtype)
    c_l1 = asctile.copy(c, [0, 0], c_shape, asctile.TensorLocation.L1)
    c_l0a = asctile.copy(c_l1, [0, 0], c_shape, asctile.TensorLocation.L0A)
    result = c_l0a @ b
    asctile.copy_out(result, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, torch_dtype, pyasc_dtype", [
    (32, 64, 64, torch.float16, asctile.float16),
    (64, 64, 64, torch.bfloat16, asctile.bfloat16),
])
def test_matmul_l0c_l1(m, k, n, torch_dtype, pyasc_dtype):
    a = (torch.rand((m, k), dtype=torch_dtype) - .5) * 10
    b = (torch.rand((k, n), dtype=torch_dtype) - .5) * 10
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_l0c_l1_kernel[1](a, b, c, a.shape, b.shape, c.shape, pyasc_dtype)
    c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch_dtype)
    c_ref = c_ref.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asctile.jit(always_compile=True)
def matmul_transpose_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                            a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr,
                            tile_a: asctile.ConstExpr, tile_b: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    a = asctile.copy_in(a_gm, [0, 0], tile_a, asctile.TensorLocation.L1)
    b = asctile.copy_in(b_gm, [0, 0], tile_b, asctile.TensorLocation.L1)
    a_0 = asctile.copy(a, [0, 0], location=asctile.TensorLocation.L0A)
    a_0_transpose = asctile.transpose(a_0)
    b_0 = asctile.copy(b, [0, 0], location=asctile.TensorLocation.L0B).T
    c = a_0_transpose @ b_0
    asctile.copy_out(c, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, dtype, tile_a, tile_b", [
    (64, 64, 64, torch.float16, [64, 64], [64, 64]),
    (128, 32, 64, torch.float16, [32, 128], [64, 32]),
    (32, 32, 32, torch.bfloat16, [32, 32], [32, 32]),
    (32, 128, 64, torch.bfloat16, [128, 32], [64, 128]),
    (32, 32, 32, torch.float32, [32, 32], [32, 32]),
    (64, 32, 128, torch.float32, [32, 64], [128, 32]),
    (32, 32, 32, torch.float8_e5m2, [32, 32], [32, 32]),
    (32, 32, 32, torch.float8_e4m3fn, [32, 32], [32, 32]),
])
def test_matmul_transpose(m, k, n, dtype, tile_a, tile_b):
    a = ((torch.rand(k, m) - .5) * 10).to(dtype)
    b = ((torch.rand(n, k) - .5) * 10).to(dtype)
    c = torch.zeros((a.shape[1], b.shape[0]), dtype=torch.float32)
    matmul_transpose_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_a, tile_b)
    c_ref = a.T.to(torch.float32) @ b.T.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asctile.jit(always_compile=True)
def matmul_bias_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, bias_ptr: asctile.GlobalAddress,
                       c_ptr: asctile.GlobalAddress, a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr,
                       bias_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr, n_tiles: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    bias_gm = asctile.global_tensor(bias_ptr, bias_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    bias_l1 = asctile.copy_in(bias_gm, [0], bias_shape, asctile.TensorLocation.L1)
    tile_n = c_shape[1] // n_tiles
    for j in asctile.range(n_tiles, unroll_factor=n_tiles):
        bias_bt = asctile.copy(bias_l1, [j * tile_n], [tile_n], asctile.TensorLocation.BT)
        a_l0a = asctile.copy_in(a_gm, [0, 0], [c_shape[0], a_shape[1]], asctile.TensorLocation.L0A)
        b_l0b = asctile.copy_in(b_gm, [0, j * tile_n], [b_shape[0], tile_n], asctile.TensorLocation.L0B)
        c = asctile.matmul(a_l0a, b_l0b, bias_bt)
        asctile.copy_out(c, c_gm, [0, j * tile_n])


@pytest.mark.parametrize("m, k, n, dtype, bias_dtype, n_tiles", [
    (64, 64, 64, torch.float16, torch.float16, 2),
    (64, 128, 64, torch.bfloat16, torch.bfloat16, 4),
    (32, 32, 32, torch.float32, torch.float32, 1),
    (64, 64, 64, torch.float16, torch.float16, 1),
    (32, 32, 32, torch.float8_e5m2, torch.float32, 1),
    (32, 32, 32, torch.float8_e4m3fn, torch.float32, 1),
])
def test_matmul_with_bias(m, k, n, dtype, bias_dtype, n_tiles):
    a = ((torch.rand(m, k) - .5) * 10).to(dtype)
    b = ((torch.rand(k, n) - .5) * 10).to(dtype)
    bias = (torch.rand((n, ), dtype=bias_dtype) - .5) * 10
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_bias_kernel[1](a, b, bias, c, a.shape, b.shape, bias.shape, c.shape, n_tiles)
    c_ref = a.to(torch.float32) @ b.to(torch.float32) + bias
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asctile.jit(always_compile=True)
def matmul_acc_bias_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, bias_ptr: asctile.GlobalAddress,
                           c_ptr: asctile.GlobalAddress, a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr,
                           bias_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr, k_tiles: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    bias_gm = asctile.global_tensor(bias_ptr, bias_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    bias_c1 = asctile.copy_in(bias_gm, [0], bias_shape, asctile.TensorLocation.L1)
    bias = asctile.copy(bias_c1, [0], bias_shape, asctile.TensorLocation.BT)
    acc = asctile.zeros_acc(c_shape, dtype=asctile.float32, bias=bias)
    k_offset = a_shape[1] // k_tiles
    a_l1 = asctile.copy_in(a_gm, [0, 0], a_shape, asctile.TensorLocation.L1)
    b_l1 = asctile.copy_in(b_gm, [0, 0], b_shape, asctile.TensorLocation.L1)
    for i in range(k_tiles, unroll_factor=1):
        a_i = asctile.copy(a_l1, [0, i * k_offset], [a_shape[0], k_offset], asctile.TensorLocation.L0A)
        b_i = asctile.copy(b_l1, [i * k_offset, 0], [k_offset, b_shape[1]], asctile.TensorLocation.L0B)
        asctile.matmul_acc(acc, a_i, b_i)
    asctile.copy_out(acc, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, dtype, bias_dtype, k_tiles", [
    (64, 64, 64, torch.float16, torch.float16, 2),
    (64, 128, 64, torch.bfloat16, torch.bfloat16, 4),
    (32, 32, 32, torch.float32, torch.float32, 1),
    (64, 64, 64, torch.float16, torch.float16, 1),
    (32, 32, 32, torch.float8_e5m2, torch.float32, 1),
])
def test_matmul_acc_with_bias(m, k, n, dtype, bias_dtype, k_tiles):
    a = ((torch.rand(m, k) - .5) * 10).to(dtype)
    b = ((torch.rand(k, n) - .5) * 10).to(dtype)
    bias = (torch.rand((n, ), dtype=bias_dtype) - .5) * 10
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_acc_bias_kernel[1](a, b, bias, c, a.shape, b.shape, bias.shape, c.shape, k_tiles)
    c_ref = a.to(torch.float32) @ b.to(torch.float32) + bias
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


@asctile.jit(always_compile=True)
def matmul_gm_l1_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                        a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr,
                        tile_k: asctile.ConstExpr):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    acc = asctile.zeros_acc(c_shape, dtype=asctile.float32)
    k_tiles = asctile.ceildiv(a_shape[1], tile_k)
    for i in range(k_tiles, unroll_factor=1):
        a_l1 = asctile.copy_in(a_gm, [0, i * tile_k], [a_shape[0], tile_k], asctile.TensorLocation.L1)
        b_l1 = asctile.copy_in(b_gm, [i * tile_k, 0], [tile_k, b_shape[1]], asctile.TensorLocation.L1)
        a_l0a = asctile.copy(a_l1, [0, 0], [a_shape[0], tile_k], asctile.TensorLocation.L0A)
        b_l0b = asctile.copy(b_l1, [0, 0], [tile_k, b_shape[1]], asctile.TensorLocation.L0B)
        asctile.matmul_acc(acc, a_l0a, b_l0b)
    asctile.copy_out(acc, c_gm, [0, 0])


@pytest.mark.parametrize("m, k, n, tile_k, dtype", [
    (256, 16, 64, 64, torch.float32),
    (256, 16, 65, 64, torch.float32),
    (256, 32, 65, 64, torch.float8_e4m3fn),
])
def test_matmul_gm_l1(m, k, n, tile_k, dtype):
    dtype = torch.float32
    a = torch.rand((m, k)).to(dtype)
    b = torch.rand((k, n)).to(dtype)
    c = torch.zeros((m, n), dtype=torch.float32)
    matmul_gm_l1_kernel[1](a, b, c, a.shape, b.shape, c.shape, tile_k)
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)


def pack_scale(input):
    block_x = 2
    block_y = 16
    target_shape = [
        asctile.ceildiv(input.shape[0], block_y) * block_y,
        asctile.ceildiv(input.shape[1], block_x) * block_x
    ]
    input = torch.nn.functional.pad(input,
                                    pad=(0, target_shape[1] - input.shape[1], 0, target_shape[0] - input.shape[0]),
                                    mode='constant', value=0)
    num_block_x = input.shape[1] // block_x
    num_block_y = input.shape[0] // block_y
    result = torch.empty([num_block_y, num_block_x, block_y, block_x], dtype=input.dtype)
    for bx, by, cx, cy in itertools.product(range(num_block_x), range(num_block_y), range(block_x), range(block_y)):
        data = input[block_y * by + cy][block_x * bx + cx]
        result[by][bx][cy][cx] = data
    return result


@asctile.jit
def matmul_scale_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, sa_ptr: asctile.GlobalAddress,
                        sb_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress, m: asctile.ConstExpr,
                        n: asctile.ConstExpr, k: asctile.ConstExpr, gs_m: asctile.ConstExpr, gs_n: asctile.ConstExpr,
                        gs_k: asctile.ConstExpr, scale_offset_a: asctile.ConstExpr, scale_offset_b: asctile.ConstExpr,
                        use_accum: asctile.ConstExpr):

    scale_k = asctile.ceildiv(k, 64)
    scale_m = asctile.ceildiv(m, 16)
    scale_n = asctile.ceildiv(n, 16)

    a_gm = asctile.global_tensor(a_ptr, [m, k])
    b_gm = asctile.global_tensor(b_ptr, [k, n])
    c_gm = asctile.global_tensor(c_ptr, [m, n])

    sa_gm = asctile.global_tensor(sa_ptr, [gs_m, gs_k, 16, 2])
    sb_gm = asctile.global_tensor(sb_ptr, [gs_n, gs_k, 16, 2])

    a1 = asctile.copy_in(a_gm, [0, 0], [m, k], asctile.TensorLocation.L1)
    b1 = asctile.copy_in(b_gm, [0, 0], [k, n], asctile.TensorLocation.L1)
    sa = asctile.copy_in(sa_gm, scale_offset_a + [0, 0], [scale_m, scale_k, 16, 2], location=asctile.TensorLocation.L1)
    sb = asctile.copy_in(sb_gm, scale_offset_b + [0, 0], [scale_n, scale_k, 16, 2], location=asctile.TensorLocation.L1)

    a0 = asctile.copy(a1, [0, 0], [m, k], location=asctile.TensorLocation.L0A)
    b0 = asctile.copy(b1, [0, 0], [k, n], location=asctile.TensorLocation.L0B)
    sa0 = asctile.copy(sa, [0, 0, 0, 0], [scale_m, scale_k, 16, 2], asctile.TensorLocation.L0A)
    sb0 = asctile.copy(sb, [0, 0, 0, 0], [scale_n, scale_k, 16, 2], asctile.TensorLocation.L0B)

    a0 = asctile.bind_scale(a0, sa0)
    b0 = asctile.bind_scale(b0, sb0)
    if use_accum:
        accum = asctile.zeros_acc([m, n], dtype=asctile.float32)
        asctile.matmul_acc(accum, a0, b0)
        asctile.copy_out(accum, c_gm, [0, 0])
    else:
        c = a0 @ b0
        asctile.copy_out(c, c_gm, [0, 0])


def matmul_scale_reference(a, b, a1, b1):
    a1_ = a1.view(torch.uint8).to(torch.float32) - 127
    b1_ = b1.view(torch.uint8).to(torch.float32) - 127
    scale_a = torch.repeat_interleave(torch.pow(2, a1_), 32, dim=1)
    scale_b = torch.repeat_interleave(torch.pow(2, b1_), 32, dim=0)

    if scale_a.shape[0] < a.shape[0] or scale_a.shape[1] < a.shape[1]:
        scale_a = torch.nn.functional.pad(scale_a,
                                          pad=(0, a.shape[1] - scale_a.shape[1], 0, a.shape[0] - scale_a.shape[0]),
                                          mode='constant', value=1.0)
    if scale_b.shape[0] < b.shape[0] or scale_b.shape[1] < b.shape[1]:
        scale_b = torch.nn.functional.pad(scale_b,
                                          pad=(0, b.shape[1] - scale_b.shape[1], 0, b.shape[0] - scale_b.shape[0]),
                                          mode='constant', value=1.0)

    result = (a.to(torch.float32) * scale_a) @ (b.to(torch.float32) * scale_b)
    return result


@pytest.mark.no_interpret
@pytest.mark.parametrize("accum_mmad", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@pytest.mark.parametrize("m, n, k, scale_m, scale_n, scale_k, offsets_a, offsets_b", [
    (31, 32, 64, 32, 32, 2, [0, 0], [0, 0]),
    (32, 31, 64, 32, 32, 2, [0, 0], [0, 0]),
    (128, 32, 128, 128, 32, 4, [0, 0], [0, 0]),
    (32, 128, 128, 32, 128, 4, [0, 0], [0, 0]),
    (32, 32, 64, 64, 64, 2, [0, 0], [0, 0]),
    (32, 32, 64, 32, 32, 4, [0, 0], [0, 0]),
    (32, 32, 64, 32, 32, 4, [0, 1], [0, 1]),
    (32, 32, 64, 64, 64, 4, [1, 1], [1, 1]),
    (32, 32, 64, 32, 32, 2, [1, 0], [0, 0]),
    (32, 32, 64, 32, 32, 2, [1, 0], [1, 0]),
])
def test_matmul_scale(dtype, m, n, k, accum_mmad, scale_m, scale_n, scale_k, offsets_a, offsets_b, require_c310):
    require_c310()

    a = torch.randn([m, k]).to(dtype)
    b = torch.randn([k, n]).to(dtype)
    # Use scale range 1..8
    a1 = torch.randint(127, 130, (
        scale_m,
        scale_k,
    )).to(dtype=torch.uint8).view(dtype=torch.int8)
    b1 = torch.randint(127, 130, (
        scale_k,
        scale_n,
    )).to(dtype=torch.uint8).view(dtype=torch.int8)
    c = torch.zeros([m, n], dtype=torch.float32)

    scale_a = pack_scale(a1)
    scale_b = pack_scale(b1.T)

    ref_offsets_a = [offsets_a[0] * 16, offsets_a[1] * 2]
    ref_offsets_b = [offsets_b[0] * 16, offsets_b[1] * 2]
    sub_scale_a = a1[ref_offsets_a[0]:ref_offsets_a[0] + m, ref_offsets_a[1]:ref_offsets_a[1] + k // 32]
    sub_scale_b = b1[ref_offsets_b[1]:ref_offsets_b[1] + k // 32, ref_offsets_b[0]:ref_offsets_b[0] + n]
    print(offsets_a, offsets_b)

    c_ref = matmul_scale_reference(a, b, sub_scale_a, sub_scale_b)
    matmul_scale_kernel[1](a, b, scale_a, scale_b, c, m, n, k, scale_a.shape[0], scale_b.shape[0], scale_a.shape[1],
                           offsets_a, offsets_b, accum_mmad)
    torch.testing.assert_close(c_ref, c, atol=1e-3, rtol=1e-3)
