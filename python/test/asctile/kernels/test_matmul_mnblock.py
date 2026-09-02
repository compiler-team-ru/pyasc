# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import pytest
import torch


@asctile.jit(always_compile=True)
def matmul_kernel(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
                  a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr, c_shape: asctile.ConstExpr,
                  m_tile: asctile.ConstExpr[int], m_tiles_per_block: asctile.ConstExpr[int],
                  n_tile: asctile.ConstExpr[int], n_tiles_per_block: asctile.ConstExpr[int]):
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, c_shape)
    blockId = asctile.block_idx()
    m_elems_per_block = m_tile * m_tiles_per_block
    m_base_off = (m_elems_per_block * blockId) % a_shape[0]
    n_base_off = ((m_elems_per_block * blockId) // a_shape[0]) * (n_tile * n_tiles_per_block)
    for j in range(n_tiles_per_block):
        b_offset = n_base_off + j * n_tile
        b_j = asctile.copy_in(b_gm, [0, b_offset], [b_shape[0], n_tile], asctile.TensorLocation.L0B)
        for i in range(m_tiles_per_block):
            a_offset = m_base_off + i * m_tile
            a_i = asctile.copy_in(a_gm, [a_offset, 0], [m_tile, a_shape[1]], asctile.TensorLocation.L0A)
            c_ij = a_i @ b_j
            asctile.copy_out(c_ij, c_gm, [a_offset, b_offset])


def eval_tiles(a: torch.Tensor, b: torch.Tensor, core_num, m_tile, n_tile, n_tiles_per_block):
    m, k = a.shape
    _, n = b.shape

    assert m_tile % 16 == 0, "M tile should be multiple of 16 elements"
    assert m_tile * k * a.element_size() <= 64 * 1024, "M tile size should be <= 64 Kb"
    assert m % m_tile == 0, "M dimension should be multiple of M tiles"
    m_tiles_num = m // m_tile

    assert n_tile % 16 == 0, "N tile should be multiple of 16 elements"
    assert n_tile * k * b.element_size() <= 64 * 1024, "N tile size should be <= 64 Kb"
    assert n % n_tile == 0, "N dimension should be multiple of N tiles"
    n_tiles_num = n // n_tile

    assert (m_tiles_num * n_tiles_num) % core_num == 0, "Tiles should be distributed evenly across cores"
    tiles_per_core = (m_tiles_num * n_tiles_num) // core_num

    assert n_tiles_num % n_tiles_per_block == 0, "N tiles should be distributed evenly across cores"
    assert tiles_per_core % n_tiles_per_block == 0, "Number of M tiles within 1 core should be integer"

    m_tiles_per_block = tiles_per_core // n_tiles_per_block
    assert m_tiles_num % m_tiles_per_block == 0, "M tiles should be distributed evenly across cores"

    return m_tiles_per_block


def matmul_launch(a: torch.Tensor, b: torch.Tensor, core_num: int, m_tile: int, n_tile: int,
                  n_tiles_per_block: int) -> torch.Tensor:
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    m_tiles_per_block = eval_tiles(a, b, core_num, m_tile, n_tile, n_tiles_per_block)
    matmul_kernel[core_num](a, b, c, a.shape, b.shape, c.shape, m_tile, m_tiles_per_block, n_tile, n_tiles_per_block)
    return c


@pytest.mark.parametrize("m, k, n, core_num, m_tile, n_tile, n_tiles_per_block", [
    (16, 16, 16, 1, 16, 16, 1),
    (16, 160, 16, 1, 16, 16, 1),
    (32, 16, 16, 2, 16, 16, 1),
    (16, 16, 32, 2, 16, 16, 1),
    (32, 16, 32, 2, 16, 16, 1),
    (32, 16, 32, 2, 16, 16, 2),
    (32, 16, 32, 4, 16, 16, 1),
    (480, 224, 1344, 30, 48, 32, 7),
])
def test_matmul_mnblock(m: int, k: int, n: int, core_num: int, m_tile: int, n_tile: int, n_tiles_per_block: int):
    dtype = torch.float16
    a = torch.rand((m, k), dtype=dtype)
    b = torch.rand((k, n), dtype=dtype)
    c = matmul_launch(a, b, core_num, m_tile, n_tile, n_tiles_per_block)
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    torch.testing.assert_close(c, c_ref)
