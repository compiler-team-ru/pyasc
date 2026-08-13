# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Tuple
import pytest

import asc
from asc.runtime import config
import asc.lib.runtime as rt
import asc.lib.host as host

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("torch is not installed", allow_module_level=True)

BATCH_A = 2
BATCH_B = 2
BATCH = 2


@asc.jit
def matmul_kernel(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress, tiling: asc.adv.TCubeTiling,
                  workspace: asc.GlobalAddress):
    tiling.share_l1_size = asc.property(asc.TOTAL_L1_SIZE)
    tiling.share_l0c_size = asc.property(asc.TOTAL_L0C_SIZE)
    offset_a, offset_b, offset_c = calc_offsets(tiling)
    a_global = asc.GlobalTensor()
    b_global = asc.GlobalTensor()
    c_global = asc.GlobalTensor()
    a_global.set_global_buffer(a + offset_a)
    b_global.set_global_buffer(b + offset_b)
    c_global.set_global_buffer(c + offset_c)
    pipe = asc.TPipe()
    matmul = asc.adv.Matmul(
        a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_global.dtype, False, asc.LayoutMode.NORMAL),
        b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_global.dtype, False, asc.LayoutMode.NORMAL),
        c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, c_global.dtype, False, asc.LayoutMode.NORMAL),
    )
    asc.adv.register_matmul(pipe, workspace, matmul, tiling)
    matmul.set_tensor_a(a_global)
    matmul.set_tensor_b(b_global)
    matmul.iterate_batch(c_global, BATCH_A, BATCH_B, False)
    asc.pipe_barrier(asc.PipeID.PIPE_ALL)


@asc.jit
def calc_offsets(tiling: asc.adv.TCubeTiling) -> Tuple[int, int, int]:
    block_idx = asc.get_block_idx()
    temp0 = tiling.m.ceildiv(tiling.single_core_m)
    temp1 = tiling.k_a.ceildiv(tiling.single_core_k)
    temp2 = tiling.used_core_num // temp1
    m_index = (block_idx % temp2) % temp0
    n_index = (block_idx % temp2) // temp0
    offset_a = m_index * tiling.k_a * tiling.single_core_m
    offset_b = n_index * tiling.single_core_n
    offset_c = m_index * tiling.n * tiling.single_core_m + n_index * tiling.single_core_n
    return offset_a, offset_b, offset_c


def matmul_launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, workspace: torch.Tensor,
                  tiling: asc.adv.TCubeTiling) -> torch.Tensor:
    matmul_kernel[tiling.used_core_num, rt.current_stream()](a, b, c, tiling, workspace)
    return c


def generate_tiling(m, n, k, dtype):
    matmul_tiling = host.MultiCoreMatmulTiling(host.get_ascendc_platform())

    host_dtype = host.DataType.DT_FLOAT if dtype == torch.float32 else host.DataType.DT_FLOAT16
    matmul_tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host_dtype, False)
    matmul_tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host_dtype, False)
    matmul_tiling.set_c_type(host.TPosition.VECCALC, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    matmul_tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)

    matmul_tiling.set_dim(16)
    matmul_tiling.set_org_shape(m, n, k)
    matmul_tiling.set_shape(m, n, k)
    matmul_tiling.set_traverse(host.MatrixTraverse.FIRSTM)
    matmul_tiling.set_buffer_space(-1, -1, -1)

    matmul_tiling.set_a_layout(BATCH_A, m, 1, 1, k)
    matmul_tiling.set_b_layout(BATCH_B, k, 1, 1, n)
    matmul_tiling.set_c_layout(BATCH, m, 1, 1, n)
    matmul_tiling.set_batch_num(BATCH)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    return tiling


param_list = [
    [torch.float32, (64, 64, 64)],
    [torch.float16, (32, 32, 32)],
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_matmul_iterate_batch(dtype, size, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    m, k, n = size
    a1 = (torch.rand((m, k), dtype=dtype, device=device) - .5) * 10
    a2 = (torch.rand((m, k), dtype=dtype, device=device) - .5) * 10
    a = torch.vstack((a1, a2))
    b1 = (torch.rand((k, n), dtype=dtype, device=device) - .5) * 10
    b2 = (torch.rand((k, n), dtype=dtype, device=device) - .5) * 10
    b = torch.vstack((b1, b2))
    batch = 2
    for idx in range(BATCH_A):
        a_tmp = (torch.rand((m, k), dtype=dtype, device=device) - .5) * 10
        if idx == 0:
            a = a_tmp
        else:
            a = torch.vstack((a, a_tmp))
    for idx in range(BATCH_B):
        b_tmp = (torch.rand((k, n), dtype=dtype, device=device) - .5) * 10
        if idx == 0:
            b = b_tmp
        else:
            b = torch.vstack((b, b_tmp))

    ida_num = batch // BATCH_A
    idb_num = batch // BATCH_B
    for idx in range(batch):
        ida = idx // ida_num
        idb = idx // idb_num
        matmul_tmp = a[ida * m:(ida + 1) * m].to(torch.float32) @ b[idb * k:(idb + 1) * k].to(torch.float32)
        if idx == 0:
            matmul = matmul_tmp
        else:
            matmul = torch.vstack((matmul, matmul_tmp))
    c = torch.zeros((batch * m, n), dtype=torch.float32, device=device)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    tiling = generate_tiling(m, n, k, dtype)
    c = matmul_launch(a, b, c, workspace, tiling)
    assert torch.allclose(c, matmul, atol=1e-3)
