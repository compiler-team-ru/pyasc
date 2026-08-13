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
        a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_global.dtype),
        b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_global.dtype),
        c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, c_global.dtype),
    )
    asc.adv.register_matmul(pipe, workspace, matmul, tiling)
    matmul.set_tensor_a(a_global)
    matmul.set_tensor_b(b_global)
    matmul.iterate_all(c_global)
    matmul.end()
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
    assert a.shape[1] == b.shape[0], "Matrices must be compatible for a multiplication"
    matmul_kernel[tiling.used_core_num, rt.current_stream()](a, b, c, tiling, workspace)
    return c


def generate_tiling(m, n, k, dtype):
    matmul_tiling = host.MultiCoreMatmulTiling(host.get_ascendc_platform())

    host_dtype = host.DataType.DT_FLOAT if dtype == torch.float32 else host.DataType.DT_FLOAT16
    matmul_tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host_dtype, False)
    matmul_tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host_dtype, False)
    matmul_tiling.set_c_type(host.TPosition.VECCALC, host.CubeFormat.ND, host.DataType.DT_FLOAT)

    matmul_tiling.set_dim(16)
    matmul_tiling.set_org_shape(m, n, k)
    matmul_tiling.set_shape(m, n, k)
    matmul_tiling.set_traverse(host.MatrixTraverse.FIRSTM)
    matmul_tiling.set_buffer_space(-1, -1, -1)

    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    return tiling


param_list = [
    [torch.float16, (256, 64, 128)],
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_matmul_iterate_all(dtype, size, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    m, k, n = size
    if dtype in {torch.float16, torch.float32}:
        a = (torch.rand((m, k), dtype=dtype, device=device) - 0.5) * 10
        b = (torch.rand((k, n), dtype=dtype, device=device) - 0.5) * 10
        c = torch.zeros((m, n), dtype=torch.float32, device=device)
    else:
        a = torch.randint(-5, 5, (m, k), dtype=dtype, device=device)
        b = torch.randint(-5, 5, (k, n), dtype=dtype, device=device)
        c = torch.zeros((m, n), dtype=torch.int32, device=device)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    matmul = a.to(torch.float32) @ b.to(torch.float32)
    if dtype == torch.int8:
        matmul = a.to(torch.int32) @ b.to(torch.int32)
    tiling = generate_tiling(m, n, k, dtype)
    c = matmul_launch(a, b, c, workspace, tiling)
    assert torch.allclose(c, matmul, atol=1e-3)
