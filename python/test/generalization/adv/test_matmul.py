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
def matmul_kernel(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress, alpha: float,
                  tiling: asc.adv.TCubeTiling, bias: asc.GlobalAddress, workspace: asc.GlobalAddress):
    tiling.share_l1_size = asc.property(asc.TOTAL_L1_SIZE)
    tiling.share_l0c_size = asc.property(asc.TOTAL_L0C_SIZE)
    offset_a, offset_b, offset_c, offset_bias = calc_offsets(tiling)
    a_global = asc.GlobalTensor()
    b_global = asc.GlobalTensor()
    c_global = asc.GlobalTensor()
    bias_global = asc.GlobalTensor()
    a_global.set_global_buffer(a + offset_a)
    b_global.set_global_buffer(b + offset_b)
    c_global.set_global_buffer(c + offset_c)
    bias_global.set_global_buffer(bias + offset_bias)
    size = tiling.base_m * tiling.base_n * c.dtype.sizeof()
    pipe = asc.TPipe()
    relu_out_queue = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(que=relu_out_queue, num=2, len=size)
    matmul = asc.adv.Matmul(a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_global.dtype),
                            b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_global.dtype),
                            c=asc.adv.MatmulType(asc.TPosition.VECCALC, asc.CubeFormat.ND, c_global.dtype),
                            bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND,
                                                    bias_global.dtype), matmul_config=asc.adv.MatmulConfig())
    asc.adv.register_matmul(pipe, workspace, matmul, tiling)
    matmul.set_tensor_a(a_global)
    matmul.set_tensor_b(b_global)
    matmul.set_bias(bias_global)
    with matmul.iterate() as count:
        relu_out_local = relu_out_queue.alloc_tensor(c.dtype)
        matmul.get_tensor_c(relu_out_local, en_sequential_write=True)
        asc.leaky_relu(relu_out_local, relu_out_local, alpha, count=tiling.base_m * tiling.base_n)
        relu_out_queue.enque(relu_out_local)
        relu_out_local = relu_out_queue.deque(c.dtype)
        round_m = tiling.single_core_m // tiling.base_m
        start_offset = count % round_m * tiling.base_m * tiling.n + count // round_m * tiling.base_n
        params = asc.DataCopyParams(
            block_count=tiling.base_m,
            block_len=tiling.base_n * c.dtype.sizeof() // asc.property(asc.DEFAULT_C0_SIZE),
            src_stride=0,
            dst_stride=(tiling.n - tiling.base_n) * c.dtype.sizeof() // asc.property(asc.DEFAULT_C0_SIZE),
        )
        asc.data_copy(c_global[start_offset:], relu_out_local, repeat_params=params)
        relu_out_queue.free_tensor(relu_out_local)
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
    offset_bias = n_index * tiling.single_core_n
    return offset_a, offset_b, offset_c, offset_bias


def matmul_launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor, workspace: torch.Tensor,
                  alpha: float, tiling: asc.adv.TCubeTiling) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Matrices must be compatible for a multiplication"
    matmul_kernel[tiling.used_core_num, rt.current_stream()](a, b, c, alpha, tiling, bias, workspace)
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
    matmul_tiling.enable_bias(True)
    matmul_tiling.set_traverse(host.MatrixTraverse.FIRSTM)
    matmul_tiling.set_buffer_space(-1, -1, -1)

    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    return tiling


param_list = [
    [torch.float32, (1024, 256, 512)],
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_matmul(size, dtype, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    m, k, n = size
    a = (torch.rand((m, k), dtype=dtype, device=device) - .5) * 10
    b = (torch.rand((k, n), dtype=dtype, device=device) - .5) * 10
    bias = (torch.rand((1, n), dtype=torch.float32, device=device) - .5) * 10
    c = torch.zeros((m, n), dtype=torch.float32, device=device)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    alpha = 0.01
    matmul = a.to(torch.float32) @ b.to(torch.float32) + bias
    tiling = generate_tiling(m, n, k, dtype)
    c = matmul_launch(a, b, c, bias, workspace, alpha, tiling)
    res = torch.where(matmul >= 0, matmul, matmul * alpha)
    assert torch.allclose(c, res, atol=1e-3)
