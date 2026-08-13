# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Tuple
import numpy as np

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt


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
    asc.adv.register_matmul(pipe, workspace, matmul)
    matmul.init(tiling)
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


def matmul_launch(a: np.ndarray, b: np.ndarray, bias: np.ndarray, alpha: float) -> np.ndarray:
    assert a.shape[1] == b.shape[0], "Matrices must be compatible for a multiplication"
    size_m, size_k = a.shape
    _, size_n = b.shape
    c = np.empty((size_m, size_n), dtype=a.dtype)
    blocksize_m = 64
    blocksize_n = 64
    blocksize_k = 64
    single_m = blocksize_m
    single_n = blocksize_n
    single_k = size_k
    used_core_num = asc.ceildiv(size_m, single_m) * asc.ceildiv(size_n, single_n)
    tiling = asc.adv.TCubeTiling(used_core_num=used_core_num, m=size_m, k_a=size_k, k_b=size_k, n=size_n,
                                 base_m=blocksize_m, base_k=blocksize_k, base_n=blocksize_n, single_core_m=single_m,
                                 single_core_k=single_k, single_core_n=single_n, depth_a1=1, depth_b1=1, step_m=1,
                                 step_n=1, share_mode=0, is_bias=1)
    workspace = np.zeros(16 * 1024 * 1024, dtype="uint8")
    matmul_kernel[used_core_num, rt.current_stream()](a, b, c, alpha, tiling, bias, workspace)
    return c


def test_matmul(backend: config.Backend):
    config.set_platform(backend)
    rng = np.random.default_rng(seed=2025)
    m, k, n = 256, 256, 256
    a = (rng.random((m, k), dtype="float32") - .5) * 10  # [-5, 5]
    b = (rng.random((k, n), dtype="float32") - .5) * 10  # [-5, 5]
    bias = (rng.random((1, n), dtype="float32") - .5) * 10
    alpha = 0.01
    matmul = a @ b + bias
    c = matmul_launch(a, b, bias, alpha)
    np.testing.assert_allclose(c, np.where(matmul >= 0, matmul, matmul * alpha), atol=1e-3)


if __name__ == "__main__":
    test_matmul(config.Backend.Model)
