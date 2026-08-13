# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("torch is not installed", allow_module_level=True)


@asc.jit
def axpy_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, scalar: float,
                block_length: asc.ConstExpr[int], buffer_num: asc.ConstExpr[int], tile_length: asc.ConstExpr[int],
                tile_num: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)
    z_gm.set_global_buffer(z + offset)
    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
    in_queue_y = asc.TQue(asc.TPosition.VECIN, buffer_num)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
    pipe.init_buffer(que=in_queue_x, num=buffer_num, len=tile_length * x.dtype.sizeof())
    pipe.init_buffer(que=in_queue_y, num=buffer_num, len=tile_length * y.dtype.sizeof())
    pipe.init_buffer(que=out_queue_z, num=buffer_num, len=tile_length * z.dtype.sizeof())
    for i in range(tile_num):
        copy_in(i, x_gm, y_gm, z_gm, in_queue_x, in_queue_y, out_queue_z, tile_length)
        compute(z_gm, in_queue_x, in_queue_y, out_queue_z, scalar, tile_length)
        copy_out(i, z_gm, out_queue_z, tile_length)


@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress, z_gm: asc.GlobalAddress, in_queue_x: asc.TQue,
            in_queue_y: asc.TQue, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    y_local = in_queue_y.alloc_tensor(y_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)
    asc.data_copy(x_local, x_gm[i * tile_length:], count=tile_length)
    asc.data_copy(y_local, y_gm[i * tile_length:], count=tile_length)
    asc.data_copy(z_local, z_gm[i * tile_length:], count=tile_length)
    in_queue_x.enque(x_local)
    in_queue_y.enque(y_local)
    out_queue_z.enque(z_local)


@asc.jit
def compute(z_gm: asc.GlobalTensor, in_queue_x: asc.TQue, in_queue_y: asc.TQue, out_queue_z: asc.TQue, scalar: float,
            tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.deque(z_gm.dtype)
    y_local = in_queue_y.deque(z_gm.dtype)
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.axpy(z_local, y_local, scalar, count=tile_length)
    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)
    in_queue_y.free_tensor(y_local)


@asc.jit
def copy_out(i: int, z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm[i * tile_length:], z_local, count=tile_length)
    out_queue_z.free_tensor(z_local)


def axpy_launch(x: torch.Tensor, y: torch.Tensor, scalar: float) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    z = torch.zeros_like(x)
    total_length = z.numel()
    use_core_num = 1
    block_length = (total_length + use_core_num - 1) // use_core_num
    tile_num = 1
    tile_length = (block_length + tile_num - 1) // tile_num
    buffer_num = 1
    axpy_kernel[use_core_num, rt.current_stream()](x, y, z, scalar, block_length, buffer_num, tile_length, tile_num)
    return z


param_list = [
    [torch.float32, (4096, )],
    [torch.float32, (8192, )],
    [torch.float16, (8192, )],
    [torch.float16, (2048, )],
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_axpy(dtype, size, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    x = torch.randn(size, dtype=dtype, device=device) * 10.0
    y = torch.randn(size, dtype=dtype, device=device) * 10.0
    scalar = 2.5
    z = axpy_launch(x, y, scalar)
    expected = y * scalar
    rtol = 1e-5
    if dtype == torch.float16:
        rtol = 1e-3
    assert torch.allclose(z, expected, rtol=rtol, atol=1e-5)
