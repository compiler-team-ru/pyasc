# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
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
def transpose_ext_kernel(x: asc.GlobalAddress, z: asc.GlobalAddress, block_length: asc.ConstExpr[int],
                         buffer_num: asc.ConstExpr[int], tile_length: asc.ConstExpr[int], tile_num: asc.ConstExpr[int],
                         tmp_buffer_len: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    z_gm.set_global_buffer(z + offset)

    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
    in_queue_tmp = asc.TQue(asc.TPosition.VECCALC, buffer_num)

    pipe.init_buffer(que=in_queue_x, num=buffer_num, len=tile_length * x.dtype.sizeof())
    pipe.init_buffer(que=out_queue_z, num=buffer_num, len=tile_length * z.dtype.sizeof())
    pipe.init_buffer(que=in_queue_tmp, num=buffer_num, len=tmp_buffer_len)

    for i in range(tile_num):
        copy_in(i, x_gm, in_queue_x, tile_length)
        compute(z_gm, in_queue_x, out_queue_z, in_queue_tmp)
        copy_out(i, z_gm, out_queue_z, tile_length)


@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, in_queue_x: asc.TQue, tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm[i * tile_length:], count=tile_length)
    in_queue_x.enque(x_local)


@asc.jit
def compute(z_gm: asc.GlobalTensor, in_queue_x: asc.TQue, out_queue_z: asc.TQue, in_queue_tmp: asc.TQue):
    x_local = in_queue_x.deque(z_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)
    tmp_buffer = in_queue_tmp.alloc_tensor(asc.uint8)

    params = asc.TransposeParamsExt(n_size=1, c_size=16, h_size=4, w_size=4,
                                    transpose_type=asc.TransposeType.TRANSPOSE_NCHW2NHWC)

    asc.transpose(z_local, x_local, tmp_buffer, params)

    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)
    in_queue_tmp.free_tensor(tmp_buffer)


@asc.jit
def copy_out(i: int, z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm[i * tile_length:], z_local, count=tile_length)
    out_queue_z.free_tensor(z_local)


def transpose_ext_launch(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    z = torch.zeros_like(x)
    use_core_num = 1
    total_length = x.numel()
    block_length = total_length
    tile_num = 1
    tile_length = block_length
    buffer_num = 1
    tmp_buffer_len = (16 + c) * 16 * 8 * 4

    transpose_ext_kernel[use_core_num, rt.current_stream()](x, z, block_length, buffer_num, tile_length, tile_num,
                                                            tmp_buffer_len)
    return z.reshape(n, h, w, c)


param_list = [
    torch.float16,
    # torch.uint16,
    torch.int16,
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_transpose_ext(dtype, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    n, c, h, w = 1, 16, 4, 4
    if dtype in {torch.float16, torch.float32}:
        x_nchw = torch.randn((n, c, h, w), dtype=dtype, device=device)
    else:
        x_nchw = torch.randint(0, 99, (n, c, h, w), dtype=dtype, device=device)
    z_nhwc = transpose_ext_launch(x_nchw)
    expected_z_nhwc = torch.permute(x_nchw, (0, 2, 3, 1))

    assert torch.allclose(z_nhwc, expected_z_nhwc)
