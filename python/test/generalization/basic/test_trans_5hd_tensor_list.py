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
def trans_5hd_tensor_list_kernel(x: asc.GlobalAddress, z: asc.GlobalAddress, block_length: asc.ConstExpr[int],
                                 buffer_num: asc.ConstExpr[int]):
    tile_length = block_length

    x_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x)
    z_gm = asc.GlobalTensor()
    z_gm.set_global_buffer(z)

    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
    pipe.init_buffer(que=in_queue_x, num=buffer_num, len=tile_length * x.dtype.sizeof())
    pipe.init_buffer(que=out_queue_z, num=buffer_num, len=tile_length * z.dtype.sizeof())

    copy_in(x_gm, in_queue_x, tile_length)
    compute(z_gm, in_queue_x, out_queue_z)
    copy_out(z_gm, out_queue_z, tile_length)


@asc.jit
def copy_in(x_gm: asc.GlobalTensor, in_queue_x: asc.TQue, tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm, count=tile_length)
    in_queue_x.enque(x_local)


@asc.jit
def compute(z_gm: asc.GlobalTensor, in_queue_x: asc.TQue, out_queue_z: asc.TQue):
    c0_size = 16
    h = 16
    w = 16
    outer_loop_count = 4

    x_local = in_queue_x.deque(z_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)

    params = asc.TransDataTo5HDParams(dst_high_half=False, src_high_half=False, repeat_times=16, dst_rep_stride=16,
                                      src_rep_stride=1)

    for j in range(outer_loop_count):
        base_dst_offset = j * c0_size * h * w
        base_src_offset = j * c0_size * h * w

        dst_list = [
            z_local[base_dst_offset + w * 0], z_local[base_dst_offset + w * 1], z_local[base_dst_offset + w * 2],
            z_local[base_dst_offset + w * 3], z_local[base_dst_offset + w * 4], z_local[base_dst_offset + w * 5],
            z_local[base_dst_offset + w * 6], z_local[base_dst_offset + w * 7], z_local[base_dst_offset + w * 8],
            z_local[base_dst_offset + w * 9], z_local[base_dst_offset + w * 10], z_local[base_dst_offset + w * 11],
            z_local[base_dst_offset + w * 12], z_local[base_dst_offset + w * 13], z_local[base_dst_offset + w * 14],
            z_local[base_dst_offset + w * 15]
        ]

        src_list = [
            x_local[base_src_offset + h * w * 0], x_local[base_src_offset + h * w * 1],
            x_local[base_src_offset + h * w * 2], x_local[base_src_offset + h * w * 3],
            x_local[base_src_offset + h * w * 4], x_local[base_src_offset + h * w * 5],
            x_local[base_src_offset + h * w * 6], x_local[base_src_offset + h * w * 7],
            x_local[base_src_offset + h * w * 8], x_local[base_src_offset + h * w * 9],
            x_local[base_src_offset + h * w * 10], x_local[base_src_offset + h * w * 11],
            x_local[base_src_offset + h * w * 12], x_local[base_src_offset + h * w * 13],
            x_local[base_src_offset + h * w * 14], x_local[base_src_offset + h * w * 15]
        ]

        asc.trans_data_to_5hd(dst_list, src_list, params)

    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)


@asc.jit
def copy_out(z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm, z_local, count=tile_length)
    out_queue_z.free_tensor(z_local)


def trans_5hd_tensor_list_launch(x: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    use_core_num = 1
    total_length = x.numel()
    trans_5hd_tensor_list_kernel[use_core_num, rt.current_stream()](x, z, total_length, 1)
    return z


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
def test_trans_5hd_tensor_list(dtype, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"

    n, c, h, w = 1, 64, 16, 16
    if dtype in {torch.float16, torch.float32}:
        x_nchw = torch.randn((n, c, h, w), dtype=dtype, device=device)
    else:
        x_nchw = torch.randint(0, 99, (n, c, h, w), dtype=dtype, device=device)

    z_from_kernel = trans_5hd_tensor_list_launch(x_nchw)

    c1_size = 4
    c0_size = 16
    expected_z = x_nchw.view(n, c1_size, c0_size, h, w).permute(0, 1, 3, 4, 2)

    assert torch.allclose(z_from_kernel.view(expected_z.shape), expected_z)
