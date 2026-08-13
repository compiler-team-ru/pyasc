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
def transdata_to_5hd_kernel(src: asc.GlobalAddress, dst: asc.GlobalAddress, n: asc.ConstExpr[int],
                            c: asc.ConstExpr[int], h: asc.ConstExpr[int], w: asc.ConstExpr[int],
                            c0: asc.ConstExpr[int]):
    data_size = n * c * h * w

    src_gm = asc.GlobalTensor()
    dst_gm = asc.GlobalTensor()
    src_gm.set_global_buffer(src)
    dst_gm.set_global_buffer(dst)

    pipe = asc.TPipe()
    in_queue_src = asc.TQue(asc.TPosition.VECIN, 1)
    out_queue_dst = asc.TQue(asc.TPosition.VECOUT, 1)
    work_queue_src1 = asc.TQue(asc.TPosition.VECCALC, 1)
    work_queue_src2 = asc.TQue(asc.TPosition.VECCALC, 1)

    pipe.init_buffer(que=in_queue_src, num=1, len=data_size * src.dtype.sizeof())
    pipe.init_buffer(que=out_queue_dst, num=1, len=data_size * dst.dtype.sizeof())
    pipe.init_buffer(que=work_queue_src1, num=1, len=16 * asc.uint64.sizeof())
    pipe.init_buffer(que=work_queue_src2, num=1, len=16 * asc.uint64.sizeof())

    copy_in(src_gm, in_queue_src, data_size)
    compute(dst_gm, in_queue_src, out_queue_dst, work_queue_src1, work_queue_src2, n, c, h, w, c0)
    copy_out(dst_gm, out_queue_dst, data_size)


@asc.jit
def copy_in(src_gm: asc.GlobalTensor, in_queue_src: asc.TQue, src_data_size: int):
    src_local = in_queue_src.alloc_tensor(src_gm.dtype)
    asc.data_copy(src_local, src_gm, count=src_data_size)
    in_queue_src.enque(src_local)


@asc.jit
def compute(dst_gm: asc.GlobalTensor, in_queue_src: asc.TQue, out_queue_dst: asc.TQue, work_queue_src1: asc.TQue,
            work_queue_src2: asc.TQue, n: int, c: int, h: int, w: int, c0: int):
    src_local = in_queue_src.deque(dst_gm.dtype)
    dst_local = out_queue_dst.alloc_tensor(dst_gm.dtype)

    params = asc.TransDataTo5HDParams(dst_high_half=False, src_high_half=False, repeat_times=16, dst_rep_stride=16,
                                      src_rep_stride=1)

    for j in range(4):
        dst_addr_local = work_queue_src1.alloc_tensor(asc.uint64)
        src_addr_local = work_queue_src2.alloc_tensor(asc.uint64)

        for i in range(16):
            dst_offset = j * c0 * h * w + w * i
            dst_addr = dst_local[dst_offset].get_phy_addr()
            dst_addr_local.set_value(i, dst_addr)

        for i in range(16):
            src_offset = j * c0 * h * w + h * w * i
            src_addr = src_local[src_offset].get_phy_addr()
            src_addr_local.set_value(i, src_addr)

        asc.trans_data_to_5hd(dst_addr_local, src_addr_local, params)

        work_queue_src1.free_tensor(dst_addr_local)
        work_queue_src2.free_tensor(src_addr_local)

    out_queue_dst.enque(dst_local)
    in_queue_src.free_tensor(src_local)


@asc.jit
def copy_out(dst_gm: asc.GlobalTensor, out_queue_dst: asc.TQue, dst_data_size: int):
    dst_local = out_queue_dst.deque(dst_gm.dtype)
    asc.data_copy(dst_gm, dst_local, count=dst_data_size)
    out_queue_dst.free_tensor(dst_local)


def transdata_to_5hd_launch(x: torch.Tensor, c0: int = 16) -> torch.Tensor:
    n, c, h, w = x.shape
    if c % c0 != 0:
        raise ValueError(f"Channel dimension {c} must be divisible by c0={c0}")

    z = torch.zeros_like(x)

    use_core_num = 1

    transdata_to_5hd_kernel[use_core_num, rt.current_stream()](x, z, n, c, h, w, c0)
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
def test_transdata_to_5hd_addr_tensor(dtype, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    n, c, h, w = 2, 32, 16, 16
    c0 = 16
    c1 = c // c0
    if dtype in {torch.float16, torch.float32}:
        x_nchw = torch.randn((n, c, h, w), dtype=dtype, device=device)
    else:
        x_nchw = torch.randint(0, 99, (n, c, h, w), dtype=dtype, device=device)
    z_nc1hwc0 = transdata_to_5hd_launch(x_nchw, c0).reshape(n, c1, h, w, c0)
    expected = x_nchw.reshape(n, c1, c0, h, w)
    expected = expected.permute((0, 1, 3, 4, 2))

    assert torch.allclose(z_nc1hwc0, expected)
