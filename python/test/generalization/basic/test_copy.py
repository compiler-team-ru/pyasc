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
def copy_kernel(x: asc.GlobalAddress, z: asc.GlobalAddress, data_size: asc.ConstExpr[int]):

    x_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x)
    z_gm = asc.GlobalTensor()
    z_gm.set_global_buffer(z)
    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, 1)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(que=in_queue_x, num=1, len=data_size * x.dtype.sizeof())
    pipe.init_buffer(que=out_queue_z, num=1, len=data_size * z.dtype.sizeof())
    copy_in(x_gm, in_queue_x, data_size)
    compute(z_gm, in_queue_x, out_queue_z, data_size)
    copy_out(z_gm, out_queue_z, data_size)


@asc.jit
def copy_in(x_gm: asc.GlobalTensor, in_queue_x: asc.TQue, data_size: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm, count=data_size)
    in_queue_x.enque(x_local)


@asc.jit
def compute(z_gm: asc.GlobalTensor, in_queue_x: asc.TQue, out_queue_z: asc.TQue, data_size: asc.ConstExpr[int]):
    src_local = in_queue_x.deque(z_gm.dtype)
    dst_local = out_queue_z.alloc_tensor(src_local.dtype)
    asc.duplicate(dst_local, 0, data_size)
    mask = 64
    repeat_time = 4
    params = asc.CopyRepeatParams(dst_stride=1, src_stride=1, dst_repeat_size=8, src_repeat_size=8)
    asc.copy(dst_local, src_local, mask, repeat_time, params)
    out_queue_z.enque(dst_local)
    in_queue_x.free_tensor(src_local)


@asc.jit
def copy_out(z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, data_size: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm, z_local, count=data_size)
    out_queue_z.free_tensor(z_local)


def copy_launch(x: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    use_core_num = 1
    total_length = x.numel()
    copy_kernel[use_core_num, rt.current_stream()](x, z, total_length)
    return z


param_list = [
    [torch.float32, (1000, )],
    [torch.float32, (1, )],
    [torch.float32, (9999, )],
    [torch.float16, (2048, )],
    [torch.int32, (8192, )],
    [torch.int16, (8192, )],
    [torch.int32, (1000, )],
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_copy(dtype, size, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    if dtype in {torch.float16, torch.float32}:
        x = torch.randn(size, dtype=dtype, device=device)
    else:
        x = torch.randint(-100, 99, size, dtype=dtype, device=device)
    z_from_kernel = copy_launch(x)
    expected_z = torch.zeros_like(x)
    mask_in_elements = 64
    repeat_time = 4
    params_dst_repeat_size_in_blocks = 8
    params_src_repeat_size_in_blocks = 8
    element_size_in_bytes = x.dtype.itemsize
    block_size_in_bytes = 32
    elements_per_block = block_size_in_bytes // element_size_in_bytes
    src_stride_between_repeats_in_elements = params_src_repeat_size_in_blocks * elements_per_block
    dst_stride_between_repeats_in_elements = params_dst_repeat_size_in_blocks * elements_per_block
    for i in range(repeat_time):
        src_start_index = i * src_stride_between_repeats_in_elements
        dst_start_index = i * dst_stride_between_repeats_in_elements
        elements_to_copy = mask_in_elements
        src_end_index = src_start_index + elements_to_copy
        dst_end_index = dst_start_index + elements_to_copy
        if src_end_index > size[0] or dst_end_index > size[0]:
            break
        expected_z[dst_start_index:dst_end_index] = x[src_start_index:src_end_index]

    assert torch.allclose(z_from_kernel, expected_z)
