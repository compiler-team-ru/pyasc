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
def copy_in(x_gm: asc.GlobalTensor, in_queue_x: asc.TQue, tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)

    swap_settings = asc.AippSwapParams(is_swap_rb=True)
    aipp_config = asc.AippParams(dtype=asc.int8, swap_params=swap_settings)
    asc.set_aipp_functions(x_gm, asc.AippInputFormat.RGB888_U8, aipp_config)
    asc.data_copy(x_local, x_gm, count=tile_length)
    in_queue_x.enque(x_local)


@asc.jit
def compute(in_queue_x: asc.TQue, out_queue_z: asc.TQue, z_gm: asc.GlobalTensor, tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.deque(z_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)

    asc.data_copy(z_local, x_local, count=tile_length)

    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)


@asc.jit
def copy_out(z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm, z_local, count=tile_length)
    out_queue_z.free_tensor(z_local)


@asc.jit
def aipp_config_kernel(x: asc.GlobalAddress, z: asc.GlobalAddress, block_length: asc.ConstExpr[int],
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
    compute(in_queue_x, out_queue_z, z_gm, tile_length)
    copy_out(z_gm, out_queue_z, tile_length)


def aipp_config_launch(x: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    use_core_num = 1
    total_length = x.numel()
    aipp_config_kernel[use_core_num, rt.current_stream()](x, z, total_length, 1)
    return z


param_list = [
    [torch.uint8, (16, 16, 3)],
    [torch.int8, (16, 16, 3)],
    [torch.float16, (16, 16, 3)],
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_aipp_config_integration(dtype, size, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"

    if dtype in {torch.float16, torch.float32}:
        x_hwc = torch.randn(size, dtype=dtype, device=device)
    else:
        x_hwc = torch.randint(0, 127, size, dtype=dtype, device=device)

    result_npu = aipp_config_launch(x_hwc)

    expected = x_hwc.clone()

    assert torch.allclose(result_npu, expected)
