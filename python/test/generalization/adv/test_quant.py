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

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("torch is not installed", allow_module_level=True)


@asc.jit(kernel_type=config.KernelType.AIV_ONLY)
def quant_kernel(src_gm: asc.GlobalAddress, dst_gm: asc.GlobalAddress, size: int, tmp_min_bytes: int,
                 use_config: asc.ConstExpr[bool]):
    # Init
    src_global = asc.GlobalTensor()
    dst_global = asc.GlobalTensor()
    src_global.set_global_buffer(src_gm)
    dst_global.set_global_buffer(dst_gm)

    pipe = asc.TPipe()
    in_que_x = asc.TQue(asc.TPosition.VECIN, 1)
    out_que = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(que=in_que_x, num=1, len=size * src_global.dtype.sizeof())
    pipe.init_buffer(que=out_que, num=1, len=size * asc.int8.sizeof())
    # Copy in
    src_local = in_que_x.alloc_tensor(src_global.dtype)
    asc.data_copy(src_local, src_global, count=size)
    in_que_x.enque(src_local)
    # Compute
    dst_local = out_que.alloc_tensor(asc.int8)
    src_local = in_que_x.deque(src_global.dtype)
    tmp_buf = asc.TBuf(asc.TPosition.VECCALC)
    pipe.init_buffer(buf=tmp_buf, len=tmp_min_bytes)
    tmp_local = tmp_buf.get(asc.uint8)
    scale = 2.0
    offset = 0.9
    if use_config:
        asc.adv.quant(dst_local, src_local, scale, offset, size, tmp_local, config=asc.adv.QuantConfig(32, 0, 0, 512))
    else:
        asc.adv.quant(dst_local, src_local, scale, offset, size, tmp_local)
    out_que.enque(dst_local)
    in_que_x.free_tensor(src_local)
    # Copy out
    dst_local = out_que.deque(asc.int8)
    asc.data_copy(dst_global, dst_local, count=size)
    out_que.free_tensor(dst_local)


def get_min_max_tmp_size(input_size: int) -> Tuple[int, int]:
    ASCEND_QUANT_TWO_TIMES = 2
    ASCEND_QUANT_ONE_REPEAT_BYTE_SIZE = 256
    ASCEND_QUANT_MEMORY_CALC = 2
    blk_size = 32
    tmp1 = input_size * ASCEND_QUANT_MEMORY_CALC
    tmp2 = ASCEND_QUANT_TWO_TIMES * ASCEND_QUANT_ONE_REPEAT_BYTE_SIZE
    min_value = ASCEND_QUANT_TWO_TIMES * ASCEND_QUANT_ONE_REPEAT_BYTE_SIZE
    max_value = (max(tmp1, tmp2) + blk_size - 1) // blk_size * blk_size
    return min_value, max_value


def quant_launch(src: torch.Tensor, use_config: bool) -> torch.Tensor:
    dst = torch.zeros_like(src, dtype=torch.int8)
    tmp_min_bytes, _ = get_min_max_tmp_size(src.numel())
    quant_kernel[1, rt.current_stream()](src, dst, src.numel(), tmp_min_bytes, use_config)
    return dst


def quant_torch(src: torch.Tensor) -> torch.Tensor:
    return torch.round((src * 2.0) + 0.9).to(torch.int8)

param_list = [
    [torch.float32, (32,)],
    [torch.float16, (32,)],
]


BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_config", [True, False])
def test_quant(dtype, size, backend: config.Backend, use_config: bool):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    src = torch.rand(size, dtype=dtype, device=device) * 8.0 - 4.0
    dst = quant_launch(src, use_config)
    assert torch.allclose(dst, quant_torch(src))
