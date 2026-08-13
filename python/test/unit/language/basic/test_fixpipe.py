# Copyright (c) 2025 ISE Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config
from asc.runtime.jit import MockTensor


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_fixpipe(mock_launcher_run):

    @asc.jit
    def kernel_fixpipe(x) -> None:
        dst_gm = asc.GlobalTensor()
        dst_gm.set_global_buffer(x)
        src_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        workspace_local = asc.LocalTensor(dtype=asc.uint64, pos=asc.TPosition.VECIN, addr=512, tile_size=1024)
        params = asc.FixpipeParamsV220(n_size=16, m_size=16, src_stride=32, dst_stride=32,
                                       quant_pre=asc.QuantModes.NoQuant, deq_scalar=0, nd_num=1, src_nd_stride=0,
                                       dst_nd_stride=0, relu_en=False, unit_flag=0, is_channel_split=False)
        fixpipe_config = asc.FixpipeConfig(asc.CO2Layout.ROW_MAJOR)
        asc.fixpipe(dst_gm, src_local, params, fixpipe_config)
        asc.fixpipe(dst_gm, src_local, workspace_local, params, fixpipe_config)

    x = MockTensor(asc.float16)
    kernel_fixpipe[1](x)
    assert mock_launcher_run.call_count == 1


def test_set_fix_pipe_pre_quant_flag_kernel(mock_launcher_run):

    @asc.jit
    def set_fix_pipe_pre_quant_flag_kernel():
        deq_scalar = 11
        asc.set_fix_pipe_pre_quant_flag(deq_scalar)

    set_fix_pipe_pre_quant_flag_kernel[1]()
    assert mock_launcher_run.call_count == 1
