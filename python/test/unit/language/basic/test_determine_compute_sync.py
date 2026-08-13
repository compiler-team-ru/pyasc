# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
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


def test_notify_next_block(mock_launcher_run):

    @asc.jit
    def kernel_notify_next_block(gm_workspace_addr: asc.GlobalAddress) -> None:
        gm_workspace = asc.GlobalTensor()
        gm_workspace.set_global_buffer(gm_workspace_addr)
        ub_workspace = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
        asc.notify_next_block(gm_workspace, ub_workspace)

    workspace_tensor = MockTensor(asc.int32)
    kernel_notify_next_block[1](workspace_tensor)
    assert mock_launcher_run.call_count == 1


def test_wait_pre_block(mock_launcher_run):

    @asc.jit
    def kernel_wait_pre_block(gm_workspace_addr: asc.GlobalAddress) -> None:
        gm_workspace = asc.GlobalTensor()
        gm_workspace.set_global_buffer(gm_workspace_addr)
        ub_workspace = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
        asc.wait_pre_block(gm_workspace, ub_workspace)

    workspace_tensor = MockTensor(asc.int32)
    kernel_wait_pre_block[1](workspace_tensor)
    assert mock_launcher_run.call_count == 1
