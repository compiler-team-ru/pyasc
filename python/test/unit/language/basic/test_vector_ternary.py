# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_axpy_kernel(mock_launcher_run):

    @asc.jit
    def axpy_kernel():
        x_local = asc.LocalTensor(dtype=asc.half, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.half, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.axpy(z_local, x_local, 2, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.axpy(z_local, x_local, 2, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.axpy(z_local, x_local, 2, mask=mask, repeat_times=1, repeat_params=params)

    axpy_kernel[1]()
    assert mock_launcher_run.call_count == 1
