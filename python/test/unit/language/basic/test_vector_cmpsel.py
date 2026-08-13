# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


def test_compare_kernel(mock_launcher_run):

    @asc.jit
    def compare_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.compare(z_local, x_local, y_local, cmp_mode=asc.CMPMODE.LT, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.compare(z_local, x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.compare(z_local, x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)
        asc.compare(x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=512, repeat_params=params)
        asc.compare(x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)

    compare_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_compare_scalar_kernel(mock_launcher_run):

    @asc.jit
    def compare_scalar_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.compare_scalar(z_local, x_local, 1, cmp_mode=asc.CMPMODE.LT, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.compare_scalar(z_local, x_local, 1, cmp_mode=asc.CMPMODE.LT, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.compare_scalar(z_local, x_local, 1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1,
                           repeat_params=params)

    compare_scalar_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_select_l2_kernel(mock_launcher_run):

    @asc.jit
    def kernel():
        x = asc.LocalTensor(asc.float16, asc.TPosition.VECIN, 0, 512)
        y = asc.LocalTensor(asc.uint32, asc.TPosition.VECIN, 0, 512)
        z = asc.LocalTensor(asc.float16, asc.TPosition.VECOUT, 0, 512)
        p = asc.LocalTensor(asc.float16, asc.TPosition.VECIN, 0, 512)
        asc.select(z, y, x, 0.0, sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE, count=512)
        asc.select(z, y, x, p, sel_mode=asc.SelMode.VSEL_CMPMASK_SPR, count=512)

    kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_select_slice_scalar_kernel(mock_launcher_run):

    @asc.jit
    def kernel():
        x = asc.LocalTensor(asc.float16, asc.TPosition.VECIN, 0, 512)
        y = asc.LocalTensor(asc.uint32, asc.TPosition.VECIN, 0, 512)
        z = asc.LocalTensor(asc.float16, asc.TPosition.VECOUT, 0, 512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        uint64_max = 2**64 - 1
        mask_list = [uint64_max, uint64_max]
        mask_contiguous = 512
        asc.select(z, y, x, 0.0, sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE, mask=mask_list, repeat_times=1,
                   repeat_params=params)
        asc.select(z, y, x, 0.0, sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE, mask=mask_contiguous, repeat_times=1,
                   repeat_params=params)
        asc.select(z, y, x, repeat_times=1, repeat_params=params)

    kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_select_slice_tensor_kernel(mock_launcher_run):

    @asc.jit
    def kernel():
        x = asc.LocalTensor(asc.float16, asc.TPosition.VECIN, 0, 512)
        y = asc.LocalTensor(asc.uint32, asc.TPosition.VECIN, 0, 512)
        z = asc.LocalTensor(asc.float16, asc.TPosition.VECOUT, 0, 512)
        p = asc.LocalTensor(asc.float16, asc.TPosition.VECIN, 0, 512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        uint64_max = 2**64 - 1
        mask_list = [uint64_max, uint64_max]
        mask_contiguous = 512
        asc.select(z, y, x, p, sel_mode=asc.SelMode.VSEL_CMPMASK_SPR, mask=mask_list, repeat_times=1,
                   repeat_params=params)
        asc.select(z, y, x, p, sel_mode=asc.SelMode.VSEL_CMPMASK_SPR, mask=mask_contiguous, repeat_times=1,
                   repeat_params=params)
        asc.select(z, x, p, repeat_times=1, repeat_params=params, sel_mode=asc.SelMode.VSEL_CMPMASK_SPR)

    kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_get_cmp_mask(mock_launcher_run):

    @asc.jit
    def kernel_get_cmp_mask() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.get_cmp_mask(x_local)

    kernel_get_cmp_mask[1]()
    assert mock_launcher_run.call_count == 1


def test_set_cmp_mask(mock_launcher_run):

    @asc.jit
    def kernel_set_cmp_mask() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        asc.set_cmp_mask(x_local)

    kernel_set_cmp_mask[1]()
    assert mock_launcher_run.call_count == 1
