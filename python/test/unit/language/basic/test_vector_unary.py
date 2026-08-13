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


def test_abs_kernel(mock_launcher_run):

    @asc.jit
    def abs_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.abs(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.abs(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.abs(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    abs_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_exp_kernel(mock_launcher_run):

    @asc.jit
    def exp_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.exp(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.exp(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.exp(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    exp_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_ln_kernel(mock_launcher_run):

    @asc.jit
    def ln_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.ln(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.ln(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.ln(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    ln_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_bitwise_not_kernel(mock_launcher_run):

    @asc.jit
    def bitwise_not_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.bitwise_not(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.bitwise_not(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.bitwise_not(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    bitwise_not_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_reciprocal_kernel(mock_launcher_run):

    @asc.jit
    def reciprocal_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.reciprocal(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.reciprocal(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.reciprocal(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    reciprocal_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_relu_kernel(mock_launcher_run):

    @asc.jit
    def relu_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.relu(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.relu(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.relu(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    relu_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_rsqrt_kernel(mock_launcher_run):

    @asc.jit
    def rsqrt_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.rsqrt(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.rsqrt(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.rsqrt(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    rsqrt_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sqrt_kernel(mock_launcher_run):

    @asc.jit
    def sqrt_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.sqrt(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.sqrt(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.sqrt(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    sqrt_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_cast_deq_kernel(mock_launcher_run):

    @asc.jit
    def cast_deq_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.cast_deq(z_local, x_local, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.cast_deq(z_local, x_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.cast_deq(z_local, x_local, mask=mask, repeat_times=1, repeat_params=params)

    cast_deq_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_gather_mask_kernel(mock_launcher_run):

    @asc.jit
    def gather_mask_kernel():
        src0_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        dst_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        pattern_value = 2
        reduce_mode = False
        gather_mask_mode = asc.GatherMaskMode.DEFAULT
        mask = 0
        params = asc.GatherMaskParams(src0_block_stride=1, repeat_times=1, src0_repeat_stride=0, src1_repeat_stride=0)
        rsvd_cnt = asc.gather_mask(dst_local, src0_local, pattern_value, reduce_mode, mask, params, gather_mask_mode)

    gather_mask_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_get_gather_mask_remain_count_kernel(mock_launcher_run):

    @asc.jit
    def get_gather_mask_remain_count_kernel():
        remain_count = asc.get_gather_mask_remain_count()

    get_gather_mask_remain_count_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_cast_kernel(mock_launcher_run):

    @asc.jit
    def cast_kernel():
        x_local = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.cast(z_local, x_local, asc.RoundMode.CAST_NONE, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.cast(z_local, x_local, asc.RoundMode.CAST_ROUND, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.cast(z_local, x_local, asc.RoundMode.CAST_FLOOR, mask=mask, repeat_times=1, repeat_params=params)

    cast_kernel[1]()
    assert mock_launcher_run.call_count == 1
