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


def test_repeat_reduce_sum_kernel(mock_launcher_run):

    @asc.jit
    def repeat_reduce_sum_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.repeat_reduce_sum(z_local, x_local, repeat_time=4, mask=128, dst_blk_stride=0, src_blk_stride=1,
                              dst_rep_stride=8, src_rep_stride=8)

    repeat_reduce_sum_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_pair_reduce_sum_kernel(mock_launcher_run):

    @asc.jit
    def pair_reduce_sum_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=128, dst_rep_stride=1, src_blk_stride=1,
                            src_rep_stride=8)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=mask, dst_rep_stride=1, src_blk_stride=1,
                            src_rep_stride=8)

    pair_reduce_sum_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_whole_reduce_sum_kernel(mock_launcher_run):

    @asc.jit
    def whole_reduce_sum_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.whole_reduce_sum(z_local, x_local, mask=128, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.whole_reduce_sum(z_local, x_local, mask=mask, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8)

    whole_reduce_sum_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_whole_reduce_max_kernel(mock_launcher_run):

    @asc.jit
    def whole_reduce_max_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8)
        asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8, order=asc.ReduceOrder.ORDER_INDEX_VALUE)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8)
        asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8, order=asc.ReduceOrder.ORDER_INDEX_VALUE)

    whole_reduce_max_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_whole_reduce_min_kernel(mock_launcher_run):

    @asc.jit
    def whole_reduce_min_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.whole_reduce_min(z_local, x_local, mask=128, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8)
        asc.whole_reduce_min(z_local, x_local, mask=128, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8, order=asc.ReduceOrder.ORDER_INDEX_VALUE)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.whole_reduce_min(z_local, x_local, mask=mask, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8)
        asc.whole_reduce_min(z_local, x_local, mask=mask, repeat_time=4, dst_rep_stride=1, src_blk_stride=1,
                             src_rep_stride=8, order=asc.ReduceOrder.ORDER_INDEX_VALUE)

    whole_reduce_min_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_reduce_max_kernel(mock_launcher_run):

    @asc.jit
    def reduce_max_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=8320)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=8320)
        shared_tmp = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECCALC, addr=0, tile_size=8320)
        asc.reduce_max(z_local, x_local, shared_tmp_buffer=shared_tmp, mask=128, repeat_time=128, src_rep_stride=65,
                       cal_index=True)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.reduce_max(z_local, x_local, shared_tmp_buffer=shared_tmp, mask=mask, repeat_time=65, src_rep_stride=8,
                       cal_index=True)
        asc.reduce_max(z_local, x_local, shared_tmp_buffer=shared_tmp, count=2048, cal_index=True)

    reduce_max_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_reduce_min_kernel(mock_launcher_run):

    @asc.jit
    def reduce_min_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=8320)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=8320)
        shared_tmp = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECCALC, addr=0, tile_size=8320)
        asc.reduce_min(z_local, x_local, shared_tmp_buffer=shared_tmp, mask=128, repeat_time=128, src_rep_stride=65,
                       cal_index=True)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.reduce_min(z_local, x_local, shared_tmp_buffer=shared_tmp, mask=mask, repeat_time=65, src_rep_stride=8,
                       cal_index=True)
        asc.reduce_min(z_local, x_local, shared_tmp_buffer=shared_tmp, count=2048, cal_index=True)

    reduce_min_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_reduce_sum_kernel(mock_launcher_run):

    @asc.jit
    def reduce_sum_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=8320)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=8320)
        shared_tmp = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECCALC, addr=0, tile_size=8320)
        asc.reduce_sum(z_local, x_local, shared_tmp_buffer=shared_tmp, mask=128, repeat_time=128, src_rep_stride=65)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.reduce_sum(z_local, x_local, shared_tmp_buffer=shared_tmp, mask=mask, repeat_time=65, src_rep_stride=8)
        asc.reduce_sum(z_local, x_local, shared_tmp_buffer=shared_tmp, count=2048)

    reduce_sum_kernel[1]()
    assert mock_launcher_run.call_count == 1
