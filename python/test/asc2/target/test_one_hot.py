# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

import asc2
import pytest
import torch

# Cases that exceed Model-simulator headroom are skipped at runtime via intrinsic
# conditions on the parametrize row itself (no opaque case labels):
#   * depth > 1024            -> covers ONE_HOT_I (depth=3511 depth-loop length).
#   * output > 512 MiB        -> covers ONE_HOT_O (~737 MB output footprint).
MODEL_MAX_DEPTH = 1024
MODEL_MAX_OUTPUT_BYTES = 512 * 1024 * 1024


@asc2.jit(static_alloc=True, reuse_ub=True)
def one_hot_kernel_1D(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, arange_ptr: asc2.GlobalAddress,
                      on_value: asc2.ConstExpr, off_value: asc2.ConstExpr, input_total: asc2.ConstExpr,
                      depth: asc2.ConstExpr, block_length: asc2.ConstExpr, block_length_tail: asc2.ConstExpr,
                      unroll_factor: asc2.ConstExpr):
    x = asc2.tensor(x_ptr, [input_total])
    y = asc2.tensor(y_ptr, [input_total * depth])
    arange_gm = asc2.tensor(arange_ptr, [depth])

    block_offset = asc2.block_idx() * block_length
    loop_count = block_length
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_length_tail

    arange_tile = asc2.load(arange_gm, [depth], offsets=[0])
    for i in asc2.range(loop_count, unroll_factor=unroll_factor, parallel=True):
        idx_pos = block_offset + i
        idx_scalar = asc2.load(x, offsets=[idx_pos])
        mask = asc2.equal(arange_tile, idx_scalar)
        result = asc2.where(mask, asc2.number(on_value, y_ptr.dtype), asc2.number(off_value, y_ptr.dtype))
        asc2.store(result, y, offsets=[idx_pos * depth])


# Shapes derived from CSV column `stc_ori_inputs`/`stc_ori_outputs` of the ops-math
# one_hot regression suite. `core_num` follows the CANN tiling selection rule from
# https://gitcode.com/cann/ops-math/blob/master/math/one_hot/op_host/arch35/one_hot_tiling_arch35.cpp
# (`GetTilingParam`, lines 75-105) with aivNum=56 for Ascend950PR_9599:
#   total <= 1024            -> 1 core
#   total <= 56 * 1024       -> ceildiv(total, 1024) cores
#   otherwise                -> 56 cores
# `unroll_factor=1` for every case: the current kernel emits depth-sized tiles
# inside a per-element loop, and parallel unroll>1 races on the splat UB slot
# used to materialise idx_scalar inside `asc2.equal`. CANN's CSV `tilingValues`
# include a CANN-side unroll hint that does not translate one-to-one to this
# kernel's loop structure.
# `axis` is recorded for documentation; the kernel produces depth-innermost output
# (functionally equivalent to axis=-1) regardless. Tests verify functional one-hot
# correctness, not GM byte-layout parity with CANN.
@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, axis, depth, on_value, off_value",
    [
        (10, 1, (9600, ), torch.int32, (9600, 2), torch.int32, -1, 2, 1, 0),
        (56, 1, (1, 1, 1, 256, 1, 318, 1), torch.int32, (1, 1, 1, 256, 1, 318, 1, 1), torch.int32, -1, 1, 1, 0),
        (56, 1, (42767, 7, 16, 16), torch.int32, (42767, 7, 16, 16, 2), torch.float16, -1, 2, 1, 0),
        (56, 1, (576, 192), torch.int32, (576, 1, 192), torch.int32, 1, 1, 1, 0),
        (1, 1, (1, 997), torch.int32, (1, 997, 64), torch.int32, -1, 64, 50, 30),
        (56, 1, (2328, 1, 1, 1, 1, 101, 1), torch.int32,
         (2328, 1, 1, 1, 1, 101, 1, 1), torch.float32, -1, 1, 10.5, 30.6),
        (56, 1, (2, 16, 256, 256), torch.int32, (2, 16, 2, 256, 256), torch.int32, 2, 2, 1, 0),
        (9, 1, (1, 9216), torch.int32, (1, 9216, 2), torch.int32, -1, 2, 50, 30),
        (4, 1, (1, 1, 1, 1, 1, 3712, 1), torch.int32,
         (1, 1, 1, 1, 1, 3712, 1, 3511), torch.float32, -1, 3511, 10.5, 30.6),
        (56, 1, (359, 167, 1, 1, 163), torch.int32, (359, 167, 1, 1, 163, 1), torch.int32, -1, 1, 1, 0),
        (1, 1, (1, 1, 593, 1, 1), torch.int32, (1, 1, 593, 1, 1, 31), torch.int32, -1, 31, 50, 30),
        (56, 1, (1, 1, 1, 1, 1, 4793, 28), torch.int32,
         (1, 1, 1, 1, 1, 4793, 28, 184), torch.float32, -1, 184, 10.5, 30.6),
        (56, 1, (1, 65536), torch.int32, (1, 65536, 2), torch.int32, -1, 2, 1, 0),
        (48, 1, (1, 1024, 2, 4, 6), torch.int32, (1, 1024, 2, 4, 6, 4), torch.int32, -1, 4, 50, 30),
        (56, 1, (1259, 1, 192, 2, 127), torch.int32, (1259, 1, 192, 2, 127, 3), torch.float32, -1, 3, 10.5, 30.6),
    ],
)
def test_one_hot(profiler, runs, core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, axis,
                 depth, on_value, off_value):
    input_total = math.prod(input_shape)
    block_length = asc2.ceildiv(input_total, core_num)
    block_length_tail = input_total - block_length * (core_num - 1)

    indices = torch.randint(0, depth, input_shape, dtype=input_dtype)
    indices_flat = indices.reshape(input_total)
    arange_t = torch.arange(depth, dtype=input_dtype)
    output_flat = torch.zeros(input_total * depth, dtype=output_dtype)

    with profiler.profile():
        for _ in range(runs):
            one_hot_kernel_1D[core_num](indices_flat, output_flat, arange_t, on_value, off_value, input_total, depth,
                                        block_length, block_length_tail, unroll_factor)

    # Golden directly expresses CANN's two-phase semantics: fill with off_value,
    # then sparse-write on_value at index positions. The kernel emits a
    # depth-innermost layout, so compare both as (input_total, depth).
    expected_flat = torch.full((input_total, depth), off_value, dtype=output_dtype)
    expected_flat.scatter_(1, indices_flat.long().reshape(-1, 1), on_value)
    torch.testing.assert_close(output_flat.reshape(input_total, depth), expected_flat, atol=1e-3, rtol=1e-3)
