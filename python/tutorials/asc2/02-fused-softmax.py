# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fused Softmax
=============

This tutorial demonstrates softmax operator and its launch on Ascend simulator.

"""

import asc2
import torch


# The functions which are executed on Ascend NPU must be marked with `@asc2.jit` decorator.
# Available parameters for @asc2.jit decorator can be seen in the documentation:
# https://compiler-team-ru.github.io/pyasc/python-api/rst/runtime/index.html
@asc2.jit(reuse_ub=True)
def fused_softmax(
        # Pointers to input and output tensors should have `asc2.GlobalAddress` type.
        input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress,
        # Scalar parameters are passed as Python types (e.g. `int`, `float`).
        # For optimization purposes it is recommended to pass scalar parameter as constants (e.g. `asc2.ConstExpr[int]`).
        # Here num_rows and num_cols are compiler-time parameter:
        num_rows: asc2.ConstExpr, num_cols: asc2.ConstExpr,
        # tile_shape is a list and must be asc2.ConstExpr
        tile_shape: asc2.ConstExpr):

    # Tensor descriptor is created from `asc2.GlobalAddress` to represent entire tensor.
    in_gm = asc2.tensor(input_ptr, [num_rows, num_cols])
    out_gm = asc2.tensor(output_ptr, [num_rows, num_cols])

    # Python expressions are used to calculate offset:
    # `asc2.block_num()` function provides number of AICOREs launched.
    rows_per_block = asc2.ceildiv(num_rows, asc2.block_num())
    # `asc2.block_idx()` function is used to get current AICORE index.
    block_offset = asc2.block_idx() * rows_per_block
    # Define the loop iterating over tiles
    ub_loop = asc2.number(asc2.ceildiv(rows_per_block, tile_shape[0]), asc2.int_)

    # `unroll_factor` parameter of `asc2.range` in `for` loop can be used to manage software pipelining. Set it to `2` to enable double buffering.
    # `parallel` parameter of `asc2.range` in `for` loop enable overlapping of store operation of `i`-th iteration and load of `i+1`-th iteration.
    # It is user responsibility to ensure that there are no data dependencies between overlapped iterations.
    for i in asc2.range(ub_loop, unroll_factor=2, parallel=True):
        row_start_offset = block_offset + i * tile_shape[0]
        # `asc2.load` is used to create 2D tile object to load from GM to UB and pad with '-inf' all values that are out of global tensor
        rows = asc2.load(in_gm, [row_start_offset, 0], [tile_shape[0], tile_shape[1]], pad_value=float('-inf'))
        # Call high-level 2D softmax
        out = asc2.softmax(rows)
        # `asc2.store` is used to move data from UB back to GM.
        asc2.store(out, out_gm, [row_start_offset, 0])


if __name__ == "__main__":
    backend = asc2.Backend.Model  # can be "Model" for simulator or "NPU" for device
    soc_version = asc2.Platform.Ascend950PR_9599  # Device version
    device_id = 0  # might be necessary to provide if more than one NPU device is present in the system
    asc2.set_platform(backend, soc_version, device_id)

    input_shape_2d = [256, 98]
    rows_per_iter = 5
    block_num = 56
    dtype = torch.float32

    # Alignment for tile_shape
    num_rows, num_cols = input_shape_2d
    ALIGNMENT_ELEMENTS = 32 // dtype.itemsize
    tile_shape = [rows_per_iter, asc2.ceildiv(num_cols, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS]

    # Allocate tensors
    in_tensor = torch.randn(input_shape_2d, dtype=dtype)
    out_tensor = torch.zeros(input_shape_2d, dtype=dtype)

    # For the kernel invocation, number of AICOREs should be provided in brackets:
    fused_softmax[block_num](in_tensor, out_tensor, input_shape_2d[0], input_shape_2d[1], tile_shape)

    expected = torch.softmax(in_tensor, dim=1)
    torch.testing.assert_close(out_tensor, expected)
