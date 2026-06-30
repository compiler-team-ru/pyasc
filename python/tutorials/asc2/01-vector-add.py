# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Vector Addition
===============

This tutorial demonstrates vector addition operator and its launch on Ascend simulator.

"""

import asc2
import torch


# The functions which are executed on Ascend NPU must be marked with `@asc2.jit` decorator.
# Available parameters for @asc2.jit decorator can be seen in the documentation:
# https://compiler-team-ru.github.io/pyasc/python-api/rst/runtime/index.html
@asc2.jit
def vector_add(
        # Pointers to input and output tensors should have `asc2.GlobalAddress` type.
        x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress,
        # Scalar parameters are passed as Python types (e.g. `int`, `float`).
        # For optimization purposes it is recommended to pass scalar parameter as constants (e.g. `asc2.ConstExpr[int]`).
        size: int,
        # tile_size is used in asc2.copy_in must be asc2.ConstExpr
        tile_size: asc2.ConstExpr):

    # Tensor descriptor is created from `asc2.GlobalAddress` to represent entire tensor.
    x_gm = asc2.global_tensor(x_ptr, [size])
    y_gm = asc2.global_tensor(y_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])

    # Python expressions are used to calculate offset and define the loop iterating over tiles:
    # `asc2.block_num()` function provides number of AICOREs launched.
    block_loop_num = asc2.ceildiv(asc2.ceildiv(size, asc2.block_num()), tile_size)
    # `asc2.block_idx()` function is used to get current AICORE index.
    block_offset = asc2.block_idx() * tile_size * block_loop_num

    # `unroll_factor` parameter of `asc2.range` in `for` loop can be used to manage software pipelining. Set it to `2` to enable double buffering.
    # `parallel` parameter of `asc2.range` in `for` loop enable overlapping of store operation of `i`-th iteration and load of `i+1`-th iteration.
    # It is user responsibility to ensure that there are no data dependencies between overlapped iterations.
    for i in asc2.range(block_loop_num, unroll_factor=2, parallel=True):
        tile_offset = block_offset + i * tile_size
        # `asc2.copy_in` is used to create tile object which is used for further calculations. Data movement from GM to UB happens in this operation.
        x = asc2.copy_in(x_gm, [tile_offset], [tile_size])
        y = asc2.copy_in(y_gm, [tile_offset], [tile_size])
        # One or more operations can be applied for tiles. It is compiler responsibility to allocate required number of memory blocks in UB.
        out = x + y
        # `asc2.copy_out` is used to move data from UB back to GM.
        asc2.copy_out(out, out_gm, [tile_offset])


if __name__ == "__main__":
    backend = asc2.Backend.Model  # can be "Model" for simulator or "NPU" for device
    soc_version = asc2.Platform.Ascend950PR_9599  # Device version
    device_id = 0  # might be necessary to provide if more than one NPU device is present in the system
    asc2.set_platform(backend, soc_version, device_id)

    size = 8192
    block_num = 16
    tile_size = 128
    dtype = torch.float32
    # Allocate tensors
    x = torch.randn(size, dtype=dtype)
    y = torch.randn(size, dtype=dtype)
    out = torch.empty_like(x)

    # For the kernel invocation, number of AICOREs should be provided in brackets:
    vector_add[block_num](x, y, out, size, tile_size)
    torch.testing.assert_close(out, x + y)
