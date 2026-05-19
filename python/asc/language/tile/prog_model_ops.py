# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable

from ..basic.sys_var import get_block_idx, get_block_num
from ..core.ir_value import PlainValue, RuntimeInt
from .tensor import Tensor
from .utils import verify_shape


def block_idx() -> PlainValue:
    """
    Returns the current block (NPU core) index.

    In the PyAsc2 programming model, kernels are executed across multiple NPU blocks (cores). This function returns the
    index of the current block, which can be used to determine which portion of the data to process.

    Returns:
        PlainValue: The current block index (0-based)
    """
    return get_block_idx()


def block_num() -> PlainValue:
    """
    Returns the total number of blocks (NPU cores) allocated for the kernel.

    This function returns the total number of NPU blocks that are executing the kernel, which was specified when
    launching the kernel.

    Returns:
        PlainValue: The total number of blocks
    """
    return get_block_num()


def num_tiles(tensor: Tensor, axis: RuntimeInt, shape: Iterable[int]) -> RuntimeInt:
    """
    Returns the number of tiles that fit along a given axis of the tensor.

    This function computes how many tiles of a given shape can be loaded from the tensor along a specified axis.
    This is useful for determining loop bounds when iterating over a tensor in tiles.

    Args:
        tensor: The source tensor.
        axis: The axis along which to count tiles (0-based).
        shape: The shape of each tile. The rank must match the tensor's rank.

    Returns:
        RuntimeInt: The number of tiles that fit along the specified axis

    Raises:
        RuntimeError: If the rank of tensor shape doesn't match the rank of tile shape
        ValueError: If axis exceeds the number of dimensions

    Note:
        If the tensor dimension is not evenly divisible by the tile dimension,
        the last tile will be a partial tile that requires masking.
    """
    shape = verify_shape(shape)
    tensor_shape = tensor.shape
    if len(tensor_shape) != len(shape):
        raise RuntimeError("rank of 'tensor_shape' must match rank of 'shape'")
    if axis >= len(shape) or axis >= len(tensor_shape):
        raise ValueError(f"axis ({axis}) exceeds number of dimensions")
    dim_size = tensor_shape[axis]
    tile_size = shape[axis]
    return (dim_size + tile_size - 1) // tile_size
