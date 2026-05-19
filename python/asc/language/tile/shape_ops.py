# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from typing import Tuple

from ..._C import ir
from ..core.utils import global_builder
from .tile import Tile, bind_tile_method
from .utils import verify_shape


def shapes_match(shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    if len(shape) > len(target_shape):
        return False
    src = shape[::-1]
    dst = target_shape[::-1]
    for i in range(0, len(dst)):
        if i < len(src) and dst[i] != src[i] and src[i] != 1:
            return False
    return True


@bind_tile_method
def broadcast_to(input: Tile, *shape: int) -> Tile:
    """
    Creates new tile of a given shape broadcasting data from the input tensor.
    This function works similar to :code:`torch.broadcast_to`.

    Args:
        input: The input tensor
        shape: The target shape

    Returns:
        Tile: A new tile with the broadcasted shape

    Raises:
        RuntimeError: If the input tile shape cannot be broadcasted to the target shape

    Examples:
        Broadcast tile to the provided shape: ::

            input = asc2.load(x, [256], offsets=[0])
            result = input.broadcast_to([16,256])

        The code above may act as the following: ::

            input:   [0,1,2,3,4, ... 255]
            result:  [[0,1,2,...255], [0,1,2,...255] ... [0,1,2,..255]]

    When lowering to Ascend C, it first it fills the BroadcastTiling structure calling GetBroadcastTilingInfo:

    .. code-block:: c++

        template <typename T, int constRank = -1, uint32_t* constDstShape = nullptr, uint32_t* constSrcShape = nullptr>
        __aicore__ inline void GetBroadcastTilingInfo(
        uint32_t rank, const uint32_t* dstShape, const uint32_t* srcShape, bool srcInnerPad, BroadcastTiling& tiling);

    Then it perform Broadcast to fill the destination tensor:

    .. code-block:: c++

        template <typename T, int constRank = -1, uint32_t* constDstShape = nullptr, uint32_t* constSrcShape = nullptr,
        bool constSrcInnerPad = false>
        __aicore__ inline void Broadcast(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t* dstShape,
        const uint32_t* srcShape, BroadcastTiling* tiling);
    """
    shape = verify_shape(shape)
    if not shapes_match(input.shape, shape):
        raise RuntimeError(f"Cannot broadcast tile with shape {input.shape} to {shape}")
    result_type = ir.clone_shaped_type(input.to_ir().get_type(), shape)
    handle = global_builder.get_ir_builder().create_asctile_BroadcastOp(result_type, input.to_ir())
    return Tile(handle)


@bind_tile_method
def reshape(input: Tile, *shape: int) -> Tile:
    """
    Reshape a tile to a new shape without changing its data.

    The total number of elements in the new shape must match the total number of elements in the input tile.

    Args:
        input: The input tile
        shape: The target shape

    Returns:
        Tile: A tile with the new shape

    Raises:
        RuntimeError: If the total number of elements doesn't match
    """
    shape = verify_shape(shape)
    if math.prod(input.shape) != math.prod(shape):
        raise RuntimeError("Result tile must have the same number of elements as input tile")
    builder = global_builder.get_ir_builder()
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), shape)
    handle = builder.create_asctile_ReshapeOp(ir_type, input.to_ir())
    return Tile(handle)


@bind_tile_method
def ravel(input: Tile) -> Tile:
    """
    Flatten a tile into a 1D tile.

    This is equivalent to :code:`reshape(input, input.size)`.

    Args:
        input: The input tile

    Returns:
        Tile: A 1D tile with all elements from the input
    """
    return reshape(input, math.prod(input.shape))


@bind_tile_method
def expand_dims(input: Tile, *axis: int) -> Tile:
    """
    Insert new dimensions of size 1 at the specified positions.

    Args:
        input: The input tile
        axis: The positions where new dimensions should be inserted (0-based)

    Returns:
        Tile: A tile with the new dimensions inserted

    Note:
        Multiple axes can be specified. Axes are processed in sorted order.
    """
    shape = list(input.shape)
    axis = sorted(set(axis))
    for ax in axis:
        shape.insert(ax, 1)
    return reshape(input, *shape)


@bind_tile_method
def squeeze(input: Tile, *axis: int) -> Tile:
    """
    Remove dimensions of size 1 from the tile.

    Args:
        input: The input tile.
        axis: The positions of dimensions to remove (0-based). If not provided, all dimensions of size 1 are removed.

    Returns:
        Tile: A tile with the specified dimensions removed

    Raises:
        RuntimeError: If attempting to squeeze a dimension that is not of size 1
    """
    shape = []
    axis = set(axis if axis else (i for i, dim in enumerate(input.shape) if dim == 1))
    for i, dim in enumerate(input.shape):
        if i not in axis:
            shape.append(dim)
            continue
        if dim != 1:
            raise RuntimeError(f"Unable to squeeze the axis {i} since its length must be 1, got {dim}")
    return reshape(input, *shape)


def transpose(input: Tile) -> Tile:
    """
    Transpose a 2D tile by swapping its dimensions.

    Args:
        input: The input tile (must be 2D)

    Returns:
        Tile: The transposed tile with swapped dimensions
    """
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), [input.shape[1], input.shape[0]])
    handle = global_builder.get_ir_builder().create_asctile_TransposeOp(ir_type, input.to_ir())
    return Tile(handle)
