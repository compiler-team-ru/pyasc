# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from typing import Iterable, Tuple

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.utils import global_builder, require_jit
from .tile import Tile, bind_tile_method
from .validation import check_dtype, check_type, verify_shape, check_data_alignment


def shapes_match(shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    if len(shape) > len(target_shape):
        return False
    src = shape[::-1]
    dst = target_shape[::-1]
    for i in range(0, len(dst)):
        if i < len(src) and dst[i] != src[i] and src[i] != 1:
            return False
    return True


def normalize_shape_args(args: tuple) -> Tuple[int, ...]:
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], Iterable) else args


@bind_tile_method
@require_jit
def broadcast_to(input: Tile, *shape: int) -> Tile:
    """
    Creates new tile of a given shape broadcasting data from the input tensor.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tensor
        shape: The target shape (can be passed as separate integers or as an iterable, e.g. list or tuple)

    Returns:
        Tile: A new tile with the broadcasted shape

    Raises:
        TypeError: If input is not a Tile or shape contains non-integer values
        RuntimeError: If the input tile shape cannot be broadcasted to the target shape or shape values are not positive

    Examples:
        Broadcast tile to the provided shape: ::

            input = asc2.load(x, [256], offsets=[0])
            result = input.broadcast_to([16,256])

        The code above may act as the following: ::

            input:   [0,1,2,3,4, ... 255]
            result:  [[0,1,2,...255], [0,1,2,...255] ... [0,1,2,..255]]
    """
    check_type("input", input, Tile)
    check_dtype("input", input, (KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32))
    shape = normalize_shape_args(shape)
    shape = verify_shape(shape)
    if input.shape == shape:
        return input
    if not shapes_match(input.shape, shape):
        raise RuntimeError(f"Cannot broadcast tile with shape {input.shape} to {shape}")
    result_type = ir.clone_shaped_type(input.to_ir().get_type(), shape)
    handle = global_builder.get_ir_builder().create_asctile_BroadcastOp(result_type, input.to_ir())
    return Tile(handle)


@bind_tile_method
@require_jit
def reshape(input: Tile, *shape: int) -> Tile:
    """
    Reshape a tile to a new shape without changing its data.

    The total number of elements in the new shape must match the total number of elements in the input tile.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tile
        shape: The target shape (can be passed as separate integers or as an iterable, e.g. list or tuple)

    Returns:
        Tile: A tile with the new shape

    Raises:
        TypeError: If input is not a Tile or shape contains non-integer values
        RuntimeError: If the total number of elements doesn't match or shape values are not positive

    Examples:
        Reshape a 1D tile to 2D: ::

            input = asc2.load(x, [256], offsets=[0])
            result = input.reshape([16, 16])

        Reshape a 2D tile to 1D: ::

            input = asc2.load(x, [32, 16], offsets=[0, 0])
            result = input.reshape([512])
    """
    check_type("input", input, Tile)
    check_dtype("input", input,
                (KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32, KT.float64))
    shape = normalize_shape_args(shape)
    shape = verify_shape(shape)
    if math.prod(input.shape) != math.prod(shape):
        raise RuntimeError(f"Reshaping tile of shape {input.shape} with {math.prod(input.shape)} elements not match "
                           f"output shape {shape} with {math.prod(shape)} elements")
    builder = global_builder.get_ir_builder()
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), shape)
    handle = builder.create_asctile_ReshapeOp(ir_type, input.to_ir())
    return Tile(handle)


@bind_tile_method
@require_jit
def ravel(input: Tile) -> Tile:
    """
    Flatten a tile into a 1D tile.

    This is equivalent to :code:`reshape(input, input.size)`.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tile

    Returns:
        Tile: A 1D tile with all elements from the input

    Raises:
        TypeError: If input is not a Tile

    Examples:
        Flatten a 2D tile to 1D: ::

            input = asc2.load(x, [32, 16], offsets=[0, 0])
            result = input.ravel()
    """
    return reshape(input, math.prod(input.shape))


@bind_tile_method
@require_jit
def expand_dims(input: Tile, *axis: int) -> Tile:
    """
    Insert new dimensions of size 1 at the specified positions.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tile
        axis: The positions where new dimensions should be inserted (0-based)

    Returns:
        Tile: A tile with the new dimensions inserted

    Raises:
        TypeError: If input is not a Tile

    Note:
        Multiple axes can be specified. Axes are processed in sorted order.

    Examples:
        Insert a dimension at axis 0: ::

            input = asc2.load(x, [256], offsets=[0])
            result = input.expand_dims(0)  # shape becomes [1, 256]

        Insert multiple dimensions: ::

            input = asc2.load(x, [32, 16], offsets=[0, 0])
            result = input.expand_dims(0, 2)  # shape becomes [1, 32, 1, 16]
    """
    check_type("input", input, Tile)
    shape = list(input.shape)
    axis = sorted(set(axis))
    for ax in axis:
        shape.insert(ax, 1)
    return reshape(input, *shape)


@bind_tile_method
@require_jit
def squeeze(input: Tile, *axis: int) -> Tile:
    """
    Remove dimensions of size 1 from the tile.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``,
    ``float64``.

    Args:
        input: The input tile.
        axis: The positions of dimensions to remove (0-based). If not provided, all dimensions of size 1 are removed.

    Returns:
        Tile: A tile with the specified dimensions removed

    Raises:
        TypeError: If input is not a Tile
        RuntimeError: If attempting to squeeze a dimension that is not of size 1

    Examples:
        Remove all dimensions of size 1: ::

            input = asc2.load(x, [1, 32, 1, 16], offsets=[0, 0, 0, 0])
            result = input.squeeze()  # shape becomes [32, 16]

        Remove a specific dimension: ::

            input = asc2.load(x, [1, 32, 16], offsets=[0, 0, 0])
            result = input.squeeze(0)  # shape becomes [32, 16]
    """
    check_type("input", input, Tile)
    shape = []
    axis = set(axis if axis else (i for i, dim in enumerate(input.shape) if dim == 1))
    for i, dim in enumerate(input.shape):
        if i not in axis:
            shape.append(dim)
            continue
        if dim != 1:
            raise RuntimeError(f"Unable to squeeze the axis {i} since its length must be 1, got {dim}")
    return reshape(input, *shape)


@bind_tile_method
@require_jit
def transpose(input: Tile, *axis: int) -> Tile:
    """
    Rearrange tile dimensions in specific order.
    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``, ``float64``.

    Args:
        input: The input tile
        axis: Order of input dimensions in result. Swaps two last dimensions when no axis provided

    Returns:
        Tile: The transposed tile with swapped dimensions

    Raises:
        TypeError: If input is not a Tile
        RuntimeError: If the input tile dtype is not supported or axis is incorrect

    Examples:
        Transpose 2d tile: ::

            input = asc2.load(x, [32, 16], offsets=[0, 0, 0])            
            result = input.transpose()  # shape becomes [32, 16], same as input.transpose(1, 0)

        Transpose a tile with 

            input = asc2.load(x, [32, 64, 16], offsets=[0, 0, 0])
            result = input.transpose(2, 0, 1)  # shape becomes [16, 32, 64]
    """
    check_type("input", input, Tile)
    check_dtype("input", input,
                (KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32, KT.float64))
    rank = len(input.shape)
    if len(axis) == 0:
        axis = list(range(0, rank))
        axis[-1], axis[-2] = axis[-2], axis[-1]
    if len(axis) != rank:
        raise RuntimeError(f"Transpose axis count {len(axis)} should match count of tensors dimensions {rank}")
    if list(axis) == list(range(0, rank)):  # Identity transformation
        return input
    if set(axis) != set(range(0, rank)):
        raise RuntimeError(f"Wrong dimensions rearrangement {axis} for tile of {rank} dimensions")
    result_shape = [input.shape[i] for i in axis]
    check_data_alignment(result_shape, input.dtype)

    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), result_shape)
    handle = global_builder.get_ir_builder().create_asctile_TransposeOp(
        ir_type, input.to_ir(),
        global_builder.get_ir_builder().get_i32_array_attr(axis))
    return Tile(handle)
