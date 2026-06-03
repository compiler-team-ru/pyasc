# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real
from typing import Iterable, Optional

from ..._C import ir
from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import RuntimeNumeric
from ..core.utils import global_builder, require_jit
from .tile import Tile, TileLocation
from .utils import constant_tile, splat_tile
from .validation import check_dtype, check_type, verify_shape


@require_jit
def full(shape: Iterable[int], value: RuntimeNumeric, dtype: Optional[DataType] = None,
         location: TileLocation = TileLocation.UB) -> Tile:
    """
    Create a tile filled with a scalar value.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        shape: The shape of the tile to create.
        value: The scalar value to fill the tile with.
        dtype: The data type of the tile. If None, inferred from the value type.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with the specified value

    Raises:
        TypeError: If value is not a numeric type, dtype is not a DataType, or location is not a TileLocation
        RuntimeError: If shape is invalid or dtype is not supported

    Examples:
        Create a tile filled with a constant integer value: ::

            tile = asc2.full([128], 42, dtype=asc2.int32)

        Create a tile filled with a floating-point value: ::

            tile = asc2.full([32, 16], 3.14, dtype=asc2.float16)

        Create a tile with dtype inferred from the value: ::

            tile = asc2.full([64], 0)       # inferred as int32
            tile = asc2.full([64], 1.5)     # inferred as float32
    """
    check_type("value", value, RuntimeNumeric)
    check_type("dtype", dtype, Optional[DataType])
    support_dtypes = (KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32)
    check_dtype("dtype", dtype, support_dtypes, optional=True)
    check_type("location", location, TileLocation)
    shape = verify_shape(shape)
    if isinstance(value, Real):
        if dtype is None:
            dtype = KT.int32 if isinstance(value, int) else KT.float32
        return constant_tile(value, shape, dtype, location)
    if dtype is None:
        check_dtype("value", value, support_dtypes)
        dtype = value.dtype
    return splat_tile(value, shape, dtype, location)


@require_jit
def full_like(input: Tile, value: RuntimeNumeric, location: TileLocation = TileLocation.UB) -> Tile:
    """
    Create a tile filled with a scalar value, with the same shape and dtype as the input tile.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tile to match shape and dtype.
        value: The scalar value to fill the tile with.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with the specified value

    Raises:
        TypeError: If input is not a Tile

    Examples:
        Create a tile filled with a value, matching another tile's shape and dtype: ::

            src = asc2.load(x_gm, [128], offsets=[0])
            tile = asc2.full_like(src, 255)
    """
    check_type("input", input, Tile)
    return full(input.shape, value, input.dtype, location)


@require_jit
def zeros(shape: Iterable[int], dtype: DataType = KT.int32, location: TileLocation = TileLocation.UB) -> Tile:
    """
    Create a tile filled with zeros.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        shape: The shape of the tile to create.
        dtype: The data type of the tile. Default is :code:`int32`.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with zeros

    Raises:
        TypeError: If dtype is not a DataType or location is not a TileLocation
        RuntimeError: If shape is invalid or dtype is not supported

    Examples:
        Create a zero-filled tile with default dtype (int32): ::

            tile = asc2.zeros([128])

        Create a zero-filled tile with a specific dtype: ::

            tile = asc2.zeros([32, 16], dtype=asc2.float16)
    """
    return full(shape, 0, dtype, location)


@require_jit
def zeros_like(input: Tile, location: TileLocation = TileLocation.UB) -> Tile:
    """
    Create a tile filled with zeros, with the same shape and dtype as the input tile.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        input: The input tile to match shape and dtype.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with zeros

    Raises:
        TypeError: If input is not a Tile

    Examples:
        Create a zero-filled tile matching another tile's shape and dtype: ::

            src = asc2.load(x_gm, [128], offsets=[0])
            tile = asc2.zeros_like(src)
    """
    check_type("input", input, Tile)
    return zeros(input.shape, input.dtype, location)


@require_jit
def zeros_acc(shape: Iterable[int], dtype: DataType) -> Tile:
    """
    Create a zero-initialized accumulator tile in L0C memory for matrix multiplication.

    This tile is specifically designed for use with :py:func:`matmul_acc` operations and is always located in
    :code:`TileLocation.L0C`.

    The supported data type is: ``float32``.

    Args:
        shape: The shape of the accumulator tile
        dtype: The data type of the accumulator

    Returns:
        Tile: A new zero-initialized accumulator tile in L0C memory

    Raises:
        TypeError: If shape contains non-integer values
        RuntimeError: If shape contains non-positive values

    Examples:
        Create an accumulator tile for matrix multiplication: ::

            acc = asc2.zeros_acc([16, 16], dtype=asc2.float32)
            result = asc2.matmul(acc, a_tile, b_tile)
    """
    check_type("dtype", dtype, DataType)
    check_dtype("dtype", dtype, KT.float32)
    shape = verify_shape(shape)
    ir_type = ir.get_asctile_TileType(shape, dtype.to_ir(), TileLocation.L0C)
    handle = global_builder.get_ir_builder().create_asctile_AccumulatorOp(ir_type)
    return Tile.from_ir(handle)


@require_jit
def concat(*inputs: Tile) -> Tile:
    """
    Concatenate tiles along the first dimension.

    All input tiles must have the same shape except for the first dimension, and must have the same data type.

    The supported data types are: ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        inputs: Two or more tiles to concatenate

    Returns:
        Tile: A new tile that is the concatenation of all input tiles along the first dimension

    Raises:
        TypeError: If any input is not a Tile
        RuntimeError: If no inputs are provided, shapes are incompatible, dtypes don't match,
            or tile dtype size does not fit an integer number of bytes

    Examples:
        Concatenate two tiles along the first dimension: ::

            tile_a = asc2.load(x_gm, [64, 32], offsets=[0, 0])
            tile_b = asc2.load(y_gm, [64, 32], offsets=[64, 0])
            result = asc2.concat(tile_a, tile_b)  # shape: [128, 32]

        Concatenate multiple tiles: ::

            tiles = [asc2.load(x_gm, [32, 16], offsets=[0, 0]), asc2.load(x_gm, [16, 16], offsets=[32, 0]),
                     asc2.load(x_gm, [8, 16], offsets=[64, 0]), asc2.maximum(tile_a, tile_b)]
            result = asc2.concat(*tiles)  # shape: [120, 16]
    """
    if not inputs or not all(isinstance(inp, Tile) for inp in inputs):
        raise TypeError("All input arguments must be tiles")
    same_shape = inputs[0].shape[1:]
    if not all(inp.shape[1:] == same_shape for inp in inputs):
        raise RuntimeError("All tiles must have the same shape except their first dimension")
    dtype = inputs[0].dtype
    if not all(inp.dtype == dtype for inp in inputs):
        raise RuntimeError("All tiles must have the same dtype")
    try:
        dtype.sizeof()
    except ValueError:
        raise RuntimeError("Tile dtype size must fit an integer number of bytes")
    result_shape = [sum(inp.shape[0] for inp in inputs), *same_shape]
    ir_type = ir.get_asctile_TileType(result_shape, dtype.to_ir(), TileLocation.UB)
    handle = global_builder.get_ir_builder().create_asctile_ConcatOp(ir_type, [inp.to_ir() for inp in inputs])
    return Tile.from_ir(handle)
