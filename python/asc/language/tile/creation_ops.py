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
from .validation import check_type, verify_shape


@require_jit
def full(shape: Iterable[int], value: RuntimeNumeric, dtype: Optional[DataType] = None,
         location: Optional[TileLocation] = TileLocation.UB) -> Tile:
    """
    Create a tile filled with a scalar value.

    Args:
        shape: The shape of the tile to create.
        value: The scalar value to fill the tile with.
        dtype: The data type of the tile. If None, inferred from the value type.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with the specified value

    Raises:
        RuntimeError: If shape contains non-integer values
    """
    check_type("value", value, RuntimeNumeric)
    check_type("dtype", dtype, Optional[DataType])
    check_type("location", location, Optional[TileLocation])
    shape = verify_shape(shape)
    if isinstance(value, Real):
        if dtype is None:
            dtype = KT.int32 if isinstance(value, int) else KT.float32
        return constant_tile(value, shape, dtype, location)
    if dtype is None:
        dtype = value.dtype
    return splat_tile(value, shape, dtype, location)


@require_jit
def full_like(input: Tile, value: RuntimeNumeric, location: Optional[TileLocation] = TileLocation.UB) -> Tile:
    """
    Create a tile filled with a scalar value, with the same shape and dtype as the input tile.

    Args:
        input: The input tile to match shape and dtype.
        value: The scalar value to fill the tile with.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with the specified value
    """
    check_type("input", input, Tile)
    return full(input.shape, value, input.dtype, location)


@require_jit
def zeros(shape: Iterable[int], dtype: DataType = KT.int32, location: Optional[TileLocation] = TileLocation.UB) -> Tile:
    """
    Create a tile filled with zeros.

    Args:
        shape: The shape of the tile to create.
        dtype: The data type of the tile. Default is :code:`int32`.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with zeros
    """
    return full(shape, 0, dtype, location)


@require_jit
def zeros_like(input: Tile, location: Optional[TileLocation] = TileLocation.UB) -> Tile:
    """
    Create a tile filled with zeros, with the same shape and dtype as the input tile.

    Args:
        input: The input tile to match shape and dtype.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile filled with zeros
    """
    check_type("input", input, Tile)
    return zeros(input.shape, input.dtype, location)


@require_jit
def zeros_acc(shape: Iterable[int], dtype: DataType) -> Tile:
    """
    Create a zero-initialized accumulator tile in L0C memory for matrix multiplication.

    This tile is specifically designed for use with :py:func:`matmul_acc` operations and is always located in
    :code:`TileLocation.L0C`.

    Args:
        shape: The shape of the accumulator tile
        dtype: The data type of the accumulator (typically :code:`float32`)

    Returns:
        Tile: A new zero-initialized accumulator tile in L0C memory

    Raises:
        RuntimeError: If shape is invalid
    """
    shape = verify_shape(shape)
    ir_type = ir.get_asctile_TileType(shape, dtype.to_ir(), TileLocation.L0C)
    handle = global_builder.get_ir_builder().create_asctile_AccumulatorOp(ir_type)
    return Tile.from_ir(handle)


@require_jit
def concat(*inputs: Tile) -> Tile:
    """
    Concatenate tiles along the first dimension.

    All input tiles must have the same shape except for the first dimension, and must have the same data type.

    Args:
        inputs: Two or more tiles to concatenate

    Returns:
        Tile: A new tile that is the concatenation of all input tiles along the first dimension

    Raises:
        RuntimeError: If no inputs are provided, inputs are not tiles, shapes are incompatible, or dtypes don't match
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
