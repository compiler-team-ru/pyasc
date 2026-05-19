# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable, List, Optional, Tuple, Union, overload

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import PlainValue, RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from ..core.utils import global_builder
from .tensor import Tensor
from .tile import Tile, TileLocation
from .utils import check_data_alignment, verify_shape


def to_ir_list(values):
    return [_mat(v, KT.int32).to_ir() for v in values]


def check_real_shape(shape: Tuple[int, ...], real_shape: Iterable[RuntimeInt]) -> None:
    if len(shape) != len(real_shape):
        raise RuntimeError(f"real_shape must have same rank as shape: {len(real_shape)} != {len(shape)}")
    for tile_dim, real_dim in zip(shape, real_shape):
        if not isinstance(real_dim, RuntimeInt):
            raise RuntimeError(f"real_shape dimension must be int or RuntimeInt, got {type(real_dim)}")
        if isinstance(real_dim, int) and real_dim > tile_dim:
            raise RuntimeError(f"real_shape dimension {real_dim} exceeds tile dimension {tile_dim}")


def infer_offsets(tensor_shape: Tuple[RuntimeInt], shape: Iterable[int], tile_id: Optional[Iterable[RuntimeInt]],
                  offsets: Optional[Iterable[RuntimeInt]]) -> List[RuntimeInt]:
    shape = tuple(shape)
    if len(tensor_shape) != len(shape):
        raise RuntimeError("rank of 'tensor_shape' must match rank of 'shape'")
    if tile_id is not None:
        return to_ir_list(idx * size for idx, size in zip(tile_id, shape))
    return to_ir_list(offsets)


def copy(tile: Tile, shape: Optional[Iterable[int]] = None, *, offsets: Optional[Iterable[RuntimeInt]] = None,
         location: TileLocation = TileLocation.UB) -> Tile:
    """
    Copy a tile to a new tile, optionally reshaping and relocating.

    Args:
        tile: The source tile to copy.
        shape: The shape of the resulting tile. If None, uses the source tile's shape.
        offsets: The offsets into the source tile for each dimension. Default is zeros.
        location: The memory location for the destination tile. Default is :code:`TileLocation.UB`.

    Returns:
        Tile: A new tile that is a copy of the source tile

    Raises:
        RuntimeError: If shape is invalid, data alignment check fails, or offsets rank mismatch
    """
    if shape is None:
        shape = tile.shape
    if offsets is None:
        offsets = (0, ) * len(tile.shape)
    shape = verify_shape(shape)
    if location == TileLocation.UB:
        check_data_alignment(shape, tile.dtype)
    offsets = infer_offsets(tile.shape, shape, None, offsets)
    ir_type = ir.get_asctile_TileType(list(shape), tile.dtype.to_ir(), location)
    handle = global_builder.get_ir_builder().create_asctile_CopyOp(ir_type, tile.to_ir(), offsets)
    return Tile(handle)


@overload
def load(tensor: Tensor, shape: Iterable[int], *, real_shape: Optional[Iterable[RuntimeInt]] = None,
         offsets: Iterable[RuntimeInt], location: TileLocation = TileLocation.UB,
         pad_value: RuntimeNumeric = 0) -> Tile:
    ...


@overload
def load(tensor: Tensor, shape: Iterable[int], *, real_shape: Optional[Iterable[RuntimeInt]] = None,
         tile_id: Iterable[RuntimeInt], location: TileLocation = TileLocation.UB,
         pad_value: RuntimeNumeric = 0) -> Tile:
    ...


@overload
def load(tensor: Tensor, *, offsets: Iterable[RuntimeInt]) -> PlainValue:
    ...


def load(tensor: Tensor, shape: Optional[Iterable[int]] = None, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
         tile_id: Optional[Iterable[RuntimeInt]] = None, offsets: Optional[Iterable[RuntimeInt]] = None,
         location: TileLocation = TileLocation.UB, pad_value: RuntimeNumeric = 0) -> Union[Tile, PlainValue]:
    """
    Load data from a tensor into a tile or scalar value.

    This function supports three modes of operation:

    1. **Load a tile with explicit offsets**: Load a tile of the given shape from the tensor
       at the specified byte offsets.

    2. **Load a tile with tile_id**: Load a tile of the given shape where the offset is
       computed as :code:`tile_id * shape` for each dimension.

    3. **Load a scalar**: When shape is not provided, load a single scalar value at the specified offsets.

    Args:
        tensor: The source tensor in global memory.
        shape: The shape of the tile to load. If None, loads a scalar value.
        real_shape: The actual shape of data when loading from a partial tile.
            Used when the tile shape is larger than the available data.
            Must match the rank of :code:`shape` and each dimension must not exceed the corresponding tile dimension.
        offsets: The offsets into the tensor for each dimension. Mutually exclusive with :code:`tile_id`.
        tile_id: The tile index for each dimension, where offset is computed as :code:`tile_id * shape`.
            Mutually exclusive with :code:`offsets`.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.
        pad_value: The value to use for padding when :code:`real_shape` is provided. Default is 0.

    Returns:
        Tile: A tile loaded from the tensor (when :code:`shape` is provided)
        PlainValue: A scalar value loaded from the tensor (when :code:`shape` is None)

    Raises:
        ValueError: If neither or both of :code:`offsets` and :code:`tile_id` are provided

    Note:
        Exactly one of :code:`offsets` or :code:`tile_id` must be provided.
    """
    if (tile_id is None) == (offsets is None):
        raise ValueError("Exactly one of 'tile_id' or 'offsets' must be provided")
    builder = global_builder.get_ir_builder()
    if shape is None:
        handle = builder.create_asctile_GetValueOp(tensor.dtype.to_ir(), tensor.to_ir(), to_ir_list(offsets))
        return PlainValue(handle)
    shape = verify_shape(shape)
    if location == TileLocation.UB:
        check_data_alignment(shape, tensor.dtype)
    offsets = infer_offsets(tensor.shape, shape, tile_id, offsets)
    ir_type = ir.get_asctile_TileType(list(shape), tensor.dtype.to_ir(), location)
    pad_value = _mat(pad_value, tensor.dtype).to_ir() if pad_value is not None else None
    real_shape_ir = []
    if real_shape is not None:
        check_real_shape(shape, real_shape)
        real_shape_ir = to_ir_list(real_shape)
    handle = builder.create_asctile_LoadOp(ir_type, tensor.to_ir(), offsets, pad_value, real_shape_ir)
    return Tile(handle)


@overload
def store(value: Tile, tensor: Tensor, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
          offsets: Iterable[RuntimeInt]) -> None:
    ...


@overload
def store(value: Tile, tensor: Tensor, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
          tile_id: Iterable[RuntimeInt]) -> None:
    ...


@overload
def store(value: RuntimeNumeric, tensor: Tensor, *, offsets: Iterable[RuntimeInt]) -> None:
    ...


def store(value: Union[Tile, RuntimeNumeric], tensor: Tensor, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
          tile_id: Optional[Iterable[RuntimeInt]] = None, offsets: Optional[Iterable[RuntimeInt]] = None) -> None:
    """
    Store data from a tile or scalar value to a tensor in global memory.

    This function supports three modes of operation:

    1. **Store a tile with explicit offsets**: Store a tile to the tensor at the specified byte offsets.

    2. **Store a tile with tile_id**: Store a tile where the offset is computed as :code:`tile_id * tile_shape` for each
       dimension.

    3. **Store a scalar**: Store a single scalar value (or a tile with one element) at the specified offsets.

    Args:
        value: The source value to store. Can be a tile, a scalar value, or a tile with exactly one element.
        tensor: The destination tensor in global memory.
        real_shape: The actual shape of data when storing a partial tile.
            Must match the rank of the tile and each dimension must not exceed the corresponding tile dimension.
            Cannot be used for scalar stores.
        offsets: The offsets into the tensor for each dimension. Mutually exclusive with :code:`tile_id`.
            Required for scalar stores.
        tile_id: The tile index for each dimension, where offset is computed as :code:`tile_id * tile_shape`.
            Mutually exclusive with :code:`offsets`. Cannot be used for scalar stores.

    Raises:
        ValueError: If neither or both of :code:`offsets` and :code:`tile_id` are provided (for tile stores), or if
                    :code:`tile_id` or :code:`real_shape` is used with scalar stores.

    Note:
        For tile stores, exactly one of :code:`offsets` or :code:`tile_id` must be provided.
        For scalar stores, :code:`offsets` must be provided and :code:`tile_id` and :code:`real_shape` cannot be used.
    """
    builder = global_builder.get_ir_builder()
    scalar_store = not isinstance(value, Tile) or value.size == 1
    if scalar_store:
        if tile_id is not None:
            raise ValueError("'tile_id' argument cannot be used when storing a scalar value or a tile with 1 element")
        if offsets is None:
            raise ValueError("'offsets' argument must be provided")
        if real_shape is not None:
            raise ValueError("'real_shape' argument cannot be used when storing a scalar value")
        value = value.to(tensor.dtype) if isinstance(value, Tile) else _mat(value, tensor.dtype)
        builder.create_asctile_SetValueOp(value.to_ir(), tensor.to_ir(), to_ir_list(offsets))
        return
    if (tile_id is None) == (offsets is None):
        raise ValueError("Exactly one of 'tile_id' or 'offsets' must be provided")
    if ir.get_tile_location(value.to_ir().get_type()) == TileLocation.UB:
        check_data_alignment(value.shape, value.dtype)
    offsets = infer_offsets(tensor.shape, value.shape, tile_id, offsets)
    real_shape_ir = []
    if real_shape is not None:
        check_real_shape(value.shape, real_shape)
        real_shape_ir = to_ir_list(real_shape)
    builder.create_asctile_StoreOp(value.to_ir(), tensor.to_ir(), offsets, real_shape_ir)
