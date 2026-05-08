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
    if shape is None:
        shape = tile.shape
    if offsets is None:
        offsets = (0, ) * len(tile.shape)
    shape = verify_shape(shape)
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
def store(value: Tile, tensor: Tensor, *, offsets: Iterable[RuntimeInt]) -> None:
    ...


@overload
def store(value: Tile, tensor: Tensor, *, tile_id: Iterable[RuntimeInt]) -> None:
    ...


@overload
def store(value: RuntimeNumeric, tensor: Tensor, *, offsets: Iterable[RuntimeInt]) -> None:
    ...


def store(value: Union[Tile, RuntimeNumeric], tensor: Tensor, *, tile_id: Optional[Iterable[RuntimeInt]] = None,
          offsets: Optional[Iterable[RuntimeInt]] = None) -> None:
    builder = global_builder.get_ir_builder()
    scalar_store = not isinstance(value, Tile) or value.size == 1
    if scalar_store:
        if tile_id is not None:
            raise ValueError("'tile_id' argument cannot be used when storing a scalar value or a tile with 1 element")
        if offsets is None:
            raise ValueError("'offsets' argument must be provided")
        value = value.to(tensor.dtype) if isinstance(value, Tile) else _mat(value, tensor.dtype)
        builder.create_asctile_SetValueOp(value.to_ir(), tensor.to_ir(), to_ir_list(offsets))
        return
    if (tile_id is None) == (offsets is None):
        raise ValueError("Exactly one of 'tile_id' or 'offsets' must be provided")
    if ir.get_tile_location(value.to_ir().get_type()) == TileLocation.UB:
        check_data_alignment(value.shape, value.dtype)
    offsets = infer_offsets(tensor.shape, value.shape, tile_id, offsets)
    builder.create_asctile_StoreOp(value.to_ir(), tensor.to_ir(), offsets)
