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


def to_ir_list(values):
    return [_mat(v, KT.int32).to_ir() for v in values]


def infer_offsets(tensor_shape: Tuple[RuntimeInt], shape: Iterable[int], tile_id: Optional[Iterable[RuntimeInt]],
                  offsets: Optional[Iterable[RuntimeInt]]) -> List[RuntimeInt]:
    shape = tuple(shape)
    if len(tensor_shape) != len(shape):
        raise RuntimeError("rank of 'tensor_shape' must match rank of 'shape'")
    if tile_id is not None:
        return to_ir_list(idx * size for idx, size in zip(tile_id, shape))
    return to_ir_list(offsets)


@overload
def load(tensor: Tensor, shape: Iterable[int], *, offsets: Iterable[RuntimeInt],
         location: TileLocation = TileLocation.UB, pad_value: RuntimeNumeric = 0) -> Tile:
    ...


@overload
def load(tensor: Tensor, shape: Iterable[int], *, tile_id: Iterable[RuntimeInt],
         location: TileLocation = TileLocation.UB, pad_value: RuntimeNumeric = 0) -> Tile:
    ...


@overload
def load(tensor: Tensor, *, offsets: Iterable[RuntimeInt]) -> PlainValue:
    ...


def load(tensor: Tensor, shape: Optional[Iterable[int]] = None, *, tile_id: Optional[Iterable[RuntimeInt]] = None,
         offsets: Optional[Iterable[RuntimeInt]] = None, location: TileLocation = TileLocation.UB,
         pad_value: RuntimeNumeric = 0) -> Union[Tile, PlainValue]:
    if (tile_id is None) == (offsets is None):
        raise ValueError("Exactly one of 'tile_id' or 'offsets' must be provided")
    if shape is None:
        handle = global_builder.get_ir_builder().create_asctile_GetValueOp(tensor.dtype.to_ir(), tensor.to_ir(),
                                                                           to_ir_list(offsets))
        return PlainValue(handle)
    if not all(isinstance(dim, int) for dim in shape):
        raise RuntimeError("shape must be integers")
    offsets = infer_offsets(tensor.shape, shape, tile_id, offsets)
    ir_type = ir.get_asctile_TileType(list(shape), tensor.dtype.to_ir(), location)
    pad_value = _mat(pad_value, tensor.dtype).to_ir() if pad_value is not None else None
    handle = global_builder.get_ir_builder().create_asctile_LoadOp(ir_type, tensor.to_ir(), offsets, pad_value)
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
    if not isinstance(value, Tile):
        if (tile_id is None) == (offsets is None):
            raise ValueError("Exactly one of 'tile_id' or 'offsets' must be provided")
        global_builder.get_ir_builder().create_asctile_SetValueOp(
            _mat(value, tensor.dtype).to_ir(), tensor.to_ir(), to_ir_list(offsets))
        return

    if not all(isinstance(dim, int) for dim in value.shape):
        raise RuntimeError("shape must be integers")
    offsets = infer_offsets(tensor.shape, value.shape, tile_id, offsets)
    global_builder.get_ir_builder().create_asctile_StoreOp(value.to_ir(), tensor.to_ir(), offsets)


def num_tiles(tensor: Tensor, axis: RuntimeInt, shape: Iterable[int]) -> RuntimeInt:
    if not all(isinstance(dim, int) for dim in shape):
        raise RuntimeError("shape must be integers")
    shape = tuple(shape)
    tensor_shape = tensor.shape
    if len(tensor_shape) != len(shape):
        raise RuntimeError("rank of 'tensor_shape' must match rank of 'shape'")
    if axis >= len(shape) or axis >= len(tensor_shape):
        raise ValueError(f"axis ({axis}) exceeds number of dimensions")
    dim_size = tensor_shape[axis]
    tile_size = shape[axis]
    return (dim_size + tile_size - 1) // tile_size
