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
from ..core.utils import global_builder, require_jit
from .tensor import Tensor
from .tile import Tile, TileLocation
from .validation import check_data_alignment, check_type, verify_runtime_ints, verify_shape


def to_ir_list(values):
    return [_mat(v, KT.int32).to_ir() for v in values]


def check_real_shape(shape: Tuple[int, ...], real_shape: Tuple[RuntimeInt, ...]) -> None:
    if len(shape) != len(real_shape):
        raise RuntimeError(f"real_shape must have same rank as shape: {len(real_shape)} != {len(shape)}")
    for tile_dim, real_dim in zip(shape, real_shape):
        if isinstance(real_dim, int) and real_dim > tile_dim:
            raise RuntimeError(f"real_shape dimension {real_dim} exceeds tile dimension {tile_dim}")


def infer_offsets(tensor_shape: Tuple[RuntimeInt], shape: Tuple[int, ...], tile_id: Optional[Iterable[RuntimeInt]],
                  offsets: Optional[Iterable[RuntimeInt]]) -> List[RuntimeInt]:
    if len(tensor_shape) != len(shape):
        raise RuntimeError("rank of 'tensor_shape' must match rank of 'shape'")
    if tile_id is not None:
        tile_id = verify_runtime_ints(tile_id, "tile_id")
        return to_ir_list(idx * size for idx, size in zip(tile_id, shape))
    offsets = verify_runtime_ints(offsets, "offsets")
    return to_ir_list(offsets)


@require_jit
def copy(tile: Tile, shape: Optional[Iterable[int]] = None, *, offsets: Optional[Iterable[RuntimeInt]] = None,
         location: TileLocation = TileLocation.UB) -> Tile:
    """
    Copy a tile to a new tile, optionally reshaping and relocating.

    **Rationale:** Unlike frameworks with simpler memory hierarchies (e.g., CUDA's global/shared/registers), Ascend NPUs
    expose multiple local memory levels (L1, L0A, L0B, L0C, UB) where local-to-local transfers are common. :code:`load`
    and :code:`store` have clear directional semantics when one endpoint is global memory ("load from global", "store to
    global"), but this breaks down for local-to-local transfers: the same L0C→L1 operation is a "store" from L0C's
    perspective yet a "load" from L1's. :code:`copy` eliminates this ambiguity by providing a direction-agnostic
    operation that clearly expresses intent regardless of which memory level you're reasoning from.

    Args:
        tile: The source tile to copy.
        shape: The shape of the resulting tile. If None, uses the source tile's shape.
            Must contain static values (e.g., :code:`ConstExpr` or compile-time constants).
        offsets: The offsets into the source tile for each dimension. Default is zeros.
        location: The memory location for the destination tile. Default is :code:`TileLocation.UB`.
            Supported location transfers: ``L1`` to ``L0A``, ``L1`` to ``L0B``, ``L1`` to ``BT``, ``L0C`` to ``L1``.

    Returns:
        Tile: A new tile that is a copy of the source tile

    Raises:
        TypeError: If tile is not a Tile or location is not a TileLocation
        RuntimeError: If shape is invalid, data alignment check fails, or offsets rank mismatch

    Examples:
        Copy a tile with the same shape: ::

            tile = asc2.load(x_gm, [128], offsets=[0])
            tile_copy = asc2.copy(tile)

        Copy a sub-tile from a larger tile with explicit shape and offsets: ::

            tile = asc2.load(x_gm, [64, 64], offsets=[0, 0])
            sub_tile = asc2.copy(tile, [32, 32], offsets=[16, 16])

        Copy a tile to a different memory location (e.g., L0A for matrix multiplication): ::

            a_l1 = asc2.load(a_gm, [64, 128], offsets=[0, 0], location=asc2.TileLocation.L1)
            a_l0a = asc2.copy(a_l1, [64, 32], offsets=[0, 0], location=asc2.TileLocation.L0A)
            b_l0b = asc2.copy(b_l1, [32, 64], offsets=[0, 0], location=asc2.TileLocation.L0B)

        Copy accumulator result from L0C to L1: ::

            acc = asc2.zeros_acc([64, 64], dtype=asc2.float32)
            asc2.matmul_acc(acc, a_l0a, b_l0b)
            result_l1 = asc2.copy(acc, location=asc2.TileLocation.L1)
    """
    check_type("tile", tile, Tile)
    check_type("location", location, TileLocation)
    if shape is None:
        shape = tile.shape
    else:
        shape = verify_shape(shape)
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


@require_jit
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
            Must contain static values (e.g., :code:`ConstExpr` or compile-time constants).
            For 1D tiles, any shape is supported. For 2D+ tiles in :code:`UB`, the last dimension
            must be aligned to 32 bytes (e.g., 8 elements for float32, 16 elements for float16).
        real_shape: Explicitly specify how many elements to load from the tensor.
            The tile will have the given :code:`shape`, but only :code:`real_shape` elements are loaded;
            remaining elements are filled with :code:`pad_value`. Must match the rank of :code:`shape`
            and each dimension must not exceed the corresponding tile dimension.
        offsets: The offsets into the tensor for each dimension. Mutually exclusive with :code:`tile_id`.
        tile_id: The tile index for each dimension, where offset is computed as :code:`tile_id * shape`.
            Mutually exclusive with :code:`offsets`.
        location: The memory location for the tile. Default is :code:`TileLocation.UB`.
            Available locations: :code:`UB`, :code:`L1`, :code:`L0A`, :code:`L0B`, :code:`BT`.
        pad_value: The value to use for padding when :code:`real_shape` is provided. Default is 0.

    Returns:
        Tile: A tile loaded from the tensor (when :code:`shape` is provided)
        PlainValue: A scalar value loaded from the tensor (when :code:`shape` is None)

    Raises:
        TypeError: If tensor is not a Tensor or location is not a TileLocation
        ValueError: If neither or both of :code:`offsets` and :code:`tile_id` are provided
        RuntimeError: If shape is invalid, data alignment check fails, offsets rank mismatch,
            or real_shape exceeds tile shape

    Note:
        Exactly one of :code:`offsets` or :code:`tile_id` must be provided.
        Only 1D and 2D tiles are fully supported and stable; higher-dimensional support is experimental.

    Examples:
        Load a 1D tile using explicit offsets: ::

            x_gm = asc2.tensor(x_ptr, [1024])
            tile = asc2.load(x_gm, [128], offsets=[256])

        Load a 1D tile using tile_id (offset = tile_id * shape): ::

            x_gm = asc2.tensor(x_ptr, [1024])
            tile = asc2.load(x_gm, [128], tile_id=[2])  # loads from offset 256

        Load a 2D tile from a 2D tensor: ::

            x_gm = asc2.tensor(x_ptr, [64, 128])
            tile = asc2.load(x_gm, [16, 32], offsets=[8, 16])

        Load a scalar value: ::

            x_gm = asc2.tensor(x_ptr, [1024])
            scalar = asc2.load(x_gm, offsets=[42])

        Load a 1D tile with padding (load fewer elements than tile shape): ::

            x_gm = asc2.tensor(x_ptr, [256])
            tile = asc2.load(x_gm, [128], offsets=[200], pad_value=2.0)

        Load a 2D tile with real_shape and padding (load fewer elements than tile shape): ::

            x_gm = asc2.tensor(x_ptr, [100, 100])
            tile = asc2.load(x_gm, [16, 16], offsets=[0, 0], real_shape=[12, 12], pad_value=-1.0)
            # tile has shape [16, 16], but only 12x12 elements loaded from tensor, rest padded with -1.0
    """
    check_type("tensor", tensor, Tensor)
    check_type("location", location, TileLocation)
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
        real_shape = verify_runtime_ints(real_shape, "real_shape")
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


@require_jit
def store(value: Union[Tile, RuntimeNumeric], tensor: Tensor, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
          tile_id: Optional[Iterable[RuntimeInt]] = None, offsets: Optional[Iterable[RuntimeInt]] = None) -> None:
    """
    Store data from a tile or scalar value to a tensor in global memory.

    This function supports three modes of operation:

    1. **Store a tile with explicit offsets**: Store a tile to the tensor at the specified byte offsets.

    2. **Store a tile with tile_id**: Store a tile where the offset is computed as :code:`tile_id * tile_shape` for each
       dimension.

    3. **Store a scalar**: Store a single scalar value (or a tile with exactly one element) at the specified offsets.

    Args:
        value: The source value to store. Can be a tile, a scalar value, or a tile with exactly one element.
            For 1D tiles, any shape is supported. For 2D+ tiles in :code:`UB`, the last dimension
            must be aligned to 32 bytes (e.g., 8 elements for float32, 16 elements for float16).
        tensor: The destination tensor in global memory.
        real_shape: Explicitly specify how many elements to store to the tensor.
            The tile has its full shape, but only :code:`real_shape` elements are written to the tensor.
            Must match the rank of the tile and each dimension must not exceed the corresponding tile dimension.
            Cannot be used for scalar stores.
        offsets: The offsets into the tensor for each dimension. Mutually exclusive with :code:`tile_id`.
            Required for scalar stores.
        tile_id: The tile index for each dimension, where offset is computed as :code:`tile_id * tile_shape`.
            Mutually exclusive with :code:`offsets`. Cannot be used for scalar stores.

    Raises:
        TypeError: If value is not a Tile or numeric, or tensor is not a Tensor
        ValueError: If neither or both of :code:`offsets` and :code:`tile_id` are provided (for tile stores), or if
            :code:`tile_id` or :code:`real_shape` is used with scalar stores
        RuntimeError: If data alignment check fails, offsets rank mismatch, or real_shape exceeds tile shape

    Note:
        For tile stores, exactly one of :code:`offsets` or :code:`tile_id` must be provided.
        For scalar stores, :code:`offsets` must be provided and :code:`tile_id` and :code:`real_shape` cannot be used.
        Tiles from :code:`UB` and :code:`L0C` memory locations can be stored to global memory.
        Only 1D and 2D tiles are fully supported and stable; higher-dimensional support is experimental.

    Examples:
        Store a 1D tile using explicit offsets: ::

            out_gm = asc2.tensor(out_ptr, [1024])
            asc2.store(tile, out_gm, offsets=[256])

        Store a 1D tile using tile_id (offset = tile_id * tile_shape): ::

            out_gm = asc2.tensor(out_ptr, [1024])
            asc2.store(tile, out_gm, tile_id=[2])  # stores at offset 2 * tile.shape[0]

        Store a 2D tile to a 2D tensor: ::

            out_gm = asc2.tensor(out_ptr, [64, 128])
            asc2.store(tile, out_gm, offsets=[8, 16])

        Store a scalar value: ::

            out_gm = asc2.tensor(out_ptr, [1024])
            asc2.store(42.0, out_gm, offsets=[0])

        Store a 2D tile with explicit real_shape (store fewer elements than tile shape): ::

            out_gm = asc2.tensor(out_ptr, [100, 100])
            asc2.store(tile, out_gm, offsets=[0, 0], real_shape=[12, 12])
            # tile has shape [16, 16], but only 12x12 elements stored to tensor

        Store an accumulator tile (from L0C) directly to global memory: ::

            acc = asc2.zeros_acc([64, 128], dtype=asc2.float32)
            asc2.matmul_acc(acc, a_l0a, b_l0b)
            out_gm = asc2.tensor(out_ptr, [64, 128])
            asc2.store(acc, out_gm, offsets=[0, 0])
    """
    check_type("value", value, (Tile, RuntimeNumeric))
    check_type("tensor", tensor, Tensor)
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
        real_shape = verify_runtime_ints(real_shape, "real_shape")
        check_real_shape(value.shape, real_shape)
        real_shape_ir = to_ir_list(real_shape)
    builder.create_asctile_StoreOp(value.to_ir(), tensor.to_ir(), offsets, real_shape_ir)
