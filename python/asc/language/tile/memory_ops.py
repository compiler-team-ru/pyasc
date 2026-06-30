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
from ..core.ir_value import IRHandle, PlainValue, RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from ..core.utils import global_builder, require_jit
from .global_tensor import GlobalTensor
from .local_tensor import LocalTensor, TensorLocation
from .validation import check_data_alignment, check_type, verify_runtime_ints, verify_shape


def to_ir_list(values: Iterable[RuntimeInt]) -> List[IRHandle]:
    return [_mat(v, KT.int32).to_ir() for v in values]


def verify_offsets(offsets: Iterable[RuntimeInt], rank: int) -> Tuple[RuntimeInt, ...]:
    return verify_runtime_ints(offsets, "offsets", rank)


def verify_real_shape(real_shape: Iterable[RuntimeInt], shape: Tuple[int, ...]) -> Tuple[RuntimeInt, ...]:
    real_shape = verify_runtime_ints(real_shape, "real_shape", len(shape))
    for tensor_dim, real_dim in zip(shape, real_shape):
        if isinstance(real_dim, int) and real_dim > tensor_dim:
            raise RuntimeError(f"real_shape[{real_dim}] (which is {real_shape}) "
                               f"exceeds tensor dimension #{tensor_dim} (which is {shape})")
    return real_shape


@require_jit
def copy(src: LocalTensor, offsets: Optional[Iterable[RuntimeInt]] = None, shape: Optional[Iterable[int]] = None,
         location: TensorLocation = TensorLocation.UB) -> LocalTensor:
    """
    Copy a local tensor to a new local tensor, optionally reshaping and relocating.

    **Rationale:** Unlike frameworks with simpler memory hierarchies (e.g., CUDA's global/shared/registers), Ascend NPUs
    expose multiple local memory levels (L1, L0A, L0B, L0C, UB) where local-to-local transfers are common. ``copy_in``
    and ``copy_out`` have clear directional semantics when one endpoint is global memory ("copy in = to local", "copy
    out = to global"), but this breaks down for local-to-local transfers: the same L0C→L1 operation is a "copy out" from
    L0C's perspective yet a "copy in" from L1's. Local ``copy`` eliminates this ambiguity by providing
    a direction-agnostic operation that clearly expresses intent regardless of which memory level you're reasoning from.

    Args:
        src: The source tensor to copy.
        offsets: The offsets into the source tensor for each dimension. Default is zeros.
        shape: The shape of the resulting tensor. If None, uses the source tensor's shape.
            Must contain static values (e.g., ``ConstExpr`` or compile-time constants).
        location: The memory location for the destination tensor. Default is ``TensorLocation.UB``.
            Supported location transfers: ``L1`` to ``L0A``, ``L1`` to ``L0B``, ``L1`` to ``BT``, ``L0C`` to ``L1``.

    Returns:
        LocalTensor: A new tensor that is a copy of the source tensor

    Raises:
        TypeError: If src is not a LocalTensor or location is not a TensorLocation
        RuntimeError: If shape is invalid, data alignment check fails, or offsets rank mismatch

    Examples:
        Copy a tensor with the same shape: ::

            src = asc2.copy_in(x_gm, [0], [128])
            result = asc2.copy(src)

        Copy a sub-tensor from a larger tensor with explicit shape and offsets: ::

            src = asc2.copy_in(x_gm, [0, 0], [64, 64])
            result = asc2.copy(src, [16, 16], [32, 32])

        Copy a tensor to a different memory location (e.g., L0A for matrix multiplication): ::

            a_l1 = asc2.copy_in(a_gm, [0, 0], [64, 128], asc2.TensorLocation.L1)
            a_l0a = asc2.copy(a_l1, [0, 0], [64, 32], asc2.TensorLocation.L0A)
            b_l0b = asc2.copy(b_l1, [0, 0], [32, 64], asc2.TensorLocation.L0B)

        Copy accumulator result from L0C to L1: ::

            acc = asc2.zeros_acc([64, 64], dtype=asc2.float32)
            asc2.matmul_acc(acc, a_l0a, b_l0b)
            result_l1 = asc2.copy(acc, location=asc2.TensorLocation.L1)
    """
    check_type("src", src, LocalTensor)
    check_type("location", location, TensorLocation)
    if shape is None:
        shape = src.shape
    else:
        shape = verify_shape(shape, src.rank)
    if offsets is None:
        offsets = (0, ) * len(src.shape)
    else:
        offsets = verify_offsets(offsets, src.rank)
    if location == TensorLocation.UB:
        check_data_alignment(shape, src.dtype)
    ir_type = ir.get_asctile_TileType(list(shape), src.dtype.to_ir(), location)
    handle = global_builder.get_ir_builder().create_asctile_CopyOp(ir_type, src.to_ir(), to_ir_list(offsets))
    return LocalTensor(handle)


@overload
def copy_in(src: GlobalTensor, offsets: Iterable[RuntimeInt], shape: Iterable[int],
            location: TensorLocation = TensorLocation.UB, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
            pad_value: RuntimeNumeric = 0) -> LocalTensor:
    ...


@overload
def copy_in(src: GlobalTensor, offsets: Iterable[RuntimeInt]) -> PlainValue:
    ...


@require_jit
def copy_in(src: GlobalTensor, offsets: Iterable[RuntimeInt], shape: Optional[Iterable[int]] = None,
            location: TensorLocation = TensorLocation.UB, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
            pad_value: RuntimeNumeric = 0) -> Union[LocalTensor, PlainValue]:
    """
    Copy data from a global tensor into a local tensor or scalar value.

    This function supports two modes of operation:

    1. **Load a local tensor with offsets**: Load values to a local tensor of the given shape from the global tensor at
       the specified offsets.

    2. **Load a scalar**: When shape is not provided, load a single scalar value at the specified offsets.

    Args:
        src: The source global tensor.
        offsets: The offsets into the global tensor for each dimension.
        shape: The shape of the local tensor to load. If None, loads a scalar value.
            Must contain static values (e.g., ``ConstExpr`` or compile-time constants).
            For 1D tensors, any shape is supported. For 2D+ tensors in ``UB``, the last dimension must be aligned to 32
            bytes (e.g., 8 elements for float32, 16 elements for float16).
        location: The memory location for the local tensor. Default is ``TensorLocation.UB``.
            Available locations: ``UB``, ``L1``, ``L0A``, ``L0B``, ``BT``.
        real_shape: Explicitly specify how many elements to load from the global tensor.
            The local tensor will have the given ``shape``, but only ``real_shape`` elements are loaded;
            remaining elements are filled with ``pad_value``. Must match the rank of ``shape`` and each dimension must
            not exceed the corresponding tensor dimension.
        pad_value: The value to use for padding when ``real_shape`` is provided. Default is 0.

    Returns:
        LocalTensor: A local tensor loaded from the global tensor (when ``shape`` is provided)
        PlainValue: A scalar value loaded from the global tensor (when ``shape`` is None)

    Raises:
        TypeError: If src is not a GlobalTensor or location is not a TensorLocation
        RuntimeError: If shape is invalid, data alignment check fails, offsets rank mismatch,
            or real_shape exceeds tensor shape

    Note:
        Only 1D and 2D tensors are fully supported and stable; higher-dimensional support is experimental.

    Examples:
        Copy a 1D tensor using explicit offsets: ::

            x_gm = asc2.global_tensor(x_ptr, [1024])
            result = asc2.copy_in(x_gm, [256], [128])

        Copy a 2D tensor from a 2D global tensor: ::

            x_gm = asc2.global_tensor(x_ptr, [64, 128])
            result = asc2.copy_in(x_gm, [8, 16], [16, 32])

        Copy a scalar value: ::

            x_gm = asc2.global_tensor(x_ptr, [1024])
            scalar = asc2.copy_in(x_gm, [42])

        Copy a 1D tensor with padding (load fewer elements than tensor shape): ::

            x_gm = asc2.global_tensor(x_ptr, [256])
            result = asc2.copy_in(x_gm, [200], [128], pad_value=2.0)

        Copy a 2D tensor with real_shape and padding (load fewer elements than tensor shape): ::

            x_gm = asc2.global_tensor(x_ptr, [100, 100])
            result = asc2.copy_in(x_gm, [0, 0], [16, 16], real_shape=[12, 12], pad_value=-1.0)
            # result has shape [16, 16], but only 12x12 elements loaded from global tensor, rest padded with -1.0
    """
    check_type("src", src, GlobalTensor)
    check_type("location", location, TensorLocation)
    builder = global_builder.get_ir_builder()
    offsets = to_ir_list(verify_offsets(offsets, src.rank))
    if shape is None:
        handle = builder.create_asctile_GetValueOp(src.dtype.to_ir(), src.to_ir(), offsets)
        return PlainValue(handle)
    shape = verify_shape(shape, src.rank)
    if location == TensorLocation.UB:
        check_data_alignment(shape, src.dtype)
    ir_type = ir.get_asctile_TileType(list(shape), src.dtype.to_ir(), location)
    pad_value = _mat(pad_value, src.dtype).to_ir() if pad_value is not None else None
    real_shape = [] if real_shape is None else to_ir_list(verify_real_shape(real_shape, shape))
    handle = builder.create_asctile_LoadOp(ir_type, src.to_ir(), offsets, pad_value, real_shape)
    return LocalTensor(handle)


@overload
def copy_out(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[RuntimeInt], *,
             real_shape: Optional[Iterable[RuntimeInt]] = None) -> None:
    ...


@overload
def copy_out(src: RuntimeNumeric, dst: GlobalTensor, offsets: Iterable[RuntimeInt]) -> None:
    ...


@require_jit
def copy_out(src: Union[LocalTensor, RuntimeNumeric], dst: GlobalTensor, offsets: Iterable[RuntimeInt], *,
             real_shape: Optional[Iterable[RuntimeInt]] = None) -> None:
    """
    Copy data from a local tensor or scalar value to a global tensor.

    This function supports two modes of operation:

    1. **Store a local tensor with offsets**: Store a local tensor to the global tensor at the specified offsets.

    2. **Store a scalar**: Store a single value (or a local tensor with exactly one element) at the specified offsets.

    Args:
        src: The source value to store. Can be a local tensor, a scalar value, or a local tensor with one element.
            For 1D tensors, any shape is supported. For 2D+ tensors in ``UB``, the last dimension
            must be aligned to 32 bytes (e.g., 8 elements for float32, 16 elements for float16).
        dst: The destination global tensor.
        offsets: The offsets into the global tensor for each dimension.
        real_shape: Explicitly specify how many elements to store to the global tensor.
            The local tensor has its full shape, but only ``real_shape`` elements are written to the global tensor.
            Must match the rank of the local tensor and each dimension must not exceed the corresponding tensor
            dimension. Cannot be used for scalar stores.

    Raises:
        TypeError: If src is not a LocalTensor or numeric, or dst is not a GlobalTensor
        ValueError: If ``real_shape`` is used with scalar stores
        RuntimeError: If data alignment check fails, offsets rank mismatch, or real_shape exceeds tensor shape

    Note:
        Local tensors from ``UB`` and ``L0C`` memory locations can be stored to global memory.
        Only 1D and 2D tensors are fully supported and stable; higher-dimensional support is experimental.

    Examples:
        Copy a 1D tensor using explicit offsets: ::

            out_gm = asc2.global_tensor(out_ptr, [1024])
            src = asc2.copy_in(x_gm, [0], [128])
            asc2.copy_out(src, out_gm, [256])

        Copy a 2D tensor to a 2D global tensor: ::

            out_gm = asc2.global_tensor(out_ptr, [64, 128])
            src = asc2.copy_in(x_gm, [0, 0], [16, 32])
            asc2.copy_out(src, out_gm, [8, 16])

        Copy a scalar value: ::

            out_gm = asc2.global_tensor(out_ptr, [1024])
            asc2.copy_out(42.0, out_gm, [0])

        Copy a 2D tensor with explicit real_shape (store fewer elements than tensor shape): ::

            out_gm = asc2.global_tensor(out_ptr, [100, 100])
            src = asc2.copy_in(x_gm, [0, 0], [16, 16])
            asc2.copy_out(src, out_gm, [0, 0], real_shape=[12, 12])
            # src has shape [16, 16], but only 12x12 elements stored to global tensor

        Copy an accumulator tensor (from L0C) directly to global memory: ::

            acc = asc2.zeros_acc([64, 128], dtype=asc2.float32)
            asc2.matmul_acc(acc, a_l0a, b_l0b)
            out_gm = asc2.global_tensor(out_ptr, [64, 128])
            asc2.copy_out(acc, out_gm, [0, 0])
    """
    check_type("src", src, (LocalTensor, RuntimeNumeric))
    check_type("dst", dst, GlobalTensor)
    builder = global_builder.get_ir_builder()
    offsets = to_ir_list(verify_offsets(offsets, dst.rank))
    scalar_store = not isinstance(src, LocalTensor) or src.size == 1
    if scalar_store:
        if real_shape is not None:
            raise ValueError("'real_shape' argument cannot be used when storing a scalar value")
        src = src.to(dst.dtype) if isinstance(src, LocalTensor) else _mat(src, dst.dtype)
        builder.create_asctile_SetValueOp(src.to_ir(), dst.to_ir(), offsets)
        return
    if src.location == TensorLocation.UB:
        check_data_alignment(src.shape, src.dtype)
    real_shape = [] if real_shape is None else to_ir_list(verify_real_shape(real_shape, src.shape))
    builder.create_asctile_StoreOp(src.to_ir(), dst.to_ir(), offsets, real_shape)
