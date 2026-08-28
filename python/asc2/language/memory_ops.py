# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable, List, Optional, Tuple, Union, overload

from asc._C import ir
from asc.language.core.dtype import KnownTypes as KT
from asc.language.core.ir_value import IRHandle, PlainValue, RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from asc.language.core.utils import global_builder, require_jit

from .global_tensor import GlobalTensor
from .local_tensor import LocalTensor
from .tensor_location import TensorLocation, TensorLocLike
from .utils import cast_tensor_location as cast_loc
from .validation import check_dtype, check_runtime_int, check_type, verify_location, verify_runtime_ints, verify_shape


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
         location: Optional[TensorLocLike] = None) -> LocalTensor:
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
        location: The memory location for the destination tensor. Default is ``src.location``.
            Supported location transfers: ``L1`` to ``L0A``, ``L1`` to ``L0B``, ``L1`` to ``BT``, ``L0C`` to ``L1``,
            ``L0C`` to ``UB``, ``UB`` to ``L1``.

    Returns:
        LocalTensor: A new tensor that is a copy of the source tensor

    Raises:
        TypeError: If src is not a LocalTensor or location is not a TensorLocation-like
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

        Copy matmul result from L0C to UB for further processing: ::

            result = asc2.matmul(a_l0a, b_l0b)
            result_ub = asc2.copy(result, location=asc2.TensorLocation.UB)

        Alternatively, the ``to`` method can be used to transform the tensor location: ::

            ub_tensor = asc2.copy_in(x_gm, [0], [128], asc2.TensorLocation.UB)
            l1_tensor = ub_tensor.to(asc2.TensorLocation.L1)
    """
    check_type("src", src, LocalTensor)
    location = src.location if location is None else location
    if src.location == TensorLocation.L1:
        location = verify_location(location, allow=(TensorLocation.L0A, TensorLocation.L0B, TensorLocation.BT))
    elif src.location == TensorLocation.L0C:
        location = verify_location(location, allow=(TensorLocation.L1, TensorLocation.UB))
    elif src.location == TensorLocation.UB:
        location = verify_location(location, allow=TensorLocation.L1)
    elif src.location != TensorLocation.Auto:
        raise RuntimeError(f"'src' tensor location must be L1, L0C, or UB, got {src.location.name}")
    if shape is None:
        shape = src.shape
    else:
        shape = verify_shape(shape, src.rank)
    if offsets is None:
        offsets = (0, ) * len(src.shape)
    else:
        offsets = verify_offsets(offsets, src.rank)
    ir_type = ir.get_asctile_LocalTensorType(list(shape), src.dtype.to_ir(), location)
    handle = global_builder.get_ir_builder().create_asctile_CopyOp(ir_type, src.to_ir(), to_ir_list(offsets))
    return cast_loc(LocalTensor(handle))


@overload
def copy_in(src: GlobalTensor, offsets: Iterable[RuntimeInt], shape: Iterable[int],
            location: TensorLocLike = TensorLocation.Auto, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
            pad_value: Optional[RuntimeNumeric] = None) -> LocalTensor:
    ...


@overload
def copy_in(src: GlobalTensor, offsets: Iterable[RuntimeInt]) -> PlainValue:
    ...


@require_jit
def copy_in(src: GlobalTensor, offsets: Iterable[RuntimeInt], shape: Optional[Iterable[int]] = None,
            location: TensorLocLike = TensorLocation.Auto, *, real_shape: Optional[Iterable[RuntimeInt]] = None,
            pad_value: Optional[RuntimeNumeric] = None) -> Union[LocalTensor, PlainValue]:
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
        location: The memory location for the local tensor. Default is ``TensorLocation.Auto``.
            Available locations: ``UB``, ``L1``, ``L0A``, ``L0B``, ``BT``.
        real_shape: Explicitly specify how many elements to load from the global tensor.
            The local tensor will have the given ``shape``, but only ``real_shape`` elements are loaded;
            remaining elements are filled with ``pad_value``. Must match the rank of ``shape`` and each dimension must
            not exceed the corresponding tensor dimension. Not supported for ``TensorLocation.L1``, ``L0A``, or ``L0B``.
            If ``pad_value`` is specified but ``real_shape`` is not, ``real_shape`` defaults to ``shape``.
        pad_value: The value to use for padding when ``real_shape`` is provided. If neither ``real_shape`` nor
            ``pad_value`` is specified, no padding is applied (pad sections contain uninitialized data).

    Returns:
        LocalTensor: A local tensor loaded from the global tensor (when ``shape`` is provided)
        PlainValue: A scalar value loaded from the global tensor (when ``shape`` is None)

    Raises:
        TypeError: If src is not a GlobalTensor or location is not a TensorLocation-like
        RuntimeError: If shape is invalid, data alignment check fails, offsets rank mismatch,
            real_shape exceeds tensor shape, or ``real_shape`` is used with ``TensorLocation.L1``, ``L0A``, or ``L0B``

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
    location = verify_location(
        location,
        allow=(TensorLocation.UB, TensorLocation.L1, TensorLocation.L0A, TensorLocation.L0B, TensorLocation.BT))
    if real_shape is not None and location not in (TensorLocation.Auto, TensorLocation.UB):
        raise RuntimeError(f"'real_shape' argument is not supported with {location.name} tensor location")
    builder = global_builder.get_ir_builder()
    offsets = to_ir_list(verify_offsets(offsets, src.rank))
    if shape is None:
        handle = builder.create_asctile_GetValueOp(src.dtype.to_ir(), src.to_ir(), offsets)
        return PlainValue(handle)
    shape = verify_shape(shape, src.rank)
    ir_type = ir.get_asctile_LocalTensorType(list(shape), src.dtype.to_ir(), location)
    if pad_value is not None and real_shape is None:
        real_shape = shape
    if real_shape is not None and pad_value is None:
        pad_value = 0
    pad_value = _mat(pad_value if pad_value is not None else 0, src.dtype).to_ir()
    real_shape = [] if real_shape is None else to_ir_list(verify_real_shape(real_shape, shape))
    handle = builder.create_asctile_LoadOp(ir_type, src.to_ir(), offsets, pad_value, real_shape)
    return cast_loc(LocalTensor(handle))


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
    verify_location(src.location, "src", (TensorLocation.UB, TensorLocation.L0C))
    real_shape = [] if real_shape is None else to_ir_list(verify_real_shape(real_shape, src.shape))
    builder.create_asctile_StoreOp(src.to_ir(), dst.to_ir(), offsets, real_shape)


@require_jit
def gather(src: GlobalTensor, offsets: Iterable[RuntimeInt], index: LocalTensor, dim: int, check_bounds: bool = True,
           real_shape: Optional[RuntimeInt] = None, pad_value: Optional[RuntimeNumeric] = None) -> LocalTensor:
    """
    Gather individual elements from global tensor

    Args:
        src: Source global tensor
        offsets: Offsets in ``src`` tensor for dimensions 0..dim
        index: Tensor stores indexes, should have rank 1 and integer dtype
        dim: Dimension of ``src`` used for indexing
        check_bounds: When set to True fill with ``pad_value`` elements with invalid index. When set to False no bound checking will
            be performed. By default set to True
        real_shape: How many elements in index are processed, if not specified process all elements in ``index``
        pad_value: Value to pad invalid indexes or innermost dimension to 32 byte border. When not specified 0 is used

    Returns:
        A Local tensor with shape ``[index.shape[0], src.shape[dim+1], ... src.shape[src.rank-1]]``, dtype same as src, last dimension is aligned to 32 bytes

    Notes:
        Gather take individual elements from specified dimension by index: result[i] = src[offsets[0],...offsets[dim-1], offsets[dim]+index[i]].
        When dim is the last dimension of src, copy individual elements, otherwise copy tensor of size len(src)-dim-1 for
        each element in index.
        Dimensions dim-1,...src.rank-1 of ``src`` should be static. Last dimension of result will be aligned to 32 bytes and padded by pad value

    Raises:
        ValueError: offsets rank mismatch, dim out of range of src rank
        TypeError: If ``src`` is not a GlobalTensor, ``src`` have dynamic dimension after ``dim``, ``index`` not a LocalTensor, ``index`` have floating point type

    Examples:

        Reads by outermost dimension. ``src`` shape is [1024, 128], ``tile`` have shape [256, 128]: ::

            index = asc2.copy_in(index, [0], [256])
            tile = asc2.gather(src, [0], index, dim=0)

        Read individual elements from specified row of global tensor, ``tile`` have shape [256]: ::

            index = asc2.copy_in(index, [0], [256])
            tile = asc2.gather(src, [row, 0], index, dim=1)

        Read even rows from tensor: ::

            src = [[0, 1, 2, 3, 4, 5, 6, 7],
                   [8, 9, 10, 11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20, 21, 22, 23],
                   [24, 25, 26, 27, 28, 29, 30, 31],
                   ....
                   [248, 249, 250, 251, 252, 253, 254, 255] # shape is [32, 8]
            index = [0, 2, 4, 6, ..., 30]
            ...
            index = asc2.copy_in(index, [0], [16])   # shape is [16]
            tile = asc2.gather(src, [0], index, dim=0) # result is [16, 8]

            tile = [[0, 1, 2, 3, 4, 5, 6, 7],
                   [16, 17, 18, 19, 20, 21, 22, 23],
                   ...,
                   [240, 241, 242, 243, 244, 245, 246, 247]]


    """
    check_type("src", src, GlobalTensor)
    offsets = to_ir_list(verify_runtime_ints(offsets, "offsets", dim + 1))
    check_type("index", index, LocalTensor)
    check_dtype("index", index, (KT.int8, KT.int16, KT.int32, KT.int64))
    if len(index.shape) > 1:
        raise TypeError(f"'index' tensor have rank {index.rank}, supports only rank 1")
    verify_location(index.location, "index", allow=TensorLocation.UB)
    if dim < 0 or dim >= src.rank:
        raise ValueError(f"'dim' of value {dim} is not valid for tensor 'src' of rank {src.rank}")
    if dim == len(src.shape) - 1:
        raise NotImplementedError("Gather for last dimension not implemented")
    if real_shape is not None:
        check_runtime_int("real_shape", real_shape)
        real_shape = _mat(real_shape, KT.int32).to_ir()
    if pad_value is not None:
        pad_value = _mat(pad_value, src.dtype).to_ir()
    target_shape = list(index.shape)
    for i in range(dim + 1, src.rank):
        if src.shape.is_dynamic_dim(i):
            raise ValueError(f"All dimensions {dim+1}..{src.rank-1} of 'src' tensor should have static size")
        else:
            target_shape += [src.shape[i]]
    dtype_size = src.dtype.sizeof()
    elements_in_block = ir.ub_block_size // dtype_size
    target_shape[-1] = (target_shape[-1] + (elements_in_block - 1)) // elements_in_block * elements_in_block
    ir_type = ir.get_asctile_LocalTensorType(target_shape, src.dtype.to_ir(), TensorLocation.UB)
    index = cast_loc(index, TensorLocation.UB)
    handle = global_builder.get_ir_builder().create_asctile_GatherOp(ir_type, src.to_ir(), offsets, index.to_ir(),
                                                                     real_shape, dim, check_bounds, pad_value)
    return cast_loc(LocalTensor(handle))


@require_jit
def scatter(index: LocalTensor, src: LocalTensor, dst: GlobalTensor, offsets: Iterable[RuntimeInt], dim: int,
            check_bounds: bool = True, real_shape: Optional[RuntimeInt] = None) -> None:
    """
    Modify individual elements or subtensors of global tensor ``dst`` specified by ``index`` with values from ``src``:

    Args:
        index: Local tensor stores indexes
        src: Local tensor stores modified data
        dst: Global tensor to store data
        offsets: Offsets in ``dst`` tensor for 0..dim dimensions
        dim: Dimension of ``dst`` used for indexing
        check_bounds: If True there will be checking of invalid indexes on writes, when set to False, user must be sure
            that there is not invalid indexes present in ``index``. Default value is True
        real_shape: How many elements in index are processed, if not specified process all elements in ``index``

    Note:
        This operation equal to ``dst[offsets[0],...offsets[dim-1],offsets[dim]+index[i]] = src[i]``, for each value from index.
        If ``dim`` is last dimension of ``dst`` single element copied, otherwise entire subtensor
        of shape ``[1,...,1,src.shape[1],...src.shape[n]]`` modified.
        ``index`` should have rank 1 and integer dtype. Last dimensions of ``dst`` after dim should be static.
        Also ``dst`` and ``src`` should have same dtype

    Raises:
        ValueError: offsets rank mismatch, dim out of range of src rank
        TypeError: wrong type of shape of input arguments

    Examples:

        Updates ``result_tensor`` by outermost dimension. ``result_gm`` shape is [1024, 128]: ::

            index = asc2.copy_in(index_tensor, [0], [256])
            data = asc2.copy_in(changes_tensor, [0, 0], [256, 128])
            asc2.scatter(index, data, result_gm, [0], dim=0)

        Updates individual elements in row ``row`` of ``result_gm``. ``result_gm`` shape is [1024, 128]: ::

            index = asc2.copy_in(index_tensor, [0], [256])
            data = asc2.copy_in(changes_tensor, [0], [256])
            asc2.scatter(index, data, result_gm, [row, 0], dim=1)

        Read even rows from tensor: ::

            result_gm = [[0, 1, 2, 3, 4, 5, 6, 7],
                         [8, 9, 10, 11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20, 21, 22, 23],
                         [24, 25, 26, 27, 28, 29, 30, 31],
                         ....
                         [248, 249, 250, 251, 252, 253, 254, 255] # shape is [32, 8]
            index_gm = [0, 2, 4, 6, ..., 30]
            data_dm = [[0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       ...
                       [15, 15, 15, 15, 15, 15, 15, 15]]

            index = asc2.copy_in(index_gm, [0], [16])
            data = asc2.copy_in(data_gm, [0], [16, 8])
            asc2.scatter(index, data, result_gm, [row, 0], dim=1)

            result_gm = [[0, 0, 0, 0, 0, 0, 0, 0],
                         [16, 17, 18, 19, 20, 21, 22, 23],
                         [1, 1, 1, 1, 1, 1, 1, 1],
                         ...,
                         [240, 241, 242, 243, 244, 245, 246, 247]]


    """
    check_type("dst", dst, GlobalTensor)
    check_type("index", index, LocalTensor)
    check_type("src", src, LocalTensor)
    check_dtype("index", index, (KT.int8, KT.int16, KT.int32, KT.int64))
    offsets = to_ir_list(verify_runtime_ints(offsets, "offsets", dim + 1))
    if dim < 0 or dim >= dst.rank:
        raise ValueError(f"'dim' value {dim} out of tensor 'dst' rank {dst.rank}")
    if dim == dst.rank - 1:
        raise NotImplementedError("Scatter on last dimension not implemented")
    if dst.rank != src.rank + dim:
        raise TypeError(f"Tensor 'src' should have rank {dst.rank - dim}")
    if index.rank > 1:
        raise TypeError(f"'index' tensor have rank {dst.rank}, supports only rank 1")
    if dst.dtype != src.dtype:
        raise TypeError(f"'src' and 'dst' have different element types: {src.dtype} and {dst.dtype}")
    for i in range(dim + 1, len(dst.shape)):
        if dst.shape.is_dynamic_dim(i):
            raise ValueError(f"All dimensions {dim}..{dst.rank-1} of 'dst' tensor should have static size")
        if i != dst.rank - 1 and dst.shape[i] != src.shape[i - dim]:
            raise TypeError(f"Tensor 'src' dimension {i-dim} with size {src.shape[i-dim]} and tensor 'dst'"
                            f"dimension {i} with size {dst.shape[i]} are different")
    if real_shape is not None:
        check_runtime_int("real_shape", real_shape)
        real_shape = _mat(real_shape, KT.int32).to_ir()
    src = cast_loc(src, TensorLocation.UB)
    index = cast_loc(index, TensorLocation.UB)
    global_builder.get_ir_builder().create_asctile_ScatterOp(src.to_ir(), index.to_ir(), dst.to_ir(), offsets,
                                                             real_shape, dim, check_bounds)
