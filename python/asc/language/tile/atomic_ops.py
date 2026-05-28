# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.utils import global_builder, require_jit
from .tensor import Tensor
from .tile import Tile
from .validation import check_type, verify_runtime_ints


def op_atomic_impl(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt], kind: ir.AtomicKind):
    check_type("tile", tile, Tile)
    check_type("tensor", tensor, Tensor)
    offsets = [_mat(v, KT.int32).to_ir() for v in verify_runtime_ints(offsets, "offsets")]
    global_builder.get_ir_builder().create_asctile_AtomicRMWOp(tile.to_ir(), tensor.to_ir(), offsets, kind)


@require_jit
def atomic_add(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt]) -> None:
    """
    Atomically add tile elements to a tensor at specified offsets.

    Performs an atomic read-modify-write operation, adding each element of :code:`tile` to the corresponding element in
    :code:`tensor` at the given :code:`offsets`.

    Args:
        tile: The source tile whose elements will be added
        tensor: The destination tensor in global memory
        offsets: The offsets into the tensor for each dimension
    """
    return op_atomic_impl(tile, tensor, offsets, ir.AtomicKind.Add)


@require_jit
def atomic_max(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt]) -> None:
    """
    Atomically compute the maximum between tile elements and tensor elements at specified offsets.

    Performs an atomic read-modify-write operation, storing the maximum of each element in :code:`tile` and the
    corresponding element in :code:`tensor` at the given :code:`offsets`.

    Args:
        tile: The source tile containing comparison values
        tensor: The destination tensor in global memory
        offsets: The offsets into the tensor for each dimension
    """
    return op_atomic_impl(tile, tensor, offsets, ir.AtomicKind.Max)


@require_jit
def atomic_min(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt]) -> None:
    """
    Atomically compute the minimum between tile elements and tensor elements at specified offsets.

    Performs an atomic read-modify-write operation, storing the minimum of each element in :code:`tile` and the
    corresponding element in :code:`tensor` at the given :code:`offsets`.

    Args:
        tile: The source tile containing comparison values
        tensor: The destination tensor in global memory
        offsets: The offsets into the tensor for each dimension
    """
    return op_atomic_impl(tile, tensor, offsets, ir.AtomicKind.Min)
