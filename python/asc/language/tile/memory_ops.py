# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable, Optional

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from ..core.utils import global_builder
from .tensor import Tensor
from .tile import Tile


def load(tensor: Tensor, offsets: Iterable[RuntimeInt], shape: Iterable[int],
         pad_value: Optional[RuntimeNumeric] = None) -> Tile:
    if not all(isinstance(dim, int) for dim in shape):
        raise RuntimeError("shape must be integers")
    ir_type = ir.get_asctile_TileType(list(shape), tensor.dtype.to_ir(), ir.TileLocation.UB)
    offsets = [_mat(v, KT.int32).to_ir() for v in offsets]
    pad_value = _mat(pad_value, tensor.dtype) if pad_value is not None else None
    handle = global_builder.get_ir_builder().create_asctile_LoadOp(ir_type, tensor.to_ir(), offsets, pad_value)
    return Tile(handle)


def store(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt]) -> None:
    offsets = [_mat(v, KT.int32).to_ir() for v in offsets]
    global_builder.get_ir_builder().create_asctile_StoreOp(tile.to_ir(), tensor.to_ir(), offsets)
