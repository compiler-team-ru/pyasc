# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ..core.utils import global_builder
from .tile import Tile, bind_tile_method


@bind_tile_method(name="relu")
def relu(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asctile_ReluOp(input.to_ir().get_type(), input.to_ir())
    return Tile(handle)


@bind_tile_method(name="__neg__")
def negative(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    result_dtype = input.dtype
    if result_dtype.is_int():
        handle = input * (-1)
    else:
        handle = builder.create_arith_NegFOp(input.to_ir())
    return Tile(handle)
