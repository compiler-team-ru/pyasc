# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ..core.ir_value import PlainValue
from ..core.utils import global_builder
from .tile import Tile, bind_tile_method


@bind_tile_method(name="sum")
def reduce_sum(input: Tile) -> PlainValue:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asctile_ReduceSumAs1dOp(input.dtype.to_ir(), input.to_ir())
    return PlainValue(handle)


@bind_tile_method(name="max")
def reduce_max(input: Tile) -> PlainValue:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asctile_ReduceMaxAs1dOp(input.dtype.to_ir(), input.to_ir())
    return PlainValue(handle)


@bind_tile_method(name="min")
def reduce_min(input: Tile) -> PlainValue:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asctile_ReduceMinAs1dOp(input.dtype.to_ir(), input.to_ir())
    return PlainValue(handle)
