# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real
from typing import Union

from ..._C import ir
from ..core.ir_value import PlainValue, RuntimeNumeric
from ..core.utils import global_builder
from .tile import Tile
from .utils import constant_tile, splat_tile


def where(mask: Tile, src0: Tile, src1: Union[Tile, RuntimeNumeric]) -> Tile:
    dtype = src0.dtype
    shape = src0.shape
    if isinstance(src1, Real):
        src1 = constant_tile(src1, shape, dtype)
    elif isinstance(src1, PlainValue):
        src1 = splat_tile(src1, shape, dtype)
    handle = global_builder.get_ir_builder().create_asctile_SelectOp(src0.to_ir().get_type(), mask.to_ir(),
                                                                     src0.to_ir(), src1.to_ir())
    return Tile(handle)
