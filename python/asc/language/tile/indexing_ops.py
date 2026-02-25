# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Union

from ..core.ir_value import RuntimeNumeric
from ..core.utils import check_type, global_builder
from .tile import Tile
from .utils import create_tile, infer_common_dtype


def where(mask: Tile, src0: Union[Tile, RuntimeNumeric], src1: Union[Tile, RuntimeNumeric]) -> Tile:
    check_type("mask", mask, Tile)
    if not isinstance(src0, Tile) and not isinstance(src1, Tile):
        raise RuntimeError(f"At least one operand must be tile, got {type(src0)} and {type(src1)}")
    src_dtype = infer_common_dtype(src0, src1)
    src0 = create_tile(src0, src_dtype, mask.shape)
    src1 = create_tile(src1, src_dtype, mask.shape)
    handle = global_builder.get_ir_builder().create_asctile_SelectOp(src0.to_ir().get_type(), mask.to_ir(),
                                                                     src0.to_ir(), src1.to_ir())
    return Tile(handle)
