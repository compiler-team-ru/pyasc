# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real

from ..._C import ir
from ..core.dtype import DataType
from ..core.ir_value import PlainValue
from ..core.tensor import TensorShape
from ..core.utils import check_type, global_builder
from .tile import Tile


def constant_tile(value: Real, shape: TensorShape, dtype: DataType, loc: ir.TileLocation = ir.TileLocation.UB) -> Tile:
    check_type("value", value, Real)
    builder = global_builder.get_ir_builder()
    attr_builders = {
        "int8": builder.get_i8_attr,
        "int16": builder.get_i16_attr,
        "int32": builder.get_i32_attr,
        "int64": builder.get_i64_attr,
        "float16": builder.get_f16_attr,
        "float32": builder.get_f32_attr,
        "float64": builder.get_f64_attr,
    }
    attr_builder = attr_builders.get(str(dtype))
    if attr_builder is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ir_type = ir.get_asctile_TileType(shape, dtype.to_ir(), loc)
    splat_attr = ir.get_splat_attr(ir_type, attr_builder(value))
    handle = builder.create_arith_ConstantOp(splat_attr)
    return Tile.from_ir(handle)


def splat_tile(value: PlainValue, shape: TensorShape, dtype: DataType,
               loc: ir.TileLocation = ir.TileLocation.UB) -> Tile:
    ir_type = ir.get_asctile_TileType(shape, dtype.to_ir(), loc)
    handle = global_builder.get_ir_builder().create_asctile_SplatOp(ir_type, value.cast(dtype).to_ir())
    return Tile.from_ir(handle)
