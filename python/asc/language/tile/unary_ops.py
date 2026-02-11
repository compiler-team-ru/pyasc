# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable

from ..core.ir_value import IRHandle
from ..core.utils import global_builder
from .tile import Tile, bind_tile_method


def op_unary_impl(input: Tile, build: Callable[..., IRHandle]) -> Tile:
    result_dtype = input.dtype
    if result_dtype.is_float():
        handle = build(input.to_ir())
    else:
        raise RuntimeError(f"Operation not support this dtype: {result_dtype}")
    return Tile(handle)


@bind_tile_method
def cos(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CosOp)


@bind_tile_method
def sin(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinOp)


@bind_tile_method
def tan(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanOp)


@bind_tile_method
def sinh(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinhOp)


@bind_tile_method
def cosh(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CoshOp)


@bind_tile_method
def tanh(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanhOp)


@bind_tile_method
def exp(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_ExpOp)


@bind_tile_method
def log(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_LogOp)


@bind_tile_method
def log2(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_Log2Op)


@bind_tile_method
def floor(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_FloorOp)


@bind_tile_method
def ceil(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CeilOp)


@bind_tile_method
def abs(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_AbsFOp)


@bind_tile_method
def erf(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_ErfOp)


@bind_tile_method
def exp2(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_Exp2Op)


@bind_tile_method
def rsqrt(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_RsqrtOp)


@bind_tile_method
def sqrt(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SqrtOp)


@bind_tile_method
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
