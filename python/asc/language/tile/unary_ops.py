# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, Union

from ..core.dtype import KnownTypes as KT
from ..core.ir_value import PlainValue, RuntimeNumeric, IRHandle, materialize_ir_value as _mat
from ..core.utils import check_type, global_builder
from .tile import Tile, bind_tile_method


def op_unary_impl(input: Union[Tile, RuntimeNumeric], build: Callable[..., IRHandle],
                  support_scalar: bool = False) -> Union[Tile, PlainValue]:
    constraint = Union[Tile, RuntimeNumeric] if support_scalar else Tile
    check_type("input", input, constraint)
    is_scalar = not isinstance(input, Tile)
    if is_scalar:
        input = _mat(input, KT.float32)
    result_dtype = input.dtype
    if result_dtype.is_float():
        handle = build(input.to_ir())
    else:
        raise RuntimeError(f"Operation not support this dtype: {result_dtype}")
    if is_scalar:
        return PlainValue(handle)
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
    return op_unary_impl(input, builder.create_math_ExpOp, support_scalar=True)


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
    return op_unary_impl(input, builder.create_math_ErfOp, support_scalar=True)


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
    return op_unary_impl(input, builder.create_math_SqrtOp, support_scalar=True)


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


@bind_tile_method
def softmax(input: Tile) -> Tile:
    check_type("input", input, Tile)
    if input.dtype not in [KT.float32, KT.half]:
        raise RuntimeError("Only float and half types are supported.")
    if len(input.shape) > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported")
    handle = global_builder.get_ir_builder().create_asctile_SoftmaxOp(input.to_ir().get_type(), input.to_ir())
    return Tile(handle)
