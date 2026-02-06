# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real
from typing import Callable, Union

from ..._C import ir
from ...common.compat import isinstance
from ..core.ir_value import IRHandle, PlainValue, RuntimeNumeric
from ..core.utils import global_builder
from .tile import Tile, bind_tile_method
from .utils import constant_tile, splat_tile


def op_binary_impl(input: Tile, other: Union[Tile, RuntimeNumeric], build_int: Callable[..., IRHandle],
                   build_float: Callable[..., IRHandle]) -> Tile:
    # TODO: cast to common dtype
    result_dtype = input.dtype
    if isinstance(other, Real):
        other = constant_tile(other, input.shape, result_dtype)
    elif isinstance(other, PlainValue):
        other = splat_tile(other, input.shape, result_dtype)
    if result_dtype.is_int():
        handle = build_int(input.to_ir(), other.to_ir())
    elif result_dtype.is_float():
        handle = build_float(input.to_ir(), other.to_ir())
    else:
        raise RuntimeError(f"Unexpected result tile dtype: {result_dtype}")
    return Tile(handle)


def op_compare_impl(input: Tile, other: Union[Tile, RuntimeNumeric], pred_int: ir.CmpIPredicate,
                    pred_float: ir.CmpFPredicate) -> Tile:
    builder = global_builder.get_ir_builder()
    dtype = input.dtype
    if isinstance(other, Real):
        other = constant_tile(other, input.shape, dtype)
    elif isinstance(other, PlainValue):
        other = splat_tile(other, input.shape, dtype)
    if dtype.is_int():
        handle = builder.create_arith_CmpIOp(pred_int, input.to_ir(), other.to_ir())
    elif dtype.is_float():
        handle = builder.create_arith_CmpFOp(pred_float, input.to_ir(), other.to_ir())
    else:
        raise RuntimeError(f"Unexpected result tile dtype: {dtype}")
    return Tile(handle)


@bind_tile_method(name="__eq__")
def equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.eq, ir.CmpFPredicate.OEQ)


@bind_tile_method(name="__ne__")
def not_equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.ne, ir.CmpFPredicate.ONE)


@bind_tile_method(name="__gt__")
def greater(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.sgt, ir.CmpFPredicate.OGT)


@bind_tile_method(name="__ge__")
def greater_equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.sge, ir.CmpFPredicate.OGE)


@bind_tile_method(name="__lt__")
def less(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.slt, ir.CmpFPredicate.OLT)


@bind_tile_method(name="__le__")
def less_equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.sle, ir.CmpFPredicate.OLE)


@bind_tile_method(name="__add__")
def add(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_AddIOp, builder.create_arith_AddFOp)


@bind_tile_method(name="__sub__")
def sub(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_SubIOp, builder.create_arith_SubFOp)


@bind_tile_method(name="__mul__")
def mul(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MulIOp, builder.create_arith_MulFOp)
