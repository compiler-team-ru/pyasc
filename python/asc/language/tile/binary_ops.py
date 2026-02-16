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
from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import IRHandle, PlainValue, RuntimeInt, RuntimeNumeric
from ..core.utils import global_builder
from .tile import BinaryOperandTypeError, Tile, bind_tile_method
from .utils import constant_tile, splat_tile


def infer_tile_dtype(value: Union[Tile, PlainValue, Real]) -> DataType:
    if isinstance(value, (Tile, PlainValue)):
        return value.dtype
    if isinstance(value, bool):
        return KT.int1
    if isinstance(value, int):
        return KT.int32
    if isinstance(value, float):
        return KT.float32
    raise BinaryOperandTypeError(f"Unable to obtain dtype of {value.__class__.__name__}")


def infer_common_dtype(lhs: Union[Tile, RuntimeNumeric], rhs: Union[Tile, RuntimeNumeric]) -> DataType:
    lhs_dtype = infer_tile_dtype(lhs)
    rhs_dtype = infer_tile_dtype(rhs)
    if lhs_dtype == rhs_dtype:
        return lhs_dtype
    if not lhs_dtype.is_numeric() or not rhs_dtype.is_numeric():
        raise RuntimeError(f"Operand dtypes must be numeric, got {lhs_dtype} and {rhs_dtype}")
    if lhs_dtype.is_unsigned() or rhs_dtype.is_unsigned():
        raise NotImplementedError(f"Unsigned dtype operands not supported, got {lhs_dtype} and {rhs_dtype}")
    if lhs_dtype.is_signed() and rhs_dtype.is_signed() and lhs_dtype.bitwidth != rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.bitwidth > rhs_dtype.bitwidth else rhs_dtype
    if lhs_dtype.is_float() and rhs_dtype.is_float() and lhs_dtype.bitwidth != rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.bitwidth > rhs_dtype.bitwidth else rhs_dtype
    if lhs_dtype.bitwidth == rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.is_float() else rhs_dtype
    raise RuntimeError(f"Unable to infer common dtype between {lhs_dtype} and {rhs_dtype}")


def create_tile(value: Union[Tile, RuntimeNumeric], dtype: DataType) -> Tile:
    if isinstance(value, Tile):
        return value.to(dtype)
    if isinstance(value, Real):
        return constant_tile(value, input.shape, dtype)
    if isinstance(value, PlainValue):
        return splat_tile(value, input.shape, dtype)
    raise BinaryOperandTypeError(f"Tile cannot be created from {value.__class__.__name__}")


def op_binary_impl(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric],
                   build_int: Callable[..., IRHandle], build_float: Callable[..., IRHandle]) -> Tile:
    if not isinstance(input, Tile) and not isinstance(other, Tile):
        raise BinaryOperandTypeError(f"At least one operand must be tile, got {type(input)} and {type(other)}")
    result_dtype = infer_common_dtype(input, other)
    input = create_tile(input, result_dtype)
    other = create_tile(other, result_dtype)
    if result_dtype.is_int():
        handle = build_int(input.to_ir(), other.to_ir())
    elif result_dtype.is_float():
        handle = build_float(input.to_ir(), other.to_ir())
    else:
        raise RuntimeError(f"Unexpected result tile dtype: {result_dtype}")
    return Tile(handle)


def op_compare_impl(input: Tile, other: Union[Tile, RuntimeNumeric], pred_int: ir.CmpIPredicate,
                    pred_float: ir.CmpFPredicate) -> Tile:
    if not isinstance(input, Tile) and not isinstance(other, Tile):
        raise BinaryOperandTypeError(f"At least one operand must be tile, got {type(input)} and {type(other)}")
    result_dtype = infer_common_dtype(input, other)
    input = create_tile(input, result_dtype)
    other = create_tile(other, result_dtype)
    builder = global_builder.get_ir_builder()
    if result_dtype.is_int():
        handle = builder.create_arith_CmpIOp(pred_int, input.to_ir(), other.to_ir())
    elif result_dtype.is_float():
        handle = builder.create_arith_CmpFOp(pred_float, input.to_ir(), other.to_ir())
    else:
        raise RuntimeError(f"Unexpected result tile dtype: {result_dtype}")
    return Tile(handle)


@bind_tile_method(name="__eq__", binary_op=True)
def equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.eq, ir.CmpFPredicate.OEQ)


@bind_tile_method(name="__ne__", binary_op=True)
def not_equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.ne, ir.CmpFPredicate.ONE)


@bind_tile_method(name="__gt__", binary_op=True)
def greater(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.sgt, ir.CmpFPredicate.OGT)


@bind_tile_method(name="__ge__", binary_op=True)
def greater_equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.sge, ir.CmpFPredicate.OGE)


@bind_tile_method(name="__lt__", binary_op=True)
def less(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.slt, ir.CmpFPredicate.OLT)


@bind_tile_method(name="__le__", binary_op=True)
def less_equal(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CmpIPredicate.sle, ir.CmpFPredicate.OLE)


@bind_tile_method(name="__add__", binary_op=True)
def add(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_AddIOp, builder.create_arith_AddFOp)


@bind_tile_method(name="__sub__", binary_op=True)
def sub(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_SubIOp, builder.create_arith_SubFOp)


@bind_tile_method(name="__mul__", binary_op=True)
def mul(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MulIOp, builder.create_arith_MulFOp)


@bind_tile_method(name="__truediv__", binary_op=True)
def div(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_DivSIOp, builder.create_arith_DivFOp)


def maximum(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MaxSIOp, builder.create_arith_MaximumFOp)


def minimum(input: Tile, other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MinSIOp, builder.create_arith_MinimumFOp)


@bind_tile_method(name="__lshift__", binary_op=True)
def left_shift(input: Tile, other: RuntimeInt) -> Tile:
    builder = global_builder.get_ir_builder()
    result_dtype = input.dtype
    if isinstance(other, int) and other >= 0:
        other = constant_tile(other, input.shape, result_dtype)
    elif isinstance(other, PlainValue) and other.dtype.is_int():
        other = splat_tile(other, input.shape, result_dtype)
    else:
        raise BinaryOperandTypeError(f"Left shift requires positive integer operand, got {other!r}")
    handle = builder.create_arith_ShLIOp(input.to_ir(), other.to_ir())
    return Tile(handle)


@bind_tile_method(name="__rshift__", binary_op=True)
def right_shift(input: Tile, other: RuntimeInt) -> Tile:
    builder = global_builder.get_ir_builder()
    result_dtype = input.dtype
    if isinstance(other, int) and other >= 0:
        other = constant_tile(other, input.shape, result_dtype)
    elif isinstance(other, PlainValue) and other.dtype.is_int():
        other = splat_tile(other, input.shape, result_dtype)
    else:
        raise BinaryOperandTypeError(f"Right shift requires positive integer operand, got {other!r}")
    handle = builder.create_arith_ShRSIOp(input.to_ir(), other.to_ir())
    return Tile(handle)
