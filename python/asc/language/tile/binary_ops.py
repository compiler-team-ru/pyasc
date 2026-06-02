# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Callable, Optional, Tuple, TypeVar, Union

from ..._C import ir
from ...common.compat import isinstance
from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import IRHandle, RuntimeInt, RuntimeNumeric
from ..core.utils import global_builder, require_jit
from .tile import BinaryOperandTypeError, Tile, TileLocation, bind_tile_method
from .utils import create_tile, infer_common_dtype, infer_common_shape
from .validation import check_dtype, check_runtime_int, check_type

T = TypeVar("T")

common_support_dtypes = (KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32)
compare_support_dtypes = (KT.int16, KT.int32, KT.float16, KT.bfloat16, KT.float32)


def check_numeric_tile_like(name: str, value: Any, support_dtypes: Tuple[DataType, ...]) -> None:
    check_type(name, value, (Tile, RuntimeNumeric), BinaryOperandTypeError)
    if isinstance(value, Tile):
        check_dtype(name, value, support_dtypes)


def op_binary_impl(
    input: Union[Tile, RuntimeNumeric],
    other: Union[Tile, RuntimeNumeric],
    build_int: Callable[..., IRHandle],
    build_float: Callable[..., IRHandle],
    support_dtypes: Tuple[DataType, ...],
) -> Tile:
    check_numeric_tile_like("input", input, support_dtypes)
    check_numeric_tile_like("other", other, support_dtypes)
    if not isinstance(input, Tile) and not isinstance(other, Tile):
        raise BinaryOperandTypeError(f"At least one operand must be tile, got {type(input)} and {type(other)}")
    result_dtype = infer_common_dtype(input, other)
    result_shape = infer_common_shape(input, other)
    input = create_tile(input, result_dtype, result_shape)
    other = create_tile(other, result_dtype, result_shape)
    if result_dtype.is_int():
        handle = build_int(input.to_ir(), other.to_ir())
    elif result_dtype.is_float():
        handle = build_float(input.to_ir(), other.to_ir())
    else:
        raise RuntimeError(f"Unexpected result tile dtype: {result_dtype}")
    return Tile(handle)


def op_compare_impl(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric],
                    mode: ir.CompareMode) -> Tile:
    check_numeric_tile_like("input", input, compare_support_dtypes)
    check_numeric_tile_like("other", other, compare_support_dtypes)
    if not isinstance(input, Tile) and not isinstance(other, Tile):
        raise BinaryOperandTypeError(f"At least one operand must be tile, got {type(input)} and {type(other)}")
    result_dtype = infer_common_dtype(input, other)
    result_shape = infer_common_shape(input, other)
    input = create_tile(input, result_dtype, result_shape)
    other = create_tile(other, result_dtype, result_shape)
    builder = global_builder.get_ir_builder()
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), builder.get_i1_type())
    handle = builder.create_asctile_CmpOp(ir_type, input.to_ir(), other.to_ir(), mode)
    return Tile(handle)


def set_docstring(name: str, support_dtypes: Tuple[DataType, ...], rhs_scalar_only: bool = False) -> Callable[[T], T]:
    dtypes_str = ", ".join(f":code:`{dtype}`" for dtype in support_dtypes)
    other_type = "scalar" if rhs_scalar_only else "tile or scalar"
    examples = "" if rhs_scalar_only else f"""
        Compute the {name} between elements of two tiles: ::

            input = asc2.load(tensor_a, [32, 16], offsets=[1, 4])
            other = asc2.load(tensor_b, [32, 16], offsets=[2, 8])
            result = asc2.{{fn_name}}(input, other)

    """
    examples += f"""
        Compute the {name} of tile elements and a given scalar value: ::

            input = asc2.load(tensor, [32, 16], offsets=[0, 0])
            result = asc2.{{fn_name}}(input, 2)
    """

    def decorator(fn: T) -> T:
        doc = f"""
    Computes the element-wise {name} of :code:`input` and :code:`other`.

    The supported data types for the inputs are: {dtypes_str}.

    Args:
        input: The left operand (tile or scalar)
        other: The right operand ({other_type})

    Returns:
        Tile: The result of {name}

    Raises:
        RuntimeError: If neither operand is a :code:`Tile`

    Note:
        At least one of input operands must be :code:`Tile`.

    Examples:
        {examples}
        """
        fn.__doc__ = doc.format(fn_name=fn.__name__)
        return fn

    return decorator


@bind_tile_method(name="__eq__", binary_op="==")
@require_jit
@set_docstring("'equality' comparison", compare_support_dtypes)
def equal(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CompareMode.EQ)


@bind_tile_method(name="__ne__", binary_op="!=")
@require_jit
@set_docstring("'inequality' comparison", compare_support_dtypes)
def not_equal(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CompareMode.NE)


@bind_tile_method(name="__gt__", binary_op=">")
@require_jit
@set_docstring("'greater' comparison", compare_support_dtypes)
def greater(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CompareMode.GT)


@bind_tile_method(name="__ge__", binary_op=">=")
@require_jit
@set_docstring("'greater or equal' comparison", compare_support_dtypes)
def greater_equal(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CompareMode.GE)


@bind_tile_method(name="__lt__", binary_op="<")
@require_jit
@set_docstring("'less' comparison", compare_support_dtypes)
def less(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CompareMode.LT)


@bind_tile_method(name="__le__", binary_op="<=")
@require_jit
@set_docstring("'less or equal' comparison", compare_support_dtypes)
def less_equal(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    return op_compare_impl(input, other, ir.CompareMode.LE)


@bind_tile_method(name="__add__", binary_op="+")
@require_jit
@set_docstring("addition", common_support_dtypes)
def add(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_AddIOp, builder.create_arith_AddFOp, common_support_dtypes)


@bind_tile_method(name="__sub__", binary_op="-")
@require_jit
@set_docstring("subtraction", common_support_dtypes)
def sub(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_SubIOp, builder.create_arith_SubFOp, common_support_dtypes)


@bind_tile_method(name="__mul__", binary_op="*")
@require_jit
@set_docstring("multiplication", common_support_dtypes)
def mul(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MulIOp, builder.create_arith_MulFOp, common_support_dtypes)


@bind_tile_method(name="__truediv__", binary_op="/")
@require_jit
@set_docstring("division", (KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))
def div(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_DivSIOp, builder.create_arith_DivFOp,
                          (KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))


@require_jit
@set_docstring("maximum", common_support_dtypes)
def maximum(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MaxSIOp, builder.create_arith_MaximumFOp,
                          common_support_dtypes)


@require_jit
@set_docstring("minimum", common_support_dtypes)
def minimum(input: Union[Tile, RuntimeNumeric], other: Union[Tile, RuntimeNumeric]) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_binary_impl(input, other, builder.create_arith_MinSIOp, builder.create_arith_MinimumFOp,
                          common_support_dtypes)


@bind_tile_method(name="__lshift__", binary_op="<<")
@require_jit
@set_docstring("left shift (bitwise)", (KT.int16, KT.int32, KT.int64), rhs_scalar_only=True)
def left_shift(input: Tile, other: RuntimeInt) -> Tile:
    check_type("input", input, Tile, BinaryOperandTypeError)
    check_runtime_int("other", other, BinaryOperandTypeError)
    check_dtype("input", input, (KT.int16, KT.int32, KT.int64))
    other = create_tile(other, input.dtype, input.shape)
    handle = global_builder.get_ir_builder().create_arith_ShLIOp(input.to_ir(), other.to_ir())
    return Tile(handle)


@bind_tile_method(name="__rshift__", binary_op=">>")
@require_jit
@set_docstring("right shift (bitwise)", (KT.int16, KT.int32, KT.int64), rhs_scalar_only=True)
def right_shift(input: Tile, other: RuntimeInt) -> Tile:
    check_type("input", input, Tile, BinaryOperandTypeError)
    check_runtime_int("other", other, BinaryOperandTypeError)
    check_dtype("input", input, (KT.int16, KT.int32, KT.int64))
    other = create_tile(other, input.dtype, input.shape)
    handle = global_builder.get_ir_builder().create_arith_ShRSIOp(input.to_ir(), other.to_ir())
    return Tile(handle)


def check_matmul_arguments(input: Tile, other: Tile, hf32: bool) -> None:
    if not isinstance(input, Tile) or not isinstance(other, Tile):
        raise BinaryOperandTypeError(f"Input operands must be tiles, got {type(input)} and {type(other)}")
    if input.dtype != other.dtype:
        raise RuntimeError(f"Input tiles must have the same types, got {input.dtype} and {other.dtype}")
    if input.dtype not in (KT.float32, KT.float16, KT.bfloat16):
        raise RuntimeError(f"Input tiles have unsupported types: {input.dtype}")
    if len(input.shape) != 2 or len(other.shape) != 2:
        raise RuntimeError(f"Input tiles must have two dims, got {len(input.shape)} and {len(other.shape)}")
    if input.shape[1] != other.shape[0]:
        raise RuntimeError(f"Input tiles have incompatible shapes: {input.shape}, {other.shape}")
    if hf32 and input.dtype != KT.float32:
        raise RuntimeError("HF32 mode can only be set when input tile dtype is float32")


def check_matmul_bias(bias: Optional[Tile], size: int) -> None:
    if bias is None:
        return
    check_type("bias", bias, Tile)
    check_dtype("bias", bias, KT.float32)
    if len(bias.shape) != 1:
        raise RuntimeError(f"Bias must be 1D tile, got shape {bias.shape}")
    if bias.shape[0] != size:
        raise RuntimeError(f"Bias shape {bias.shape[0]} must match last output dimension {size}")


@bind_tile_method(name="__matmul__", binary_op="@")
def matmul(input: Tile, other: Tile, bias: Optional[Tile] = None, *, hf32: bool = False) -> Tile:
    """
    Computes the matrix multiplication of :code:`input` and :code:`other` with optional :code:`bias`.

    Args:
        input: The left operand (2D tile)
        other: The right operand (2D tile)
        bias: Optional bias tile (1D tile of float32)
        hf32: Enable the rounding to HF32 for input tiles with float32 dtype

    Returns:
        Tile: The result of the matrix multiplication

    Raises:
        RuntimeError: If input tiles are not 2D, have incompatible shapes, or unsupported dtype

    Note:
        Input tiles must have either :code:`float16` or :code:`float32` data type and compatible shapes.
        Result tile type is always :code:`float32`.
        Bias must be a 1D tile of :code:`float32` with shape matching the last dimension of the output.
    """
    check_matmul_arguments(input, other, hf32)
    check_matmul_bias(bias, other.shape[1])
    builder = global_builder.get_ir_builder()
    ir_type = ir.get_asctile_TileType([input.shape[0], other.shape[1]], KT.float32.to_ir(), TileLocation.L0C)
    bias_ir = bias.to_ir() if bias is not None else None
    handle = builder.create_asctile_MatmulOp(ir_type, input.to_ir(), other.to_ir(), bias_ir, hf32)
    return Tile(handle)


def matmul_acc(input: Tile, other: Tile, acc: Tile, *, hf32: bool = False) -> None:
    """
    Computes the matrix multiplication of :code:`input` and :code:`other` with accumulator :code:`acc`.

    Args:
        input: The left operand (2D tile).
        other: The right operand (2D tile).
        acc: Accumulator of matmul result (2D tile). Must be created with :py:func:`asc2.zeros_acc`.
        hf32: Enable the rounding to HF32 for input tiles with float32 dtype.

    Raises:
        RuntimeError: If input tiles are not 2D, have incompatible shapes, unsupported dtype,
                      or accumulator has wrong shape/dtype

    Note:
        Input tiles must have either :code:`float16` or :code:`float32` data type and compatible shapes.
        Accumulator tile type is always :code:`float32`.
    """
    check_type("acc", acc, Tile)
    check_matmul_arguments(input, other, hf32)
    if len(acc.shape) != 2:
        raise RuntimeError(f"Accumulation tile must have two dims, got {len(acc.shape)}")
    if acc.dtype != KT.float32:
        raise RuntimeError(f"Accumulation tile have unsupported types: {acc.dtype}")
    global_builder.get_ir_builder().create_asctile_MatmulAccOp(input.to_ir(), other.to_ir(), acc.to_ir(), hf32)
