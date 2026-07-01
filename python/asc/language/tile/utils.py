# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real
from typing import Optional, Tuple, Union

from ..._C import ir
from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import PlainValue, RuntimeNumeric
from ..core.tensor import TensorShape
from ..core.utils import global_builder
from .local_tensor import BinaryOperandTypeError, LocalTensor, TensorLocation
from .validation import check_dtype, check_type


def constant_tile(value: Real, shape: TensorShape, dtype: DataType,
                  loc: TensorLocation = TensorLocation.UB) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    attr_builders = {
        "int8": builder.get_i8_attr,
        "int16": builder.get_i16_attr,
        "int32": builder.get_i32_attr,
        "int64": builder.get_i64_attr,
        "float16": builder.get_f16_attr,
        "bfloat16": builder.get_bf16_attr,
        "float32": builder.get_f32_attr,
        "float64": builder.get_f64_attr,
    }
    attr_builder = attr_builders.get(str(dtype))
    if attr_builder is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ir_type = ir.get_asctile_LocalTensorType(shape, dtype.to_ir(), loc)
    splat_attr = ir.get_splat_attr(ir_type, attr_builder(value))
    handle = builder.create_arith_ConstantOp(splat_attr)
    return LocalTensor.from_ir(handle)


def splat_tile(value: PlainValue, shape: TensorShape, dtype: DataType,
               loc: TensorLocation = TensorLocation.UB) -> LocalTensor:
    ir_type = ir.get_asctile_LocalTensorType(shape, dtype.to_ir(), loc)
    handle = global_builder.get_ir_builder().create_asctile_SplatOp(ir_type, value.cast(dtype).to_ir())
    return LocalTensor.from_ir(handle)


def create_tile(value: Union[LocalTensor, RuntimeNumeric], dtype: DataType, shape: Tuple[int, ...]) -> LocalTensor:
    if isinstance(value, LocalTensor):
        return value.to(dtype).broadcast_to(*shape)
    if isinstance(value, Real):
        return constant_tile(value, shape, dtype)
    if isinstance(value, PlainValue):
        return splat_tile(value, shape, dtype)
    raise BinaryOperandTypeError(f"LocalTensor cannot be created from {value.__class__.__name__}")


def infer_tile_dtype(value: Union[LocalTensor, PlainValue, Real]) -> DataType:
    if isinstance(value, (LocalTensor, PlainValue)):
        return value.dtype
    if isinstance(value, bool):
        return KT.int1
    if isinstance(value, int):
        return KT.int32
    if isinstance(value, float):
        return KT.float32
    raise BinaryOperandTypeError(f"Unable to obtain dtype of {value.__class__.__name__}")


def infer_common_dtype(lhs: Union[LocalTensor, RuntimeNumeric], rhs: Union[LocalTensor, RuntimeNumeric]) -> DataType:
    lhs_dtype = infer_tile_dtype(lhs)
    rhs_dtype = infer_tile_dtype(rhs)
    if lhs_dtype == rhs_dtype:
        return lhs_dtype
    if not lhs_dtype.is_numeric() or not rhs_dtype.is_numeric():
        raise RuntimeError(f"Operand dtypes must be numeric, got {lhs_dtype} and {rhs_dtype}")
    if lhs_dtype.is_unsigned() or rhs_dtype.is_unsigned():
        raise NotImplementedError(f"Unsigned dtype operands not supported, got {lhs_dtype} and {rhs_dtype}")
    lhs_is_tile = isinstance(lhs, LocalTensor)
    rhs_is_tile = isinstance(rhs, LocalTensor)
    if lhs_is_tile and not rhs_is_tile:
        return lhs.dtype
    if rhs_is_tile and not lhs_is_tile:
        return rhs.dtype
    if lhs_dtype.is_signed() and rhs_dtype.is_signed() and lhs_dtype.bitwidth != rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.bitwidth > rhs_dtype.bitwidth else rhs_dtype
    if lhs_dtype.is_float() and rhs_dtype.is_float() and lhs_dtype.bitwidth != rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.bitwidth > rhs_dtype.bitwidth else rhs_dtype
    if lhs_dtype.bitwidth == rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.is_float() else rhs_dtype
    raise RuntimeError(f"Unable to infer common dtype between {lhs_dtype} and {rhs_dtype}")


def infer_common_shape(lhs: Union[LocalTensor, RuntimeNumeric], rhs: Union[LocalTensor,
                                                                           RuntimeNumeric]) -> Tuple[int, ...]:
    lhs_is_tile = isinstance(lhs, LocalTensor)
    rhs_is_tile = isinstance(rhs, LocalTensor)
    if not lhs_is_tile and not rhs_is_tile:
        raise TypeError("At least one operand must be a LocalTensor")
    if not lhs_is_tile:
        return rhs.shape
    if not rhs_is_tile:
        return lhs.shape
    if lhs.shape == rhs.shape:
        return lhs.shape
    rank = max(len(lhs.shape), len(rhs.shape))
    lhs_padded = (1, ) * (rank - len(lhs.shape)) + lhs.shape
    rhs_padded = (1, ) * (rank - len(rhs.shape)) + rhs.shape
    result = []
    for dim_lhs, dim_rhs in zip(lhs_padded, rhs_padded):
        if dim_lhs != dim_rhs and dim_lhs != 1 and dim_rhs != 1:
            raise RuntimeError(f"Shapes are not broadcastable: {lhs.shape} vs. {rhs.shape}")
        result.append(max(dim_lhs, dim_rhs))
    return tuple(result)


def check_bias(bias: Optional[LocalTensor], size: int) -> None:
    if bias is None:
        return
    check_type("bias", bias, LocalTensor)
    check_dtype("bias", bias, (KT.bfloat16, KT.float16, KT.float32), optional=True)
    if len(bias.shape) != 1:
        raise RuntimeError(f"Bias must be 1D tensor, got shape {bias.shape}")
    if bias.size != size:
        raise RuntimeError(f"Bias shape {bias.size} must match last output dimension {size}")
