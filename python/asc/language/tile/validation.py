# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Iterable, Protocol, Tuple, Type, TypeGuard, Union

from ..._C import ir
from ...common.compat import isinstance
from ..core.dtype import DataType
from ..core.ir_value import PlainValue, RuntimeInt
from ..core.utils import get_type_name


class DataTyped(Protocol):
    dtype: DataType


def check_data_alignment(shape: Tuple[int, ...], dtype: DataType) -> None:
    try:
        dtype_size = dtype.sizeof()
    except ValueError:  # sizeof might be not supported
        return
    if len(shape) > 1 and shape[-1] % (ir.ub_block_size // dtype_size) != 0:
        raise RuntimeError(f"Last dimension of tile must be aligned by {ir.ub_block_size} bytes, "
                           f"got {shape[-1]} x {dtype_size} bytes")


def check_type(name: str, value: Any, constraint: Union[Type, Tuple[Type, ...]],
               exc_type: Type[Exception] = TypeError) -> None:
    if isinstance(value, constraint):
        return
    raise exc_type(f"'{name}' argument must be {get_type_name(constraint)}, got {value.__class__.__name__}")


def check_dtype(name: str, value: DataTyped, dtypes: Tuple[DataType]) -> None:
    if value.dtype not in dtypes:
        dtypes_str = ", ".join(map(str, dtypes))
        raise RuntimeError(f"'{name}' dtype must be one of {dtypes_str}, got {value.dtype}")


def is_runtime_int(value: Any) -> TypeGuard[RuntimeInt]:
    return isinstance(value, int) or isinstance(value, PlainValue) and value.dtype.is_signed()


def check_runtime_int(name: str, value: Any, exc_type: Type[Exception] = TypeError) -> None:
    if not is_runtime_int(value):
        raise exc_type(f"'{name}' argument must be int or integer PlainValue, got {value.__class__.__name__}")


def check_runtime_float(name: str, value: Any, exc_type: Type[Exception] = TypeError) -> None:
    if isinstance(value, float) or isinstance(value, PlainValue) and value.dtype.is_float():
        return
    raise exc_type(f"'{name}' argument must be float or PlainValue with float dtype, got {value.__class__.__name__}")


def iterable_to_non_empty_tuple(iterable: Any, name: str) -> tuple:
    if not isinstance(iterable, Iterable):
        raise TypeError(f"'{name}' must be Iterable")
    values = iterable if isinstance(iterable, tuple) else tuple(iterable)
    if len(values) < 1:
        raise RuntimeError(f"'{name}' must have at least one value")
    return values


def verify_runtime_ints(values: Iterable[Any], name: str) -> Tuple[RuntimeInt, ...]:
    values = iterable_to_non_empty_tuple(values, name)
    if not all(is_runtime_int(value) for value in values):
        raise TypeError(f"All values in '{name}' must be int or integer PlainValue")
    return values


def verify_shape(shape: Iterable[int], name: str = "shape") -> Tuple[int, ...]:
    shape = iterable_to_non_empty_tuple(shape, name)
    if not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"All values in '{name}' must be integers")
    if any(dim <= 0 for dim in shape):
        raise RuntimeError(f"All values in '{name}' must be positive")
    return shape
