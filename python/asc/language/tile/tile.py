# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import functools
import inspect
from typing import Callable, Final, Optional, TypeVar, overload
from typing_extensions import Self

from ..._C import ir
from ..core.dtype import DataType
from ..core.tensor import TensorShape
from ..core.ir_value import IRHandle, IRValue, RuntimeInt
from ..core.utils import global_builder

T = TypeVar("T")


class Tile(IRValue):

    def __init__(self, handle: IRHandle) -> None:
        super().__init__()
        self.handle: Final = handle
        ir_type = handle.get_type()
        self.dtype: Final = DataType.from_ir(ir.get_element_type(ir_type))
        self.shape: Final = TensorShape(ir.get_shape(ir_type))
        if len(self.shape) < 1:
            raise RuntimeError("Tile shape must have at least one dimension")
        try:
            dtype_size = self.dtype.sizeof()
        except ValueError:  # sizeof might be not supported
            return
        if self.shape[-1] % (ir.ub_block_size // dtype_size) != 0:
            raise RuntimeError(f"Last dimension of tile must be aligned by {ir.ub_block_size} bytes, "
                               f"got {self.shape[-1]} x {dtype_size} bytes")

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    def to(self: Tile, dtype: DataType) -> Tile:
        if self.dtype == dtype:
            return self
        from_i = self.dtype.is_signed()
        from_f = self.dtype.is_float()
        to_i = dtype.is_signed()
        to_f = dtype.is_float()
        if (not from_f and not from_i) or (not to_f and not to_i):
            cast_supported = False
        elif self.dtype.bitwidth == dtype.bitwidth and (from_f and to_i or from_i and to_f):
            cast_supported = True
        elif self.dtype.bitwidth != dtype.bitwidth and (from_i and to_i) or (from_f and to_f):
            cast_supported = True
        if not cast_supported:
            raise RuntimeError(f"Cast from {self.dtype} to {dtype} is not supported")
        ir_type = ir.clone_shaped_type(self.to_ir().get_type(), dtype.to_ir())
        handle = global_builder.get_ir_builder().create_asctile_CastOp(ir_type, self.to_ir())
        return Tile(handle)

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        return self / other

    def __lshift__(self, other: RuntimeInt) -> Self:
        ...

    def __rshift__(self, other: RuntimeInt) -> Self:
        ...

    def __eq__(self, other: Self) -> Self:
        ...

    def __ne__(self, other: Self) -> Self:
        ...

    def __gt__(self, other: Self) -> Self:
        ...

    def __ge__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> Self:
        ...

    def __le__(self, other: Self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __pos__(self) -> Self:
        return self

    def sin(self) -> Self:
        ...

    def cos(self) -> Self:
        ...

    def tan(self) -> Self:
        ...

    def sinh(self) -> Self:
        ...

    def cosh(self) -> Self:
        ...

    def tanh(self) -> Self:
        ...

    def exp(self) -> Self:
        ...

    def exp2(self) -> Self:
        ...

    def log(self) -> Self:
        ...

    def log2(self) -> Self:
        ...

    def floor(self) -> Self:
        ...

    def ceil(self) -> Self:
        ...

    def abs(self) -> Self:
        ...

    def erf(self) -> Self:
        ...

    def sqrt(self) -> Self:
        ...

    def rsqrt(self) -> Self:
        ...


class BinaryOperandTypeError(TypeError):
    """Exception for dunder methods implementing binary operators"""
    pass


class Binder:

    def __init__(self, name: Optional[str] = None, binary_op: bool = False) -> None:
        self.name = name
        self.binary_op = binary_op

    def __call__(self, fn: T) -> T:
        name = self.name or fn.__name__
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) < 1:
            raise ValueError("Bound function must have at least one parameter")
        params[0] = params[0].replace(name="self")
        new_sig = sig.replace(parameters=params)

        if self.binary_op:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except BinaryOperandTypeError:
                    return NotImplemented

        else:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

        wrapper.__signature__ = new_sig
        setattr(Tile, name, wrapper)
        return fn


@overload
def bind_tile_method(fn: T) -> T:
    ...


@overload
def bind_tile_method(name: str, binary_op: bool = False) -> Callable[[T], T]:
    ...


def bind_tile_method(fn: Optional[T] = None, *, name: Optional[str] = None, binary_op: bool = False):
    binder = Binder(name, binary_op)
    if fn is None:
        return binder
    return binder(fn)
