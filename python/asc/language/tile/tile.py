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
from ..core.ir_value import IRHandle, IRValue

T = TypeVar("T")


class Tile(IRValue):

    def __init__(self, handle: IRHandle) -> None:
        super().__init__()
        self.handle: Final = handle
        ir_type = handle.get_type()
        self.dtype: Final = DataType.from_ir(ir.get_element_type(ir_type))
        self.shape: Final = TensorShape(ir.get_shape(ir_type))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __neg__(self, other: Self) -> Self:
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

    def __floor__(self) -> Self:
        ...

    def __ceil__(self) -> Self:
        ...

    def __abs__(self) -> Self:
        ...

    def erf(self) -> Self:
        ...

    def sqrt(self) -> Self:
        ...

    def rsqrt(self) -> Self:
        ...


class Binder:

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    def __call__(self, fn: T) -> T:
        name = self.name or fn.__name__
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) < 1:
            raise ValueError("Bound function must have at least one parameter")
        params[0] = params[0].replace(name="self")
        new_sig = sig.replace(parameters=params)

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
def bind_tile_method(name: str) -> Callable[[T], T]:
    ...


def bind_tile_method(fn: Optional[T] = None, *, name: Optional[str] = None):
    binder = Binder(name)
    if fn is None:
        return binder
    return binder(fn)
