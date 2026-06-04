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
import math
from typing import Callable, Final, Optional, Tuple, TypeVar, Union, overload
from typing_extensions import Self, TypeAlias

from ..._C import ir
from ..core.dtype import DataType
from ..core.ir_value import IRHandle, IRValue, PlainValue, RuntimeInt, RuntimeNumeric
from .validation import check_type

T = TypeVar("T")

TileLocation: TypeAlias = ir.TileLocation


class Tile(IRValue):
    """
    A tile is a multi-dimensional array of values in local memory (Unified Buffer, L1 Cache, etc.)

    Each element is of :py:attr:`dtype` type and number of elements is defined by :py:attr:`shape` tuple.
    """

    dtype: DataType
    """Tile element type"""

    shape: Tuple[int]
    """Tile shape"""

    size: int
    """Number of elements"""

    def __init__(self, handle: IRHandle) -> None:
        """
        This constructor is not called by user.

        Use :py:func:`load`, :py:func:`zeros`, or other functions to create a tile.
        """
        super().__init__()
        check_type("handle", handle, IRHandle)
        self.handle: Final = handle
        ir_type = handle.get_type()
        self.dtype: Final = DataType.from_ir(ir.get_element_type(ir_type))
        self.shape: Final = tuple(ir.get_shape(ir_type))
        if len(self.shape) < 1:
            raise RuntimeError("Tile shape must have at least one dimension")
        self.size: Final = math.prod(self.shape)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    def to(self, dtype: DataType) -> Self:
        """Forwards to :py:func:`cast` function."""
        from .creation_ops import cast
        return cast(self, dtype)

    @property
    def T(self) -> Self:
        """Transpose a 2D tile by swapping its dimensions (see :py:func:`transpose`)."""
        from .shape_ops import transpose
        return transpose(self)

    # Binary operations

    def __add__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __sub__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __mul__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __truediv__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __floordiv__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        return self / other

    def __lshift__(self, other: RuntimeInt) -> Self:
        ...

    def __rshift__(self, other: RuntimeInt) -> Self:
        ...

    def __matmul__(self, other: Self) -> Self:
        ...

    def __eq__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __ne__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __gt__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __ge__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __lt__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __le__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    # Reduction operations

    @overload
    def sum(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    @overload
    def sum(self) -> PlainValue:
        ...

    @overload
    def max(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    @overload
    def max(self) -> PlainValue:
        ...

    @overload
    def min(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    @overload
    def min(self) -> PlainValue:
        ...

    def prod(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    # Unary operations

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

    def exp2(self) -> Self:
        ...

    def sqrt(self) -> Self:
        ...

    def rsqrt(self) -> Self:
        ...

    def relu(self) -> Self:
        ...

    # Shape operations

    def broadcast_to(self, *shape: int) -> Self:
        ...

    def reshape(self, *shape: int) -> Self:
        ...

    def ravel(self) -> Self:
        ...

    def expand_dims(self, *axis: int) -> Self:
        ...

    def squeeze(self, *axis: int) -> Self:
        ...

    def transpose(self) -> Self:
        ...


class BinaryOperandTypeError(TypeError):
    """Exception for dunder methods implementing binary operators"""
    pass


class Binder:

    def __init__(self, name: Optional[str] = None, binary_op: Optional[str] = None, unary_op: Optional[str] = None):
        self.name = name
        self.binary_op = binary_op
        self.unary_op = unary_op

    def __call__(self, fn: T) -> T:
        fn_name = fn.__name__
        name = self.name or fn_name
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) < 1:
            raise ValueError("Bound function must have at least one parameter")
        if not fn.__doc__:
            fn.__doc__ = ""
        if self.binary_op:
            call_kind = "via a binary operator"
            call_func = f"{fn_name}(input, other)"
            call_alias = f"input {self.binary_op} other"
        elif self.unary_op:
            call_kind = "via an unary operator"
            call_func = f"{fn_name}(input)"
            call_alias = f"{self.unary_op}input"
        else:
            call_kind = "as a member function"
            call_func = f"{fn_name}(input, ...)"
            call_alias = f"input.{name}(...)"
        fn.__doc__ += f"""
    This function can also be called {call_kind} on :py:class:`Tile`,
    as :code:`{call_alias}` instead of :code:`{call_func}`.
        """
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
        wrapper.__doc__ = wrapper.__doc__ = f"Forwards to :py:func:`{fn_name}` function."
        setattr(Tile, name, wrapper)
        return fn


@overload
def bind_tile_method(fn: T) -> T:
    ...


@overload
def bind_tile_method(name: str, binary_op: Optional[str] = None, unary_op: Optional[str] = None) -> Callable[[T], T]:
    ...


def bind_tile_method(fn: Optional[T] = None, *, name: Optional[str] = None, binary_op: Optional[str] = None,
                     unary_op: Optional[str] = None):
    binder = Binder(name, binary_op, unary_op)
    if fn is None:
        return binder
    return binder(fn)
