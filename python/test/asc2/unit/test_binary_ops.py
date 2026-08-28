# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass, field
import operator
from typing import Callable, Optional, Tuple, Type

import asc2
import pytest

from .helpers import all_dtypes

compare_valid_dtypes = (asc2.int8, asc2.int16, asc2.int32, asc2.float16, asc2.bfloat16, asc2.float32)
bitwise_valid_dtypes = (asc2.int8, asc2.int16, asc2.int32, asc2.int64)
shift_valid_dtypes = (asc2.int16, asc2.int32, asc2.int64)


@dataclass(frozen=True)
class InvalidInputCase:
    lhs: object
    rhs: object
    expected_exception: Type[Exception]
    match: str
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class BinaryOpSpec:
    fn: Callable
    valid_dtypes: Tuple[asc2.DataType,
                        ...] = (asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.bfloat16, asc2.float32)
    supports_tile_tile: bool = True
    supports_tile_scalar: bool = True
    supports_scalar_tile: bool = True
    operator_fn: Optional[Callable] = None
    invalid_cases: Tuple[InvalidInputCase, ...] = field(default_factory=tuple)
    invalid_dtypes: Tuple[asc2.DataType, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "invalid_dtypes", tuple(d for d in all_dtypes if d not in self.valid_dtypes))

    def __str__(self):
        return self.fn.__name__


invalid_lhs_string = InvalidInputCase("string", 1, TypeError, "must be", "lhs_string")
invalid_rhs_string = InvalidInputCase(1, "string", TypeError, "must be", "rhs_string")
invalid_lhs_none = InvalidInputCase(None, 1, TypeError, "must be", "lhs_none")
invalid_rhs_none = InvalidInputCase(1, None, TypeError, "must be", "rhs_none")
invalid_lhs_list = InvalidInputCase([1, 2], 1, TypeError, "must be", "lhs_list")
invalid_rhs_list = InvalidInputCase(1, [1, 2], TypeError, "must be", "rhs_list")
invalid_both_scalars = InvalidInputCase(1, 2, TypeError, "must be", "both_scalars")

common_invalid_cases = (invalid_lhs_string, invalid_rhs_string, invalid_lhs_none, invalid_rhs_none, invalid_lhs_list,
                        invalid_rhs_list, invalid_both_scalars)
shift_invalid_cases = (invalid_lhs_string, invalid_lhs_none, invalid_lhs_list,
                       InvalidInputCase(None, 2.5, TypeError, "must be", "rhs_float"))

fn_overload = [False, True]

specs = (
    BinaryOpSpec(fn=asc2.add, operator_fn=operator.add, invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.sub, operator_fn=operator.sub, invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.mul, operator_fn=operator.mul, invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.div, valid_dtypes=(asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.float32),
                 operator_fn=operator.truediv, invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.maximum, operator_fn=None, invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.minimum, operator_fn=None, invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.equal, valid_dtypes=compare_valid_dtypes, operator_fn=operator.eq,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.not_equal, valid_dtypes=compare_valid_dtypes, operator_fn=operator.ne,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.greater, valid_dtypes=compare_valid_dtypes, operator_fn=operator.gt,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.greater_equal, valid_dtypes=compare_valid_dtypes, operator_fn=operator.ge,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.less, valid_dtypes=compare_valid_dtypes, operator_fn=operator.lt,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.less_equal, valid_dtypes=compare_valid_dtypes, operator_fn=operator.le,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.bitwise_and, valid_dtypes=bitwise_valid_dtypes, operator_fn=operator.and_,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.bitwise_or, valid_dtypes=bitwise_valid_dtypes, operator_fn=operator.or_,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.bitwise_xor, valid_dtypes=bitwise_valid_dtypes, operator_fn=operator.xor,
                 invalid_cases=common_invalid_cases),
    BinaryOpSpec(fn=asc2.left_shift, valid_dtypes=shift_valid_dtypes, supports_tile_tile=False,
                 supports_scalar_tile=False, operator_fn=operator.lshift, invalid_cases=shift_invalid_cases),
    BinaryOpSpec(fn=asc2.right_shift, valid_dtypes=shift_valid_dtypes, supports_tile_tile=False,
                 supports_scalar_tile=False, operator_fn=operator.rshift, invalid_cases=shift_invalid_cases),
)


@pytest.mark.parametrize("spec, dtype, use_overload", [(s, d, o)
                                                       for s in specs
                                                       for d in s.valid_dtypes
                                                       for o in fn_overload
                                                       if s.supports_tile_tile and (not o or s.operator_fn)])
def test_tile_tile(jit_test, mock_launch, zero_tile, spec: BinaryOpSpec, dtype, use_overload):

    @jit_test
    def kernel():
        x, y = zero_tile([32, 32], dtype, n=2)
        (spec.operator_fn if use_overload else spec.fn)(x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("spec, dtype, use_overload", [(s, d, o)
                                                       for s in specs
                                                       for d in s.valid_dtypes
                                                       for o in fn_overload
                                                       if s.supports_tile_scalar and (not o or s.operator_fn)])
def test_tile_scalar(jit_test, mock_launch, zero_tile, scalar_value, spec: BinaryOpSpec, dtype, use_overload):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], dtype)
        y = scalar_value(dtype)
        (spec.operator_fn if use_overload else spec.fn)(x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("spec, dtype, use_overload", [(s, d, o)
                                                       for s in specs
                                                       for d in s.valid_dtypes
                                                       for o in fn_overload
                                                       if s.supports_scalar_tile and (not o or s.operator_fn)])
def test_scalar_tile(jit_test, mock_launch, zero_tile, scalar_value, spec: BinaryOpSpec, dtype, use_overload):

    @jit_test
    def kernel():
        x = scalar_value(dtype)
        y = zero_tile([32, 32], dtype)
        (spec.operator_fn if use_overload else spec.fn)(x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("spec, dtype", [(s, d) for s in specs for d in s.invalid_dtypes])
def test_invalid_dtype(jit_test, zero_tile, spec: BinaryOpSpec, dtype):

    @jit_test
    def kernel():
        x, y = zero_tile([32, 32], dtype, n=2)
        spec.fn(x, y)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


@pytest.mark.parametrize("spec, case", [(s, c) for s in specs for c in s.invalid_cases])
def test_invalid_operand_type(jit_test, spec: BinaryOpSpec, case: InvalidInputCase):

    @jit_test
    def kernel():
        spec.fn(case.lhs, case.rhs)

    with pytest.raises(case.expected_exception, match=case.match):
        kernel[1]()
