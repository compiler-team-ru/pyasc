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

from .helpers import all_dtypes, all_locations


@dataclass(frozen=True)
class InvalidInputCase:
    input: object
    expected_exception: Type[Exception]
    match: str
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class UnaryOpSpec:
    fn: Callable
    valid_dtypes: Tuple[asc2.DataType, ...] = (asc2.float16, asc2.float32)
    supports_scalar: bool = False
    supported_locations: Tuple[asc2.TensorLocation, ...] = (asc2.TensorLocation.UB, )
    operator_fn: Optional[Callable] = None
    invalid_cases: Tuple[InvalidInputCase, ...] = field(default_factory=tuple)
    invalid_dtypes: Tuple[asc2.DataType, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "invalid_dtypes", tuple(d for d in all_dtypes if d not in self.valid_dtypes))

    def __str__(self):
        return self.fn.__name__


invalid_string = InvalidInputCase("string", TypeError, "must be", "string")
invalid_none = InvalidInputCase(None, TypeError, "must be", "none")
invalid_list = InvalidInputCase([1, 2], TypeError, "must be", "list")
common_invalid_cases = (invalid_string, invalid_none, invalid_list)

fn_overload = [False, True]

specs = (
    UnaryOpSpec(fn=asc2.cos, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.sin, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.tan, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.sinh, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.cosh, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.tanh, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.exp, supports_scalar=True, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.log, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.log2, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.floor, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.ceil, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.abs, valid_dtypes=(asc2.int8, asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.float32),
                invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.erf, supports_scalar=True, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.exp2, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.rsqrt, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.sqrt, supports_scalar=True, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.relu, supported_locations=(asc2.TensorLocation.UB, asc2.TensorLocation.L0C),
                invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.negative,
                valid_dtypes=(asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.bfloat16, asc2.float32),
                operator_fn=operator.neg),
    UnaryOpSpec(fn=asc2.bitwise_not, valid_dtypes=(asc2.int8, asc2.int16, asc2.int32, asc2.int64),
                operator_fn=operator.invert, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asc2.softmax, invalid_cases=common_invalid_cases),
)


@pytest.mark.parametrize(
    "spec, dtype, use_overload",
    [(s, d, o) for s in specs for d in s.valid_dtypes for o in fn_overload if not o or s.operator_fn])
def test_tile(jit_test, mock_launch, zero_tile, spec: UnaryOpSpec, dtype, use_overload):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], dtype)
        (spec.operator_fn if use_overload else spec.fn)(x)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("spec, dtype", [(s, d) for s in specs for d in s.valid_dtypes if s.supports_scalar])
def test_scalar(jit_test, mock_launch, scalar_value, spec: UnaryOpSpec, dtype):

    @jit_test
    def kernel():
        x = scalar_value(dtype)
        spec.fn(x)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("spec, dtype", [(s, d) for s in specs for d in s.invalid_dtypes])
def test_invalid_dtype(jit_test, zero_tile, spec: UnaryOpSpec, dtype):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], dtype)
        spec.fn(x)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


@pytest.mark.parametrize("spec, case", [(s, c) for s in specs for c in s.invalid_cases])
def test_invalid_operand_type(jit_test, spec: UnaryOpSpec, case: InvalidInputCase):

    @jit_test
    def kernel():
        spec.fn(case.input)

    with pytest.raises(case.expected_exception, match=case.match):
        kernel[1]()


@pytest.mark.parametrize("spec, loc",
                         [(s, loc) for s in specs for loc in all_locations if loc not in s.supported_locations])
def test_invalid_location(jit_test, zero_tile, spec: UnaryOpSpec, loc):
    dtype = spec.valid_dtypes[0]

    @jit_test
    def kernel():
        x = zero_tile([32, 32], dtype, loc)
        spec.fn(x)

    with pytest.raises(RuntimeError, match="location"):
        kernel[1]()
