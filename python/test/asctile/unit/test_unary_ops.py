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

import asctile
import pytest

from .helpers import all_dtypes


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
    valid_dtypes: Tuple[asctile.DataType, ...] = (asctile.float16, asctile.float32)
    supports_scalar: bool = False
    operator_fn: Optional[Callable] = None
    invalid_cases: Tuple[InvalidInputCase, ...] = field(default_factory=tuple)
    invalid_dtypes: Tuple[asctile.DataType, ...] = field(init=False)

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
    UnaryOpSpec(fn=asctile.cos, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.sin, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.tan, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.sinh, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.cosh, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.tanh, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.exp, supports_scalar=True, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.log, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.log2, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.floor, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.ceil, invalid_cases=common_invalid_cases),
    UnaryOpSpec(
        fn=asctile.abs,
        valid_dtypes=(asctile.int8, asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.float32),
        invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.erf, supports_scalar=True, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.exp2, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.rsqrt, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.sqrt, supports_scalar=True, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.relu, invalid_cases=common_invalid_cases),
    UnaryOpSpec(
        fn=asctile.negative,
        valid_dtypes=(asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.bfloat16, asctile.float32),
        operator_fn=operator.neg),
    UnaryOpSpec(fn=asctile.bitwise_not, valid_dtypes=(asctile.int8, asctile.int16, asctile.int32, asctile.int64),
                operator_fn=operator.invert, invalid_cases=common_invalid_cases),
    UnaryOpSpec(fn=asctile.softmax, invalid_cases=common_invalid_cases),
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


def test_softmax_3d_error(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([2, 3, 4], asctile.float32)
        asctile.softmax(x)

    with pytest.raises(RuntimeError, match="dimensionality"):
        kernel[1]()


def test_rms_norm(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 128], asctile.float32)
        gamma = zero_tile([128], asctile.float32)
        asctile.rms_norm(x, gamma, 1e-6)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_rms_norm_3d_error(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([2, 3, 4], asctile.float32)
        gamma = zero_tile([4], asctile.float32)
        asctile.rms_norm(x, gamma, 1e-6)

    with pytest.raises(RuntimeError, match="dimensionality"):
        kernel[1]()


def test_layer_norm(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 128], asctile.float32)
        gamma = zero_tile([128], asctile.float32)
        beta = zero_tile([128], asctile.float32)
        asctile.layer_norm(x, gamma, beta, 1e-6)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_layer_norm_3d_error(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([2, 3, 4], asctile.float32)
        gamma = zero_tile([4], asctile.float32)
        beta = zero_tile([4], asctile.float32)
        asctile.layer_norm(x, gamma, beta, 1e-6)

    with pytest.raises(RuntimeError, match="dimensionality"):
        kernel[1]()
