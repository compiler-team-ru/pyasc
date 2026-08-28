# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import pytest

from .helpers import all_dtypes, non_ub_locations

valid_dtypes = (asc2.int8, asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.bfloat16, asc2.float32)
invalid_dtypes = tuple(d for d in all_dtypes if d not in valid_dtypes)


@pytest.mark.parametrize("dtype", valid_dtypes)
def test_full(jit_test, mock_launch, dtype):

    @jit_test
    def kernel():
        value = 2.0 if dtype.is_float() else 2
        asc2.full([32, 32], value, dtype)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_full_infer_int_dtype(jit_test, mock_launch):

    @jit_test
    def kernel():
        asc2.full([32], 42)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_full_infer_float_dtype(jit_test, mock_launch):

    @jit_test
    def kernel():
        asc2.full([32], 3.14)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("dtype", valid_dtypes)
def test_zeros(jit_test, mock_launch, dtype):

    @jit_test
    def kernel():
        asc2.zeros([32, 32], dtype)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("dtype", valid_dtypes)
def test_full_like(jit_test, mock_launch, zero_tile, dtype):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], dtype)
        value = 2.0 if dtype.is_float() else 2
        asc2.full_like(x, value)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("dtype", valid_dtypes)
def test_zeros_like(jit_test, mock_launch, zero_tile, dtype):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], dtype)
        asc2.zeros_like(x)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_zeros_acc(jit_test, mock_launch):

    @jit_test
    def kernel():
        asc2.zeros_acc([64, 256], asc2.float32)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_zeros_acc_with_bias(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        bias = zero_tile([256], asc2.float16, asc2.TensorLocation.BT)
        asc2.zeros_acc([64, 256], asc2.float32, bias=bias)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("src_dtype", valid_dtypes)
@pytest.mark.parametrize("dst_dtype", valid_dtypes)
def test_cast(jit_test, mock_launch, zero_tile, src_dtype, dst_dtype):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], src_dtype)
        asc2.cast(x, dst_dtype)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_cast_scalar(jit_test, mock_launch):

    @jit_test
    def kernel():
        asc2.cast(2.0, asc2.float16)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_cast_scalar_with_round_mode(jit_test):

    @jit_test
    def kernel():
        asc2.cast(2.0, asc2.float16, round_mode=asc2.RoundMode.Floor)

    with pytest.raises(RuntimeError, match="round_mode"):
        kernel[1]()


@pytest.mark.parametrize("dtype", valid_dtypes)
def test_concat(jit_test, mock_launch, zero_tile, dtype):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], dtype)
        y = zero_tile([64, 32], dtype)
        asc2.concat(x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_full_invalid_value(jit_test):

    @jit_test
    def kernel():
        asc2.full([32, 32], "invalid", asc2.float32)

    with pytest.raises(TypeError, match="value"):
        kernel[1]()


@pytest.mark.parametrize("dtype", invalid_dtypes)
def test_full_invalid_dtype(jit_test, dtype):

    @jit_test
    def kernel():
        asc2.full([32, 32], 2.0, dtype)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


@pytest.mark.parametrize("loc", non_ub_locations)
def test_full_invalid_location(jit_test, loc):

    @jit_test
    def kernel():
        asc2.full([32, 32], 2.0, asc2.float32, loc)

    with pytest.raises(RuntimeError, match="location"):
        kernel[1]()


def test_zeros_acc_invalid_dtype(jit_test):

    @jit_test
    def kernel():
        asc2.zeros_acc([64, 256], asc2.float16)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


def test_cast_invalid_input(jit_test):

    @jit_test
    def kernel():
        asc2.cast("invalid", asc2.float32)

    with pytest.raises(TypeError, match="input"):
        kernel[1]()


def test_cast_invalid_dtype_arg(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], asc2.float32)
        asc2.cast(x, "invalid")

    with pytest.raises(TypeError, match="dtype"):
        kernel[1]()


def test_concat_invalid_input(jit_test):

    @jit_test
    def kernel():
        asc2.concat("invalid")

    with pytest.raises(TypeError, match="tensors"):
        kernel[1]()


def test_concat_dtype_mismatch(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], asc2.float32)
        y = zero_tile([64, 32], asc2.float16)
        asc2.concat(x, y)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


def test_concat_shape_mismatch(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 32], asc2.float32)
        y = zero_tile([64, 64], asc2.float32)
        asc2.concat(x, y)

    with pytest.raises(RuntimeError, match="shape"):
        kernel[1]()
