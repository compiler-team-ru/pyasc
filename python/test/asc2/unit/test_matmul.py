# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import operator

import asc2
import pytest

from .helpers import all_dtypes, all_locations

valid_dtypes = (asc2.float16, asc2.bfloat16, asc2.float32)
invalid_dtypes = tuple(d for d in all_dtypes if d not in valid_dtypes)


@pytest.mark.parametrize("dtype", valid_dtypes)
@pytest.mark.parametrize("fn", [asc2.matmul, operator.matmul])
def test_matmul(jit_test, mock_launch, zero_tile, fn, dtype):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], dtype, asc2.TensorLocation.L0A)
        y = zero_tile([128, 256], dtype, asc2.TensorLocation.L0B)
        fn(x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("dtype", valid_dtypes)
def test_matmul_with_bias(jit_test, mock_launch, zero_tile, dtype):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], dtype, asc2.TensorLocation.L0A)
        y = zero_tile([128, 256], dtype, asc2.TensorLocation.L0B)
        bias = zero_tile([256], dtype, asc2.TensorLocation.BT)
        asc2.matmul(x, y, bias)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_matmul_hf32(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], asc2.float32, asc2.TensorLocation.L0A)
        y = zero_tile([64, 64], asc2.float32, asc2.TensorLocation.L0B)
        asc2.matmul(x, y, hf32=True)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_matmul_acc(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        acc = zero_tile([64, 128], asc2.float32, asc2.TensorLocation.L0C)
        x = zero_tile([64, 128], asc2.float32, asc2.TensorLocation.L0A)
        y = zero_tile([128, 128], asc2.float32, asc2.TensorLocation.L0B)
        asc2.matmul_acc(acc, x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("dtype", invalid_dtypes)
def test_invalid_dtype(jit_test, zero_tile, dtype):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], dtype, asc2.TensorLocation.L0A)
        y = zero_tile([128, 256], dtype, asc2.TensorLocation.L0B)
        asc2.matmul(x, y)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


@pytest.mark.parametrize("invalid_lhs", ["string", None, [1, 2]])
def test_invalid_lhs(jit_test, zero_tile, invalid_lhs):

    @jit_test
    def kernel():
        y = zero_tile([128, 256], asc2.float32, asc2.TensorLocation.L0B)
        asc2.matmul(invalid_lhs, y)

    with pytest.raises(TypeError, match="must be"):
        kernel[1]()


@pytest.mark.parametrize("invalid_rhs", ["string", None, [1, 2]])
def test_invalid_rhs(jit_test, zero_tile, invalid_rhs):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asc2.float32, asc2.TensorLocation.L0A)
        asc2.matmul(x, invalid_rhs)

    with pytest.raises(TypeError, match="must be"):
        kernel[1]()


@pytest.mark.parametrize("lhs_shape, rhs_shape, match", [
    ([64], [128, 256], "two dims"),
    ([64, 128], [64, 256], "incompatible shapes"),
])
def test_invalid_shape(jit_test, zero_tile, lhs_shape, rhs_shape, match):

    @jit_test
    def kernel():
        x = zero_tile(lhs_shape, asc2.float32, asc2.TensorLocation.L0A)
        y = zero_tile(rhs_shape, asc2.float32, asc2.TensorLocation.L0B)
        asc2.matmul(x, y)

    with pytest.raises(RuntimeError, match=match):
        kernel[1]()


def test_invalid_dtype_mismatch(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asc2.float16, asc2.TensorLocation.L0A)
        y = zero_tile([128, 256], asc2.float32, asc2.TensorLocation.L0B)
        asc2.matmul(x, y)

    with pytest.raises(RuntimeError, match="same types"):
        kernel[1]()


def test_invalid_hf32_non_float32(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asc2.float16, asc2.TensorLocation.L0A)
        y = zero_tile([128, 256], asc2.float16, asc2.TensorLocation.L0B)
        asc2.matmul(x, y, hf32=True)

    with pytest.raises(RuntimeError, match="HF32.*float32"):
        kernel[1]()


@pytest.mark.parametrize("loc", [loc for loc in all_locations if loc != asc2.TensorLocation.L0A])
def test_invalid_location_lhs(jit_test, zero_tile, loc):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asc2.float32, loc)
        y = zero_tile([128, 256], asc2.float32, asc2.TensorLocation.L0B)
        asc2.matmul(x, y)

    with pytest.raises(RuntimeError, match="location"):
        kernel[1]()


@pytest.mark.parametrize("loc", [loc for loc in all_locations if loc != asc2.TensorLocation.L0B])
def test_invalid_location_rhs(jit_test, zero_tile, loc):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asc2.float32, asc2.TensorLocation.L0A)
        y = zero_tile([128, 256], asc2.float32, loc)
        asc2.matmul(x, y)

    with pytest.raises(RuntimeError, match="location"):
        kernel[1]()
