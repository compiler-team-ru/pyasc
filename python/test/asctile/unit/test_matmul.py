# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import operator

import asctile
import pytest

from .helpers import all_dtypes

valid_dtypes = (asctile.float16, asctile.bfloat16, asctile.float32)
invalid_dtypes = tuple(d for d in all_dtypes if d not in valid_dtypes)


@pytest.mark.parametrize("dtype", valid_dtypes)
@pytest.mark.parametrize("fn", [asctile.matmul, operator.matmul])
def test_matmul(jit_test, mock_launch, zero_tile, fn, dtype):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], dtype, asctile.TensorLocation.L0A)
        y = zero_tile([128, 256], dtype, asctile.TensorLocation.L0B)
        fn(x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("dtype", valid_dtypes)
def test_matmul_with_bias(jit_test, mock_launch, zero_tile, dtype):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], dtype, asctile.TensorLocation.L0A)
        y = zero_tile([128, 256], dtype, asctile.TensorLocation.L0B)
        bias = zero_tile([256], dtype, asctile.TensorLocation.BT)
        asctile.matmul(x, y, bias)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_matmul_hf32(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], asctile.float32, asctile.TensorLocation.L0A)
        y = zero_tile([64, 64], asctile.float32, asctile.TensorLocation.L0B)
        asctile.matmul(x, y, hf32=True)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_matmul_acc(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        acc = zero_tile([64, 128], asctile.float32, asctile.TensorLocation.L0C)
        x = zero_tile([64, 128], asctile.float32, asctile.TensorLocation.L0A)
        y = zero_tile([128, 128], asctile.float32, asctile.TensorLocation.L0B)
        asctile.matmul_acc(acc, x, y)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("dtype", invalid_dtypes)
def test_invalid_dtype(jit_test, zero_tile, dtype):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], dtype, asctile.TensorLocation.L0A)
        y = zero_tile([128, 256], dtype, asctile.TensorLocation.L0B)
        asctile.matmul(x, y)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


@pytest.mark.parametrize("invalid_lhs", ["string", None, [1, 2]])
def test_invalid_lhs(jit_test, zero_tile, invalid_lhs):

    @jit_test
    def kernel():
        y = zero_tile([128, 256], asctile.float32, asctile.TensorLocation.L0B)
        asctile.matmul(invalid_lhs, y)

    with pytest.raises(TypeError, match="must be"):
        kernel[1]()


@pytest.mark.parametrize("invalid_rhs", ["string", None, [1, 2]])
def test_invalid_rhs(jit_test, zero_tile, invalid_rhs):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asctile.float32, asctile.TensorLocation.L0A)
        asctile.matmul(x, invalid_rhs)

    with pytest.raises(TypeError, match="must be"):
        kernel[1]()


@pytest.mark.parametrize("lhs_shape, rhs_shape, match", [
    ([64], [128, 256], "two dims"),
    ([64, 128], [64, 256], "incompatible shapes"),
])
def test_invalid_shape(jit_test, zero_tile, lhs_shape, rhs_shape, match):

    @jit_test
    def kernel():
        x = zero_tile(lhs_shape, asctile.float32, asctile.TensorLocation.L0A)
        y = zero_tile(rhs_shape, asctile.float32, asctile.TensorLocation.L0B)
        asctile.matmul(x, y)

    with pytest.raises(RuntimeError, match=match):
        kernel[1]()


def test_invalid_dtype_mismatch(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asctile.float16, asctile.TensorLocation.L0A)
        y = zero_tile([128, 256], asctile.float32, asctile.TensorLocation.L0B)
        asctile.matmul(x, y)

    with pytest.raises(RuntimeError, match="same types"):
        kernel[1]()


def test_invalid_hf32_non_float32(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([64, 128], asctile.float16, asctile.TensorLocation.L0A)
        y = zero_tile([128, 256], asctile.float16, asctile.TensorLocation.L0B)
        asctile.matmul(x, y, hf32=True)

    with pytest.raises(RuntimeError, match="HF32.*float32"):
        kernel[1]()
