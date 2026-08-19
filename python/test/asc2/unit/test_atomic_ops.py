# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
from asc.runtime.jit import MockTensor
import pytest

from .helpers import all_dtypes

valid_dtypes = (asc2.int16, asc2.int32, asc2.float16, asc2.bfloat16, asc2.float32)
invalid_dtypes = tuple(d for d in all_dtypes if d not in valid_dtypes)


@pytest.mark.parametrize("fn", [asc2.atomic_add, asc2.atomic_max, asc2.atomic_min])
@pytest.mark.parametrize("dtype", valid_dtypes)
def test_atomic(jit_test, mock_launch, fn, dtype):

    @jit_test
    def kernel(out_ptr: asc2.GlobalAddress):
        out_gm = asc2.global_tensor(out_ptr, [128])
        src = asc2.copy_in(out_gm, [0], [128])
        fn(src, out_gm, [0])

    kernel[1](MockTensor(dtype))
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("fn", [asc2.atomic_add, asc2.atomic_max, asc2.atomic_min])
@pytest.mark.parametrize("dtype", invalid_dtypes)
def test_atomic_invalid_dtype(jit_test, fn, dtype):

    @jit_test
    def kernel(out_ptr: asc2.GlobalAddress):
        out_gm = asc2.global_tensor(out_ptr, [128])
        src = asc2.copy_in(out_gm, [0], [128])
        fn(src, out_gm, [0])

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1](MockTensor(dtype))


@pytest.mark.parametrize("fn", [asc2.atomic_add, asc2.atomic_max, asc2.atomic_min])
def test_atomic_invalid_src_location(jit_test, zero_tile, fn):

    @jit_test
    def kernel(out_ptr: asc2.GlobalAddress):
        out_gm = asc2.global_tensor(out_ptr, [128])
        src = zero_tile([128], asc2.float32, asc2.TensorLocation.L0C)
        fn(src, out_gm, [0])

    with pytest.raises(RuntimeError, match="location"):
        kernel[1](MockTensor(asc2.float32))


@pytest.mark.parametrize("fn", [asc2.atomic_add, asc2.atomic_max, asc2.atomic_min])
def test_atomic_dtype_mismatch(jit_test, zero_tile, fn):

    @jit_test
    def kernel(out_ptr: asc2.GlobalAddress):
        out_gm = asc2.global_tensor(out_ptr, [128])
        src = zero_tile([128], asc2.float16)
        fn(src, out_gm, [0])

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1](MockTensor(asc2.float32))


def test_atomic_invalid_src_type(jit_test):

    @jit_test
    def kernel(out_ptr: asc2.GlobalAddress):
        out_gm = asc2.global_tensor(out_ptr, [128])
        asc2.atomic_add("invalid", out_gm, [0])

    with pytest.raises(TypeError, match="must be"):
        kernel[1](MockTensor(asc2.float32))


def test_atomic_invalid_dst_type(jit_test, zero_tile):

    @jit_test
    def kernel(out_ptr: asc2.GlobalAddress):
        src = zero_tile([128], asc2.float32)
        asc2.atomic_add(src, "invalid", [0])

    with pytest.raises(TypeError, match="must be"):
        kernel[1](MockTensor(asc2.float32))
