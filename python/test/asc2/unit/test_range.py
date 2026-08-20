# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import pytest
from asc.runtime.jit import MockValue


class TestRange:

    def test_single_arg(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.range(10):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_start_stop(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.range(2, 10):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_start_stop_step(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.range(0, 10, 2):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_unroll_factor(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.range(10, unroll_factor=4):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_gm_barrier(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.range(10, gm_barrier=True):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_builtin_range_with_params(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in range(0, 10, 1, unroll_factor=2, gm_barrier=True):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_runtime_bound(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel(n: int):
            for _ in asc2.range(n):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1](MockValue(asc2.int32))
        assert mock_launch.call_count == 1

    def test_invalid_unroll_factor_type(self, jit_test):

        @jit_test
        def kernel():
            for _ in asc2.range(10, unroll_factor="invalid"):
                pass

        with pytest.raises(TypeError, match="unroll_factor"):
            kernel[1]()

    def test_invalid_gm_barrier_type(self, jit_test):

        @jit_test
        def kernel():
            for _ in asc2.range(10, gm_barrier="invalid"):
                pass

        with pytest.raises(TypeError, match="gm_barrier"):
            kernel[1]()

    def test_invalid_unroll_factor_zero(self, jit_test):

        @jit_test
        def kernel():
            for _ in asc2.range(10, unroll_factor=0):
                pass

        with pytest.raises(ValueError, match="1 or greater"):
            kernel[1]()


class TestStaticRange:

    def test_single_arg(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.static_range(10):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_start_stop(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.static_range(2, 10):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_start_stop_step(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            for _ in asc2.static_range(0, 10, 2):
                x = zero_tile([128], asc2.float32)
                x + 1.0

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_runtime_bound_fails(self, jit_test):

        @jit_test
        def kernel(n: int):
            for _ in asc2.static_range(n):
                pass

        with pytest.raises(TypeError):
            kernel[1](MockValue(asc2.int32))

    def test_too_few_args(self, jit_test):

        @jit_test
        def kernel():
            for _ in asc2.static_range():
                pass

        with pytest.raises(ValueError, match="1 to 3"):
            kernel[1]()

    def test_too_many_args(self, jit_test):

        @jit_test
        def kernel():
            for _ in asc2.static_range(1, 2, 3, 4):
                pass

        with pytest.raises(ValueError, match="1 to 3"):
            kernel[1]()
