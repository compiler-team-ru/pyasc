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

from .helpers import all_locations


class TestInline:

    def test_inline(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asc2.inline('constexpr int32_t x = 42;')

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_inline_with_args(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            asc2.inline('auto ptr = $0;', [x_ptr])

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_inline_before_function(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asc2.inline('constexpr int32_t x = 42;', before_function=True)

        kernel[1]()
        assert mock_launch.call_count == 1


class TestInlineVf:

    def test_without_inputs(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asc2.inline_vf('// noop', [32, 64], asc2.float32)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_with_inputs(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            y = zero_tile([32, 64], asc2.float32)
            asc2.inline_vf('// noop', [32, 64], asc2.float32, [x, y])

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_invalid_code_type(self, jit_test):

        @jit_test
        def kernel():
            asc2.inline_vf(123, [32], asc2.float32)

        with pytest.raises(TypeError, match="code"):
            kernel[1]()

    def test_invalid_dtype_type(self, jit_test):

        @jit_test
        def kernel():
            asc2.inline_vf('// noop', [32], "invalid")

        with pytest.raises(TypeError, match="dtype"):
            kernel[1]()

    def test_invalid_input_type(self, jit_test):

        @jit_test
        def kernel():
            asc2.inline_vf('// noop', [32], asc2.float32, ["invalid"])

        with pytest.raises(TypeError, match="inputs"):
            kernel[1]()

    @pytest.mark.parametrize("loc", [loc for loc in all_locations if loc != asc2.TensorLocation.UB])
    def test_invalid_input_location(self, jit_test, zero_tile, loc):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32, loc)
            asc2.inline_vf('// noop', [32, 64], asc2.float32, [x])

        with pytest.raises(RuntimeError, match="location"):
            kernel[1]()
