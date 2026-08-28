# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import pytest

valid_src_dtypes = (asc2.int16, asc2.int32, asc2.float16, asc2.bfloat16, asc2.float32)


class TestWhere:

    @pytest.mark.parametrize("dtype", valid_src_dtypes)
    def test_tile_tile(self, jit_test, mock_launch, zero_tile, dtype):

        @jit_test
        def kernel():
            a = zero_tile([32, 64], dtype)
            b = zero_tile([32, 64], dtype)
            mask = a > b
            asc2.where(mask, a, b)

        kernel[1]()
        assert mock_launch.call_count == 1

    # NOTE: where with scalar operands is not tested because check_dtype in where()
    # doesn't handle plain scalars (no .dtype attribute) - implementation limitation.

    def test_invalid_mask_dtype(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            mask = zero_tile([32, 64], asc2.float32)
            src0 = zero_tile([32, 64], asc2.float32)
            src1 = zero_tile([32, 64], asc2.float32)
            asc2.where(mask, src0, src1)

        with pytest.raises(RuntimeError, match="dtype"):
            kernel[1]()

    def test_invalid_src_dtype(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            a = zero_tile([32, 64], asc2.int8)
            b = zero_tile([32, 64], asc2.int8)
            mask = a > b
            asc2.where(mask, a, b)

        with pytest.raises(RuntimeError, match="dtype"):
            kernel[1]()

    def test_invalid_mask_type(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            src0 = zero_tile([32, 64], asc2.float32)
            src1 = zero_tile([32, 64], asc2.float32)
            asc2.where("invalid", src0, src1)

        with pytest.raises(TypeError, match="mask"):
            kernel[1]()

    def test_invalid_src0_type(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            a = zero_tile([32, 64], asc2.float32)
            b = zero_tile([32, 64], asc2.float32)
            mask = a > b
            asc2.where(mask, "invalid", b)

        with pytest.raises(TypeError, match="src0"):
            kernel[1]()

    def test_invalid_src1_type(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            a = zero_tile([32, 64], asc2.float32)
            b = zero_tile([32, 64], asc2.float32)
            mask = a > b
            asc2.where(mask, a, "invalid")

        with pytest.raises(TypeError, match="src1"):
            kernel[1]()


class TestMask:

    def test_count(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            y = zero_tile([32, 64], asc2.float32)
            with asc2.mask(count=8):
                x + y

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_bits(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            y = zero_tile([32, 64], asc2.float32)
            with asc2.mask(bits=[0, 64], other=-1):
                x + y

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_no_args(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            with asc2.mask():
                x + 1

        with pytest.raises(ValueError, match="must be provided"):
            kernel[1]()

    def test_bits_wrong_count(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            with asc2.mask(bits=[0, 64, 128]):
                x + 1

        with pytest.raises(ValueError, match="must have 2 values"):
            kernel[1]()

    def test_both_count_and_bits(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            with asc2.mask(count=8, bits=[0, 64]):
                x + 1

        with pytest.raises(ValueError, match="not both"):
            kernel[1]()

    def test_invalid_other_type(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            with asc2.mask(count=8, other="invalid"):
                x + 1

        with pytest.raises(TypeError, match="other"):
            kernel[1]()
