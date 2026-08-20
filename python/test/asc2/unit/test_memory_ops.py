# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
from asc.runtime.jit import MockTensor, MockValue
import pytest

from .helpers import non_ub_l0c_locations

valid_dtypes = (asc2.int8, asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.bfloat16, asc2.float32)
copy_in_locs = (asc2.TensorLocation.UB, asc2.TensorLocation.L1, asc2.TensorLocation.L0A, asc2.TensorLocation.L0B)


class TestCopyIn:

    @pytest.mark.parametrize("dtype", valid_dtypes)
    @pytest.mark.parametrize("loc", copy_in_locs)
    def test_copy_in(self, jit_test, mock_launch, dtype, loc):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [64, 128])
            asc2.copy_in(x_gm, [0, 0], [64, 128], loc)

        kernel[1](MockTensor(dtype))
        assert mock_launch.call_count == 1

    def test_scalar(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [128])
            asc2.copy_in(x_gm, [0])

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_with_padding(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [256])
            asc2.copy_in(x_gm, [0], [128], real_shape=[64], pad_value=0.0)

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_pad_value_without_real_shape(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [128])
            asc2.copy_in(x_gm, [0], [128], pad_value=0.0)

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_real_shape_without_pad_value(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [128])
            asc2.copy_in(x_gm, [0], [128], real_shape=[64])

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_real_shape_exceeds_tensor(self, jit_test):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [64])
            asc2.copy_in(x_gm, [0], [128], real_shape=[256])

        with pytest.raises(RuntimeError, match="exceeds"):
            kernel[1](MockTensor(asc2.float32))

    def test_invalid_src_type(self, jit_test):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            asc2.copy_in("invalid", [0], [128])

        with pytest.raises(TypeError, match="src"):
            kernel[1](MockTensor(asc2.float32))

    def test_real_shape_with_l1(self, jit_test):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [256])
            asc2.copy_in(x_gm, [0], [128], asc2.TensorLocation.L1, real_shape=[64])

        with pytest.raises(RuntimeError, match="real_shape"):
            kernel[1](MockTensor(asc2.float32))

    def test_with_plain_value_offset(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress, offset: int):
            x_gm = asc2.global_tensor(x_ptr, [256])
            asc2.copy_in(x_gm, [offset], [128])

        kernel[1](MockTensor(asc2.float32), MockValue(asc2.int32))
        assert mock_launch.call_count == 1


class TestCopyOut:

    @pytest.mark.parametrize("dtype", valid_dtypes)
    def test_copy_out(self, jit_test, mock_launch, zero_tile, dtype):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress):
            out_gm = asc2.global_tensor(out_ptr, [128])
            src = zero_tile([128], dtype)
            asc2.copy_out(src, out_gm, [0])

        kernel[1](MockTensor(dtype))
        assert mock_launch.call_count == 1

    def test_scalar(self, jit_test, mock_launch):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress):
            out_gm = asc2.global_tensor(out_ptr, [128])
            asc2.copy_out(42.0, out_gm, [0])

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_with_real_shape(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress):
            out_gm = asc2.global_tensor(out_ptr, [128])
            src = zero_tile([128], asc2.float32)
            asc2.copy_out(src, out_gm, [0], real_shape=[64])

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_scalar_with_real_shape(self, jit_test):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress):
            out_gm = asc2.global_tensor(out_ptr, [128])
            asc2.copy_out(42.0, out_gm, [0], real_shape=[64])

        with pytest.raises(ValueError, match="real_shape"):
            kernel[1](MockTensor(asc2.float32))

    def test_from_l0c(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress):
            out_gm = asc2.global_tensor(out_ptr, [128])
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.L0C)
            asc2.copy_out(src, out_gm, [0])

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    @pytest.mark.parametrize("loc", non_ub_l0c_locations)
    def test_invalid_location(self, jit_test, zero_tile, loc):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress):
            out_gm = asc2.global_tensor(out_ptr, [128])
            src = zero_tile([128], asc2.float32, loc)
            asc2.copy_out(src, out_gm, [0])

        with pytest.raises(RuntimeError, match="location"):
            kernel[1](MockTensor(asc2.float32))

    def test_invalid_dst_type(self, jit_test, zero_tile):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress):
            src = zero_tile([128], asc2.float32)
            asc2.copy_out(src, "invalid", [0])

        with pytest.raises(TypeError, match="dst"):
            kernel[1](MockTensor(asc2.float32))

    def test_with_plain_value_offset(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel(out_ptr: asc2.GlobalAddress, offset: int):
            out_gm = asc2.global_tensor(out_ptr, [256])
            src = zero_tile([128], asc2.float32)
            asc2.copy_out(src, out_gm, [offset])

        kernel[1](MockTensor(asc2.float32), MockValue(asc2.int32))
        assert mock_launch.call_count == 1


class TestCopy:

    def test_ub_to_l1(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [128])
            ub = asc2.copy_in(x_gm, [0], [128])
            asc2.copy(ub, location=asc2.TensorLocation.L1)

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_l0c_to_ub(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.L0C)
            asc2.copy(src, location=asc2.TensorLocation.UB)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l0c_to_l1(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.L0C)
            asc2.copy(src, location=asc2.TensorLocation.L1)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l1_to_l0a(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.L1)
            asc2.copy(src, location=asc2.TensorLocation.L0A)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l1_to_l0b(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.L1)
            asc2.copy(src, location=asc2.TensorLocation.L0B)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l1_to_bt(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.L1)
            asc2.copy(src, location=asc2.TensorLocation.BT)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_invalid_src_location(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.L0A)
            asc2.copy(src, location=asc2.TensorLocation.UB)

        with pytest.raises(RuntimeError, match="location"):
            kernel[1]()

    def test_invalid_transfer(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asc2.float32, asc2.TensorLocation.UB)
            asc2.copy(src, location=asc2.TensorLocation.L0A)

        with pytest.raises(RuntimeError, match="location"):
            kernel[1]()

    def test_with_offsets_and_shape(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([64, 128], asc2.float32, asc2.TensorLocation.L1)
            asc2.copy(src, [16, 16], [32, 32], asc2.TensorLocation.L0A)

        kernel[1]()
        assert mock_launch.call_count == 1


class TestTo:

    def test_to_location(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [128])
            ub = asc2.copy_in(x_gm, [0], [128])
            ub.to(asc2.TensorLocation.L1)

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_to_dtype(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [128])
            ub = asc2.copy_in(x_gm, [0], [128])
            ub.to(asc2.float16)

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1

    def test_to_same_location(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asc2.GlobalAddress):
            x_gm = asc2.global_tensor(x_ptr, [128])
            ub = asc2.copy_in(x_gm, [0], [128])
            ub.to(asc2.TensorLocation.UB)

        kernel[1](MockTensor(asc2.float32))
        assert mock_launch.call_count == 1


def test_copy_in_misaligned(jit_test):

    @jit_test
    def kernel(x_ptr: asc2.GlobalAddress):
        x_gm = asc2.global_tensor(x_ptr, [32, 33])
        asc2.copy_in(x_gm, [0, 0], [32, 33])

    with pytest.raises(RuntimeError, match="aligned"):
        kernel[1](MockTensor(asc2.float32))
