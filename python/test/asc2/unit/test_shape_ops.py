# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2
import pytest
import torch

from .helpers import all_dtypes, non_ub_locations

valid_dtypes = (asc2.int8, asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.bfloat16, asc2.float32)
reshape_dtypes = valid_dtypes + (asc2.float64, )
transpose_dtypes = (asc2.int8, asc2.int16, asc2.int32, asc2.float16, asc2.bfloat16, asc2.float32)


class TestBroadcastTo:

    @pytest.mark.parametrize("dtype", valid_dtypes)
    def test_broadcast(self, jit_test, mock_launch, zero_tile, dtype):

        @jit_test
        def kernel():
            x = zero_tile([1, 64], dtype)
            x.broadcast_to(32, 64)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_same_shape(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.broadcast_to(32, 64)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_incompatible(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.broadcast_to(64, 32)

        with pytest.raises(RuntimeError, match="Cannot broadcast"):
            kernel[1]()

    @pytest.mark.parametrize("loc", non_ub_locations)
    def test_invalid_location(self, jit_test, zero_tile, loc):

        @jit_test
        def kernel():
            x = zero_tile([1, 64], asc2.float32, loc)
            x.broadcast_to(32, 64)

        with pytest.raises(RuntimeError, match="location"):
            kernel[1]()

    def test_invalid_input_type(self, jit_test):

        @jit_test
        def kernel():
            asc2.broadcast_to("invalid", 32, 64)

        with pytest.raises(TypeError, match="input"):
            kernel[1]()


class TestReshape:

    @pytest.mark.parametrize("dtype", reshape_dtypes)
    def test_reshape(self, jit_test, mock_launch, zero_tile, dtype):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], dtype)
            x.reshape(64, 32)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_same_shape(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.reshape(32, 64)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_to_1d(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.reshape(2048)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_element_mismatch(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.reshape(32, 32)

        with pytest.raises(RuntimeError, match="not match"):
            kernel[1]()

    def test_invalid_input_type(self, jit_test):

        @jit_test
        def kernel():
            asc2.reshape("invalid", 32, 64)

        with pytest.raises(TypeError, match="input"):
            kernel[1]()


class TestRavel:

    def test_ravel(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.ravel()

        kernel[1]()
        assert mock_launch.call_count == 1


class TestExpandDims:

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_expand_dims(self, jit_test, mock_launch, zero_tile, axis):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.expand_dims(axis)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_multiple_axes(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.expand_dims(0, 2)

        kernel[1]()
        assert mock_launch.call_count == 1


class TestSqueeze:

    def test_squeeze_all(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([1, 32, 1, 64], asc2.float32)
            x.squeeze()

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_squeeze_specific(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([1, 32, 64], asc2.float32)
            x.squeeze(0)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_non_unit_dim(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.squeeze(0)

        with pytest.raises(RuntimeError, match="must be 1"):
            kernel[1]()


class TestTranspose:

    def test_2d(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.transpose()

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_3d(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([16, 32, 64], asc2.float32)
            x.transpose(2, 0, 1)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_identity(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.transpose(0, 1)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_wrong_axis_count(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            x.transpose(0, 1, 2)

        with pytest.raises(RuntimeError, match="axis count"):
            kernel[1]()

    def test_wrong_permutation(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64, 128], asc2.float32)
            x.transpose(0, 0, 1)

        with pytest.raises(RuntimeError, match="rearrangement"):
            kernel[1]()

    @pytest.mark.parametrize("dtype", [d for d in all_dtypes if d not in transpose_dtypes])
    def test_invalid_dtype(self, jit_test, zero_tile, dtype):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], dtype)
            x.transpose()

        with pytest.raises(RuntimeError, match="dtype"):
            kernel[1]()

    @pytest.mark.parametrize("loc", non_ub_locations)
    def test_invalid_location_3d(self, jit_test, zero_tile, loc):

        @jit_test
        def kernel():
            x = zero_tile([16, 32, 64], asc2.float32, loc)
            x.transpose(2, 0, 1)

        with pytest.raises(RuntimeError, match="location"):
            kernel[1]()

    @pytest.mark.parametrize("loc", (asc2.TensorLocation.L0A, asc2.TensorLocation.L0B, asc2.TensorLocation.L1))
    def test_non_ub_align(self, jit_test, mock_launch, zero_tile, loc):

        @jit_test
        def kernel():
            x = zero_tile([8, 64], asc2.float16, loc)
            x.T

        kernel[1]()
        assert mock_launch.call_count == 1


class TestBroadcastShapes:

    def test_single_shape(self):
        assert asc2.broadcast_shapes([32]) == (32, )

    def test_single_shape_multidim(self):
        assert asc2.broadcast_shapes((4, 32)) == (4, 32)

    def test_two_identical_shapes(self):
        assert asc2.broadcast_shapes([4, 32], [4, 32]) == (4, 32)

    def test_two_shapes_different_ranks(self):
        assert asc2.broadcast_shapes([1, 32], [4, 32]) == (4, 32)

    def test_two_shapes_rank_padding(self):
        assert asc2.broadcast_shapes([32], [4, 32]) == (4, 32)

    def test_two_shapes_both_broadcast(self):
        assert asc2.broadcast_shapes([1, 32], [4, 1]) == (4, 32)

    def test_three_shapes(self):
        assert asc2.broadcast_shapes([1, 32], [4, 1], [4, 32]) == (4, 32)

    def test_three_shapes_different_ranks(self):
        assert asc2.broadcast_shapes([32], [1, 1], [4, 32]) == (4, 32)

    def test_scalar_like_shapes(self):
        assert asc2.broadcast_shapes([1], [32]) == (32, )

    def test_high_dim(self):
        assert asc2.broadcast_shapes([2, 1, 32], [1, 4, 1]) == (2, 4, 32)

    def test_matches_torch(self):
        shapes = [(1, 32), (4, 1), (4, 32)]
        result = asc2.broadcast_shapes(*shapes)
        expected = torch.broadcast_shapes(*shapes)
        assert result == expected

    def test_no_shapes_raises(self):
        with pytest.raises(ValueError):
            asc2.broadcast_shapes()

    def test_incompatible_shapes_raises(self):
        with pytest.raises(RuntimeError, match="not broadcastable"):
            asc2.broadcast_shapes([3, 32], [4, 32])

    def test_incompatible_shapes_different_ranks_raises(self):
        with pytest.raises(RuntimeError, match="not broadcastable"):
            asc2.broadcast_shapes([2, 32], [3, 32])

    def test_non_positive_dim_raises(self):
        with pytest.raises(RuntimeError, match="positive"):
            asc2.broadcast_shapes([0, 32])

    def test_negative_dim_raises(self):
        with pytest.raises(RuntimeError, match="positive"):
            asc2.broadcast_shapes([-1, 32])

    def test_non_integer_dim_raises(self):
        with pytest.raises(TypeError, match="integers"):
            asc2.broadcast_shapes([1.5, 32])

    def test_empty_shape_raises(self):
        with pytest.raises(RuntimeError, match="at least one value"):
            asc2.broadcast_shapes([])

    def test_unpacked_generator_input(self):
        shapes = [[4, 32], [1, 32]]
        assert asc2.broadcast_shapes(*shapes) == (4, 32)


class TestBroadcastTensors:

    def test_broadcast_tensors(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([1, 64], asc2.float32)
            y = zero_tile([32, 1], asc2.float32)
            asc2.broadcast_tensors(x, y)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_incompatible(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            y = zero_tile([64, 32], asc2.float32)
            asc2.broadcast_tensors(x, y)

        with pytest.raises(RuntimeError, match="not broadcastable"):
            kernel[1]()

    def test_single_tensor(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asc2.float32)
            asc2.broadcast_tensors(x)

        kernel[1]()
        assert mock_launch.call_count == 1
