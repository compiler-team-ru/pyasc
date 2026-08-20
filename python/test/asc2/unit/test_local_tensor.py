# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2


def test_T(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], asc2.float32)
        result = x.T
        assert result.shape == (64, 32)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_floordiv(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([128], asc2.float32)
        y = zero_tile([128], asc2.float32)
        result = x // y
        assert result.shape == (128, )

    kernel[1]()
    assert mock_launch.call_count == 1


def test_rfloordiv(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([128], asc2.float32)
        result = 2.0 // x
        assert result.shape == (128, )

    kernel[1]()
    assert mock_launch.call_count == 1


def test_pos(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([128], asc2.float32)
        result = +x
        assert result.shape == (128, )

    kernel[1]()
    assert mock_launch.call_count == 1


def test_abs(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([128], asc2.float32)
        result = x.abs()
        assert result.shape == (128, )

    kernel[1]()
    assert mock_launch.call_count == 1


def test_ceildiv(jit_test, mock_launch):

    @jit_test
    def kernel():
        result = asc2.ceildiv(7, 3)
        assert result == 3

    kernel[1]()
    assert mock_launch.call_count == 1
