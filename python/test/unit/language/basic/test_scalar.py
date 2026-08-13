# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_scalar_cast(mock_launcher_run):

    @asc.jit
    def scalar_cast_kernel():
        value_in = 2.5
        dtype = asc.int32
        round_mode = asc.RoundMode.CAST_ROUND
        value_out = asc.scalar_cast(value_in, dtype, round_mode)

    scalar_cast_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_scalar_get_sff_value(mock_launcher_run):

    @asc.jit
    def scalar_get_sff_value_kernel():
        value_in = 28
        one_count_0 = asc.scalar_get_sff_value(value_in, 0)
        one_count_1 = asc.scalar_get_sff_value(value_in, 1)

    scalar_get_sff_value_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_scalar_get_count_of_value(mock_launcher_run):

    @asc.jit
    def scalar_get_count_of_value_kernel():
        value_in = 16
        zero_count = asc.scalar_get_count_of_value(value_in, 0)
        one_count = asc.scalar_get_count_of_value(value_in, 1)

    scalar_get_count_of_value_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_scalar_count_leading_zero(mock_launcher_run):

    @asc.jit
    def scalar_count_leading_zero_kernel():
        value_in = 0x0fffffffffffffff
        leading_zeros = asc.scalar_count_leading_zero(value_in)

    scalar_count_leading_zero_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_count_bits_cnt_same_as_sign_bit(mock_launcher_run):

    @asc.jit
    def count_bits_cnt_same_as_sign_bit_kernel():
        value_in = 0x0f00000000000000
        result = asc.count_bits_cnt_same_as_sign_bit(value_in)
        zero_result = asc.count_bits_cnt_same_as_sign_bit(0)
        neg_one_result = asc.count_bits_cnt_same_as_sign_bit(-1)

    count_bits_cnt_same_as_sign_bit_kernel[1]()
    assert mock_launcher_run.call_count == 1
