# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference implementation: https://github.com/sgl-project/sgl-kernel-npu/blob/c28ea2a940a53c00f3a0322d9576210a7f5ae92f/python/sgl_kernel_npu/sgl_kernel_npu/kimi_k3/attn_residual.py

import pytest

from ...target.kimi_k3.test_attn_residual import run_attn_residual_test


@pytest.mark.parametrize("num_tokens, num_valid_blocks, hidden_size, unroll_factor", [
    (4, 2, 128, 2),
    (8, 2, 128, 2),
    (2, 4, 128, 2),
    (16, 1, 256, 1),
])
def test_attn_residual(profiler, runs, num_tokens, num_valid_blocks, hidden_size, unroll_factor):
    run_attn_residual_test(profiler, runs, num_tokens, num_valid_blocks, hidden_size, unroll_factor)
