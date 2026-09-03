# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

from .matmul_v3 import FullLoadMode, run_matmul_v3_test

test_cases = [
    # (36, (1500, 1669, 113, 256, 256, 128, 256, 256, 32), torch.float32, False, True, FullLoadMode.NONE, True, False,
    #    (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)), # FAILED with UB overflow
    (36, (45000, 92, 32, 336, 96, 32, 336, 96, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (10000, 200, 256, 144, 208, 256, 144, 208, 64), torch.float16, False, False, FullLoadMode.B, False, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (46500, 88, 104, 336, 96, 64, 336, 96, 16), torch.float32, False, False, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (39000, 116, 132, 256, 128, 64, 256, 128, 16), torch.float32, False, False, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (45000, 124, 124, 256, 128, 64, 256, 128, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (49500, 144, 128, 224, 144, 64, 224, 144, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (192000, 66, 64, 400, 80, 64, 400, 80, 16), torch.float32, False, False, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    # (36, (250000, 120, 145, 256, 128, 64, 256, 128, 16), torch.float32, False, False, FullLoadMode.B, True, False,
    #    (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)), # FAILED: Accuracy mismatch
    (36, (96000, 104, 64, 288, 112, 64, 288, 112, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    # (36, (75000, 116, 116, 256, 128, 64, 256, 128, 16), torch.float32, False, True, FullLoadMode.B, True, False,
    #    (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)), # FAILED: Accuracy mismatch
    (36, (150000, 76, 64, 400, 80, 64, 400, 80, 16), torch.float16, False, True, FullLoadMode.B, False, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (40960, 280, 256, 112, 288, 64, 112, 288, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (102400, 168, 64, 176, 176, 64, 176, 176, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (180000, 84, 64, 336, 96, 64, 336, 96, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    (36, (250000, 145, 120, 192, 160, 64, 192, 160, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    # (36, (307200, 200, 128, 144, 208, 64, 144, 208, 16), torch.float32, False, True, FullLoadMode.B, True, False,
    #    (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)), # FAILED: Accuracy mismatch
    (36, (375000, 148, 148, 192, 160, 64, 192, 160, 16), torch.float32, False, True, FullLoadMode.B, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)),
    # (36, (4096, 13664, 32, 256, 256, 32, 256, 256, 32), torch.float16, True, False, FullLoadMode.NONE, False, False,
    #    (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3)), # FAILED: Accuracy mismatch
    (36, (4800, 2864, 128, 320, 192, 64, 320, 192, 16), torch.float32, False, False, FullLoadMode.NONE, True, False,
     (1, 1, 1, 2, 2), (0, 1), (1e-3, 1e-3))
]


@pytest.mark.parametrize(
    "core_num, tiling_data, dtype, is_a_transpose_l0, is_b_transpose_l0, full_load_mode, enable_hf32_mode, has_bias, double_buffering, input_range, accuracy",
    test_cases, ids=["_".join(map(str, tc[1][:3])) for tc in test_cases])
def test_matmul_v3(profiler, runs, core_num, tiling_data, dtype, is_a_transpose_l0, is_b_transpose_l0, full_load_mode,
                   enable_hf32_mode, has_bias, double_buffering, input_range, accuracy):
    run_matmul_v3_test(profiler, runs, core_num, tiling_data, dtype, is_a_transpose_l0, is_b_transpose_l0,
                       full_load_mode, enable_hf32_mode, has_bias, double_buffering, input_range, accuracy, l0c2ub=True)
