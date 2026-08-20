# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2


def test_block_idx(jit_test, mock_launch):

    @jit_test
    def kernel():
        idx = asc2.block_idx()
        assert idx.dtype == asc2.int32

    kernel[1]()
    assert mock_launch.call_count == 1


def test_block_num(jit_test, mock_launch):

    @jit_test
    def kernel():
        num = asc2.block_num()
        assert num.dtype == asc2.int32

    kernel[1]()
    assert mock_launch.call_count == 1
