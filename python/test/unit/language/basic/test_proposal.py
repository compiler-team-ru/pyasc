# Copyright (c) 2025 ISE Group, Harbin Institute of Technology.
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
    

def test_get_sort_len(mock_launcher_run):
    
    @asc.jit
    def kernel_get_sort_len() -> None:
        length = asc.get_sort_len(100)
     
    kernel_get_sort_len[1]()
    assert mock_launcher_run.call_count == 1


def test_get_sort_offset(mock_launcher_run):
    
    @asc.jit
    def kernel_get_sort_offset() -> None:
        offset = asc.get_sort_offset(10)
     
    kernel_get_sort_offset[1]()
    assert mock_launcher_run.call_count == 1