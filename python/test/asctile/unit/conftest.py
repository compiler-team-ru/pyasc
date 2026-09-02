# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, Tuple, Union
from unittest.mock import patch

import asctile
from asctile.language.utils import constant_tile
import pytest


@pytest.fixture
def jit_test():
    return asctile.jit(always_compile=True, capture_exceptions=False)


def verify_module(mod, options):
    mod.verify(raising=True)


@pytest.fixture(autouse=True)
def mock_launch():

    with patch("asc.runtime.jit.JITFunction._run_compiler", side_effect=verify_module, return_value=None):
        with patch("asc.runtime.jit.JITFunction._run_launcher", return_value=None) as mock:
            yield mock


@pytest.fixture
def zero_tile():

    def zeros(shape: Tuple[int, ...], dtype: asctile.DataType = asctile.float32,
              loc: asctile.TensorLocation = asctile.TensorLocation.UB, *, n: int = 1):
        tiles = tuple(constant_tile(0, shape, dtype, loc) for _ in range(n))
        return tiles[0] if n == 1 else tiles

    return zeros


@pytest.fixture
def scalar_value() -> Callable[[asctile.DataType], Union[int, float]]:
    return lambda dtype: 2.0 if dtype.is_float() else 2
