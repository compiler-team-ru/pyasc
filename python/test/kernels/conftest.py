# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from asc.runtime import config


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--backend", default=config.Backend.Model, help="Runtime backend for testing")
    parser.addoption("--platform", default=config.Platform.Ascend910B1, help="Runtime platform for testing")


@pytest.fixture
def backend(request: pytest.FixtureRequest):
    return request.config.getoption("--backend")


@pytest.fixture
def platform(request: pytest.FixtureRequest):
    return request.config.getoption("--platform")
