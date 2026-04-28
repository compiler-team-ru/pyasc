# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from asc.lib.profiling import Profiler, task_time_median
from asc.runtime import config


class StubProfiler:

    def profile(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--backend", type=config.Backend, default=config.Backend.Model, help="Runtime backend")
    parser.addoption("--platform", type=config.Platform, default=config.Platform.Ascend950PR_9599,
                     help="Runtime platform")
    parser.addoption("--device", type=int, default=0, help="Device ID")
    parser.addoption("--profile", action="store_true", help="Enable NPU profiling (if available)")


def pytest_configure(config):
    config.profiling_results = []


def pytest_terminal_summary(terminalreporter, config):
    if not config.profiling_results:
        return
    terminalreporter.write_sep("=", "Profiling results")
    for entry in config.profiling_results:
        terminalreporter.write_line(f"{entry['test']}: {entry['duration']} μs")


@pytest.fixture
def backend(request: pytest.FixtureRequest):
    return request.config.getoption("--backend")


@pytest.fixture
def platform(request: pytest.FixtureRequest):
    return request.config.getoption("--platform")


@pytest.fixture
def device_id(request: pytest.FixtureRequest):
    return request.config.getoption("--device")


def require_c310_impl(platform: config.Platform):
    if config.platform_to_arch(platform) != config.CompilationArch.C310:
        pytest.skip(f"{platform.value} platform is not supported")


@pytest.fixture
def require_c310():
    return require_c310_impl


@pytest.fixture
def profiler(request, tmp_path_factory, backend):
    if backend != config.Backend.NPU or not request.config.getoption("--profile"):
        yield StubProfiler()
        return
    profiler = Profiler(str(tmp_path_factory.mktemp("profiler")))
    yield profiler
    request.config.profiling_results.append({
        "test": request.node.nodeid,
        "duration": task_time_median(profiler.last_result.tasks, skip=1),
    })
