# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from contextlib import ExitStack
from importlib import import_module
import os
from unittest.mock import patch

from asc.lib.profiling import Profiler, task_time_median
from asc.runtime import config
import pytest


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
    parser.addoption("--device", type=int, help="Device ID")
    parser.addoption("--profile", action="store_true", help="Enable NPU profiling (if available)")
    parser.addoption("--runs", type=int, default=1, help="Number of kernel launches")
    parser.addoption("--compile-only", action="store_true", help="Stop after the kernel compilation (do not launch)")


def pytest_configure(config):
    config.profiling_results = []


def pytest_make_parametrize_id(val):
    if callable(val):
        return val.__name__
    return str(val)


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
    return request.config.getoption("--device", default=None)


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


@pytest.fixture
def runs(request: pytest.FixtureRequest):
    return request.config.getoption("--runs")


@pytest.fixture(scope="session", autouse=True)
def compile_only(request: pytest.FixtureRequest):
    if not request.config.getoption("--compile-only"):
        yield False
        return
    os.environ["PYASC_DRY_RUN"] = "1"
    with ExitStack() as stack:
        try:
            import_module("numpy")
            stack.enter_context(patch("numpy.testing.assert_allclose"))
        except ModuleNotFoundError:
            pass
        try:
            import_module("torch")
            stack.enter_context(patch("torch.testing.assert_close"))
            for fn in ("torch.allclose", "torch.equal"):
                stack.enter_context(patch(fn)).return_value = True
        except ModuleNotFoundError:
            pass
        yield True
