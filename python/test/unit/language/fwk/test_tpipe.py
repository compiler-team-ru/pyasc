# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

import asc
from asc.runtime import config


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_alloc_event_id(mock_launcher_run):

    @asc.jit
    def kernel_alloc_event_id() -> None:
        pipe = asc.TPipe()
        pipe_ptr = asc.get_tpipe_ptr()
        event_id = pipe_ptr.alloc_event_id(event=asc.HardEvent.V_S)

    kernel_alloc_event_id[1]()
    assert mock_launcher_run.call_count == 1


def test_destroy(mock_launcher_run):

    @asc.jit
    def kernel_destroy() -> None:
        pipe = asc.TPipe()
        pipe.destroy()

    kernel_destroy[1]()
    assert mock_launcher_run.call_count == 1


def test_fetch_event_id(mock_launcher_run):

    @asc.jit
    def kernel_fetch_event_id() -> None:
        pipe = asc.TPipe()
        asc.get_tpipe_ptr().fetch_event_id(event=asc.HardEvent.V_S)

    kernel_fetch_event_id[1]()
    assert mock_launcher_run.call_count == 1


@pytest.mark.skip(reason="CPU DEBUG ONLY")
def test_get_base_addr(mock_launcher_run):

    @asc.jit
    def kernel_get_base_addr() -> None:
        pipe = asc.TPipe()
        pipe.get_base_addr(logic_pos=asc.TPosition.VECCALC)

    kernel_get_base_addr[1]()
    assert mock_launcher_run.call_count == 1


def test_init(mock_launcher_run):

    @asc.jit
    def kernel_init() -> None:
        pipe = asc.TPipe()
        asc.get_tpipe_ptr().init()

    kernel_init[1]()
    assert mock_launcher_run.call_count == 1


def test_init_buf_pool(mock_launcher_run):

    @asc.jit
    def kernel_init_buf_pool() -> None:
        pipe = asc.TPipe()
        buf_pool1 = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)
        buf_pool0 = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)
        pipe.init_buf_pool(buf_pool=buf_pool0, len=256)
        pipe.init_buf_pool(buf_pool=buf_pool0, len=256, share_buf=buf_pool1)

    kernel_init_buf_pool[1]()
    assert mock_launcher_run.call_count == 1


def test_init_buffer(mock_launcher_run):

    @asc.jit
    def kernel_init_buffer() -> None:
        pipe = asc.TPipe()
        que = asc.TQue(asc.TPosition.VECIN, 1)
        buf = asc.TBuf(asc.TPosition.A1)
        pipe.init_buffer(que=que, num=1, len=128)
        pipe.init_buffer(buf=buf, len=128)

    kernel_init_buffer[1]()
    assert mock_launcher_run.call_count == 1


def test_init_method(mock_launcher_run):

    @asc.jit
    def kernel_init_method() -> None:
        pipe = asc.TPipe()
        pipe.init()

    kernel_init_method[1]()
    assert mock_launcher_run.call_count == 1


def test_release_event_id(mock_launcher_run):

    @asc.jit
    def kernel_release_event_id() -> None:
        pipe = asc.TPipe()
        pipe_ptr = asc.get_tpipe_ptr()
        event_id = pipe_ptr.alloc_event_id(event=asc.HardEvent.V_S)
        pipe_ptr.release_event_id(id=event_id, event=asc.HardEvent.V_S)

    kernel_release_event_id[1]()
    assert mock_launcher_run.call_count == 1


def test_reset(mock_launcher_run):

    @asc.jit
    def kernel_reset() -> None:
        pipe = asc.TPipe()
        pipe.reset()

    kernel_reset[1]()
    assert mock_launcher_run.call_count == 1
