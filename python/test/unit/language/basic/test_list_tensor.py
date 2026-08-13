# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config
from asc.runtime.jit import MockTensor


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_tensor_desc_set_shape_addr(mock_launcher_run):

    @asc.jit
    def kernel_tensor_desc_set_shape_addr() -> None:
        tensor_desc = asc.TensorDesc()
        tensor_desc.set_shape_addr(0)

    kernel_tensor_desc_set_shape_addr[1]()
    assert mock_launcher_run.call_count == 1


def test_tensor_desc_get_dim(mock_launcher_run):

    @asc.jit
    def kernel_tensor_desc_get_dim() -> None:
        tensor_desc = asc.TensorDesc()
        dim = tensor_desc.get_dim()

    kernel_tensor_desc_get_dim[1]()
    assert mock_launcher_run.call_count == 1


def test_tensor_desc_get_index(mock_launcher_run):

    @asc.jit
    def kernel_tensor_desc_get_index() -> None:
        tensor_desc = asc.TensorDesc()
        index = tensor_desc.get_index()

    kernel_tensor_desc_get_index[1]()
    assert mock_launcher_run.call_count == 1


def test_tensor_desc_get_shape(mock_launcher_run):

    @asc.jit
    def kernel_tensor_desc_get_shape() -> None:
        tensor_desc = asc.TensorDesc()
        offset = 0
        shape = tensor_desc.get_shape(offset)

    kernel_tensor_desc_get_shape[1]()
    assert mock_launcher_run.call_count == 1


def test_tensor_desc_get_data_ptr(mock_launcher_run):

    @asc.jit
    def kernel_tensor_desc_get_data_ptr() -> None:
        tensor_desc = asc.TensorDesc()
        data_ptr = tensor_desc.get_data_ptr()

    kernel_tensor_desc_get_data_ptr[1]()
    assert mock_launcher_run.call_count == 1


def test_tensor_desc_get_data_obj(mock_launcher_run):

    @asc.jit
    def kernel_tensor_desc_get_data_obj() -> None:
        tensor_desc = asc.TensorDesc()
        data_obj = tensor_desc.get_data_obj()

    kernel_tensor_desc_get_data_obj[1]()
    assert mock_launcher_run.call_count == 1


def test_init(mock_launcher_run):

    @asc.jit
    def kernel_init(x: asc.GlobalAddress) -> None:
        x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
        x_desc = asc.ListTensorDesc()

    x = MockTensor(asc.uint8)
    kernel_init[1](x)
    assert mock_launcher_run.call_count == 1


def test_init_func(mock_launcher_run):

    @asc.jit
    def kernel_init_func(x: asc.GlobalAddress) -> None:
        x_desc = asc.ListTensorDesc()
        x_desc.init(data=x, length=0xffffffff, shape_size=0xffffffff)

    x = MockTensor(asc.uint8)
    kernel_init_func[1](x)
    assert mock_launcher_run.call_count == 1


def test_get_desc(mock_launcher_run):

    @asc.jit
    def kernel_get_desc(x: asc.GlobalAddress) -> None:
        x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
        y = asc.TensorDesc()
        x_desc.get_desc(y, index=0)

    x = MockTensor(asc.uint8)
    kernel_get_desc[1](x)
    assert mock_launcher_run.call_count == 1


def test_get_data_ptr(mock_launcher_run):

    @asc.jit
    def kernel_get_data_ptr(x: asc.GlobalAddress) -> None:
        x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
        x_ptr = x_desc.get_data_ptr(index=0, dtype=asc.float16)

    x = MockTensor(asc.uint8)
    kernel_get_data_ptr[1](x)
    assert mock_launcher_run.call_count == 1


def test_get_size(mock_launcher_run):

    @asc.jit
    def kernel_get_size(x: asc.GlobalAddress) -> None:
        x_desc = asc.ListTensorDesc(data=x, length=0xffffffff, shape_size=0xffffffff)
        x_size = x_desc.get_size()

    x = MockTensor(asc.uint8)
    kernel_get_size[1](x)
    assert mock_launcher_run.call_count == 1
