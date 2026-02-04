# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from typing import Iterable, Optional
from typing_extensions import Self

from ..._C import ir
from ..core.dtype import DataType, int32
from ..core.tensor import GlobalAddress
from ..core.ir_value import IRHandle, IRValue, RuntimeInt, materialize_ir_value as mat
from ..core.utils import global_builder


class Tensor(IRValue):

    def __init__(self, *, handle: Optional[IRHandle] = None) -> None:
        super().__init__()
        self.handle = handle
        self.dtype = DataType.from_ir(ir.get_element_type(self.handle.get_type()))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


def tensor(base: GlobalAddress, shape: Iterable[RuntimeInt]) -> Tensor:
    if all(isinstance(dim, int) for dim in shape):
        static_shape = list(shape)
        dynamic_shape = []
    else:
        dynamic_shape = [mat(dim, int32) for dim in shape]
        static_shape = [ir.dynshape] * len(dynamic_shape)
    ir_type = ir.get_asctile_TensorType(static_shape, base.dtype.to_ir())
    handle = global_builder.get_ir_builder().create_asctile_TensorOp(ir_type, base.to_ir(), dynamic_shape)
    return Tensor.from_ir(handle=handle)
