# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from typing import Generator, Iterable, Optional
from typing_extensions import Self

from ..._C import ir
from ..core.dtype import DataType, int32
from ..core.tensor import GlobalAddress
from ..core.ir_value import IRHandle, IRValue, PlainValue, RuntimeInt, materialize_ir_value as mat
from ..core.utils import check_type, global_builder


class RuntimeShape:

    def __init__(self, ir_tensor: IRHandle, shape: Iterable[int]) -> None:
        self.ir_tensor = ir_tensor
        self.shape = tuple(shape)

    def normalize_index(self, index: int) -> int:
        check_type("index", index, int)
        rank = len(self.shape)
        if index < 0:
            index = rank + index
        if index < 0 or index >= rank:
            raise IndexError(f"shape index {index} out of range")
        return index

    def __getitem__(self, index: int) -> RuntimeInt:
        index = self.normalize_index(index)
        dim = self.shape[index]
        if dim != ir.dynshape:
            return dim
        return PlainValue(global_builder.get_ir_builder().create_asctile_DimOp(int32.to_ir(), self.ir_tensor, index))

    def __len__(self) -> int:
        return len(self.shape)

    def __iter__(self) -> Generator[RuntimeInt, None, None]:
        for i in range(len(self)):
            yield self[i]

    def is_static(self) -> bool:
        return all(dim != ir.dynshape for dim in self.shape)

    def is_dynamic_dim(self, index: int) -> bool:
        index = self.normalize_index(index)
        return self.shape[index] == ir.dynshape


class Tensor(IRValue):

    def __init__(self, *, handle: Optional[IRHandle] = None) -> None:
        super().__init__()
        self.handle = handle
        ir_type = self.handle.get_type()
        self.dtype = DataType.from_ir(ir.get_element_type(ir_type))
        self.shape = RuntimeShape(self.handle, ir.get_shape(ir_type))

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle


def tensor(base: GlobalAddress, shape: Iterable[RuntimeInt]) -> Tensor:
    static_sizes = []
    dynamic_sizes = []
    for dim in shape:
        if isinstance(dim, int):
            static_sizes.append(dim)
        elif isinstance(dim, PlainValue):
            static_sizes.append(ir.dynshape)
            dynamic_sizes.append(mat(dim, int32).to_ir())
        else:
            raise TypeError(f"Tensor dimension should be RuntimeInt, got {dim.__class__.__name__}")
    ir_type = ir.get_asctile_TensorType(static_sizes, base.dtype.to_ir())
    handle = global_builder.get_ir_builder().create_asctile_TensorOp(ir_type, base.to_ir(), dynamic_sizes)
    return Tensor.from_ir(handle=handle)
