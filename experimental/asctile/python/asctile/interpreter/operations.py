# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import builtins
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np

from asc.language.core.dtype import DataType

from ..language.validation import check_type
from .patch import override
from .types import GlobalAddress, GlobalTensor, LocalTensor, dtype_to_numpy


class BlockContext:

    def __init__(self) -> None:
        self.block_idx: int = 0
        self.block_num: int = 1

    def set(self, block_idx: int, block_num: int) -> None:
        self.block_idx = int(block_idx)
        self.block_num = int(block_num)


block_context = BlockContext()


@override
def block_idx() -> int:
    return block_context.block_idx


@override
def block_num() -> int:
    return block_context.block_num


@override
def range(*args: Any, **kwargs: Any) -> builtins.range:
    del kwargs
    return builtins.range(*args)


@override
def global_tensor(base: GlobalAddress, shape: Iterable[Any], l2_cache: bool = True) -> GlobalTensor:
    del l2_cache
    return GlobalTensor(base, shape)


def clip(offsets: Iterable[Any], shape: Tuple[int, ...],
         bounds: Tuple[int, ...]) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    offsets = tuple(map(int, offsets))
    if len(offsets) != len(bounds) or len(shape) != len(bounds):
        raise ValueError(f"offsets {offsets} and shape {shape} must have the rank of {bounds}")
    if any(off < 0 for off in offsets):
        raise IndexError(f"offsets {offsets} are out of bounds for shape {bounds}")
    counts = tuple(max(0, min(dim, total - off)) for off, dim, total in zip(offsets, shape, bounds))
    region = tuple(slice(off, off + cnt) for off, cnt in zip(offsets, counts))
    head = tuple(slice(None, cnt) for cnt in counts)
    return region, head


def verify_real_shape(real_shape: Iterable[int], shape: Tuple[int, ...]) -> Tuple[int, ...]:
    real_shape = tuple(map(int, real_shape))
    if len(real_shape) != len(shape) or any(real > dim for real, dim in zip(real_shape, shape)):
        raise ValueError(f"real_shape {real_shape} must match the rank of {shape} and not exceed it")
    return real_shape


@override
def copy_in(src: GlobalTensor, offsets: Iterable[Any], shape: Iterable[int], location: Any = None, *,
            real_shape: Optional[Iterable[int]] = None, pad_value: Optional[Union[int, float]] = None) -> LocalTensor:
    del location
    check_type("src", src, GlobalTensor)
    shape = tuple(map(int, shape))
    if pad_value is not None and real_shape is None:
        real_shape = shape
    if real_shape is not None and pad_value is None:
        pad_value = 0
    load_shape = shape if real_shape is None else verify_real_shape(real_shape, shape)
    region, head = clip(offsets, load_shape, src.shape)
    if real_shape is None:
        data = np.empty(shape, dtype=src.data.dtype)
    else:
        data = np.full(shape, pad_value, dtype=src.data.dtype)
    data[head] = src.data[region]
    return LocalTensor(data)


@override
def copy_out(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[Any], *,
             real_shape: Optional[Iterable[int]] = None) -> None:
    check_type("src", src, LocalTensor)
    check_type("dst", dst, GlobalTensor)
    store_shape = src.shape if real_shape is None else verify_real_shape(real_shape, src.shape)
    region, head = clip(offsets, store_shape, dst.shape)
    dst.data[region] = src.data[head]


@override
def copy(src: LocalTensor, offsets: Iterable[Any], shape: Iterable[int], location: Any = None) -> LocalTensor:
    del location
    check_type("src", src, LocalTensor)
    shape = tuple(map(int, shape))
    region, head = clip(offsets, shape, src.shape)
    data = np.empty(shape, dtype=src.data.dtype)
    data[head] = src.data[region]
    return LocalTensor(data)


@override
def zeros_acc(shape: Iterable[int], dtype: DataType) -> LocalTensor:
    return LocalTensor(np.zeros(tuple(map(int, shape)), dtype=dtype_to_numpy(dtype)))


@override
def matmul_acc(acc: LocalTensor, input: LocalTensor, other: LocalTensor) -> None:
    check_type("acc", acc, LocalTensor)
    check_type("input", input, LocalTensor)
    check_type("other", other, LocalTensor)
    acc.data += input.data.astype(acc.data.dtype) @ other.data.astype(acc.data.dtype)


@override
def exp(input: LocalTensor) -> LocalTensor:
    return LocalTensor(np.exp(input.data))


def reduce_impl(op: Callable[..., Any], input: LocalTensor, dims: Tuple[int, ...], keep_dims: bool) -> Any:
    check_type("input", input, LocalTensor)
    if not dims:
        value = op(input.data)
        return LocalTensor(np.full([1] * input.rank, value, dtype=input.data.dtype)) if keep_dims else value
    return LocalTensor(op(input.data, axis=dims, keepdims=keep_dims))


@override
def reduce_max(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Any:
    return reduce_impl(np.max, input, dims, keep_dims)


@override
def reduce_sum(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Any:
    return reduce_impl(np.sum, input, dims, keep_dims)
