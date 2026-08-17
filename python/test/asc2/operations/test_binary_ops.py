# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes

import asc2
import pytest
import torch

# vector_vector, vector_scalar, scalar_vector
VV, VS, SV = "VV", "VS", "SV"
NO_MASK, COUNT_MASK, BIT_MASK = "NO_MASK", "COUNT_MASK", "BIT_MASK"
all_formats = [VV, VS, SV]

binary_ops = [
    (asc2.add, torch.add, all_formats,
     [torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]),
    (asc2.div, torch.div, all_formats, [torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]),
    (asc2.mul, torch.mul, all_formats,
     [torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]),
    (asc2.sub, torch.sub, all_formats,
     [torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]),
    (asc2.bitwise_and, torch.bitwise_and, all_formats, [torch.int8, torch.int16, torch.int32, torch.int64]),
    (asc2.bitwise_or, torch.bitwise_or, all_formats, [torch.int8, torch.int16, torch.int32, torch.int64]),
    (asc2.bitwise_xor, torch.bitwise_xor, all_formats, [torch.int8, torch.int16, torch.int32, torch.int64]),
    (asc2.left_shift, torch.bitwise_left_shift, [VS], [torch.int16, torch.int32, torch.int64]),
    (asc2.right_shift, torch.bitwise_right_shift, [VS], [torch.int16, torch.int32, torch.int64]),
    (asc2.maximum, torch.maximum, all_formats,
     [torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]),
    (asc2.minimum, torch.minimum, all_formats,
     [torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]),
]


@asc2.jit(always_compile=True)
def kernel(x_ptr, y_ptr, z_ptr, block_length: asc2.ConstExpr, fmt: asc2.ConstExpr, op: asc2.ConstExpr,
           mask_type: asc2.ConstExpr, count: asc2.ConstExpr, other: asc2.ConstExpr, hibits: asc2.ConstExpr,
           lowbits: asc2.ConstExpr) -> None:
    if fmt == VV:
        xt = asc2.copy_in(asc2.global_tensor(x_ptr, [32]), [0], [block_length])
        yt = asc2.copy_in(asc2.global_tensor(y_ptr, [32]), [0], [block_length])
    elif fmt == VS:
        xt = asc2.copy_in(asc2.global_tensor(x_ptr, [32]), [0], [block_length])
        yt = asc2.copy_in(asc2.global_tensor(y_ptr, [1]), [0])
    elif fmt == SV:
        xt = asc2.copy_in(asc2.global_tensor(x_ptr, [1]), [0])
        yt = asc2.copy_in(asc2.global_tensor(y_ptr, [32]), [0], [block_length])

    if mask_type == NO_MASK:
        zt = op(xt, yt)
        asc2.copy_out(zt, asc2.global_tensor(z_ptr, [32]), [0])
    elif mask_type == COUNT_MASK:
        with asc2.mask(count=count, other=other):
            zt = op(xt, yt)
            asc2.copy_out(zt, asc2.global_tensor(z_ptr, [32]), [0])
    elif mask_type == BIT_MASK:
        with asc2.mask(bits=[hibits, lowbits], other=other):
            zt = op(xt, yt)
            asc2.copy_out(zt, asc2.global_tensor(z_ptr, [32]), [0])


def handle_mask(gold, mask_type, count, other, hibits, lowbits) -> torch.Tensor:
    if mask_type == NO_MASK:
        return gold

    def uint64_to_binary_tensor(value) -> torch.Tensor:
        binary_tensor = [bit == '1' for bit in bin(value)[2:]]
        pad_amount = (64 - len(binary_tensor))
        if 64 > len(binary_tensor):
            binary_tensor.extend([False] * pad_amount)
        return torch.tensor(binary_tensor[0:64])

    size, dtype = gold.size(0), gold.dtype
    # In bytes
    REPEAT_BLOCK_SIZE = 256
    max_elem_count = REPEAT_BLOCK_SIZE // dtype.itemsize
    if mask_type == COUNT_MASK:
        mask = torch.arange(max_elem_count) < count
    elif mask_type == BIT_MASK:
        hi = uint64_to_binary_tensor(hibits)
        lo = uint64_to_binary_tensor(lowbits)
        mask = torch.cat((hi, lo), dim=0)
    repeats = (size + max_elem_count - 1) // max_elem_count
    total_mask = torch.tile(mask, (repeats, ))[0:size]
    others = torch.full((size, ), other, dtype=dtype)
    return torch.where(total_mask, gold, others)


@pytest.mark.parametrize("mask_type", [NO_MASK, COUNT_MASK])
@pytest.mark.parametrize("asc_op, torch_op, fmt, dtype", [(asc_op, torch_op, f, d)
                                                          for asc_op, torch_op, fmts, dtypes in binary_ops
                                                          for f in fmts
                                                          for d in dtypes])
def test_binary_operations(require_c310, asc_op, torch_op, fmt, dtype, mask_type):
    if any((
            dtype in (torch.bfloat16, torch.int8, torch.int64),
            asc_op is asc2.div and not dtype.is_floating_point,
            asc_op in (asc2.bitwise_and, asc2.bitwise_or, asc2.bitwise_xor) and dtype != torch.int16,
    )):
        require_c310()
    if mask_type == COUNT_MASK and dtype in (torch.int8, torch.int64):
        pytest.skip("L0 API has incorrect assertion")

    size = 32

    def create_input(input_dtype: torch.dtype, is_vector: bool):
        if is_vector:
            if input_dtype.is_floating_point:
                return torch.randn((size, ), dtype=input_dtype).clamp(1, 100)
            elif input_dtype.is_signed:
                return torch.randint(1, 100, (size, ), dtype=input_dtype)
        else:
            return torch.tensor([2], dtype=input_dtype)

    if fmt == VV:
        x = create_input(dtype, True)
        y = create_input(dtype, True)
    elif fmt == VS:
        x = create_input(dtype, True)
        y = create_input(dtype, False)
    elif fmt == SV:
        x = create_input(dtype, False)
        y = create_input(dtype, True)
    count, other, hibits, lowbits = 0, 0, 0x0000000000000000, 0x0000000000000000
    if mask_type == NO_MASK:
        pass
    elif mask_type == COUNT_MASK:
        count, other = (23, 7)
    elif mask_type == BIT_MASK:
        hibits, lowbits = 0x0000000000000000, 0xFFFE000000000000
        other = 7
    z = torch.zeros(size, dtype=dtype)
    hi_number = ctypes.c_uint64(hibits).value
    low_number = ctypes.c_uint64(lowbits).value
    kernel[1](x, y, z, size, fmt, asc_op, mask_type, count, other, hi_number, low_number)
    if isinstance(x, (int, float)):
        x = torch.tensor(x, dtype=dtype)
    if isinstance(y, (int, float)):
        y = torch.tensor(y, dtype=dtype)
    gold = torch_op(x, y).to(dtype)
    gold = handle_mask(gold, mask_type, count, other, hibits, lowbits)
    torch.testing.assert_close(z, gold)
