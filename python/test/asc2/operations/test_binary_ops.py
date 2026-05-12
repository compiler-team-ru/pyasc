import asc
from asc.runtime import config
import asc2
import ctypes
import pytest
import torch

# vector_vector, vector_scalar, scalar_vector
VV, VS, SV = "VV", "VS", "SV"
NO_MASK, COUNT_MASK, BIT_MASK = "NO_MASK", "COUNT_MASK", "BIT_MASK"
USE_CORE_NUM = 1

binary_ops = [
    (asc2.add, torch.add, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32], [NO_MASK, COUNT_MASK]),
    (asc2.div, torch.div, [VV, VS, SV], [torch.float32], [NO_MASK, COUNT_MASK]),
    (asc2.mul, torch.mul, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32], [NO_MASK, COUNT_MASK]),
    (asc2.sub, torch.sub, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32], [NO_MASK, COUNT_MASK]),
    (asc2.left_shift, torch.bitwise_left_shift, [VS], [torch.int32], [NO_MASK, COUNT_MASK]),
    (asc2.right_shift, torch.bitwise_right_shift, [VS], [torch.int32], [NO_MASK, COUNT_MASK]),
    (asc2.maximum, torch.maximum, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32], [NO_MASK, COUNT_MASK]),
    (asc2.minimum, torch.minimum, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32], [NO_MASK, COUNT_MASK]),
]


@asc2.jit(always_compile=True)
def kernel(x_ptr, y_ptr, z_ptr, block_length: asc.ConstExpr, fmt: asc.ConstExpr, op: asc.ConstExpr,
           mask_type: asc.ConstExpr, count: asc.ConstExpr, other: asc.ConstExpr, hibits: asc.ConstExpr,
           lowbits: asc.ConstExpr) -> None:
    if fmt == VV:
        xt = asc2.load(asc2.tensor(x_ptr, [32]), [block_length], offsets=[0])
        yt = asc2.load(asc2.tensor(y_ptr, [32]), [block_length], offsets=[0])
    elif fmt == VS:
        xt = asc2.load(asc2.tensor(x_ptr, [32]), [block_length], offsets=[0])
        yt = y_ptr
    elif fmt == SV:
        xt = x_ptr
        yt = asc2.load(asc2.tensor(y_ptr, [32]), [block_length], offsets=[0])

    if mask_type == NO_MASK:
        zt = op(xt, yt)
        asc2.store(zt, asc2.tensor(z_ptr, [32]), offsets=[0])
    elif mask_type == COUNT_MASK:
        with asc2.mask(count=count, other=other):
            zt = op(xt, yt)
            asc2.store(zt, asc2.tensor(z_ptr, [32]), offsets=[0])
    elif mask_type == BIT_MASK:
        with asc2.mask(bits=[hibits, lowbits], other=other):
            zt = op(xt, yt)
            asc2.store(zt, asc2.tensor(z_ptr, [32]), offsets=[0])


def handle_mask(gold, mask_type, count, other, hibits, lowbits) -> torch.Tensor:

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

    if mask_type == NO_MASK:
        return gold
    elif mask_type == COUNT_MASK:
        mask = torch.arange(max_elem_count) < count
    elif mask_type == BIT_MASK:
        hi = uint64_to_binary_tensor(hibits)
        lo = uint64_to_binary_tensor(lowbits)
        mask = torch.cat((hi, lo), dim=0)

    repeats = (size + max_elem_count - 1) // max_elem_count
    total_mask = torch.tile(mask, (repeats, ))[0:size]
    others = torch.full((size, ), other)
    return torch.where(total_mask, gold, others)


@pytest.mark.parametrize("asc_op, torch_op, fmt, dtype, mask_type",
                         [(asc_op, torch_op, f, d, m)
                          for asc_op, torch_op, fmts, dtypes, mask_types in binary_ops
                          for f in fmts
                          for d in dtypes
                          for m in mask_types])
def test_binary_operations(backend, platform, device_id, require_c310, asc_op, torch_op, fmt, dtype, mask_type):
    if dtype == torch.bfloat16:
        require_c310(platform)
    config.set_platform(backend, platform, device_id, check=False)

    def create_input(input_dtype: torch.dtype, is_vector: bool):
        if is_vector:
            if input_dtype.is_floating_point:
                return torch.randn((size, ), dtype=input_dtype, device=device).clamp(1, 100)
            elif input_dtype.is_signed:
                return torch.randint(1, 100, (size, ), dtype=input_dtype, device=device)
        else:
            return torch.tensor(2, dtype=input_dtype)

    size = 32
    block_length = size // USE_CORE_NUM
    device = "cpu"

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
    kernel[1](x, y, z, block_length, fmt, asc_op, mask_type, count, other, hi_number, low_number)
    if isinstance(x, (int, float)):
        x = torch.tensor(x, dtype=dtype)
    if isinstance(y, (int, float)):
        y = torch.tensor(y, dtype=dtype)

    gold = torch_op(x, y)
    gold = handle_mask(gold, mask_type, count, other, hibits, lowbits)

    if dtype == torch.float32:
        assert torch.allclose(z, gold, atol=1e-3), f"Failed {asc_op.__name__}"
    else:
        assert torch.equal(z, gold), f"Failed {asc_op.__name__}"
