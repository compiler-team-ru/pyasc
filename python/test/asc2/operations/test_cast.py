import asc2
import pytest
import torch

USE_CORE_NUM = 1
SIZE = 128


@asc2.jit(always_compile=True)
def cast_kernel(x_ptr, z_ptr, size: asc2.ConstExpr, dst_dtype: asc2.ConstExpr) -> None:
    x_gm = asc2.tensor(x_ptr, [size])
    z_gm = asc2.tensor(z_ptr, [size])
    tile = asc2.load(x_gm, [size], offsets=[0])
    casted = tile.to(dst_dtype)
    asc2.store(casted, z_gm, offsets=[0])


@pytest.mark.parametrize("dst_dtype, torch_src, torch_dst", [
    # float -> float
    (asc2.float16, torch.bfloat16, torch.float16),
    (asc2.float32, torch.bfloat16, torch.float32),
    (asc2.bfloat16, torch.float16, torch.bfloat16),
    (asc2.float32, torch.float16, torch.float32),
    (asc2.bfloat16, torch.float32, torch.bfloat16),
    (asc2.float16, torch.float32, torch.float16),
    # int -> float
    (asc2.float16, torch.int8, torch.float16),
    (asc2.float16, torch.int16, torch.float16),
    (asc2.float32, torch.int16, torch.float32),
    (asc2.float32, torch.int32, torch.float32),
    (asc2.float16, torch.int32, torch.float16),
    (asc2.float32, torch.int64, torch.float32),
    # float -> int
    (asc2.int32, torch.bfloat16, torch.int32),
    (asc2.int8, torch.float16, torch.int8),
    (asc2.int16, torch.float16, torch.int16),
    (asc2.int32, torch.float16, torch.int32),
    (asc2.int16, torch.float32, torch.int16),
    (asc2.int32, torch.float32, torch.int32),
    (asc2.int64, torch.float32, torch.int64),
    # int -> int
    (asc2.int16, torch.int8, torch.int16),
    (asc2.int32, torch.int8, torch.int32),
    (asc2.int32, torch.int16, torch.int32),
    (asc2.int16, torch.int32, torch.int16),
    (asc2.int64, torch.int32, torch.int64),
    (asc2.int32, torch.int64, torch.int32),
])
def test_cast(backend, platform, device_id, require_c310, dst_dtype, torch_src, torch_dst):
    if ((torch_src == torch.bfloat16 and torch_dst == torch.float16)
            or (torch_src == torch.float16 and torch_dst == torch.bfloat16)
            or (torch_src == torch.int8 and torch_dst == torch.int16)
            or (torch_src == torch.int8 and torch_dst == torch.int32)
            or (torch_src == torch.int16 and torch_dst == torch.int32)
            or (torch_src == torch.int32 and torch_dst == torch.float16)):
        require_c310(platform)
    asc2.set_platform(backend, platform, device_id, check=False)
    device = "cpu"

    def create_input(dtype: torch.dtype):
        if dtype.is_floating_point:
            return torch.randn(SIZE, dtype=dtype, device=device)
        if dtype.is_signed:
            return torch.randint(-100, 100, (SIZE, ), dtype=dtype, device=device)

    def get_expected(x, src_dtype: torch.dtype, dst_dtype: torch.dtype):
        if dst_dtype.is_floating_point:
            if dst_dtype == torch.float16:
                return x.to(torch.float16)
            if dst_dtype == torch.float32:
                return x.to(torch.float32)
            if dst_dtype == torch.bfloat16:
                return x.to(torch.bfloat16)
        if src_dtype.is_floating_point:
            rounded = torch.round(x)
            if dst_dtype == torch.int8:
                return torch.clamp(rounded, min=-128, max=127).to(torch.int8)
            if dst_dtype == torch.int16:
                return torch.clamp(rounded, min=-32768, max=32767).to(torch.int16)
            if dst_dtype == torch.int32:
                return rounded.to(torch.int32)
            if dst_dtype == torch.int64:
                return rounded.to(torch.int64)
        src_bits = src_dtype.itemsize * 8
        dst_bits = dst_dtype.itemsize * 8
        if src_bits < dst_bits:
            return x.to(dst_dtype)
        if src_bits > dst_bits:
            if dst_dtype == torch.int8:
                return torch.clamp(x.to(torch.int32), min=-128, max=127).to(torch.int8)
            if dst_dtype == torch.int16:
                return torch.clamp(x.to(torch.int32), min=-32768, max=32767).to(torch.int16)
            if dst_dtype == torch.int32:
                return torch.clamp(x.to(torch.int64), min=-2147483648, max=2147483647).to(torch.int32)
        return x.to(dst_dtype)

    x = create_input(torch_src)
    z = torch.zeros(SIZE, dtype=torch_dst, device=device)
    cast_kernel[USE_CORE_NUM](x, z, SIZE, dst_dtype)
    expected = get_expected(x, torch_src, torch_dst)
    if torch_dst.is_floating_point:
        atol = 1e-3 if torch_dst == torch.float16 else 1e-2 if torch_dst == torch.bfloat16 else 1e-6
        torch.testing.assert_close(z, expected, atol=atol, rtol=atol)
    else:
        torch.testing.assert_close(z, expected)
