from typing import Tuple

import asc2
import pytest
import torch

SIZE = 32
TILE_SIZE = 128
SINGLE_CORE = 1
MULTI_CORE = 16

SELECT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32]
SELECT_SCALAR_SOURCE_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
CONDITION_PATTERNS = ["all_true", "all_false", "alternating", "first_true", "last_true"]


def create_tensor(dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_floating_point:
        return torch.rand(SIZE, dtype=dtype, device="cpu")
    if dtype.is_signed:
        return torch.randint(-100, 100, (SIZE, ), dtype=dtype, device="cpu")


@asc2.jit(always_compile=True)
def where_kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, z_ptr: asc2.GlobalAddress, op: asc2.ConstExpr):
    x = asc2.tensor(x_ptr, [SIZE])
    y = asc2.tensor(y_ptr, [SIZE])
    z = asc2.tensor(z_ptr, [SIZE])
    xt = asc2.load(x, [SIZE], offsets=[0])
    yt = asc2.load(y, [SIZE], offsets=[0])
    zt = asc2.where(op(xt, yt), xt, yt)
    asc2.store(zt, z, offsets=[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32])
@pytest.mark.parametrize("asc_op, torch_op", [
    (asc2.equal, torch.eq),
    (asc2.not_equal, torch.ne),
    (asc2.greater, torch.gt),
    (asc2.greater_equal, torch.ge),
    (asc2.less, torch.lt),
    (asc2.less_equal, torch.le),
])
def test_where_ops(backend, platform, device_id, require_c310, asc_op, torch_op, dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310(platform)
    asc2.set_platform(backend, platform, device_id, check=False)
    x = create_tensor(dtype)
    y = create_tensor(dtype)
    result = torch.zeros_like(x)
    where_kernel[1](x, y, result, asc_op)
    expected = torch.where(torch_op(x, y), x, y)
    torch.testing.assert_close(result, expected)


@asc2.jit(always_compile=True)
def where_scalar_kernel(x_ptr: asc2.GlobalAddress, scalar, z_ptr: asc2.GlobalAddress, op: asc2.ConstExpr):
    x = asc2.tensor(x_ptr, [SIZE])
    z = asc2.tensor(z_ptr, [SIZE])
    xt = asc2.load(x, [SIZE], offsets=[0])
    zt = asc2.where(op(xt, scalar), asc2.number(0.0, x_ptr.dtype), asc2.number(1.0, x_ptr.dtype))
    asc2.store(zt, z, offsets=[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32])
@pytest.mark.parametrize("asc_op, torch_op", [
    (asc2.equal, torch.eq),
    (asc2.not_equal, torch.ne),
    (asc2.greater, torch.gt),
    (asc2.greater_equal, torch.ge),
    (asc2.less, torch.lt),
    (asc2.less_equal, torch.le),
])
def test_where_scalar_ops(backend, platform, device_id, require_c310, asc_op, torch_op, dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310(platform)
    asc2.set_platform(backend, platform, device_id, check=False)
    x = create_tensor(dtype)
    y = torch.tensor(0 if dtype.is_signed else 0.5, dtype=dtype)
    result = torch.zeros_like(x)
    where_scalar_kernel[1](x, y, result, asc_op)
    expected = torch.where(torch_op(x, y), torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
    torch.testing.assert_close(result, expected)


@asc2.jit(always_compile=True)
def where_with_cond_kernel(cond_ptr: asc2.GlobalAddress, x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress,
                           out_ptr: asc2.GlobalAddress, size: asc2.ConstExpr[int], tile_size: asc2.ConstExpr[int],
                           tile_per_block: asc2.ConstExpr[int]):
    cond_gm = asc2.tensor(cond_ptr, [size])
    x_gm = asc2.tensor(x_ptr, [size])
    y_gm = asc2.tensor(y_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        c = asc2.load(cond_gm, [tile_size], offsets=[tile_offset])
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        y = asc2.load(y_gm, [tile_size], offsets=[tile_offset])
        out = asc2.where(c != 0, x, y)
        asc2.store(out, out_gm, offsets=[tile_offset])


@asc2.jit(always_compile=True)
def where_scalar_source_kernel(x_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, size: asc2.ConstExpr[int],
                               tile_size: asc2.ConstExpr[int], tile_per_block: asc2.ConstExpr[int],
                               scalar_value: asc2.ConstExpr, scalar_on_true: asc2.ConstExpr):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        scalar = asc2.number(scalar_value, x_ptr.dtype)
        if scalar_on_true:
            out = asc2.where(x > 0, scalar, x)
        else:
            out = asc2.where(x > 0, x, scalar)
        asc2.store(out, out_gm, offsets=[tile_offset])


def check_dtype(platform: asc2.Platform, dtype: torch.dtype, require_c310):
    if dtype not in (torch.float16, torch.float32):
        require_c310(platform)


def make_data(size: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    x_i = torch.randint(-8, 9, (size, ), generator=generator, dtype=torch.int32)
    y_i = torch.randint(-9, 10, (size, ), generator=generator, dtype=torch.int32)
    # Force x == y at every 7th position so the test also exercises the case
    # where both arms agree: the kernel output must equal that shared value
    # regardless of the condition tile.
    overlap = torch.arange(size, dtype=torch.int32) % 7 == 0
    x_i = torch.where(overlap, y_i, x_i)
    if dtype.is_floating_point:
        return x_i.to(torch.float32).to(dtype), y_i.to(torch.float32).to(dtype)
    return x_i.to(dtype), y_i.to(dtype)


def make_condition(size: int, pattern: str, dtype: torch.dtype) -> torch.Tensor:
    if pattern == "all_true":
        cond = torch.ones(size, dtype=torch.int32, device="cpu")
    elif pattern == "all_false":
        cond = torch.zeros(size, dtype=torch.int32, device="cpu")
    elif pattern == "alternating":
        cond = torch.arange(size, dtype=torch.int32, device="cpu") % 2
    elif pattern == "first_true":
        cond = torch.zeros(size, dtype=torch.int32, device="cpu")
        cond[0] = 1
    elif pattern == "last_true":
        cond = torch.zeros(size, dtype=torch.int32, device="cpu")
        cond[size - 1] = 1
    else:
        raise ValueError(f"Unknown condition pattern: {pattern}")
    if dtype.is_floating_point:
        return cond.to(torch.float32).to(dtype)
    return cond.to(dtype)


def where_with_cond_launch(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, core_num: int = SINGLE_CORE,
                           tile_size: int = TILE_SIZE) -> torch.Tensor:
    out = torch.empty_like(x)
    size = out.numel()
    num_tiles = asc2.ceildiv(size, tile_size)
    where_with_cond_kernel[core_num](cond, x, y, out, size, tile_size, asc2.ceildiv(num_tiles, core_num))
    return out


def where_scalar_source_launch(x: torch.Tensor, *, scalar_value: int, scalar_on_true: bool, core_num: int = SINGLE_CORE,
                               tile_size: int = TILE_SIZE) -> torch.Tensor:
    out = torch.empty_like(x)
    size = out.numel()
    num_tiles = asc2.ceildiv(size, tile_size)
    where_scalar_source_kernel[core_num](x, out, size, tile_size, asc2.ceildiv(num_tiles, core_num), scalar_value,
                                         scalar_on_true)
    return out


@pytest.mark.parametrize("dtype", SELECT_DTYPES, ids=str)
def test_where_condition_dtypes(backend: asc2.Backend, platform: asc2.Platform, device_id: int, require_c310,
                                dtype: torch.dtype):
    check_dtype(platform, dtype, require_c310)
    asc2.set_platform(backend, platform, device_id)
    size = TILE_SIZE
    x, y = make_data(size, dtype)
    cond = make_condition(size, "alternating", dtype)
    out = where_with_cond_launch(cond, x, y)
    expected = torch.where(cond.bool(), x, y)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("pattern", CONDITION_PATTERNS)
def test_where_condition_patterns(backend: asc2.Backend, platform: asc2.Platform, device_id: int, pattern: str):
    asc2.set_platform(backend, platform, device_id)
    size = TILE_SIZE
    x, y = make_data(size, torch.float32)
    cond = make_condition(size, pattern, torch.float32)
    out = where_with_cond_launch(cond, x, y)
    expected = torch.where(cond.bool(), x, y)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("logical_size", [1, 2, 127, 128, 129, 255, 256])
def test_where_logical_border_prefixes(backend: asc2.Backend, platform: asc2.Platform, device_id: int,
                                       logical_size: int):
    asc2.set_platform(backend, platform, device_id)
    num_tiles = asc2.ceildiv(logical_size, TILE_SIZE)
    physical_size = TILE_SIZE * asc2.ceildiv(num_tiles, SINGLE_CORE) * SINGLE_CORE
    x, y = make_data(physical_size, torch.float32)
    cond = make_condition(physical_size, "alternating", torch.float32)
    out = where_with_cond_launch(cond, x, y)
    expected = torch.where(cond[:logical_size].bool(), x[:logical_size], y[:logical_size])
    torch.testing.assert_close(out[:logical_size], expected)


def test_where_multicore_unrolled(backend: asc2.Backend, platform: asc2.Platform, device_id: int):
    asc2.set_platform(backend, platform, device_id)
    size = TILE_SIZE * MULTI_CORE * 2
    x, y = make_data(size, torch.float32)
    cond = make_condition(size, "alternating", torch.float32)
    out = where_with_cond_launch(cond, x, y, core_num=MULTI_CORE)
    expected = torch.where(cond.bool(), x, y)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", SELECT_SCALAR_SOURCE_DTYPES, ids=str)
@pytest.mark.parametrize("scalar_on_true, scalar_value", [(False, -7), (True, 7)],
                         ids=["tensor_then_scalar", "scalar_then_tensor"])
def test_where_scalar_source_layouts(backend: asc2.Backend, platform: asc2.Platform, device_id: int, require_c310,
                                     scalar_on_true: bool, scalar_value: int, dtype: torch.dtype):
    check_dtype(platform, dtype, require_c310)
    asc2.set_platform(backend, platform, device_id)
    x, _ = make_data(TILE_SIZE, dtype)
    out = where_scalar_source_launch(x, scalar_value=scalar_value, scalar_on_true=scalar_on_true)
    scalar_tensor = torch.full_like(x, scalar_value)
    if scalar_on_true:
        expected = torch.where(x > 0, scalar_tensor, x)
    else:
        expected = torch.where(x > 0, x, scalar_tensor)
    torch.testing.assert_close(out, expected)
