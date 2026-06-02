import asc2
import pytest
import torch

supported_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_full_1d(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]
    fill_value = 7 if dtype.is_signed or dtype.is_floating_point else 7
    if dtype.is_floating_point:
        fill_value = 3.5

    @asc2.jit(always_compile=True)
    def kernel(out_ptr, shape: asc2.ConstExpr, value: asc2.ConstExpr):
        out_gm = asc2.tensor(out_ptr, shape)
        tile = asc2.full(shape, value, dtype=out_gm.dtype)
        asc2.store(tile, out_gm, offsets=[0])

    out = torch.zeros(shape, dtype=dtype)
    kernel[1](out, shape, fill_value)
    expected = torch.full(shape, fill_value, dtype=dtype)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_full_2d(dtype: torch.dtype):
    last_dim = 16 if dtype == torch.float16 else 8
    shape = [4, last_dim]
    fill_value = 2.5 if dtype.is_floating_point else 5

    @asc2.jit(always_compile=True)
    def kernel(out_ptr, shape: asc2.ConstExpr, value: asc2.ConstExpr):
        out_gm = asc2.tensor(out_ptr, shape)
        tile = asc2.full(shape, value, dtype=out_gm.dtype)
        asc2.store(tile, out_gm, offsets=[0, 0])

    out = torch.zeros(shape, dtype=dtype)
    kernel[1](out, shape, fill_value)
    expected = torch.full(shape, fill_value, dtype=dtype)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_full_like(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]
    fill_value = 3.5 if dtype.is_floating_point else 9

    @asc2.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asc2.ConstExpr, value: asc2.ConstExpr):
        in_gm = asc2.tensor(in_ptr, shape)
        out_gm = asc2.tensor(out_ptr, shape)
        src = asc2.load(in_gm, shape, offsets=[0])
        tile = asc2.full_like(src, value)
        asc2.store(tile, out_gm, offsets=[0])

    inp = torch.ones(shape, dtype=dtype)
    out = torch.zeros(shape, dtype=dtype)
    kernel[1](inp, out, shape, fill_value)
    expected = torch.full(shape, fill_value, dtype=dtype)
    torch.testing.assert_close(out, expected)


def test_full_like_2d():
    shape = [4, 8]

    @asc2.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asc2.ConstExpr):
        in_gm = asc2.tensor(in_ptr, shape)
        out_gm = asc2.tensor(out_ptr, shape)
        src = asc2.load(in_gm, shape, offsets=[0, 0])
        tile = asc2.full_like(src, 7)
        asc2.store(tile, out_gm, offsets=[0, 0])

    inp = torch.ones(shape, dtype=torch.int32)
    out = torch.zeros(shape, dtype=torch.int32)
    kernel[1](inp, out, shape)
    expected = torch.full(shape, 7, dtype=torch.int32)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_zeros_1d(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]

    @asc2.jit(always_compile=True)
    def kernel(out_ptr, shape: asc2.ConstExpr):
        out_gm = asc2.tensor(out_ptr, shape)
        tile = asc2.zeros(shape, dtype=out_gm.dtype)
        asc2.store(tile, out_gm, offsets=[0])

    out = torch.ones(shape, dtype=dtype)
    kernel[1](out, shape)
    expected = torch.zeros(shape, dtype=dtype)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_zeros_2d(dtype: torch.dtype):
    last_dim = 16 if dtype == torch.float16 else 8
    shape = [4, last_dim]

    @asc2.jit(always_compile=True)
    def kernel(out_ptr, shape: asc2.ConstExpr):
        out_gm = asc2.tensor(out_ptr, shape)
        tile = asc2.zeros(shape, dtype=out_gm.dtype)
        asc2.store(tile, out_gm, offsets=[0, 0])

    out = torch.ones(shape, dtype=dtype)
    kernel[1](out, shape)
    expected = torch.zeros(shape, dtype=dtype)
    torch.testing.assert_close(out, expected)


def test_zeros_default_dtype():
    shape = [16]

    @asc2.jit(always_compile=True)
    def kernel(out_ptr, shape: asc2.ConstExpr):
        out_gm = asc2.tensor(out_ptr, shape)
        tile = asc2.zeros(shape)
        asc2.store(tile, out_gm, offsets=[0])

    out = torch.ones(shape, dtype=torch.int32)
    kernel[1](out, shape)
    expected = torch.zeros(shape, dtype=torch.int32)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_zeros_like(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]

    @asc2.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asc2.ConstExpr):
        in_gm = asc2.tensor(in_ptr, shape)
        out_gm = asc2.tensor(out_ptr, shape)
        src = asc2.load(in_gm, shape, offsets=[0])
        tile = asc2.zeros_like(src)
        asc2.store(tile, out_gm, offsets=[0])

    inp = torch.ones(shape, dtype=dtype)
    out = torch.ones(shape, dtype=dtype)
    kernel[1](inp, out, shape)
    expected = torch.zeros(shape, dtype=dtype)
    torch.testing.assert_close(out, expected)


def test_zeros_like_2d():
    shape = [4, 8]

    @asc2.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asc2.ConstExpr):
        in_gm = asc2.tensor(in_ptr, shape)
        out_gm = asc2.tensor(out_ptr, shape)
        src = asc2.load(in_gm, shape, offsets=[0, 0])
        tile = asc2.zeros_like(src)
        asc2.store(tile, out_gm, offsets=[0, 0])

    inp = torch.ones(shape, dtype=torch.float32)
    out = torch.ones(shape, dtype=torch.float32)
    kernel[1](inp, out, shape)
    expected = torch.zeros(shape, dtype=torch.float32)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_concat_two_tiles_1d(require_c310, dtype: torch.dtype):
    require_c310()
    shape_a = [16]
    shape_b = [16]
    out_shape = [32]

    @asc2.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, out_ptr, shape_a: asc2.ConstExpr, shape_b: asc2.ConstExpr, out_shape: asc2.ConstExpr):
        a_gm = asc2.tensor(a_ptr, shape_a)
        b_gm = asc2.tensor(b_ptr, shape_b)
        out_gm = asc2.tensor(out_ptr, out_shape)
        tile_a = asc2.load(a_gm, shape_a, offsets=[0])
        tile_b = asc2.load(b_gm, shape_b, offsets=[0])
        result = asc2.concat(tile_a, tile_b)
        asc2.store(result, out_gm, offsets=[0])

    a = torch.arange(16, dtype=dtype)
    b = torch.arange(16, 32, dtype=dtype)
    out = torch.zeros(out_shape, dtype=dtype)
    kernel[1](a, b, out, shape_a, shape_b, out_shape)
    expected = torch.cat([a, b])
    torch.testing.assert_close(out, expected)


def test_concat_three_tiles_1d(require_c310):
    require_c310()
    shape_a = [8]
    shape_b = [8]
    shape_c = [8]
    out_shape = [24]

    @asc2.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, shape_a: asc2.ConstExpr, shape_b: asc2.ConstExpr, shape_c: asc2.ConstExpr,
               out_shape: asc2.ConstExpr):
        a_gm = asc2.tensor(a_ptr, shape_a)
        b_gm = asc2.tensor(b_ptr, shape_b)
        c_gm = asc2.tensor(c_ptr, shape_c)
        out_gm = asc2.tensor(out_ptr, out_shape)
        tile_a = asc2.load(a_gm, shape_a, offsets=[0])
        tile_b = asc2.load(b_gm, shape_b, offsets=[0])
        tile_c = asc2.load(c_gm, shape_c, offsets=[0])
        result = asc2.concat(tile_a, tile_b, tile_c)
        asc2.store(result, out_gm, offsets=[0])

    a = torch.arange(8, dtype=torch.float32)
    b = torch.arange(8, 16, dtype=torch.float32)
    c = torch.arange(16, 24, dtype=torch.float32)
    out = torch.zeros(out_shape, dtype=torch.float32)
    kernel[1](a, b, c, out, shape_a, shape_b, shape_c, out_shape)
    expected = torch.cat([a, b, c])
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_concat_two_tiles_2d(require_c310, dtype: torch.dtype):
    require_c310()
    last_dim = 16 if dtype == torch.float16 else 8
    shape_a = [2, last_dim]
    shape_b = [2, last_dim]
    out_shape = [4, last_dim]

    @asc2.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, out_ptr, shape_a: asc2.ConstExpr, shape_b: asc2.ConstExpr, out_shape: asc2.ConstExpr):
        a_gm = asc2.tensor(a_ptr, shape_a)
        b_gm = asc2.tensor(b_ptr, shape_b)
        out_gm = asc2.tensor(out_ptr, out_shape)
        tile_a = asc2.load(a_gm, shape_a, offsets=[0, 0])
        tile_b = asc2.load(b_gm, shape_b, offsets=[0, 0])
        result = asc2.concat(tile_a, tile_b)
        asc2.store(result, out_gm, offsets=[0, 0])

    a = torch.arange(2 * last_dim, dtype=dtype).reshape(2, last_dim)
    b = torch.arange(2 * last_dim, 4 * last_dim, dtype=dtype).reshape(2, last_dim)
    out = torch.zeros(out_shape, dtype=dtype)
    kernel[1](a, b, out, shape_a, shape_b, out_shape)
    expected = torch.cat([a, b], dim=0)
    torch.testing.assert_close(out, expected)


def test_concat_different_first_dim(require_c310):
    require_c310()
    shape_a = [4]
    shape_b = [12]
    out_shape = [16]

    @asc2.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, out_ptr, shape_a: asc2.ConstExpr, shape_b: asc2.ConstExpr, out_shape: asc2.ConstExpr):
        a_gm = asc2.tensor(a_ptr, shape_a)
        b_gm = asc2.tensor(b_ptr, shape_b)
        out_gm = asc2.tensor(out_ptr, out_shape)
        tile_a = asc2.load(a_gm, shape_a, offsets=[0])
        tile_b = asc2.load(b_gm, shape_b, offsets=[0])
        result = asc2.concat(tile_a, tile_b)
        asc2.store(result, out_gm, offsets=[0])

    a = torch.arange(4, dtype=torch.float32)
    b = torch.arange(4, 16, dtype=torch.float32)
    out = torch.zeros(out_shape, dtype=torch.float32)
    kernel[1](a, b, out, shape_a, shape_b, out_shape)
    expected = torch.cat([a, b])
    torch.testing.assert_close(out, expected)
