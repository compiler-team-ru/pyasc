import asc2
import pytest
import torch

# dim, tensor_shape, tile_shape, tile_id, offsets, is_static
tests = [
    # STATIC
    # tile_id
    (1, [64], [16], [0], None, True),
    (1, [64], [16], [3], None, True),
    (2, [128, 128], [32, 32], [0, 0], None, True),
    (2, [128, 128], [32, 32], [3, 3], None, True),
    (2, [1024, 512], [128, 64], [1, 2], None, True),
    (1, [37], [8], [3], None, True),
    (2, [123, 456], [13, 16], [2, 5], None, True),

    # offsets
    (1, [64], [16], None, [0], True),
    (1, [64], [16], None, [48], True),
    (2, [128, 128], [32, 32], None, [96, 64], True),
    (1, [53], [16], None, [22], True),
    (2, [257, 511], [19, 24], None, [40, 64], True),
    (2, [8, 7], [8, 8], None, [0, 0], True),
    (2, [8, 15], [8, 8], None, [0, 0], True),
    (2, [16, 7], [8, 8], None, [0, 0], True),
    (2, [16, 15], [8, 8], None, [0, 0], True),
    (2, [7, 8], [8, 8], None, [0, 0], True),
    (2, [7, 16], [8, 8], None, [0, 0], True),
    (2, [17, 8], [8, 8], None, [0, 0], True),
    (2, [17, 16], [8, 8], None, [0, 0], True),
    (2, [17, 7], [8, 8], None, [0, 0], True),
    (2, [17, 19], [8, 8], None, [0, 0], True),

    # DYNAMIC
    # tile_id
    (1, [32], [8], [1], None, False),
    (1, [32], [8], [3], None, False),
    (2, [16, 2048], [4, 512], [0, 3], None, False),
    (2, [16, 2048], [8, 256], [1, 7], None, False),
    (2, [512, 512], [64, 64], [4, 4], None, False),
    (1, [99], [16], [5], None, False),
    (2, [1000, 1000], [33, 40], [1, 1], None, False),

    # offsets
    (1, [32], [8], None, [16], False),
    (1, [32], [8], None, [24], False),
    (2, [16, 2048], [4, 512], None, [8, 1024], False),
    (2, [16, 2048], [12, 1024], None, [4, 0], False),
    (2, [512, 512], [64, 64], None, [128, 256], False),
    (1, [77], [24], None, [48], False),
    (2, [150, 300], [21, 40], None, [10, 20], False),

    # Scalar load, store tests
    (1, [32], None, None, [0], True),
    (2, [32, 32], None, None, [0, 0], True),
    (1, [1024], None, None, [0], True),
    (2, [512, 512], None, None, [0, 0], True),
]


@pytest.fixture(autouse=True)
def set_platform(backend: asc2.Backend, platform: asc2.Platform, device_id: int):
    asc2.set_platform(backend, platform, device_id, check=False)


@asc2.jit(always_compile=True)
def kernel_static(x_ptr, y_ptr, z_ptr, tensor_shape: asc2.ConstExpr, tile_shape: asc2.ConstExpr,
                  tile_id: asc2.ConstExpr, offsets: asc2.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, tensor_shape), tile_shape, tile_id=tile_id, offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, tensor_shape), tile_shape, tile_id=tile_id, offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, tensor_shape), tile_id=tile_id, offsets=offsets)


@asc2.jit(always_compile=True)
def kernel_dynamic_1D(x_ptr, y_ptr, z_ptr, ts0, tile_shape: asc2.ConstExpr, tile_id: asc2.ConstExpr,
                      offsets: asc2.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, [ts0]), tile_shape, tile_id=tile_id, offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, [ts0]), tile_shape, tile_id=tile_id, offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, [ts0]), tile_id=tile_id, offsets=offsets)


@asc2.jit(always_compile=True)
def kernel_dynamic_2D(x_ptr, y_ptr, z_ptr, ts0, ts1, tile_shape: asc2.ConstExpr, tile_id: asc2.ConstExpr,
                      offsets: asc2.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, [ts0, ts1]), tile_shape, tile_id=tile_id, offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, [ts0, ts1]), tile_shape, tile_id=tile_id, offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, [ts0, ts1]), tile_id=tile_id, offsets=offsets)


@asc2.jit(always_compile=True)
def kernel_scalar_load_store(x_ptr, y_ptr, z_ptr, tensor_shape: asc2.ConstExpr, offsets: asc2.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, tensor_shape), offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, tensor_shape), offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, tensor_shape), offsets=offsets)


@pytest.mark.parametrize("dim, tensor_shape, tile_shape, tile_id, offsets, is_static", tests)
def test_load_store(platform, require_c310, dim, tensor_shape, tile_shape, tile_id, offsets, is_static):
    if dim == 2 and not is_static:
        require_c310(platform)
    x, y = [torch.randn(tensor_shape) for _ in range(2)]
    device = "cpu"
    z = torch.zeros(tensor_shape, dtype=torch.float32, device=device)
    if is_static:
        if tile_shape is None:
            kernel_scalar_load_store[1](x, y, z, tensor_shape, offsets)
        else:
            kernel_static[1](x, y, z, tensor_shape, tile_shape, tile_id, offsets)
    else:
        if dim == 1:
            kernel_dynamic_1D[1](x, y, z, tensor_shape[0], tile_shape, tile_id, offsets)
        else:
            kernel_dynamic_2D[1](x, y, z, tensor_shape[0], tensor_shape[1], tile_shape, tile_id, offsets)
    if tile_id is not None:
        actual_offsets = [i * s for i, s in zip(tile_id, tile_shape)]
    else:
        actual_offsets = offsets
    if tile_shape is not None:
        slices = tuple(slice(off, off + size) for off, size in zip(actual_offsets, tile_shape))
    else:
        slices = tuple(actual_offsets)
    z_expected = torch.zeros_like(z)
    z_expected[slices] = x[slices] + y[slices]
    torch.testing.assert_close(z, z_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("tensor_shape, offsets", (
    ((16, ), (0, )),
    ((16, ), (7, )),
    ((16, 16), (0, 0)),
    ((16, 16), (7, 7)),
))
def test_store_1elem_tile(tensor_shape, offsets):
    x = torch.randn(tensor_shape, dtype=torch.float32, device="cpu")
    y = torch.zeros_like(x)

    @asc2.jit(always_compile=True)
    def kernel(x_ptr, y_ptr, tensor_shape: asc2.ConstExpr, offsets: asc2.ConstExpr):
        x = asc2.tensor(x_ptr, tensor_shape)
        s = asc2.load(x, offsets=offsets)
        y = asc2.tensor(y_ptr, tensor_shape)
        asc2.store(asc2.full([1], s), y, offsets=offsets)

    kernel[1](x, y, tensor_shape, offsets)
    y_ref = y.clone()
    y_ref[offsets] = x[offsets]
    torch.testing.assert_close(y, y_ref)


@asc2.jit(always_compile=True)
def kernel_load_padding(x_ptr, out_ptr, input_shape: asc2.ConstExpr, tile_shape: asc2.ConstExpr,
                        offsets: asc2.ConstExpr, pad_value: asc2.ConstExpr) -> None:
    x_gm = asc2.tensor(x_ptr, input_shape)
    out_gm = asc2.tensor(out_ptr, tile_shape)
    tile = asc2.load(x_gm, shape=tile_shape, offsets=offsets, pad_value=pad_value)
    asc2.store(tile, out_gm, offsets=[0, 0])


@pytest.mark.parametrize(
    "input_shape, tile_shape, offsets",
    (
        ([16, 16], [8, 8], [0, 0]),
        ([16, 4], [8, 16], [0, 0]),
        ([12, 12], [24, 16], [0, 0]),
        ([9, 9], [8, 8], [5, 4]),
    ),
)
def test_load_padding(platform, require_c310, input_shape, tile_shape, offsets):
    require_c310(platform)
    pad_value = -1000.0
    x = torch.arange(1, input_shape[0] * input_shape[1] + 1, dtype=torch.float32, device="cpu").reshape(input_shape)
    out = torch.full(tile_shape, pad_value, dtype=torch.float32, device="cpu")
    kernel_load_padding[1](x, out, input_shape, tile_shape, offsets, pad_value)
    out_expected = torch.full(tile_shape, pad_value, dtype=torch.float32, device="cpu")
    row_start, col_start = offsets
    tile_rows, tile_cols = tile_shape
    src_rows, src_cols = input_shape
    available_rows = max(0, min(src_rows, row_start + tile_rows) - row_start)
    available_cols = max(0, min(src_cols, col_start + tile_cols) - col_start)
    real_rows, real_cols = available_rows, available_cols
    valid_rows = min(real_rows, input_shape[0] - row_start) if row_start < input_shape[0] else 0
    valid_cols = min(real_cols, input_shape[1] - col_start) if col_start < input_shape[1] else 0
    if valid_rows > 0 and valid_cols > 0:
        out_expected[0:valid_rows, 0:valid_cols] = x[row_start:row_start + valid_rows, col_start:col_start + valid_cols]
    torch.testing.assert_close(out, out_expected)
