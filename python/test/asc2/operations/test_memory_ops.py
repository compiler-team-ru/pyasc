import asc
from asc.runtime import config
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
    (2, [128, 128], [32, 32], None, [0, 0], True),
    (2, [128, 128], [32, 32], None, [96, 64], True),
    (2, [1024, 512], [128, 64], None, [256, 128], True),
    (1, [53], [16], None, [22], True),
    (2, [257, 511], [19, 24], None, [40, 64], True),

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
def set_platform(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform, check=False)


@asc2.jit(always_compile=True)
def kernel_static(x_ptr, y_ptr, z_ptr, tensor_shape: asc.ConstExpr, tile_shape: asc.ConstExpr, tile_id: asc.ConstExpr,
                  offsets: asc.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, tensor_shape), tile_shape, tile_id=tile_id, offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, tensor_shape), tile_shape, tile_id=tile_id, offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, tensor_shape), tile_id=tile_id, offsets=offsets)


@asc2.jit(always_compile=True)
def kernel_dynamic_1D(x_ptr, y_ptr, z_ptr, ts0, tile_shape: asc.ConstExpr, tile_id: asc.ConstExpr,
                      offsets: asc.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, [ts0]), tile_shape, tile_id=tile_id, offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, [ts0]), tile_shape, tile_id=tile_id, offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, [ts0]), tile_id=tile_id, offsets=offsets)


@asc2.jit(always_compile=True)
def kernel_dynamic_2D(x_ptr, y_ptr, z_ptr, ts0, ts1, tile_shape: asc.ConstExpr, tile_id: asc.ConstExpr,
                      offsets: asc.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, [ts0, ts1]), tile_shape, tile_id=tile_id, offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, [ts0, ts1]), tile_shape, tile_id=tile_id, offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, [ts0, ts1]), tile_id=tile_id, offsets=offsets)


@asc2.jit(always_compile=True)
def kernel_scalar_load_store(x_ptr, y_ptr, z_ptr, tensor_shape: asc.ConstExpr, offsets: asc.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, tensor_shape), offsets=offsets)
    yt = asc2.load(asc2.tensor(y_ptr, tensor_shape), offsets=offsets)
    zt = xt + yt
    asc2.store(zt, asc2.tensor(z_ptr, tensor_shape), offsets=offsets)


@pytest.mark.parametrize("dim, tensor_shape, tile_shape, tile_id, offsets, is_static", tests, ids=str)
def test_load_store(dim, tensor_shape, tile_shape, tile_id, offsets, is_static):
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
), ids=str)
def test_store_1elem_tile(tensor_shape, offsets):
    x = torch.randn(tensor_shape, dtype=torch.float32, device="cpu")
    y = torch.zeros_like(x)

    @asc2.jit(always_compile=True)
    def kernel(x_ptr, y_ptr, tensor_shape: asc.ConstExpr, offsets: asc.ConstExpr):
        x = asc2.tensor(x_ptr, tensor_shape)
        s = asc2.load(x, offsets=offsets)
        y = asc2.tensor(y_ptr, tensor_shape)
        asc2.store(asc2.full([1], s), y, offsets=offsets)

    kernel[1](x, y, tensor_shape, offsets)
    y_ref = y.clone()
    y_ref[offsets] = x[offsets]
    torch.testing.assert_close(y, y_ref)
