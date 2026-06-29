import asc2
import pytest
import torch


@pytest.fixture(autouse=True)
def require_c310_auto(require_c310):
    require_c310()


@asc2.jit(always_compile=True)
def softmax_1d_kernel(dst_ptr, src_ptr, length: asc2.ConstExpr) -> None:
    dst = asc2.tensor(dst_ptr, [length])
    src = asc2.tensor(src_ptr, [length])
    src_tile = asc2.load(src, [0], [length])
    dst_tile = asc2.softmax(src_tile)
    asc2.store(dst_tile, dst, [0])
    asc2.store(src_tile, src, [0])


@asc2.jit(always_compile=True)
def softmax_2d_kernel(dst_ptr, src_ptr, shape: asc2.ConstExpr) -> None:
    dst = asc2.tensor(dst_ptr, shape)
    src = asc2.tensor(src_ptr, shape)
    src_tile = asc2.load(src, [0, 0], shape)
    dst_tile = asc2.softmax(src_tile)
    asc2.store(dst_tile, dst, [0, 0])
    asc2.store(src_tile, src, [0, 0])


@pytest.mark.parametrize("dtype, shape", [
    (torch.float16, [1, 16]),
    (torch.float16, [4, 1024]),
    (torch.float32, [1, 8]),
    (torch.float32, [4, 256]),
])
def test_softmax_2d(dtype: torch.dtype, shape):
    src = torch.rand(shape, dtype=dtype)
    dst = torch.zeros_like(src)
    softmax_2d_kernel[1](dst, src, shape)
    torch.testing.assert_close(dst, torch.softmax(src, dim=1))


@pytest.mark.parametrize("dtype, length", [
    (torch.float16, 16),
    (torch.float16, 1024),
    (torch.float32, 8),
    (torch.float32, 1024),
    (torch.float32, 1040),
])
def test_softmax_1d(dtype: torch.dtype, length):
    src = torch.rand(length, dtype=dtype)
    dst = torch.zeros_like(src)
    softmax_1d_kernel[1](dst, src, length)
    torch.testing.assert_close(dst, torch.softmax(src, dim=0))
