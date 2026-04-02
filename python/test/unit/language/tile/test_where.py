import pytest
import torch

import asc
from asc.runtime import config
import asc2

SIZE = 32


@pytest.fixture(autouse=True)
def set_platform(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform, check=False)


def create_tensor(dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_floating_point:
        return torch.rand(SIZE, dtype=dtype, device="cpu")
    if dtype.is_signed:
        return torch.randint(-100, 100, SIZE, dtype=dtype, device="cpu")


@asc2.jit(always_compile=True)
def where_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, z_ptr: asc.GlobalAddress, op: asc.ConstExpr):
    x = asc2.tensor(x_ptr, [SIZE])
    y = asc2.tensor(y_ptr, [SIZE])
    z = asc2.tensor(z_ptr, [SIZE])
    xt = asc2.load(x, [SIZE], offsets=[0])
    yt = asc2.load(y, [SIZE], offsets=[0])
    zt = asc2.where(op(xt, yt), xt, yt)
    asc2.store(zt, z, offsets=[0])


def where_launch(x: torch.Tensor, y: torch.Tensor, op) -> torch.Tensor:
    z = torch.zeros_like(x)
    where_kernel[1](x, y, z, op)
    return z


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)  # TODO: enable int types
@pytest.mark.parametrize("asc_op, torch_op", [
    (asc2.equal, torch.eq),
    (asc2.not_equal, torch.ne),
    (asc2.greater, torch.gt),
    (asc2.greater_equal, torch.ge),
    (asc2.less, torch.lt),
    (asc2.less_equal, torch.le),
])
def test_where_ops(asc_op, torch_op, dtype):
    x = create_tensor(dtype)
    y = create_tensor(dtype)
    result = where_launch(x, y, asc_op)
    expected = torch.where(torch_op(x, y), x, y)
    torch.testing.assert_close(result, expected)


@asc2.jit(always_compile=True)
def where_scalar_kernel(x_ptr: asc.GlobalAddress, scalar, z_ptr: asc.GlobalAddress, op: asc.ConstExpr):
    x = asc2.tensor(x_ptr, [SIZE])
    z = asc2.tensor(z_ptr, [SIZE])
    xt = asc2.load(x, [SIZE], offsets=[0])
    zt = asc2.where(op(xt, scalar), asc.number(0.0, asc.float16), xt)
    asc2.store(zt, z, offsets=[0])


def where_scalar_launch(x: torch.Tensor, y, op) -> torch.Tensor:
    z = torch.zeros_like(x)
    where_scalar_kernel[1](x, y, z, op)
    return z


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)  # TODO: enable int types
@pytest.mark.parametrize("asc_op, torch_op", [
    (asc2.equal, torch.eq),
    (asc2.not_equal, torch.ne),
    (asc2.greater, torch.gt),
    (asc2.greater_equal, torch.ge),
    (asc2.less, torch.lt),
    (asc2.less_equal, torch.le),
])
def test_where_and_scalar_ops(asc_op, torch_op, dtype):
    x = create_tensor(dtype)
    y = 0.5
    result = where_scalar_launch(x, y, asc_op)
    expected = torch.where(torch_op(x, y), 0, x)
    torch.testing.assert_close(result, expected)
