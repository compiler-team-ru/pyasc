import pytest
import torch

import asc
from asc.runtime import config
import asc2

SIZE = 32


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)  # TODO: enable int types
@pytest.mark.parametrize("asc_op, torch_op", [
    (asc2.equal, torch.eq),
    (asc2.not_equal, torch.ne),
    (asc2.greater, torch.gt),
    (asc2.greater_equal, torch.ge),
    (asc2.less, torch.lt),
    (asc2.less_equal, torch.le),
])
def test_where_ops(backend, platform, require_c310, asc_op, torch_op, dtype):
    if dtype == torch.bfloat16:
        require_c310(platform)
    config.set_platform(backend, platform, check=False)
    x = create_tensor(dtype)
    y = create_tensor(dtype)
    result = torch.zeros_like(x)
    where_kernel[1](x, y, result, asc_op)
    expected = torch.where(torch_op(x, y), x, y)
    torch.testing.assert_close(result, expected)


@asc2.jit(always_compile=True)
def where_scalar_kernel(x_ptr: asc.GlobalAddress, scalar, z_ptr: asc.GlobalAddress, op: asc.ConstExpr):
    x = asc2.tensor(x_ptr, [SIZE])
    z = asc2.tensor(z_ptr, [SIZE])
    xt = asc2.load(x, [SIZE], offsets=[0])
    zt = asc2.where(op(xt, scalar), asc.number(0.0, x_ptr.dtype), asc.number(1.0, x_ptr.dtype))
    asc2.store(zt, z, offsets=[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)  # TODO: enable int types
@pytest.mark.parametrize("asc_op, torch_op", [
    (asc2.equal, torch.eq),
    (asc2.not_equal, torch.ne),
    (asc2.greater, torch.gt),
    (asc2.greater_equal, torch.ge),
    (asc2.less, torch.lt),
    (asc2.less_equal, torch.le),
])
def test_where_and_scalar_ops(backend, platform, require_c310, asc_op, torch_op, dtype):
    if dtype == torch.bfloat16:
        require_c310(platform)
    config.set_platform(backend, platform, check=False)
    x = create_tensor(dtype)
    y = torch.tensor(0.5, dtype=dtype)
    result = torch.zeros_like(x)
    where_scalar_kernel[1](x, y, result, asc_op)
    expected = torch.where(torch_op(x, y), torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
    torch.testing.assert_close(result, expected)
