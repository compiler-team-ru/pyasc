import pytest
import torch

import asc
from asc.runtime import config
import asc2

SIZE = 32
USE_CORE_NUM = 1


@asc2.jit
def where_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, z_ptr: asc.GlobalAddress,
                 block_length: asc.ConstExpr, op: asc.ConstExpr):
    x = asc2.tensor(x_ptr, [SIZE])
    y = asc2.tensor(y_ptr, [SIZE])
    z = asc2.tensor(z_ptr, [SIZE])
    xt = asc2.load(x, [block_length * asc.get_block_idx()], [block_length])
    yt = asc2.load(y, [block_length * asc.get_block_idx()], [block_length])
    zt = asc2.where(op(xt, yt), xt, yt)
    asc2.store(zt, z, [block_length * asc.get_block_idx()])


def where_launch(x: torch.Tensor, y: torch.Tensor, op) -> torch.Tensor:
    z = torch.zeros_like(x)
    total_length = z.numel()
    block_length = total_length // USE_CORE_NUM
    where_kernel[USE_CORE_NUM](x, y, z, block_length, op)
    return z


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("asc_op, torch_op", [
    (asc2.equal, torch.eq),
    (asc2.not_equal, torch.ne),
    (asc2.greater, torch.gt),
    (asc2.greater_equal, torch.ge),
    (asc2.less, torch.lt),
    (asc2.less_equal, torch.le),
])
def test_where_ops(asc_op, torch_op, dtype):
    config.set_platform(config.Backend.Model, check=False)

    x = torch.rand(SIZE, dtype=dtype)
    y = torch.rand(SIZE, dtype=dtype)

    result = where_launch(x, y, asc_op)
    expected = torch.where(torch_op(x, y), x, y)
    torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
