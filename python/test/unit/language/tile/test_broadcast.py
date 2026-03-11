import pytest
import torch

import asc
from asc.runtime import config
import asc2

test_cases = [
    (torch.float32, [32], [50, 32], [0], [0, 0]),
    # (torch.float16, [1, 64], [100, 64], [0, 0], [0, 0]),
]


@pytest.fixture(autouse=True)
def set_platform(backend: config.Backend, platform: config.Platform):
    if platform != config.Platform.Ascend910_9599:
        pytest.skip("platform is not supported")
    config.set_platform(backend, platform, check=False)


@asc2.jit(always_compile=True)
def broadcast_kernel(x_ptr: asc.GlobalAddress, z_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr,
                     output_shape: asc.ConstExpr, input_offsets: asc.ConstExpr, output_offsets: asc.ConstExpr):
    x = asc2.tensor(x_ptr, input_shape)
    z = asc2.tensor(z_ptr, output_shape)
    xt = asc2.load(x, input_shape, offsets=input_offsets)
    zt = xt.broadcast_to(*output_shape)
    asc2.store(zt, z, offsets=output_offsets)


@pytest.mark.parametrize("dtype, input_shape, output_shape, input_offsets, output_offsets", test_cases)
def test_broadcast(dtype: torch.dtype, input_shape, output_shape, input_offsets, output_offsets):
    input = torch.rand(input_shape, dtype=dtype)
    result = torch.zeros(output_shape, dtype=dtype)
    broadcast_kernel[1](input, result, input_shape, output_shape, input_offsets, output_offsets)
    expected = input.broadcast_to(output_shape)
    torch.testing.assert_close(result, expected)
