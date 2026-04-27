import math

import asc
from asc.runtime import config
import asc2
import pytest
import torch

USE_CORE_NUM = 4

# asc_op, torch_op, tensor_shape, dtype
atomic_ops = [
    # asc2.atomic_add
    (asc2.atomic_add, torch.add, [128], torch.float32),
    (asc2.atomic_add, torch.add, [128], torch.float16),
    (asc2.atomic_add, torch.add, [128], torch.int32),
    # asc2.atomic_max
    (asc2.atomic_max, torch.maximum, [128], torch.float32),
    (asc2.atomic_max, torch.maximum, [128], torch.float16),
    (asc2.atomic_max, torch.maximum, [128], torch.int32),
    # asc2.atomic_min
    (asc2.atomic_min, torch.minimum, [128], torch.float32),
    (asc2.atomic_min, torch.minimum, [128], torch.float16),
    (asc2.atomic_min, torch.minimum, [128], torch.int32),
]


@pytest.fixture(autouse=True)
def set_platform(backend: config.Backend, platform: config.Platform, device_id: int):
    config.set_platform(backend, platform, device_id, check=False)


@asc2.jit(always_compile=True)
def kernel(x_ptr: asc.GlobalAddress, z_ptr: asc.GlobalAddress, tensor_shape: asc.ConstExpr, tile_length: asc.ConstExpr,
           op: asc.ConstExpr):
    offset_x = asc.get_block_idx() * tile_length
    xt = asc2.load(asc2.tensor(x_ptr, tensor_shape), [tile_length], offsets=[offset_x])
    xt += asc2.full_like(xt, 10)  # temporary tile to keep TQue synchronization valid
    op(xt, asc2.tensor(z_ptr, [tile_length]), offsets=[0])


@pytest.mark.parametrize("asc_op, torch_op, tensor_shape, dtype",
                         [(asc_op, torch_op, tensor_shape, dtype)
                          for asc_op, torch_op, tensor_shape, dtype in atomic_ops])
def test_atomic_op(asc_op, torch_op, tensor_shape, dtype):

    def create_input(shape):
        if dtype == torch.float32:
            res = torch.randn(tuple(shape), dtype=dtype, device=device)
            res = torch.clamp(res, 1, 100)
        else:
            res = torch.randint(1, 100, tuple(shape), dtype=dtype, device=device)
        return res

    size = math.prod(tensor_shape)
    tile_length = size // USE_CORE_NUM
    device = "cpu"
    x = create_input(tensor_shape)
    z = create_input([tile_length])
    torch_z = z.clone()
    kernel[USE_CORE_NUM](x, z, tensor_shape, tile_length, asc_op)
    expected_z = torch_z
    for i in range(USE_CORE_NUM):
        x_block = x[i * tile_length:(i + 1) * tile_length] + 10
        expected_z = torch_op(expected_z, x_block)
    torch.testing.assert_close(z, expected_z, atol=1e-3, rtol=1e-3)
