import math

import asc2
import pytest
import torch

USE_CORE_NUM = 4


@asc2.jit(always_compile=True)
def kernel(x_ptr: asc2.GlobalAddress, z_ptr: asc2.GlobalAddress, tensor_shape: asc2.ConstExpr,
           tile_length: asc2.ConstExpr, op: asc2.ConstExpr):
    offset_x = asc2.block_idx() * tile_length
    xt = asc2.load(asc2.tensor(x_ptr, tensor_shape), [tile_length], offsets=[offset_x])
    xt += 10  # temporary tile to keep TQue synchronization valid
    op(xt, asc2.tensor(z_ptr, [tile_length]), offsets=[0])


@pytest.mark.parametrize("dtype", (torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("asc_op, torch_op", (
    (asc2.atomic_add, torch.add),
    (asc2.atomic_max, torch.maximum),
    (asc2.atomic_min, torch.minimum),
))
def test_atomic_op(require_c310, asc_op, torch_op, dtype):
    if dtype == torch.bfloat16:
        require_c310()  # due to use of addition in test kernel

    def create_input(shape):
        if dtype == torch.float32:
            res = torch.randn(tuple(shape), dtype=dtype)
            res = torch.clamp(res, 1, 100)
        else:
            res = torch.randint(1, 100, tuple(shape), dtype=dtype)
        return res

    tensor_shape = [128]
    size = math.prod(tensor_shape)
    tile_length = size // USE_CORE_NUM
    x = create_input(tensor_shape)
    z = create_input([tile_length])
    torch_z = z.clone()
    kernel[USE_CORE_NUM](x, z, tensor_shape, tile_length, asc_op)
    expected_z = torch_z
    for i in range(USE_CORE_NUM):
        x_block = x[i * tile_length:(i + 1) * tile_length] + 10
        expected_z = torch_op(expected_z, x_block)
    torch.testing.assert_close(z, expected_z)
