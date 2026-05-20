import math

import asc
from asc.runtime import config
import asc2
import pytest
import torch


@pytest.fixture(autouse=True)
def set_platform(backend: config.Backend, platform: config.Platform, device_id: int, require_c310):
    require_c310(platform)
    config.set_platform(backend, platform, device_id, check=False)


@pytest.mark.parametrize("data_shape, load_shape, ub_shape, offsets", [
    [[16, 16], [16, 16], [16, 16], [0, 0]],
    [[16, 16], [8, 8], [16, 16], [0, 0]],
    [[16, 16], [8, 8], [16, 16], [3, 1]],
    [[32, 16], [32, 16], [32, 16], [0, 0]],
    [[128, 64], [31, 15], [32, 16], [1, 3]],
])
def test_transpose(data_shape, load_shape, ub_shape, offsets):
    count = math.prod(data_shape)
    input = torch.arange(0, count, dtype=torch.float32, device="cpu").reshape(data_shape)
    if not load_shape:
        load_shape = ub_shape
    result = torch.zeros([ub_shape[1], ub_shape[0]], dtype=torch.float32, device="cpu")

    @asc2.jit(always_compile=True)
    def kernel(input_ptr, result_ptr, data_shape: asc.ConstExpr, load_shape: asc.ConstExpr, ub_shape: asc.ConstExpr,
               offsets: asc.ConstExpr):
        g_input = asc2.tensor(input_ptr, data_shape)
        tile = asc2.load(g_input, offsets=offsets, shape=ub_shape, real_shape=load_shape)
        g_output = asc2.tensor(result_ptr, [ub_shape[1], ub_shape[0]])
        asc2.store(tile.transpose(), g_output, offsets=[0, 0], real_shape=[load_shape[1], load_shape[0]])

    kernel[1](input, result, data_shape, load_shape, ub_shape, offsets)
    expected = input[offsets[0]:offsets[0] + load_shape[0], offsets[1]:offsets[1] + load_shape[1]].transpose(0, 1)
    result_filled = result[:load_shape[1], :load_shape[0]]
    torch.testing.assert_close(result_filled, expected)
