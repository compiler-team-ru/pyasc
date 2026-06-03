import math

import asc2
import pytest
import torch


@pytest.mark.parametrize("data_shape, load_shape, ub_shape, offsets", [
    [[16, 16], [16, 16], [16, 16], [0, 0]],
    [[16, 16], [8, 8], [16, 16], [0, 0]],
    [[16, 16], [8, 8], [16, 16], [3, 1]],
    [[32, 16], [32, 16], [32, 16], [0, 0]],
    [[128, 64], [31, 15], [32, 16], [1, 3]],
    [[64, 32], [64, 32], [64, 32], [0, 0]],
    [[128, 64], [31, 31], [32, 32], [1, 3]],
])
@pytest.mark.parametrize(
    "dtype",
    (torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_transpose(require_c310, data_shape, load_shape, ub_shape, offsets, dtype: torch.dtype):
    require_c310()
    if ub_shape[-1] % (32 // dtype.itemsize) != 0:
        pytest.skip("data is not 32 byte aligned")
    count = math.prod(data_shape)
    input = torch.arange(0, count, dtype=dtype).reshape(data_shape)
    if not load_shape:
        load_shape = ub_shape
    result = torch.zeros([ub_shape[1], ub_shape[0]], dtype=dtype)

    @asc2.jit(always_compile=True)
    def kernel(input_ptr, result_ptr, data_shape: asc2.ConstExpr, load_shape: asc2.ConstExpr, ub_shape: asc2.ConstExpr,
               offsets: asc2.ConstExpr):
        g_input = asc2.tensor(input_ptr, data_shape)
        tile = asc2.load(g_input, offsets=offsets, shape=ub_shape, real_shape=load_shape)
        g_output = asc2.tensor(result_ptr, [ub_shape[1], ub_shape[0]])
        asc2.store(tile.transpose(), g_output, offsets=[0, 0], real_shape=[load_shape[1], load_shape[0]])

    kernel[1](input, result, data_shape, load_shape, ub_shape, offsets)
    expected = input[offsets[0]:offsets[0] + load_shape[0], offsets[1]:offsets[1] + load_shape[1]].transpose(0, 1)
    result_filled = result[:load_shape[1], :load_shape[0]]
    torch.testing.assert_close(result_filled, expected)
