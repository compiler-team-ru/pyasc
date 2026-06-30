import asc2
import pytest
import torch

unary_ops = [
    (asc2.abs, torch.abs, [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]),
    (asc2.ceil, torch.ceil, [torch.float16, torch.float32]),
    (asc2.cos, torch.cos, [torch.float16, torch.float32]),
    (asc2.cosh, torch.cosh, [torch.float16, torch.float32]),
    (asc2.erf, torch.erf, [torch.float16, torch.float32]),
    (asc2.exp, torch.exp, [torch.float16, torch.float32]),
    (asc2.exp2, torch.exp2, [torch.float16, torch.float32]),
    (asc2.floor, torch.floor, [torch.float16, torch.float32]),
    (asc2.log, torch.log, [torch.float16, torch.float32]),
    (asc2.log2, torch.log2, [torch.float16, torch.float32]),
    (asc2.negative, torch.neg, [torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]),
    (asc2.relu, torch.relu, [torch.float16, torch.float32]),
    (asc2.rsqrt, torch.rsqrt, [torch.float16, torch.float32]),
    (asc2.sin, torch.sin, [torch.float16, torch.float32]),
    (asc2.sinh, torch.sinh, [torch.float16, torch.float32]),
    (asc2.sqrt, torch.sqrt, [torch.float16, torch.float32]),
    (asc2.tan, torch.tan, [torch.float16, torch.float32]),
    (asc2.tanh, torch.tanh, [torch.float16, torch.float32]),
]


@asc2.jit(always_compile=True)
def kernel(x_ptr: asc2.GlobalAddress, z_ptr: asc2.GlobalAddress, block_length: asc2.ConstExpr,
           op: asc2.ConstExpr) -> None:
    xt = asc2.load(asc2.global_tensor(x_ptr, [32]), [0], [block_length])
    zt = op(xt)
    asc2.store(zt, asc2.global_tensor(z_ptr, [32]), [0])


@pytest.mark.parametrize("asc_op, torch_op, dtype",
                         [(asc_op, torch_op, d) for asc_op, torch_op, dtypes in unary_ops for d in dtypes])
def test_unary_operations(require_c310, asc_op, torch_op, dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    def create_input(dtype: torch.dtype):
        if dtype.is_floating_point:
            return torch.randn((size, ), dtype=dtype).clamp(1, 100)
        elif dtype.is_signed:
            return torch.randint(1, 100, (size, ), dtype=dtype)

    size = 32
    x = create_input(dtype)
    z = torch.zeros(size, dtype=dtype)
    kernel[1](x, z, size, asc_op)
    torch.testing.assert_close(z, torch_op(x))
