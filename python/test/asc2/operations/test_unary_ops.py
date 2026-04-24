import asc
from asc.runtime import config
import asc2
import pytest
import torch

USE_CORE_NUM = 1

# asc2 op - torch op - (output type, input type)
unary_ops = [
    (asc2.abs, torch.abs, [(torch.float32, torch.float32)]),
    (asc2.ceil, torch.ceil, [(torch.float32, torch.float32)]),
    (asc2.cos, torch.cos, [(torch.float32, torch.float32)]),
    (asc2.cosh, torch.cosh, [(torch.float32, torch.float32)]),
    (asc2.erf, torch.erf, [(torch.float32, torch.float32)]),
    (asc2.exp, torch.exp, [(torch.float32, torch.float32)]),
    (asc2.exp2, torch.exp2, [(torch.float32, torch.float32)]),
    (asc2.floor, torch.floor, [(torch.float32, torch.float32)]),
    (asc2.log, torch.log, [(torch.float32, torch.float32)]),
    (asc2.negative, torch.neg, [(torch.float32, torch.float32)]),
    (asc2.relu, torch.relu, [(torch.float32, torch.float32)]),
    (asc2.rsqrt, torch.rsqrt, [(torch.float32, torch.float32)]),
    (asc2.sin, torch.sin, [(torch.float32, torch.float32)]),
    (asc2.sinh, torch.sinh, [(torch.float32, torch.float32)]),
    (asc2.sqrt, torch.sqrt, [(torch.float32, torch.float32)]),
    (asc2.tan, torch.tan, [(torch.float32, torch.float32)]),
    (asc2.tanh, torch.tanh, [(torch.float32, torch.float32)]),
]


@pytest.fixture(autouse=True)
def set_platform(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform, check=False)


@asc2.jit(always_compile=True)
def kernel(x_ptr: asc.GlobalAddress, z_ptr: asc.GlobalAddress, block_length: asc.ConstExpr, op: asc.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, [32]), [block_length], offsets=[0])
    zt = op(xt)
    asc2.store(zt, asc2.tensor(z_ptr, [32]), offsets=[0])


@pytest.mark.parametrize("asc_op, torch_op, dtypes",
                         [(asc_op, torch_op, d) for asc_op, torch_op, dtypes in unary_ops for d in dtypes])
def test_unary_operations(asc_op, torch_op, dtypes):

    def create_input(input_dtype):
        if input_dtype == torch.float32:
            res = torch.randn((size, ), dtype=input_dtype, device=device)
            res = torch.clamp(res, 1, 100)
        else:
            res = torch.randint(1, 100, (size, ), dtype=input_dtype, device=device)

        return res

    dtype_z, dtype_x = dtypes
    size = 32
    block_length = size // USE_CORE_NUM
    device = "cpu"

    x = create_input(dtype_x)
    z = torch.zeros(size, dtype=dtype_z)

    kernel[1](x, z, block_length, asc_op)

    if dtype_z == torch.float32:
        assert torch.allclose(z, torch_op(x), atol=1e-3), f"Failed {asc_op.__name__}"
    else:
        assert torch.equal(z, torch_op(x)), f"Failed {asc_op.__name__}"
