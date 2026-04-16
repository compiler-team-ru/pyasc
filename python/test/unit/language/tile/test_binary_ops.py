import asc
from asc.runtime import config
import asc2
import pytest
import torch

# vector_vector, vector_scalar, scalar_vector
VV, VS, SV = "VV", "VS", "SV"
USE_CORE_NUM = 1

binary_ops = [
    (asc2.add, torch.add, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32]),
    (asc2.div, torch.div, [VV, VS, SV], [torch.float32]),
    (asc2.mul, torch.mul, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32]),
    (asc2.sub, torch.sub, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32]),
    (asc2.left_shift, torch.bitwise_left_shift, [VS], [torch.int32]),
    (asc2.right_shift, torch.bitwise_right_shift, [VS], [torch.int32]),
    (asc2.maximum, torch.maximum, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32]),
    (asc2.minimum, torch.minimum, [VV, VS, SV], [torch.bfloat16, torch.float32, torch.int32]),
]


@asc2.jit(always_compile=True)
def kernel(x_ptr, y_ptr, z_ptr, block_length: asc.ConstExpr, fmt: asc.ConstExpr, op: asc.ConstExpr) -> None:
    if fmt == VV:
        xt = asc2.load(asc2.tensor(x_ptr, [32]), [block_length], offsets=[0])
        yt = asc2.load(asc2.tensor(y_ptr, [32]), [block_length], offsets=[0])
    elif fmt == VS:
        xt = asc2.load(asc2.tensor(x_ptr, [32]), [block_length], offsets=[0])
        yt = y_ptr
    elif fmt == SV:
        xt = x_ptr
        yt = asc2.load(asc2.tensor(y_ptr, [32]), [block_length], offsets=[0])

    zt = op(xt, yt)
    asc2.store(zt, asc2.tensor(z_ptr, [32]), offsets=[0])


def local_ids(obj) -> str:
    if callable(obj):
        return obj.__name__
    return str(obj)


@pytest.mark.parametrize("asc_op, torch_op, fmt, dtype", [(asc_op, torch_op, f, d)
                                                          for asc_op, torch_op, fmts, dtypes in binary_ops
                                                          for f in fmts
                                                          for d in dtypes], ids=local_ids)
def test_binary_operations(backend, platform, require_platform_95, asc_op, torch_op, fmt, dtype):
    if dtype == torch.bfloat16:
        require_platform_95(platform)
    config.set_platform(backend, platform, check=False)

    def create_input(input_dtype: torch.dtype, is_vector: bool):
        if is_vector:
            if input_dtype.is_floating_point:
                return torch.randn((size, ), dtype=input_dtype, device=device).clamp(1, 100)
            elif input_dtype.is_signed:
                return torch.randint(1, 100, (size, ), dtype=input_dtype, device=device)
        else:
            return torch.tensor(2, dtype=input_dtype)

    size = 32
    block_length = size // USE_CORE_NUM
    device = "cpu"

    if fmt == VV:
        x = create_input(dtype, True)
        y = create_input(dtype, True)
    elif fmt == VS:
        x = create_input(dtype, True)
        y = create_input(dtype, False)
    elif fmt == SV:
        x = create_input(dtype, False)
        y = create_input(dtype, True)

    z = torch.zeros(size, dtype=dtype)

    kernel[1](x, y, z, block_length, fmt, asc_op)
    if isinstance(x, (int, float)):
        x = torch.tensor(x, dtype=dtype)
    if isinstance(y, (int, float)):
        y = torch.tensor(y, dtype=dtype)
    if dtype == torch.float32:
        assert torch.allclose(z, torch_op(x, y), atol=1e-3), f"Failed {asc_op.__name__}"
    else:
        assert torch.equal(z, torch_op(x, y)), f"Failed {asc_op.__name__}"
