import asc
from asc.runtime import config
import asc2
import pytest
import torch

# vector_vector, vector_scalar, scalar_vector
VV, VS, SV = 0, 1, 2
USE_CORE_NUM = 1

binary_ops = [
    (asc2.add, torch.add, [VV], [
        (torch.float32, (torch.float32, torch.float32)),
        (torch.int32, (torch.int32, torch.int32)),
    ]),
    (asc2.div, torch.div, [VV], [
        (torch.float32, (torch.float32, torch.float32)),
    ]),
    (asc2.mul, torch.mul, [VV], [
        (torch.float32, (torch.float32, torch.float32)),
        (torch.int32, (torch.int32, torch.int32)),
    ]),
    (asc2.sub, torch.sub, [VV], [
        (torch.float32, (torch.float32, torch.float32)),
        (torch.int32, (torch.int32, torch.int32)),
    ]),
    (asc2.equal, torch.eq, [VV], [
        (torch.int16, (torch.float32, torch.float32)),
        (torch.int16, (torch.int32, torch.int32)),
    ]),
    (asc2.greater, torch.gt, [VV], [
        (torch.int16, (torch.float32, torch.float32)),
        (torch.int16, (torch.int32, torch.int32)),
    ]),
    (asc2.greater_equal, torch.ge, [VV], [
        (torch.int16, (torch.float32, torch.float32)),
        (torch.int16, (torch.int32, torch.int32)),
    ]),
    (asc2.less, torch.lt, [VV], [
        (torch.int16, (torch.float32, torch.float32)),
        (torch.int16, (torch.int32, torch.int32)),
    ]),
    (asc2.less_equal, torch.le, [VV], [
        (torch.int16, (torch.float32, torch.float32)),
        (torch.int16, (torch.int32, torch.int32)),
    ]),
    (asc2.not_equal, torch.ne, [VV], [
        (torch.int16, (torch.float32, torch.float32)),
        (torch.int16, (torch.int32, torch.int32)),
    ]),
    (asc2.left_shift, torch.bitwise_left_shift, [VS], [
        (torch.int32, (torch.int32, torch.int32)),
    ]),
    (asc2.right_shift, torch.bitwise_right_shift, [VS], [
        (torch.int32, (torch.int32, torch.int32)),
    ]),
    (asc2.maximum, torch.maximum, [VV], [
        (torch.float32, (torch.float32, torch.float32)),
        (torch.int32, (torch.int32, torch.int32)),
    ]),
    (asc2.minimum, torch.minimum, [VV], [
        (torch.float32, (torch.float32, torch.float32)),
        (torch.int32, (torch.int32, torch.int32)),
    ]),
]


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


@asc2.jit(always_compile=True)
def kernel(x_ptr, y_ptr, z_ptr, block_length: asc.ConstExpr, fmt: asc.ConstExpr, op: asc.ConstExpr) -> None:
    if fmt == VV:
        xt = asc2.load(asc2.tensor(x_ptr, [32]), [0], [block_length])
        yt = asc2.load(asc2.tensor(y_ptr, [32]), [0], [block_length])
    elif fmt == VS:
        xt = asc2.load(asc2.tensor(x_ptr, [32]), [0], [block_length])
        yt = y_ptr
    elif fmt == SV:
        xt = x_ptr
        yt = asc2.load(asc2.tensor(y_ptr, [32]), [0], [block_length])

    zt = op(xt, yt)
    asc2.store(zt, asc2.tensor(z_ptr, [32]), [0])


@pytest.mark.parametrize("asc_op, torch_op, fmt, dtypes", [(asc_op, torch_op, f, d)
                                                           for asc_op, torch_op, fmts, dtypes in binary_ops
                                                           for f in fmts
                                                           for d in dtypes])
def test_binary_operations(asc_op, torch_op, fmt, dtypes):

    def create_input(input_dtype, is_vector):
        if is_vector:
            if input_dtype == torch.float32:
                res = torch.randn((size, ), dtype=input_dtype, device=device)
                res = torch.clamp(res, 1, 100)
            else:
                res = torch.randint(1, 100, (size, ), dtype=input_dtype, device=device)
        else:
            if input_dtype == torch.float32:
                res = float(2.5)
            else:
                res = int(2)

        return res

    if asc_op in [asc2.equal, asc2.greater, asc2.greater_equal, asc2.less, asc2.less_equal, asc2.not_equal]:
        pytest.skip(f"{asc_op} is not enabled")

    dtype_z, (dtype_x, dtype_y) = dtypes
    size = 32
    block_length = size // USE_CORE_NUM
    device = "cpu"

    if fmt == VV:
        x = create_input(dtype_x, True)
        y = create_input(dtype_y, True)
    elif fmt == VS:
        x = create_input(dtype_x, True)
        y = create_input(dtype_y, False)
    elif fmt == SV:
        x = create_input(dtype_x, False)
        y = create_input(dtype_y, True)

    z = torch.zeros(size, dtype=dtype_z)

    kernel[1](x, y, z, block_length, fmt, asc_op)

    if dtype_z == torch.float32:
        assert torch.allclose(z, torch_op(x, y), atol=1e-3), f"Failed {asc_op.__name__}"
    else:
        assert torch.equal(z, torch_op(x, y)), f"Failed {asc_op.__name__}"
