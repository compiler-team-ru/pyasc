import asc
from asc.runtime import config
import asc2
import pytest
import torch

# asc_op, torch_op, tuple of extra_arg, tensor_shape, dtype
shape_ops = [
    # asc2.broadcast_to
    (asc2.broadcast_to, torch.broadcast_to, (4, 32), (1, 32), torch.float32),
    (asc2.broadcast_to, torch.broadcast_to, (4, 32), (1, 32), torch.float16),
    (asc2.broadcast_to, torch.broadcast_to, (4, 32), (1, 32), torch.int32),
    (asc2.broadcast_to, torch.broadcast_to, (50, 32), (1, 32), torch.float32),
    (asc2.broadcast_to, torch.broadcast_to, (50, 32), (1, 32), torch.float16),
    (asc2.broadcast_to, torch.broadcast_to, (50, 32), (1, 32), torch.int32),
    # asc2.reshape
    (asc2.reshape, torch.reshape, (64, ), (2, 32), torch.float32),
    (asc2.reshape, torch.reshape, (64, ), (2, 32), torch.float16),
    (asc2.reshape, torch.reshape, (64, ), (2, 32), torch.int32),
    # asc2.expand_dims
    (asc2.expand_dims, torch.unsqueeze, (0, ), (32, ), torch.float32),
    (asc2.expand_dims, torch.unsqueeze, (0, ), (32, ), torch.float16),
    (asc2.expand_dims, torch.unsqueeze, (0, ), (32, ), torch.int32),
    # asc2.squeeze
    (asc2.squeeze, torch.squeeze, (0, ), (1, 32), torch.float32),
    (asc2.squeeze, torch.squeeze, (0, ), (1, 32), torch.float16),
    (asc2.squeeze, torch.squeeze, (0, ), (1, 32), torch.int32),
]


@asc2.jit(always_compile=True)
def kernel(x_ptr: asc.GlobalAddress, z_ptr: asc.GlobalAddress, input_shape: asc.ConstExpr, output_shape: asc.ConstExpr,
           in_offsets: asc.ConstExpr, out_offsets: asc.ConstExpr, op: asc.ConstExpr, op_param: asc.ConstExpr) -> None:
    xt = asc2.load(asc2.tensor(x_ptr, input_shape), input_shape, offsets=in_offsets)
    zt = op(xt, *op_param)
    asc2.store(zt, asc2.tensor(z_ptr, output_shape), offsets=out_offsets)


@pytest.mark.parametrize("asc_op, torch_op, arg, shape, dtype",
                         [(asc_op, torch_op, arg, shape, dtype) for asc_op, torch_op, arg, shape, dtype in shape_ops])
def test_shape_op(backend, platform, require_platform_95, asc_op, torch_op, arg, shape, dtype):
    if asc_op is asc2.broadcast_to:
        require_platform_95(platform)
    config.set_platform(backend, platform, check=False)

    def create_input(tensor_shape):
        if dtype == torch.float32:
            res = torch.randn(tensor_shape, dtype=dtype, device=device)
            res = torch.clamp(res, 1, 100)
        else:
            res = torch.randint(1, 100, tensor_shape, dtype=dtype, device=device)
        return res

    device = "cpu"
    x = create_input(shape)
    if asc_op == asc2.expand_dims or asc_op == asc2.squeeze:
        ref_z = torch_op(x, dim=arg[0])
    else:
        ref_z = torch_op(x, arg)
    z = create_input(ref_z.shape)
    in_offsets = (0, ) * len(x.shape)
    out_offsets = (0, ) * len(ref_z.shape)
    kernel[1](x, z, x.shape, ref_z.shape, in_offsets, out_offsets, asc_op, arg)
    torch.testing.assert_close(z, ref_z, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("shape", ([32], [3, 32]), ids=str)
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.int16, torch.int32), ids=str)
def test_broadcast_dup(backend, platform, shape, dtype):
    config.set_platform(backend, platform, check=False)

    @asc2.jit(always_compile=True)
    def kernel(out_ptr, shape: asc.ConstExpr, offsets: asc.ConstExpr):
        out_tensor = asc2.tensor(out_ptr, shape)
        out = asc2.full([1], 777, out_tensor.dtype).broadcast_to(*out_tensor.shape)
        asc2.store(out, out_tensor, offsets=offsets)

    out = torch.zeros(shape, dtype=dtype)
    out_ref = torch.full_like(out, 777)
    size = tuple(out.size())
    kernel[1](out, size, [0] * len(size))
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=1e-6)
