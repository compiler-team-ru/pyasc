import asc
from asc.runtime import config
import asc2
import pytest
import torch

# asc_op, torch_op, *args, tensor_shape, dtype
shape_ops = [
    (asc2.broadcast_to, torch.broadcast_to, [4, 32], [1, 32], [torch.float16, torch.float32, torch.int32]),
    (asc2.broadcast_to, torch.broadcast_to, [50, 32], [1, 32], [torch.float16, torch.float32, torch.int32]),
    (asc2.reshape, torch.reshape, [64], [2, 32], [torch.float16, torch.float32, torch.int32]),
    (asc2.reshape, torch.reshape, [4, 16], [64], [torch.float16, torch.float32, torch.int32]),
    (asc2.ravel, torch.ravel, [], [2, 32], [torch.float16, torch.float32, torch.int32]),
    (asc2.expand_dims, torch.unsqueeze, [0], [32], [torch.float16, torch.float32, torch.int32]),
    (asc2.squeeze, torch.squeeze, [0], [1, 32], [torch.float16, torch.float32, torch.int32]),
]


def local_ids(obj) -> str:
    if callable(obj):
        return obj.__name__
    return str(obj)


@pytest.mark.parametrize("asc_op, torch_op, args, shape, dtype", [(asc_op, torch_op, args, shape, d)
                                                                  for asc_op, torch_op, args, shape, dtypes in shape_ops
                                                                  for d in dtypes], ids=local_ids)
def test_shape_op(backend, platform, device_id, require_c310, asc_op, torch_op, args, shape, dtype: torch.dtype):
    if asc_op is asc2.broadcast_to:
        require_c310(platform)
    config.set_platform(backend, platform, device_id, check=False)

    def create_input(tensor_shape):
        if dtype.is_floating_point:
            return torch.randn(tensor_shape, dtype=dtype, device="cpu").clamp(1, 100)
        elif dtype.is_signed:
            return torch.randint(1, 100, tensor_shape, dtype=dtype, device="cpu")

    x = create_input(shape)
    if asc_op in (asc2.expand_dims, asc2.squeeze):
        ref_z = torch_op(x, dim=args[0])
    elif not args:
        ref_z = torch_op(x)
    else:
        ref_z = torch_op(x, args)
    z = create_input(ref_z.shape)
    in_offsets = (0, ) * len(x.shape)
    out_offsets = (0, ) * len(ref_z.shape)
    static_alloc = False if asc_op is asc2.broadcast_to else None

    @asc2.jit(always_compile=True, static_alloc=static_alloc)
    def kernel(x_ptr, z_ptr, input_shape: asc.ConstExpr, output_shape: asc.ConstExpr, in_offsets: asc.ConstExpr,
               out_offsets: asc.ConstExpr, op: asc.ConstExpr, op_param: asc.ConstExpr) -> None:
        xt = asc2.load(asc2.tensor(x_ptr, input_shape), input_shape, offsets=in_offsets)
        zt = op(xt, *op_param)
        asc2.store(zt, asc2.tensor(z_ptr, output_shape), offsets=out_offsets)

    kernel[1](x, z, x.shape, ref_z.shape, in_offsets, out_offsets, asc_op, args)
    torch.testing.assert_close(z, ref_z, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("shape", ([32], [3, 32]), ids=str)
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.int16, torch.int32), ids=str)
def test_broadcast_dup(backend, platform, device_id, shape, dtype):
    config.set_platform(backend, platform, device_id, check=False)

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
