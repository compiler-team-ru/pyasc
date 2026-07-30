import itertools

import asc2
import pytest
import torch


@pytest.mark.parametrize("asc_op, torch_op, args, shape", (
    (asc2.broadcast_to, torch.broadcast_to, [4, 32], [1, 32]),
    (asc2.broadcast_to, torch.broadcast_to, [50, 32], [1, 32]),
    (asc2.reshape, torch.reshape, [64], [2, 32]),
    (asc2.reshape, torch.reshape, [4, 32], [128]),
    (asc2.ravel, torch.ravel, [], [2, 32]),
    (asc2.expand_dims, torch.unsqueeze, [0], [32]),
    (asc2.squeeze, torch.squeeze, [0], [1, 32]),
))
@pytest.mark.parametrize(
    "dtype",
    (torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_shape_op(require_c310, asc_op, torch_op, args, shape, dtype: torch.dtype):
    if asc_op is asc2.broadcast_to:
        require_c310()
        if dtype == torch.float64:
            pytest.skip("broadcast_to float64 support is limited")

    def create_input(tensor_shape):
        if dtype.is_floating_point:
            return torch.randn(tensor_shape, dtype=dtype).clamp(1, 100)
        elif dtype.is_signed:
            return torch.randint(1, 100, tensor_shape, dtype=dtype)

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

    @asc2.jit(always_compile=True)
    def kernel(x_ptr, z_ptr, input_shape: asc2.ConstExpr, output_shape: asc2.ConstExpr, in_offsets: asc2.ConstExpr,
               out_offsets: asc2.ConstExpr, op: asc2.ConstExpr, op_param: asc2.ConstExpr) -> None:
        xt = asc2.copy_in(asc2.global_tensor(x_ptr, input_shape), in_offsets, input_shape)
        zt = op(xt, *op_param)
        asc2.copy_out(zt, asc2.global_tensor(z_ptr, output_shape), out_offsets)

    kernel[1](x, z, x.shape, ref_z.shape, in_offsets, out_offsets, asc_op, args)
    torch.testing.assert_close(z, ref_z)


@pytest.mark.parametrize("iter_factory", (list, tuple, itertools.chain))
@pytest.mark.parametrize("asc_op, torch_op, dst_shape, input_shape", (
    (asc2.reshape, torch.reshape, [64], [2, 32]),
    (asc2.reshape, torch.reshape, [4, 32], [128]),
    (asc2.broadcast_to, torch.broadcast_to, [4, 32], [1, 32]),
))
def test_shape_op_with_list_or_tuple(require_c310, asc_op, torch_op, dst_shape, input_shape, iter_factory):
    if asc_op is asc2.broadcast_to:
        require_c310()

    x = torch.randn(input_shape, dtype=torch.float32).clamp(1, 100)
    ref_z = torch_op(x, dst_shape)
    z = torch.zeros(ref_z.shape, dtype=torch.float32)
    in_offsets = (0, ) * len(x.shape)
    out_offsets = (0, ) * len(ref_z.shape)
    wrapped_args = iter_factory(dst_shape)

    @asc2.jit(always_compile=True)
    def kernel(x_ptr, z_ptr, input_shape: asc2.ConstExpr, output_shape: asc2.ConstExpr, in_offsets: asc2.ConstExpr,
               out_offsets: asc2.ConstExpr, op: asc2.ConstExpr, op_param: asc2.ConstExpr) -> None:
        xt = asc2.copy_in(asc2.global_tensor(x_ptr, input_shape), in_offsets, input_shape)
        zt = op(xt, op_param)
        asc2.copy_out(zt, asc2.global_tensor(z_ptr, output_shape), out_offsets)

    kernel[1](x, z, x.shape, ref_z.shape, in_offsets, out_offsets, asc_op, wrapped_args)
    torch.testing.assert_close(z, ref_z)


@pytest.mark.parametrize("shape", ([32], [3, 32]))
@pytest.mark.parametrize(
    "dtype", (torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32))
def test_broadcast_dup(require_c310, shape, dtype):
    if dtype in (torch.int8, torch.int64):
        require_c310()

    @asc2.jit(always_compile=True)
    def kernel(out_ptr, shape: asc2.ConstExpr, offsets: asc2.ConstExpr):
        out_tensor = asc2.global_tensor(out_ptr, shape)
        out = asc2.full([1], 77, out_tensor.dtype).broadcast_to(*out_tensor.shape)
        asc2.copy_out(out, out_tensor, offsets)

    out = torch.zeros(shape, dtype=dtype)
    out_ref = torch.full_like(out, 77)
    size = tuple(out.size())
    kernel[1](out, size, [0] * len(size))
    torch.testing.assert_close(out, out_ref)


class TestBroadcastShapes:

    def test_single_shape(self):
        assert asc2.broadcast_shapes([32]) == (32, )

    def test_single_shape_multidim(self):
        assert asc2.broadcast_shapes((4, 32)) == (4, 32)

    def test_two_identical_shapes(self):
        assert asc2.broadcast_shapes([4, 32], [4, 32]) == (4, 32)

    def test_two_shapes_different_ranks(self):
        assert asc2.broadcast_shapes([1, 32], [4, 32]) == (4, 32)

    def test_two_shapes_rank_padding(self):
        assert asc2.broadcast_shapes([32], [4, 32]) == (4, 32)

    def test_two_shapes_both_broadcast(self):
        assert asc2.broadcast_shapes([1, 32], [4, 1]) == (4, 32)

    def test_three_shapes(self):
        assert asc2.broadcast_shapes([1, 32], [4, 1], [4, 32]) == (4, 32)

    def test_three_shapes_different_ranks(self):
        assert asc2.broadcast_shapes([32], [1, 1], [4, 32]) == (4, 32)

    def test_scalar_like_shapes(self):
        assert asc2.broadcast_shapes([1], [32]) == (32, )

    def test_high_dim(self):
        assert asc2.broadcast_shapes([2, 1, 32], [1, 4, 1]) == (2, 4, 32)

    def test_matches_torch(self):
        shapes = [(1, 32), (4, 1), (4, 32)]
        result = asc2.broadcast_shapes(*shapes)
        expected = torch.broadcast_shapes(*shapes)
        assert result == expected

    def test_no_shapes_raises(self):
        with pytest.raises(ValueError):
            asc2.broadcast_shapes()

    def test_incompatible_shapes_raises(self):
        with pytest.raises(RuntimeError, match="not broadcastable"):
            asc2.broadcast_shapes([3, 32], [4, 32])

    def test_incompatible_shapes_different_ranks_raises(self):
        with pytest.raises(RuntimeError, match="not broadcastable"):
            asc2.broadcast_shapes([2, 32], [3, 32])

    def test_non_positive_dim_raises(self):
        with pytest.raises(RuntimeError, match="positive"):
            asc2.broadcast_shapes([0, 32])

    def test_negative_dim_raises(self):
        with pytest.raises(RuntimeError, match="positive"):
            asc2.broadcast_shapes([-1, 32])

    def test_non_integer_dim_raises(self):
        with pytest.raises(TypeError, match="integers"):
            asc2.broadcast_shapes([1.5, 32])

    def test_empty_shape_raises(self):
        with pytest.raises(RuntimeError, match="at least one value"):
            asc2.broadcast_shapes([])

    def test_unpacked_generator_input(self):
        shapes = [[4, 32], [1, 32]]
        assert asc2.broadcast_shapes(*shapes) == (4, 32)
