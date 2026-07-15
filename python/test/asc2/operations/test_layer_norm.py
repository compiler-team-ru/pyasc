import asc2
import pytest
import torch


@asc2.jit(always_compile=True)
def layer_norm_1d(x_ptr: asc2.GlobalAddress, gamma_ptr: asc2.GlobalAddress, beta_ptr: asc2.GlobalAddress,
                  out_ptr: asc2.GlobalAddress, mean_ptr: asc2.GlobalAddress, var_ptr: asc2.GlobalAddress,
                  size: asc2.ConstExpr, tile_size: asc2.ConstExpr, tile_per_block: asc2.ConstExpr,
                  epsilon: asc2.ConstExpr, mv_repeat: asc2.ConstExpr) -> None:
    x_gm = asc2.global_tensor(x_ptr, [size])
    gamma_gm = asc2.global_tensor(gamma_ptr, [size])
    beta_gm = asc2.global_tensor(beta_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    mean_gm = asc2.global_tensor(mean_ptr, [asc2.ceildiv(size, tile_size) * mv_repeat])
    var_gm = asc2.global_tensor(var_ptr, [asc2.ceildiv(size, tile_size) * mv_repeat])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block):
        tile_offset = base_offset + i * tile_size
        mv_offset = (asc2.block_idx() * tile_per_block + i) * mv_repeat
        x_tile = asc2.copy_in(x_gm, [tile_offset], [tile_size])
        gamma_tile = asc2.copy_in(gamma_gm, [tile_offset], [tile_size])
        beta_tile = asc2.copy_in(beta_gm, [tile_offset], [tile_size])
        out_tile, mean_tile, var_tile = asc2.layer_norm(x_tile, gamma_tile, beta_tile, epsilon)
        asc2.copy_out(out_tile, out_gm, [tile_offset])
        mean_buf = asc2.broadcast_to(mean_tile, [mv_repeat])
        var_buf = asc2.broadcast_to(var_tile, [mv_repeat])
        asc2.copy_out(mean_buf, mean_gm, [mv_offset])
        asc2.copy_out(var_buf, var_gm, [mv_offset])


@asc2.jit(always_compile=True)
def layer_norm_2d(x_ptr: asc2.GlobalAddress, gamma_ptr: asc2.GlobalAddress, beta_ptr: asc2.GlobalAddress,
                  out_ptr: asc2.GlobalAddress, mean_ptr: asc2.GlobalAddress, var_ptr: asc2.GlobalAddress,
                  num_rows: asc2.ConstExpr, num_cols: asc2.ConstExpr, epsilon: asc2.ConstExpr,
                  mv_repeat: asc2.ConstExpr) -> None:
    x_gm = asc2.global_tensor(x_ptr, [num_rows, num_cols])
    gamma_gm = asc2.global_tensor(gamma_ptr, [num_cols])
    beta_gm = asc2.global_tensor(beta_ptr, [num_cols])
    out_gm = asc2.global_tensor(out_ptr, [num_rows, num_cols])
    mean_gm = asc2.global_tensor(mean_ptr, [num_rows * mv_repeat])
    var_gm = asc2.global_tensor(var_ptr, [num_rows * mv_repeat])
    for i in asc2.range(asc2.block_idx(), num_rows, asc2.block_num()):
        x_tile = asc2.copy_in(x_gm, [i, 0], [1, num_cols])
        gamma_tile = asc2.copy_in(gamma_gm, [0], [num_cols])
        beta_tile = asc2.copy_in(beta_gm, [0], [num_cols])
        out_tile, mean_tile, var_tile = asc2.layer_norm(x_tile, gamma_tile, beta_tile, epsilon)
        asc2.copy_out(out_tile, out_gm, [i, 0])
        mean_buf = asc2.broadcast_to(mean_tile, [mv_repeat])
        var_buf = asc2.broadcast_to(var_tile, [mv_repeat])
        asc2.copy_out(mean_buf, mean_gm, [i * mv_repeat])
        asc2.copy_out(var_buf, var_gm, [i * mv_repeat])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("core_num, size", [
    (1, 1024),
    (2, 2048),
    (4, 4096),
    (8, 8192),
    (16, 16384),
])
def test_layer_norm_1d(dtype, core_num, size):
    epsilon = 1e-3
    x = torch.randn(size, dtype=dtype)
    gamma = torch.randn(size, dtype=dtype)
    beta = torch.randn(size, dtype=dtype)
    out = torch.empty_like(x)
    tile_size = 1024
    num_tiles = asc2.ceildiv(size, tile_size)
    tile_per_block = asc2.ceildiv(num_tiles, core_num)
    mv_repeat = 16 if dtype == torch.float16 else 8
    mean = torch.empty(num_tiles * mv_repeat, dtype=dtype)
    var = torch.empty(num_tiles * mv_repeat, dtype=dtype)
    layer_norm_1d[core_num](x, gamma, beta, out, mean, var, size, tile_size, tile_per_block, epsilon=epsilon,
                            mv_repeat=mv_repeat)
    expected = torch.empty_like(x)
    expected_mean = torch.empty(num_tiles, dtype=dtype)
    expected_var = torch.empty(num_tiles, dtype=dtype)
    for i in range(num_tiles):
        start = i * tile_size
        end = min(start + tile_size, size)
        tile_x = x[start:end]
        tile_mean = tile_x.mean()
        tile_var = tile_x.var(unbiased=False)
        expected[start:end] = torch.nn.functional.layer_norm(tile_x, [end - start], gamma[start:end], beta[start:end],
                                                             eps=epsilon)
        expected_mean[i] = tile_mean
        expected_var[i] = tile_var
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
    recieve_mean = mean[::mv_repeat]
    recieve_var = var[::mv_repeat]
    torch.testing.assert_close(expected_mean, recieve_mean, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(expected_var, recieve_var, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("core_num, size", [
    (1, (2, 32)),
    (16, (16, 128)),
    (16, (16, 256)),
    (32, (64, 256)),
    (32, (64, 512)),
])
def test_layer_norm_2d(dtype, core_num, size):
    epsilon = 1e-3
    num_rows, num_cols = size
    x = torch.randn(num_rows, num_cols, dtype=dtype)
    gamma = torch.randn(num_cols, dtype=dtype)
    beta = torch.randn(num_cols, dtype=dtype)
    out = torch.empty_like(x)
    mv_repeat = 16 if dtype == torch.float16 else 8
    mean = torch.empty(num_rows * mv_repeat, dtype=dtype)
    var = torch.empty(num_rows * mv_repeat, dtype=dtype)
    layer_norm_2d[core_num](x, gamma, beta, out, mean, var, num_rows, num_cols, epsilon=epsilon, mv_repeat=mv_repeat)
    expected = torch.nn.functional.layer_norm(x, [num_cols], gamma, beta, eps=epsilon)
    expected_mean = x.mean(dim=1)
    expected_var = x.var(dim=1, unbiased=False)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
    extracted_mean = mean[::mv_repeat]
    extracted_var = var[::mv_repeat]
    torch.testing.assert_close(extracted_mean, expected_mean, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(extracted_var, expected_var, rtol=1e-3, atol=1e-3)
