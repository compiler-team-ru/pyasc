# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fused Softmax
=============

A fused (single-call) softmax over the last dimension of a 2-D matrix -- a pattern that reads and writes global memory
several times and is therefore memory-bound.

.. currentmodule:: asc.experimental

In this tutorial you will learn about:

* Row-wise tiling: each tile holds a few *complete* rows so a row's reduction never crosses a tile boundary.

* Boundary handling: padding the column tail with the reduction identity (``-inf`` for max) so no special-case branch is
  needed.

* JIT options for reusing on-chip buffers and overlapping loads with compute (multi-buffering) to hide GM latency.

* The numerically stable softmax formula, implemented by hand as a row-wise reduction function.

* The ``vf_fusion=True`` JIT option, which fuses a chain of elementwise/reduction ops into a single register-level VF
  (vector function) block.
"""

# %%
# Motivation
# ----------
#
# A naive, three-pass softmax written with ordinary tensor ops does, per row::
#
#     m = max(x)        # pass 1: read  N elements
#     e = exp(x - m)    # pass 2: read  N, write N
#     s = sum(e)        # pass 3: read  N
#     y = e / s         #         read  N, write N
#
# Each pass is a separate round-trip through global memory. A *fused* kernel keeps the tile in on-chip UB for the whole
# computation, so the row is read from and written to GM exactly once. The trick is to keep each row's whole reduction
# inside one tile: we load a tile of complete rows and reduce along the columns (dim 1), so no row needs data from any
# other tile.
#
# Row-wise softmax implementation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``row_wise_softmax`` is a *device function*: it takes a UB tensor in and returns one out, implementing the numerically
# stable softmax by hand. The runnable kernel below calls it on each tile. We subtract the row maximum before
# exponentiating, so ``exp(x - max)`` can never overflow. This assumes each whole row fits Unified Buffer, but for very
# long rows you would split the reduction across tiles (not shown here).

from asc.experimental import asctile


@asctile.jit
def row_wise_softmax(rows: asctile.LocalTensor) -> asctile.LocalTensor:
    # Per-row maximum along dim 1, kept as [tile_rows, 1] so it broadcasts back across the columns when we subtract it.
    row_max = asctile.reduce_max(rows, 1, keep_dims=True)
    # Subtract the max before exponentiating so exp(x - max) can never overflow.
    shifted = rows - row_max
    exp_vals = asctile.exp(shifted)
    sum_exp = asctile.reduce_sum(exp_vals, 1, keep_dims=True)
    return exp_vals / sum_exp


# %%
# Compute Kernel
# --------------
#
# ``fused_softmax`` is the runnable kernel. Its body loads a tile of complete rows into UB, calls the
# ``row_wise_softmax`` function on it, and stores the result back. The tiling and memory options are:
#
# * Each core owns a contiguous chunk of ``rows_per_block`` rows, starting at ``block_idx * rows_per_block``; within it
#   we step by ``tile_shape[0]`` rows per tile.
#
# * Each tile is a ``[tile_rows, tile_cols]`` block of complete rows loaded into UB. ``tile_cols`` is ``num_cols``
#   rounded up to the 32-byte alignment; the extra columns are padded with ``-inf`` -- the identity for max -- so they
#   can never raise a row's maximum and the softmax stays correct.
#
# * ``unroll_factor=2`` pipelines the next tile's load with this tile's compute (double buffering), hiding the memory
#   latency behind the compute.
#
# * ``reuse_alloc=2`` reuses a finished tile's memory region for the next tile instead of acquiring fresh buffer for
#   each unrolled loop iteration, reducing peak on-chip memory usage so the multi-buffered tiles fit.
#
# * ``vf_fusion=True`` fuses the chain of elementwise/reduction ops (max, subtract, exp, sum, divide) into a single
#   register-level vector function (VF), avoiding redundant UB reads/writes between the intermediate steps. It is
#   experimental and best suited to such elementwise chains.
#
# See also: :py:class:`asctile.CompileOptions` dataclass.
#
# Type hints on kernel arguments are optional -- only ``asctile.ConstExpr`` annotations are necessary, since they tell
# the compiler which scalars are compile-time constants. Other type hints on arguments are omitted here for brevity.


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def fused_softmax(input_ptr, output_ptr, num_rows, num_cols, tile_shape: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [num_rows, num_cols])
    out_gm = asctile.global_tensor(output_ptr, [num_rows, num_cols])
    rows_per_block = asctile.ceildiv(num_rows, asctile.block_num())
    block_offset = asctile.block_idx() * rows_per_block
    ub_loop = asctile.ceildiv(rows_per_block, tile_shape[0])
    for i in asctile.range(ub_loop, unroll_factor=2):
        row_start_offset = block_offset + i * tile_shape[0]
        rows = asctile.copy_in(in_gm, [row_start_offset, 0], [tile_shape[0], tile_shape[1]], pad_value=float("-inf"))
        out = row_wise_softmax(rows)
        asctile.copy_out(out, out_gm, [row_start_offset, 0])


# %%
# Launch and Verify
# -----------------
#
# We run ``fused_softmax`` on a single ``[256, 98]`` matrix. ``98`` is deliberately not 32-byte aligned, so the column
# padding with ``-inf`` is exercised; the row count divides evenly across the 8 cores (256 = 8 * 32 rows per core), so
# there is no row tail to handle.

if __name__ == "__main__":
    import torch

    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599)
    torch.manual_seed(0)

    num_rows, num_cols = 256, 98
    block_num = 8
    tile_rows = 16
    # The last UB dimension must be 32-byte aligned: 32 / 4 bytes (float32) = 8 elements.
    alignment = 32 // torch.float32.itemsize
    tile_cols = (num_cols + alignment - 1) // alignment * alignment  # 98 -> 104
    tile_shape = [tile_rows, tile_cols]

    x = torch.randn(num_rows, num_cols, dtype=torch.float32)
    reference = torch.softmax(x, dim=1)

    out = torch.empty_like(x)
    fused_softmax[block_num](x, out, num_rows, num_cols, tile_shape)
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
    max_diff = (out - reference).abs().max().item()
    print(f"fused_softmax: PASSED [{num_rows}x{num_cols}] (max diff={max_diff:.2e})")
