# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Matrix Multiplication
=====================

A tiled matrix multiplication ``C = A @ B`` that uses the Ascend **cube** unit -- the dedicated matmul accelerator,
distinct from the vector unit used in the previous tutorials.

.. currentmodule:: asc.experimental

In this tutorial you will learn about:

* The Ascend cube memory hierarchy and the required data flow ``GM -> L1 -> L0A/L0B -> L0C -> GM``,
  and *why* each level exists.

* Two different on-chip moves: GM -> L1 (staging) and L1 -> the cube's operand registers (L0A/L0B).

* The accumulator pattern: why the result lives in a dedicated ``L0C`` register and is updated *in place* (destination
  passing) rather than returned by value.

* Splitting the reduction dimension ``K`` into three nested levels so that big L1 tiles feed many small cube calls.

* Casting the float32 accumulator down to the storage dtype before storing it back to GM.
"""

# %%
# Motivation
# ----------
#
# The kernel we write implements the familiar **Blocked Matmul** algorithm::
#
#     for each output tile [M_i, N_j] assigned to a core:
#         acc = zeros([single_core_m, single_core_n])
#         for k in range(0, K, ...):          # split K into chunks
#             a = A[M_i, k]                    # load A tile
#             b = B[k, N_j]                    # load B tile
#             acc += a @ b                    # accumulate in the cube
#         C[M_i, N_j] = acc
#
# The twist on Ascend is that ``a @ b`` does not happen in UB (as elementwise ops do) but in the **cube unit**, which
# has its own private memories.
#
# The cube memory hierarchy can be presented as the following::
#
#     GM  ->  L1  ->  L0A (operand A)   ->  L0C (accumulator)  ->  GM
#                 ->  L0B (operand B)   ->
#
# ===== ====
# Realm Role
# ===== ====
# GM    HBM: A, B live here; results are written back here.
# L1    A staging cache between GM and the cube operand regs.
# L0A   Cube operand register for the left  matrix A.
# L0B   Cube operand register for the right matrix B.
# L0C   The cube *accumulator* register (always float32).
# ===== ====
#
# See also: :py:class:`asctile.TensorLocation` enum.
#
# L1 is a required bridge: you cannot go GM -> L0A/L0B directly. Conveniently, calling
# ``asctile.copy_in(..., location=L0A)`` lets the compiler split it into GM -> L1 -> L0A automatically. Even more,
# explicit tensor locations may be omitted completely in most cases, so they are resolved during compilation. However,
# in this tutorial we do the two steps explicitly to make the flow visible.

# %%
# Compute Kernel
# --------------

from asc.experimental import asctile


@asctile.jit
def matrix_multiplication(a_ptr, b_ptr, c_ptr, a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr,
                          single_core_m: asctile.ConstExpr[int],  # output-tile height one core computes
                          single_core_n: asctile.ConstExpr[int],  # output-tile width  one core computes
                          step_ka: asctile.ConstExpr[int],  # K-chunk loaded into L1 for A each k_mid step
                          step_kb: asctile.ConstExpr[int],  # K-chunk loaded into L1 for B each k_outer step
                          base_k: asctile.ConstExpr[int],  # K slice fed to the cube in one matmul_acc call
                          quant_type: asctile.ConstExpr,  # dtype to store the result in (e.g. float16)
                          ):

    m, k = a_shape
    _, n = b_shape
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, [m, n])

    # The cube accumulator: a zero-initialised [single_core_m, single_core_n] tensor in L0C. Note it is *not*
    # ``asctile.zeros`` (which lives in UB): L0C is a dedicated register that the cube writes to. It is always float32,
    # even when A/B are float16, so the K-reduction accumulates without loss.
    acc = asctile.zeros_acc([single_core_m, single_core_n], dtype=asctile.float32)

    # Map each core to one output tile of the [M, N] grid, in row-major order. Index arithmetic on ``block_idx()``
    # lowers to signed integer division (``//``) and modulo.
    n_blocks = asctile.ceildiv(n, single_core_n)
    m_off = single_core_m * (asctile.block_idx() // n_blocks)
    n_off = single_core_n * (asctile.block_idx() % n_blocks)

    # Three nested K loops, from coarse to fine. Each ``unroll_factor=2`` lets the compiler overlap the next tile's load
    # with the current tile's compute.
    #
    #   k_outer : step over ``step_kb``      -> load a fresh B tile into L1
    #   k_mid   : step over ``step_ka``      -> load a fresh A tile into L1
    #   k_l0    : step over ``base_k``       -> copy A/B slices L1 -> L0A/L0B, then matmul
    for k_outer in asctile.range(asctile.ceildiv(k, step_kb), unroll_factor=2):
        # GM -> L1: stage a [step_kb, single_core_n] tile of B.
        b_l1 = asctile.copy_in(b_gm, [k_outer * step_kb, n_off], [step_kb, single_core_n], asctile.TensorLocation.L1)
        for k_mid in asctile.range(asctile.ceildiv(step_kb, step_ka), unroll_factor=2):
            k_off = k_outer * step_kb + k_mid * step_ka
            # GM -> L1: stage a [single_core_m, step_ka] tile of A.
            a_l1 = asctile.copy_in(a_gm, [m_off, k_off], [single_core_m, step_ka], asctile.TensorLocation.L1)
            for k_l0 in asctile.range(asctile.ceildiv(step_ka, base_k), unroll_factor=2):
                # L1 -> L0A/L0B: feed the cube operands. ``asctile.copy`` is the local-to-local move (as opposed to
                # ``copy_in``/``copy_out`` which touch GM). Each call takes a [.., base_k] / [base_k, ..] slice.
                a_l0 = asctile.copy(a_l1, [0, k_l0 * base_k], [single_core_m, base_k], asctile.TensorLocation.L0A)
                b_l0 = asctile.copy(b_l1, [k_mid * step_ka + k_l0 * base_k, 0], [base_k, single_core_n],
                                    asctile.TensorLocation.L0B)
                # Accumulate in place: acc += a_l0 @ b_l0. ``matmul_acc`` takes the accumulator by reference
                # (destination passing) and returns nothing, because L0C is a fixed register the cube updates directly:
                # there is no "new" tensor to return.
                asctile.matmul_acc(acc, a_l0, b_l0)

    # The accumulator is float32; cast to the storage dtype (e.g. float16) before leaving L0C.
    # ``.to`` on an L0C tensor is the cube's quantisation step (F322F16 etc.).
    result = acc.to(quant_type)
    # L0C -> GM directly: ``copy_out`` accepts an L0C source, so no extra UB hop is needed.
    asctile.copy_out(result, c_gm, [m_off, n_off])


# %%
# Launch and Verify
# -----------------
#
# We run ``matrix_multiplication`` on a small float16 matmul and exercise all three nested K loops:
#
# * Shapes: ``A = [M, K] = [32, 128]``, ``B = [K, N] = [128, 128]``, ``C = [M, N] = [32, 128]``. Inputs are float16, but
#   the cube accumulates in float32 and the result is stored back as float16.
#
# * Output grid: ``single_core_m = single_core_n = 32`` gives ``ceildiv(32,32) x ceildiv(128,32) = 1 x 4 = 4`` tiles,
#   one per core (``block_num = 4``).
#
# * K reduction (K=128): ``step_kb = 64``, ``step_ka = 32``, ``base_k = 16`` -> ``k_outer = ceildiv(128,64) = 2``,
#   ``k_mid = ceildiv(64,32) = 2`` and ``k_l0 = ceildiv(32,16) = 2``; each tile's K=128 reduction is split across all
#   three levels, with the cube fed ``[32,16]`` / ``[16,32]`` L0 slices at the innermost level.

if __name__ == "__main__":
    import torch

    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599)
    torch.manual_seed(0)

    m, k, n = 32, 128, 128
    single_core_m, single_core_n = 32, 32
    step_ka, step_kb, base_k = 32, 64, 16
    block_num = 4
    quant_type = asctile.float16

    dtype = torch.float16
    a = torch.rand((m, k), dtype=dtype)
    b = torch.rand((k, n), dtype=dtype)
    c = torch.zeros((m, n), dtype=dtype)

    matrix_multiplication[block_num](a, b, c, a.shape, b.shape, single_core_m, single_core_n, step_ka, step_kb, base_k,
                                     quant_type)

    # Reference: upcast to float32 for the matmul (as the cube does internally), then cast back to the storage dtype.
    reference = (a.to(torch.float32) @ b.to(torch.float32)).to(dtype)
    torch.testing.assert_close(c, reference, atol=1e-5, rtol=1e-5)
    max_diff = (c.to(torch.float32) - reference.to(torch.float32)).abs().max().item()
    print(f"matrix_multiplication: PASSED ({m}x{k} @ {k}x{n} -> {m}x{n}, blocks={block_num}, "
          f"tile={single_core_m}x{single_core_n}, max diff={max_diff:.2e})")
