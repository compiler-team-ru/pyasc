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

This tutorial demonstrates matrix multiplication operator and its launch on Ascend simulator.

"""

import asctile
import torch


# The functions which are executed on Ascend NPU must be marked with `@asctile.jit` decorator.
# Available parameters for @asctile.jit decorator can be seen in the documentation:
# https://compiler-team-ru.github.io/pyasc/python-api/rst/runtime/index.html
@asctile.jit
def matrix_multiplication(
        # Pointers to input and output tensors should have `asctile.GlobalAddress` type.
        a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress, c_ptr: asctile.GlobalAddress,
        # Input shapes for A and B matrices
        a_shape: asctile.ConstExpr, b_shape: asctile.ConstExpr,
        # For optimization purposes it is recommended to pass scalar parameter as constants (e.g. `asctile.ConstExpr[int]`).
        single_core_m: asctile.ConstExpr, single_core_n: asctile.ConstExpr, step_ka: asctile.ConstExpr,
        step_kb: asctile.ConstExpr, base_k: asctile.ConstExpr):
    m, k = a_shape
    _, n = b_shape
    # Tensor descriptor is created from `asctile.GlobalAddress` to represent entire tensor.
    a_gm = asctile.global_tensor(a_ptr, a_shape)
    b_gm = asctile.global_tensor(b_ptr, b_shape)
    c_gm = asctile.global_tensor(c_ptr, [m, n])

    # Create 2D tensor for cube L0C accumulator
    acc = asctile.zeros_acc([single_core_m, single_core_n], dtype=asctile.float32)

    # Python expressions are used to calculate block offset:
    # - `asctile.block_idx()` function is used to get current AICORE index.
    n_blocks = asctile.ceildiv(n, single_core_n)
    m_off = single_core_m * (asctile.block_idx() / n_blocks)
    n_off = single_core_n * (asctile.block_idx() % n_blocks)

    # `unroll_factor` parameter of `asctile.range` in `for` loop can be used to manage software pipelining. Set it to `2` to enable double buffering.
    # `parallel` parameter of `asctile.range` in `for` loop enable overlapping of store operation of `i`-th iteration and load of `i+1`-th iteration.
    # It is user responsibility to ensure that there are no data dependencies between overlapped iterations.
    for k_outer in range(asctile.ceildiv(k, step_kb), unroll_factor=2):
        # Load B matrix from GM to a new local tensor in L1
        b_l1 = asctile.copy_in(b_gm, [k_outer * step_kb, n_off], [step_kb, single_core_n], asctile.TensorLocation.L1)
        for k_mid in range(asctile.ceildiv(step_kb, step_ka), unroll_factor=2):
            k_off = k_outer * step_kb + k_mid * step_ka
            # Load A matrix from GM to a new local tensor in L1
            a_l1 = asctile.copy_in(a_gm, [m_off, k_off], [single_core_m, step_ka], asctile.TensorLocation.L1)
            for k_l0 in range(asctile.ceildiv(step_ka, base_k), unroll_factor=2):
                # Copy A matrix from L1 to a new local tensor in L0A
                a_l0 = asctile.copy(a_l1, [0, k_l0 * base_k], [single_core_m, base_k], asctile.TensorLocation.L0A)
                # Copy B matrix from L1 to a new local tensor in L0B
                b_l0 = asctile.copy(b_l1, [k_mid * step_ka + k_l0 * base_k, 0], [base_k, single_core_n],
                                    asctile.TensorLocation.L0B)
                # Perform matrix multiplication with updating accumulator
                asctile.matmul_acc(acc, a_l0, b_l0)

    # `asctile.copy_out` is used to move data from L0C accumulator back to GM.
    asctile.copy_out(acc, c_gm, [m_off, n_off])


if __name__ == "__main__":
    backend = asctile.Backend.Model  # can be "Model" for simulator or "NPU" for device
    soc_version = asctile.Platform.Ascend950PR_9599  # Device version
    device_id = 0  # might be necessary to provide if more than one NPU device is present in the system
    asctile.set_platform(backend, soc_version, device_id)

    block_num = 16
    dtype = torch.float32

    m, k, n = 1024, 64, 16
    single_core_m, single_core_n = 64, 16
    step_ka, step_kb, base_k = 16, 64, 16

    # Allocate tensors for A, B and result matrix C
    a = torch.rand((m, k), dtype=dtype)
    b = torch.rand((k, n), dtype=dtype)
    c = torch.zeros((m, n), dtype=dtype)

    matrix_multiplication[block_num](a, b, c, a.shape, b.shape, single_core_m, single_core_n, step_ka, step_kb, base_k)
    c_ref = a @ b
    torch.testing.assert_close(c, c_ref)
