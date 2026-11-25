# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""AscTile-native MHA forward kernel.

This target test re-implements the TileLang ``example_mha.py`` online-softmax MHA
kernel (``O = softmax(Q @ K^T / sqrt(d)) @ V``) for a single head with
``q_len == kv_len`` using only high-level AscTile ops — no ``inline_vf``, no raw
Ascend C Reg API, no ``to_asc`` / ``from_asc`` escape hatches.

The kernel mirrors the structure of ``test_fused_infer_attention_score.py`` (the
general BNSD MIX FlashAttention) but collapses the batch/head/group dimensions:
each task is one ``BR``-row Q block of the single ``[S, D]`` head, and both AIV
sub-blocks of a core split that block's rows via ``cv_strategy(SplitByM)``.
"""

# Reference implementation: https://github.com/tile-ai/tilelang/blob/fa93ba145637dbb707529a5ad8a68cc07fe99b6d/examples/ascend/flash_attention/example_mha.py

import math

from asc.experimental import asctile
import pytest
import torch

# BR is the full task M dimension; cv_ratio=2 gives each AIV sub-block BR//2 rows.
BR = 128
BC = 128
D = 128


@asctile.jit
def bmm_qk(q_part, k_part, D: asctile.ConstExpr, s1_batch: asctile.ConstExpr, step2: asctile.ConstExpr,
           mm1_k_step: asctile.ConstExpr):
    """Q @ K^T fragmented over the head dimension (D) reduction axis.

    Args:
        q_part: L1 Q tile [s1_batch, D]
        k_part: L1 K tile (already transposed) [D, step2]
        D, s1_batch, step2, mm1_k_step: compile-time constants
    Returns:
        L0C accumulator of shape [s1_batch, step2], float32.
    """
    mm_acc = asctile.zeros_acc([s1_batch, step2], dtype=asctile.float32)
    for k in asctile.range(asctile.ceildiv(D, mm1_k_step), unroll_factor=2):
        l0q = asctile.copy(q_part, [0, k * mm1_k_step], [s1_batch, mm1_k_step], location=asctile.TensorLocation.L0A)
        l0k = asctile.copy(k_part, [k * mm1_k_step, 0], [mm1_k_step, step2], location=asctile.TensorLocation.L0B)
        asctile.matmul_acc(mm_acc, l0q, l0k)
    return mm_acc


@asctile.jit
def online_softmax(attn_weight, softmax_max, softmax_d, scale_factor: asctile.ConstExpr, s1_batch: asctile.ConstExpr,
                   step2: asctile.ConstExpr, dtype: asctile.ConstExpr):
    """One online-softmax step (FlashAttention-2 form) on this sub-block's rows.

    Uses the canonical FA-2 rescale: ``P = exp(S - m_new)`` is computed once and
    reused for both the running sum and the BMM2 operand. Compared to the older
    two-step form (``exp(S - m_tile)`` then ``* exp(m_tile - m_new)``) this saves
    one ``exp`` on the row scalars and one ``mul`` + ``broadcast`` on the full
    ``[s1_batch, step2]`` probability tile per S2 iteration.

    Args:
        attn_weight: UB QK tile [s1_batch, step2] (float32)
        softmax_max: UB running row max [s1_batch] (float32)
        softmax_d:   UB running row exp-sum [s1_batch] (float32)
        scale_factor, s1_batch, step2, dtype: compile-time constants
    Returns:
        (exp_max, x_max, softmax_d, l1p) where:
          * exp_max = exp(softmax_max - x_max): rescale factor for the prior O
          * x_max  : new running row max [s1_batch]
          * softmax_d : new running row exp-sum [s1_batch]
          * l1p : probability tile P = exp(S - x_max), cast to ``dtype``, in L1.
    """
    scaled_weight = attn_weight * scale_factor
    max_tmp = asctile.reduce_max(scaled_weight, 1)
    x_max = asctile.maximum(softmax_max, max_tmp)
    exp_max = asctile.exp(softmax_max - x_max)
    # Non-positive exponents avoid overflow; x_exp feeds both the sum and PV matmul.
    x_exp = asctile.exp(scaled_weight - x_max.reshape(s1_batch, 1).broadcast_to(s1_batch, step2))
    sum_tmp = asctile.reduce_sum(x_exp, 1)
    softmax_d = exp_max * softmax_d + sum_tmp
    p16 = x_exp.to(dtype)
    l1p = asctile.copy(p16, location=asctile.TensorLocation.L1)
    return exp_max, x_max, softmax_d, l1p


@asctile.jit
def bmm_pv(l1p, l1v, D: asctile.ConstExpr, s1_batch: asctile.ConstExpr, step2: asctile.ConstExpr,
           mm2_k_step: asctile.ConstExpr):
    """P @ V fragmented over the BC (KV) reduction axis.

    Args:
        l1p: L1 probability tile [s1_batch, step2] (cast to ``dtype``)
        l1v: L1 value tile [step2, D]
        D, s1_batch, step2, mm2_k_step: compile-time constants
    Returns:
        L0C accumulator [s1_batch, D], float32.
    """
    mm_acc = asctile.zeros_acc([s1_batch, D], dtype=asctile.float32)
    for k2 in asctile.range(asctile.ceildiv(step2, mm2_k_step), unroll_factor=2):
        l0p = asctile.copy(l1p, [0, k2 * mm2_k_step], [s1_batch, mm2_k_step], location=asctile.TensorLocation.L0A)
        l0v = asctile.copy(l1v, [k2 * mm2_k_step, 0], [mm2_k_step, D], location=asctile.TensorLocation.L0B)
        asctile.matmul_acc(mm_acc, l0p, l0v)
    return mm_acc


@asctile.jit(cv_ratio=2, reuse_alloc=2, vf_fusion=True)
def flash_attention_mha_kernel(q_ptr: asctile.GlobalAddress, k_ptr: asctile.GlobalAddress, v_ptr: asctile.GlobalAddress,
                               o_ptr: asctile.GlobalAddress, ws_o_ptr: asctile.GlobalAddress,
                               seq_len: asctile.ConstExpr, D: asctile.ConstExpr, dtype: asctile.ConstExpr,
                               scale_factor: asctile.ConstExpr, step2: asctile.ConstExpr, s2_unroll: asctile.ConstExpr,
                               mm1_k_step: asctile.ConstExpr, mm2_k_step: asctile.ConstExpr):
    """Online-softmax FlashAttention-2 forward kernel for a single head.

    Args:
        q_ptr, k_ptr, v_ptr: GM base addresses for [seq_len, D] Q/K/V.
        o_ptr: GM base address for the [seq_len, D] output (cast to ``dtype``).
        ws_o_ptr: GM workspace used to seed the O accumulator (a single
            [BR, D] zero tile) — read once per task to defeat hoisting and let
            the inner S2 loop reuse that UB region.
        seq_len, D, dtype, scale_factor, step2, s2_unroll, mm1_k_step, mm2_k_step: compile-time constants.
    """
    q_gm = asctile.global_tensor(q_ptr, [seq_len, D], l2_cache=False)
    k_gm = asctile.global_tensor(k_ptr, [seq_len, D], l2_cache=True)
    v_gm = asctile.global_tensor(v_ptr, [seq_len, D], l2_cache=True)
    o_gm = asctile.global_tensor(o_ptr, [seq_len, D])

    s1_batch = BR
    ws_o_gm = asctile.global_tensor(ws_o_ptr, [s1_batch, D])

    # block_idx() includes AIV sub-block offsets; division maps both sub-blocks
    # to one task, while the AIC uses its plain block index.
    start = asctile.block_idx() / asctile.sub_block_num()
    total_tasks = asctile.ceildiv(seq_len, BR)
    step = asctile.block_num()

    for task in asctile.range(start, total_tasks, step, unroll_factor=1):
        q_row = task * BR

        q_part = asctile.copy_in(q_gm, [q_row, 0], [s1_batch, D], location=asctile.TensorLocation.L1)

        with asctile.cv_strategy(asctile.DistribMode.SplitByM):
            softmax_max = asctile.full([s1_batch], -1e10, dtype=asctile.float32)
            softmax_d = asctile.zeros([s1_batch], dtype=asctile.float32)
            o = asctile.copy_in(ws_o_gm, [0, 0], [s1_batch, D], location=asctile.TensorLocation.UB)

        for j in asctile.range(asctile.ceildiv(seq_len, step2), unroll_factor=s2_unroll):
            s2 = j * step2
            # K is transposed during GM->L1 staging.
            k_part = asctile.copy_in(k_gm, [s2, 0], [step2, D], location=asctile.TensorLocation.L1).transpose()
            l1v = asctile.copy_in(v_gm, [s2, 0], [step2, D], location=asctile.TensorLocation.L1)
            qk = bmm_qk(q_part, k_part, D, s1_batch, step2, mm1_k_step)
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                attn_weight = asctile.copy(qk, location=asctile.TensorLocation.UB)
                exp_max, softmax_max, softmax_d, l1p = online_softmax(attn_weight, softmax_max, softmax_d, scale_factor,
                                                                      s1_batch, step2, dtype)
            pv = bmm_pv(l1p, l1v, D, s1_batch, step2, mm2_k_step)
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                mm_res = asctile.copy(pv, location=asctile.TensorLocation.UB)
                o = o * exp_max.reshape(s1_batch, 1).broadcast_to(s1_batch, D) + mm_res

        with asctile.cv_strategy(asctile.DistribMode.SplitByM):
            update = o / softmax_d.reshape(s1_batch, 1).broadcast_to(s1_batch, D)
            asctile.copy_out(update.to(dtype), o_gm, [q_row, 0])


def flash_attention_mha_reference(q, k, v, scale):
    """CPU float32 reference for the single-head MHA kernel."""
    scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * scale
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v.to(torch.float32))


# yapf: disable
# Column order: (None, seq_len, torch_dtype, s2_unroll, block_num)
test_cases = [
# PYASC_TESTS_BEGIN
    pytest.param(None, 512, torch.bfloat16, 2, 36, id="seq512-bf16"),
    pytest.param(None, 1024, torch.bfloat16, 2, 36, id="seq1024-bf16"),
    pytest.param(None, 2048, torch.bfloat16, 2, 36, id="seq2048-bf16"),
    pytest.param(None, 4096, torch.bfloat16, 2, 36, id="seq4096-bf16"),
    pytest.param(None, 8192, torch.bfloat16, 2, 36, id="seq8192-bf16"),
# PYASC_TESTS_END
]
# yapf: enable


@pytest.mark.parametrize("_, seq_len, torch_dtype, s2_unroll, block_num", test_cases)
def test_flash_attention_mha(profiler, runs, _, seq_len, torch_dtype, s2_unroll, block_num, compile_only):
    assert seq_len % BR == 0, f"seq_len must be a multiple of BR={BR}"
    assert torch_dtype in (torch.float16, torch.bfloat16)
    asctile_dtype = asctile.bfloat16 if torch_dtype == torch.bfloat16 else asctile.float16
    scale_factor = 1.0 / math.sqrt(D)

    # Unlike GQA, single-head MHA need not shrink step2 for re-read KV tiles in L1.
    step2 = min(BC, seq_len)

    # Unrolled bmm2 double-buffers the 64KB L0B: 2 * mm2_k_step * D * 2 bytes must fit.
    mm1_k_step = min(256, 16384 // step2, D)
    mm2_k_step = min(16384 // D, step2)

    torch.manual_seed(42)
    q = torch.randn(seq_len, D, dtype=torch_dtype) * 0.5
    k = torch.randn(seq_len, D, dtype=torch_dtype)
    v = torch.randn(seq_len, D, dtype=torch_dtype)
    o = torch.zeros(seq_len, D, dtype=torch_dtype)
    ws_o = torch.zeros(BR * D, dtype=torch.float32)

    with profiler.profile():
        for _ in range(runs):
            flash_attention_mha_kernel[block_num](q, k, v, o, ws_o, seq_len, D, asctile_dtype, scale_factor, step2,
                                                  s2_unroll, mm1_k_step, mm2_k_step)
    if compile_only:
        return

    o_ref = flash_attention_mha_reference(q, k, v, scale_factor).to(torch_dtype)
    torch.testing.assert_close(o, o_ref, atol=1e-2, rtol=1e-2)
