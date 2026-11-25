# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""AscTile-native relative-position attention (DSOCR) kernel.

Re-implements the TileLang ``dsocr_kernel.py`` relative-height/relative-width
attention ``O = softmax(Q @ K^T * scale + rel_h[:, key_h] + rel_w[:, key_w]) @ V``
using only high-level AscTile ops — no ``inline_vf``, no ``to_asc`` / ``from_asc``.

The bias gather (``rel_h[:, key_h[k]]`` for each query row and key column) is
realised with a single 2-D ``asctile.reindex`` per bias tensor: the per-row
stride (``i * w_aligned``) is precomputed on the host and combined with the
key-index tile at runtime to form a ``[BR, step2]`` index into the flattened
``[BR, w_aligned]`` rel-bias tile.

Inputs are permuted on the host from ``[B, S, H, D]`` (BSHD) to ``[B, H, S, D]``
(BHSD) and flattened to ``[B*H*S, D]`` so that each (batch, head) pair has a
contiguous sequence axis.  The sequence is padded to a multiple of ``BR`` so
that every query block is full; invalid key positions in the last key block are
masked to ``-inf`` via a precomputed ``key_valid`` mask.
"""

# Reference implementation: https://gitcode.com/cann/ops-tilelang/blob/master/src/cann_ops_tilelang/multimodal/dsocr_kernel.py

import math

from asc.experimental import asctile
import pytest
import torch

# TileLang uses the same BR and D. BC=128 fits C310 UB only with s2_unroll=1;
# larger unrolling requires BC=64 because AscTile keeps intermediates in UB.
BR = 128
BC = 128
D = 64


@asctile.jit
def bmm_qk(q_part, k_part, D: asctile.ConstExpr, s1_batch: asctile.ConstExpr, step2: asctile.ConstExpr,
           mm1_k_step: asctile.ConstExpr):
    """Q @ K^T fragmented over the head dimension (D) reduction axis.

    Args:
        q_part: L1 Q tile [s1_batch, D]
        k_part: L1 K tile (already transposed) [D, step2]
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
def online_softmax_with_bias(attn_weight, rel_h_ub, rel_w_ub, row_off, key_h_tile, key_w_tile, key_valid_tile,
                             softmax_max, softmax_d, scale_factor: asctile.ConstExpr, s1_batch: asctile.ConstExpr,
                             step2: asctile.ConstExpr, w_aligned: asctile.ConstExpr, dtype: asctile.ConstExpr):
    """One online-softmax step with relative-position bias on this sub-block's rows.

    Adds ``rel_h[:, key_h]`` and ``rel_w[:, key_w]`` to the scaled attention tile,
    masks invalid key positions, then runs the canonical FA-2 rescale.

    Args:
        attn_weight: UB QK tile [s1_batch, step2] (float32, unscaled)
        rel_h_ub, rel_w_ub: UB bias tiles [s1_batch, w_aligned]
        row_off: UB row offsets [s1_batch] (row_off[i] = i * w_aligned)
        key_h_tile, key_w_tile, key_valid_tile: UB key-index tiles [step2] (int32)
        softmax_max, softmax_d: UB running row stats [s1_batch] (float32)
        scale_factor, s1_batch, step2, w_aligned, dtype: compile-time constants
    Returns:
        (exp_max, x_max, softmax_d, l1p)
    """
    # Fold bias and mask into one expression: separate full-size float32
    # intermediates overflow the 253952-byte C310 UB at BC=128.
    row_off_2d = row_off.reshape(s1_batch, 1).broadcast_to(s1_batch, step2)
    key_h_2d = key_h_tile.reshape(1, step2).broadcast_to(s1_batch, step2)
    idx_h = row_off_2d + key_h_2d
    bias_h = asctile.reindex(rel_h_ub, idx_h).to(asctile.float32)

    key_w_2d = key_w_tile.reshape(1, step2).broadcast_to(s1_batch, step2)
    idx_w = row_off_2d + key_w_2d
    bias_w = asctile.reindex(rel_w_ub, idx_w).to(asctile.float32)

    # A moderate -1e4 penalty avoids overflow from the initial softmax_max.
    mask_penalty = (key_valid_tile - 1).to(asctile.float32) * 1e4
    mask_penalty_2d = mask_penalty.reshape(1, step2).broadcast_to(s1_batch, step2)

    biased_weight = attn_weight * scale_factor + bias_h + bias_w + mask_penalty_2d

    max_tmp = asctile.reduce_max(biased_weight, 1)
    x_max = asctile.maximum(softmax_max, max_tmp)
    exp_max = asctile.exp(softmax_max - x_max)
    x_exp = asctile.exp(biased_weight - x_max.reshape(s1_batch, 1).broadcast_to(s1_batch, step2))
    sum_tmp = asctile.reduce_sum(x_exp, 1)
    softmax_d = exp_max * softmax_d + sum_tmp
    # JoinByM rejoins AIV row halves in L1 for the full PV matmul; direct
    # L1->L1 copies are unsupported.
    l1p = asctile.copy(x_exp.to(dtype), location=asctile.TensorLocation.L1, distrib=asctile.DistribMode.JoinByM)
    return exp_max, x_max, softmax_d, l1p


@asctile.jit
def bmm_pv(l1p, l1v, D: asctile.ConstExpr, s1_batch: asctile.ConstExpr, step2: asctile.ConstExpr,
           mm2_k_step: asctile.ConstExpr):
    """P @ V fragmented over the BC (KV) reduction axis.

    Args:
        l1p: L1 probability tile [s1_batch, step2]
        l1v: L1 value tile [step2, D]
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
def dsocr_attn_kernel(q_ptr: asctile.GlobalAddress, k_ptr: asctile.GlobalAddress, v_ptr: asctile.GlobalAddress,
                      rel_h_ptr: asctile.GlobalAddress, rel_w_ptr: asctile.GlobalAddress,
                      key_h_ptr: asctile.GlobalAddress, key_w_ptr: asctile.GlobalAddress,
                      key_valid_ptr: asctile.GlobalAddress, row_off_ptr: asctile.GlobalAddress,
                      o_ptr: asctile.GlobalAddress, dim: asctile.ConstExpr, w_aligned: asctile.ConstExpr,
                      scale_factor: asctile.ConstExpr, step2: asctile.ConstExpr, s2_unroll: asctile.ConstExpr,
                      mm1_k_step: asctile.ConstExpr, mm2_k_step: asctile.ConstExpr, heads: asctile.ConstExpr,
                      batch: asctile.ConstExpr, seq_padded: asctile.ConstExpr, dtype: asctile.ConstExpr):
    """Relative-position attention forward kernel (MIX / cv_strategy SplitByM).

    Computes ``O = softmax(Q @ K^T * scale + rel_h_bias + rel_w_bias) @ V`` with
    online (FlashAttention-2) softmax.  The relative-position bias is gathered
    from the per-query-row rel_h / rel_w tiles using ``asctile.reindex``.

    Structure mirrors ``test_flash_attention_mha.py`` / ``test_fused_infer_attention_score.py``:
    Q in L1, K/V streamed GM->L1 per S2 step, both AIV sub-blocks of a core split
    the Q rows via ``cv_strategy(SplitByM)``.  The ``bmm_qk`` / ``bmm_pv`` matmul
    helpers run on the AIC between cv_strategy regions, which serves as a sync
    point freeing L0A/L0B between the two matmuls (avoiding the domination
    conflict).  ``online_softmax_with_bias`` runs inside the first cv_strategy
    region on the AIV sub-blocks.

    Args:
        q_ptr, k_ptr, v_ptr: GM base addresses for [B*H*seq_padded, D] Q/K/V (BHSD layout).
        rel_h_ptr, rel_w_ptr: GM base for [B*H*seq_padded, w_aligned] rel-h / rel-w bias.
        key_h_ptr, key_w_ptr: GM base for [seq_padded] int32 key index tensors (key_h[k]=k//side, key_w[k]=k%side).
        key_valid_ptr: GM base for [seq_padded] int32 mask (1 for k < seq_len, 0 otherwise).
        row_off_ptr: GM base for [BR] int32 row offsets (row_off[i] = i * w_aligned).
        o_ptr: GM base for [B*H*seq_padded, D] output.
        dim, w_aligned, scale_factor, step2, s2_unroll, mm1_k_step, mm2_k_step: compile-time constants.
        heads, batch, seq_padded: compile-time constants.
        dtype: compute dtype (bfloat16).
    """
    total_seq = batch * heads * seq_padded
    s1_batch = BR
    s1_half = s1_batch // 2
    query_blocks = seq_padded // BR
    num_key_blocks = seq_padded // step2

    q_gm = asctile.global_tensor(q_ptr, [total_seq, dim], l2_cache=False)
    k_gm = asctile.global_tensor(k_ptr, [total_seq, dim], l2_cache=True)
    v_gm = asctile.global_tensor(v_ptr, [total_seq, dim], l2_cache=True)
    o_gm = asctile.global_tensor(o_ptr, [total_seq, dim])
    rel_h_gm = asctile.global_tensor(rel_h_ptr, [total_seq, w_aligned], l2_cache=True)
    rel_w_gm = asctile.global_tensor(rel_w_ptr, [total_seq, w_aligned], l2_cache=True)
    key_h_gm = asctile.global_tensor(key_h_ptr, [seq_padded], l2_cache=True)
    key_w_gm = asctile.global_tensor(key_w_ptr, [seq_padded], l2_cache=True)
    key_valid_gm = asctile.global_tensor(key_valid_ptr, [seq_padded], l2_cache=True)
    row_off_gm = asctile.global_tensor(row_off_ptr, [s1_half])

    # block_idx() includes AIV sub-block offsets; division maps both sub-blocks
    # to one task, while the AIC uses its plain block index.
    start = asctile.block_idx() / asctile.sub_block_num()
    total_tasks = batch * heads * query_blocks
    step = asctile.block_num()
    sb = asctile.sub_block_idx()

    # Each task covers one (batch, head, query block), with explicit SplitByM
    # and half-sized UB tiles. cv_strategy would split the reindex source but
    # not its flattened indexing, making sub-block 1 gather the wrong bias.
    for task in asctile.range(start, total_tasks, step, unroll_factor=1):
        query_block = task % query_blocks
        matrix_idx = task // query_blocks
        head_idx = matrix_idx % heads
        batch_idx = matrix_idx // heads
        bh_base = (batch_idx * heads + head_idx) * seq_padded
        query_start = query_block * BR
        q_abs = bh_base + query_start
        row_base = q_abs + s1_half * sb

        # Keep full Q in L1 for the AIC matmul; only L0C->UB transfers split rows.
        q_part = asctile.copy_in(q_gm, [q_abs, 0], [s1_batch, dim], location=asctile.TensorLocation.L1)

        softmax_max = asctile.full([s1_half], -1e4, dtype=asctile.float32)
        softmax_d = asctile.zeros([s1_half], dtype=asctile.float32)
        o = asctile.zeros([s1_half, dim], dtype=asctile.float32)

        # row_off contains local flattened offsets into this sub-block's bias tile.
        rel_h_ub = asctile.copy_in(rel_h_gm, [row_base, 0], [s1_half, w_aligned])
        rel_w_ub = asctile.copy_in(rel_w_gm, [row_base, 0], [s1_half, w_aligned])
        row_off = asctile.copy_in(row_off_gm, [0], [s1_half])

        # Unrolling overlaps GM->L1 loads with cube compute; s2_unroll=2 doubles
        # live intermediates and requires BC=64 to fit UB.
        for j in asctile.range(num_key_blocks, unroll_factor=s2_unroll):
            s2 = j * step2

            # K is transposed during GM->L1 staging.
            k_part = asctile.copy_in(k_gm, [bh_base + s2, 0], [step2, dim],
                                     location=asctile.TensorLocation.L1).transpose()
            l1v = asctile.copy_in(v_gm, [bh_base + s2, 0], [step2, dim], location=asctile.TensorLocation.L1)

            qk = bmm_qk(q_part, k_part, dim, s1_batch, step2, mm1_k_step)
            attn_weight = asctile.copy(qk, location=asctile.TensorLocation.UB, distrib=asctile.DistribMode.SplitByM)
            key_h_tile = asctile.copy_in(key_h_gm, [s2], [step2])
            key_w_tile = asctile.copy_in(key_w_gm, [s2], [step2])
            key_valid_tile = asctile.copy_in(key_valid_gm, [s2], [step2])
            exp_max, softmax_max, softmax_d, l1p = online_softmax_with_bias(attn_weight, rel_h_ub, rel_w_ub, row_off,
                                                                            key_h_tile, key_w_tile, key_valid_tile,
                                                                            softmax_max, softmax_d, scale_factor,
                                                                            s1_half, step2, w_aligned, dtype)

            pv = bmm_pv(l1p, l1v, dim, s1_batch, step2, mm2_k_step)
            mm_res = asctile.copy(pv, location=asctile.TensorLocation.UB, distrib=asctile.DistribMode.SplitByM)
            o = o * exp_max.reshape(s1_half, 1).broadcast_to(s1_half, dim) + mm_res

        update = o / softmax_d.reshape(s1_half, 1).broadcast_to(s1_half, dim)
        asctile.copy_out(update.to(dtype), o_gm, [row_base, 0])


def dsocr_attn_reference(q, k, v, rel_h, rel_w, side):
    """CPU float32 reference matching ``tests/multimodal/test_dsocr.py``."""
    batch, sequence, heads, dim = q.shape
    scale = math.sqrt(1.0 / dim)
    q_head = q.permute(0, 2, 1, 3).float()
    k_head = k.permute(0, 2, 1, 3).float()
    v_head = v.permute(0, 2, 1, 3).float()
    key_idx = torch.arange(sequence)
    key_h = key_idx // side
    key_w = key_idx % side
    output = torch.empty((batch, heads, sequence, dim), dtype=torch.float32)
    for query_start in range(0, sequence, BR):
        query_end = min(query_start + BR, sequence)
        scores = torch.matmul(q_head[:, :, query_start:query_end, :], k_head.transpose(-1, -2)) * scale
        rel_h_chunk = rel_h[:, query_start:query_end, :, :side].permute(0, 2, 1, 3).float()
        rel_w_chunk = rel_w[:, query_start:query_end, :, :side].permute(0, 2, 1, 3).float()
        scores += rel_h_chunk.index_select(-1, key_h)
        scores += rel_w_chunk.index_select(-1, key_w)
        probs = torch.softmax(scores, dim=-1).to(torch.bfloat16).float()
        output[:, :, query_start:query_end, :] = torch.matmul(probs, v_head)
    return output.permute(0, 2, 1, 3).to(torch.bfloat16)


# yapf: disable
# Shapes and the 36-core launch match the TileLang suite (12 heads, dim=64, bf16).
test_cases = [
# PYASC_TESTS_BEGIN
    pytest.param(2, 14, 36, id="dsocr-2-14"),
    pytest.param(1, 40, 36, id="dsocr-1-40"),
    pytest.param(1, 64, 36, id="dsocr-1-64"),
# PYASC_TESTS_END
]
# yapf: enable


@pytest.mark.parametrize("batch, side, block_num", test_cases)
def test_dsocr_attn(profiler, runs, batch, side, block_num, compile_only):
    seq_len = side * side
    heads = 12
    dim = D
    w_aligned = ((side + 7) // 8) * 8
    # UB copy_in requires 32-byte-aligned widths (16 bf16 elements).
    w_aligned_padded = ((w_aligned + 15) // 16) * 16
    seq_padded = ((seq_len + BR - 1) // BR) * BR

    asctile_dtype = asctile.bfloat16
    scale_factor = math.sqrt(1.0 / dim)
    step2 = min(BC, seq_padded)
    s2_unroll = 1
    mm1_k_step = min(256, 16384 // step2, dim)
    mm2_k_step = min(16384 // dim, step2)

    torch.manual_seed(42)
    qkv = torch.randn((batch, seq_len, 3, heads, dim), dtype=torch.bfloat16)
    q, k, v = qkv.unbind(2)
    rel_h = torch.randn((batch, seq_len, heads, w_aligned), dtype=torch.bfloat16)
    rel_w = torch.randn((batch, seq_len, heads, w_aligned), dtype=torch.bfloat16)

    # int32 indices avoid RV_VLD alignment faults seen with int16 on the simulator.
    key_idx = torch.arange(seq_padded, dtype=torch.int32)
    key_valid = (key_idx < seq_len).to(torch.int32)
    # Keep padded reindex indices in bounds.
    key_idx_clamped = torch.minimum(key_idx, torch.tensor(seq_len - 1, dtype=torch.int32))
    key_h = (key_idx_clamped // side).to(torch.int32)
    key_w = (key_idx_clamped % side).to(torch.int32)
    # Local offsets address each sub-block's independently loaded bias tile.
    row_off = (torch.arange(BR // 2, dtype=torch.int32) * w_aligned_padded).to(torch.int32)

    def pad_seq(t):
        """[B, H, S, X] -> [B, H, seq_padded, X] (zero-padded)."""
        if seq_padded > seq_len:
            padded = torch.zeros(t.shape[0], t.shape[1], seq_padded, t.shape[3], dtype=t.dtype)
            padded[:, :, :seq_len] = t
            return padded
        return t

    def pad_w(t):
        """[B, H, S, W] -> [B, H, S, w_aligned_padded] (zero-padded)."""
        if w_aligned_padded > w_aligned:
            padded = torch.zeros(t.shape[0], t.shape[1], t.shape[2], w_aligned_padded, dtype=t.dtype)
            padded[:, :, :, :w_aligned] = t
            return padded
        return t

    q_bhsd = pad_seq(q.permute(0, 2, 1, 3).contiguous())
    k_bhsd = pad_seq(k.permute(0, 2, 1, 3).contiguous())
    v_bhsd = pad_seq(v.permute(0, 2, 1, 3).contiguous())
    rel_h_bhsd = pad_seq(pad_w(rel_h.permute(0, 2, 1, 3).contiguous()))
    rel_w_bhsd = pad_seq(pad_w(rel_w.permute(0, 2, 1, 3).contiguous()))

    q_flat = q_bhsd.reshape(-1, dim)
    k_flat = k_bhsd.reshape(-1, dim)
    v_flat = v_bhsd.reshape(-1, dim)
    rel_h_flat = rel_h_bhsd.reshape(-1, w_aligned_padded)
    rel_w_flat = rel_w_bhsd.reshape(-1, w_aligned_padded)
    o_flat = torch.zeros(batch * heads * seq_padded, dim, dtype=torch.bfloat16)
    with profiler.profile():
        for _ in range(runs):
            dsocr_attn_kernel[block_num](q_flat, k_flat, v_flat, rel_h_flat, rel_w_flat, key_h, key_w, key_valid,
                                         row_off, o_flat, dim, w_aligned_padded, scale_factor, step2, s2_unroll,
                                         mm1_k_step, mm2_k_step, heads, batch, seq_padded, asctile_dtype)
    if compile_only:
        return

    # Drop padded sequence rows between (batch, head) blocks before restoring BSHD.
    o_bhsd = torch.empty(batch, heads, seq_len, dim, dtype=torch.bfloat16)
    for b in range(batch):
        for h in range(heads):
            base = (b * heads + h) * seq_padded
            o_bhsd[b, h] = o_flat[base:base + seq_len]
    o_bshd = o_bhsd.permute(0, 2, 1, 3).contiguous()

    o_ref = dsocr_attn_reference(q, k, v, rel_h, rel_w, side)
    assert not torch.isnan(o_bshd).any()
    torch.testing.assert_close(o_bshd, o_ref, atol=0.005, rtol=0.01)
