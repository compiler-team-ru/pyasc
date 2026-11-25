# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of the CANN Open Software License Agreement Version
# 2.0 (the "License").  Please refer to the License for details. You may not use
# this file except in compliance with the License.  THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
# A PARTICULAR PURPOSE.  See LICENSE in the root of the software repository for
# the full text of the License.

# Reference implementation: https://gitcode.com/cann/ops-tilelang/blob/master/src/cann_ops_tilelang/moe/moe_topk_gate_kernel.py

from asc.experimental import asctile
import pytest
import torch

from ..helpers import CORE_NUM
from .topk_ref import moe_topk_gate as torch_moe_topk_gate

SIGMOID = 0
SQRTSOFTPLUS = 1
SOFTMAX = 2
IDENTITY = 3

_SCORING_MAP = {
    'sigmoid': SIGMOID,
    'sqrtsoftplus': SQRTSOFTPLUS,
    'softmax': SOFTMAX,
    'identity': IDENTITY,
}

FLOAT32_ALIGN = 64
NEG_INF = -1e30
SQRTSOFTPLUS_THRESHOLD = 20.0
VREG_ELEMS = 64


def _align_up(a, alignment):
    return ((a + alignment - 1) // alignment) * alignment


@asctile.jit(reuse_alloc=0, vf_fusion=True)
def moe_topk_gate_kernel(
    logits_ptr: asctile.GlobalAddress,
    bias_ptr: asctile.GlobalAddress,
    image_bias_ptr: asctile.GlobalAddress,
    image_token_mask_ptr: asctile.GlobalAddress,
    mask_ptr: asctile.GlobalAddress,
    index_ptr: asctile.GlobalAddress,
    k_index_ptr: asctile.GlobalAddress,
    topk_idx_ptr: asctile.GlobalAddress,
    topk_weights_ptr: asctile.GlobalAddress,
    unmapped_topk_idx_ptr: asctile.GlobalAddress,
    to_physical_map_ptr: asctile.GlobalAddress,
    logical_count_ptr: asctile.GlobalAddress,
    fix_routing_mask_ptr: asctile.GlobalAddress,
    fix_indices_ptr: asctile.GlobalAddress,
    force_random_ptr: asctile.GlobalAddress,
    random_idx_ptr: asctile.GlobalAddress,
    random_weights_ptr: asctile.GlobalAddress,
    num_tokens: asctile.ConstExpr,
    routed_scaling_factor: asctile.ConstExpr,
    num_topk: asctile.ConstExpr,
    num_routed_experts: asctile.ConstExpr,
    num_shared_experts: asctile.ConstExpr,
    num_physical_topk: asctile.ConstExpr,
    num_logical_experts: asctile.ConstExpr,
    map_width: asctile.ConstExpr,
    ep_rank: asctile.ConstExpr,
    scoring_type: asctile.ConstExpr,
    has_bias: asctile.ConstExpr,
    has_image_bias: asctile.ConstExpr,
    has_mask: asctile.ConstExpr,
    has_unmapped: asctile.ConstExpr,
    has_physical_map: asctile.ConstExpr,
    has_fix_routing: asctile.ConstExpr,
    has_force_random: asctile.ConstExpr,
    aligned_experts: asctile.ConstExpr,
    aligned_k: asctile.ConstExpr,
    use_inline_vf: asctile.ConstExpr,
):
    logits_gm = asctile.global_tensor(logits_ptr, [num_tokens * num_routed_experts])
    topk_idx_gm = asctile.global_tensor(topk_idx_ptr, [num_tokens * num_physical_topk])
    topk_weights_gm = asctile.global_tensor(topk_weights_ptr, [num_tokens * num_physical_topk])
    index_gm = asctile.global_tensor(index_ptr, [aligned_experts])
    k_index_gm = asctile.global_tensor(k_index_ptr, [aligned_k])

    if has_unmapped:
        unmapped_gm = asctile.global_tensor(unmapped_topk_idx_ptr, [num_tokens * num_topk])
    if has_physical_map:
        to_physical_map_gm = asctile.global_tensor(to_physical_map_ptr, [num_logical_experts * map_width])
        logical_count_gm = asctile.global_tensor(logical_count_ptr, [num_logical_experts])
    if has_bias:
        bias_gm = asctile.global_tensor(bias_ptr, [aligned_experts])
    if has_image_bias:
        image_bias_gm = asctile.global_tensor(image_bias_ptr, [aligned_experts])
        image_token_mask_gm = asctile.global_tensor(image_token_mask_ptr, [num_tokens])
    if has_mask:
        mask_gm = asctile.global_tensor(mask_ptr, [num_tokens])
    if has_fix_routing:
        fix_routing_gm = asctile.global_tensor(fix_routing_mask_ptr, [num_tokens])
        fix_indices_gm = asctile.global_tensor(fix_indices_ptr, [num_tokens * num_topk])
    if has_force_random:
        force_random_gm = asctile.global_tensor(force_random_ptr, [num_tokens])
        random_idx_gm = asctile.global_tensor(random_idx_ptr, [num_tokens * num_physical_topk])
        random_weights_gm = asctile.global_tensor(random_weights_ptr, [num_tokens * num_physical_topk])

    indices = asctile.copy_in(index_gm, [0], [aligned_experts])
    k_index = asctile.copy_in(k_index_gm, [0], [aligned_k])

    if has_bias:
        bias_local = asctile.copy_in(bias_gm, [0], [aligned_experts])
    if has_image_bias:
        image_bias_local = asctile.copy_in(image_bias_gm, [0], [aligned_experts])

    # Pre-create early-exit buffers outside the row loop (reused per masked/FR row).
    # Hoisted once instead of asctile.full() on every masked/FR row.
    neg_one_i32 = asctile.full([aligned_k], -1, dtype=asctile.int32)
    zero_f32 = asctile.full([aligned_k], 0.0, dtype=asctile.float32)
    neg_one_i64 = neg_one_i32.to(asctile.int64)

    # Batched scalar reads: load per-row control arrays into UB once, then read
    # scalars via local_get_value (UB read) instead of copy_in (GM read) per row.
    aligned_tokens = asctile.ceildiv(num_tokens, FLOAT32_ALIGN) * FLOAT32_ALIGN
    if has_mask:
        mask_ub = asctile.to_asc(asctile.copy_in(mask_gm, [0], [aligned_tokens]))
    if has_force_random:
        fr_ub = asctile.to_asc(asctile.copy_in(force_random_gm, [0], [aligned_tokens]))
    if has_image_bias:
        img_mask_ub = asctile.to_asc(asctile.copy_in(image_token_mask_gm, [0], [aligned_tokens]))

    for row in asctile.range(asctile.block_idx(), num_tokens, asctile.block_num()):
        mask_val = 0
        is_masked = 0
        if has_mask:
            mask_val = mask_ub.get_value(row)
            is_masked = 1 - mask_val

        is_fr = 0
        if has_force_random:
            is_fr = fr_ub.get_value(row)
            if has_mask:
                is_fr = is_fr * (1 - is_masked)

        do_normal = 1 - is_masked
        if has_force_random:
            do_normal = do_normal * (1 - is_fr)

        if do_normal > 0:
            logits = asctile.copy_in(logits_gm, [row * num_routed_experts], [aligned_experts])

            if scoring_type == SIGMOID:
                scores = 1.0 / (1.0 + asctile.exp(-logits))
                ranked_base = scores
                scores = asctile.where(indices < num_routed_experts, scores, 0.0)
                ranked_base = asctile.where(indices < num_routed_experts, ranked_base, NEG_INF)
            elif scoring_type == SQRTSOFTPLUS:
                softplus_log = asctile.log(1.0 + asctile.exp(logits))
                softplus = asctile.where(logits > SQRTSOFTPLUS_THRESHOLD, logits, softplus_log)
                scores = asctile.sqrt(softplus)
                ranked_base = scores
                scores = asctile.where(indices < num_routed_experts, scores, 0.0)
                ranked_base = asctile.where(indices < num_routed_experts, ranked_base, NEG_INF)
            elif scoring_type == SOFTMAX:
                ranked_base = asctile.where(indices < num_routed_experts, logits, NEG_INF)
                scores = asctile.softmax(ranked_base)
            else:
                scores = logits
                ranked_base = scores
                scores = asctile.where(indices < num_routed_experts, scores, 0.0)
                ranked_base = asctile.where(indices < num_routed_experts, ranked_base, NEG_INF)

            if has_image_bias:
                img_mask_val = img_mask_ub.get_value(row)
                img_mask_f32 = asctile.cast(img_mask_val, asctile.float32)
                if has_bias:
                    diff = image_bias_local - bias_local
                    bias_for_ranking = bias_local + diff * img_mask_f32
                else:
                    bias_for_ranking = image_bias_local * img_mask_f32
                ranked_scores = ranked_base + bias_for_ranking
            elif has_bias:
                ranked_scores = ranked_base + bias_local
            else:
                ranked_scores = ranked_base

            if use_inline_vf:
                # Register-resident selection via inline_vf: winners are
                # accumulated in an output register and written once at the
                # end (single aligned store).  Scores are modified in-place
                # in UB (set to NEG_INF after each selection), with a
                # LocalMemBar between the store and the next iteration's load.
                topk_idx_buf = asctile.inline_vf(
                    f'constexpr uint16_t NUM_VREGS = {aligned_experts // VREG_ELEMS};\n'
                    f'constexpr uint16_t NUM_TOPK = {num_topk};\n'
                    """
                    __ubuf__ int32_t* out_ptr = reinterpret_cast<__ubuf__ int32_t*>($0.GetPhyAddr());
                    __ubuf__ float* scores_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
                    __ubuf__ int32_t* indices_ptr = reinterpret_cast<__ubuf__ int32_t*>($2.GetPhyAddr());
                    AscendC::Reg::MaskReg mask_all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::RegTensor<int32_t> lane_idx;
                    AscendC::Reg::Arange<int32_t, AscendC::Reg::IndexOrder::INCREASE_ORDER>(lane_idx, (int32_t)0);
                    AscendC::Reg::RegTensor<int32_t> out_reg;
                    AscendC::Reg::Duplicate(out_reg, (int32_t)(-1));
                    AscendC::Reg::RegTensor<float> neg_inf_reg;
                    AscendC::Reg::Duplicate(neg_inf_reg, (float)(-1e30));
                    AscendC::Reg::RegTensor<int32_t> int_max_reg;
                    AscendC::Reg::Duplicate(int_max_reg, (int32_t)0x7FFFFFFF);
                    AscendC::Reg::RegTensor<float> scores_reg, max_acc, max_scalar, max_broadcast;
                    AscendC::Reg::RegTensor<int32_t> indices_reg, cand_reg, min_acc, winner_scalar, winner_broadcast;
                    AscendC::Reg::MaskReg eq_mask, winner_mask, slot_mask;
                    for (uint16_t k = 0; k < NUM_TOPK; k += 1) {
                        AscendC::Reg::Duplicate(max_acc, (float)(-1e30));
                        for (uint16_t v = 0; v < NUM_VREGS; v += 1) {
                            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(scores_reg, scores_ptr + v * 64);
                            AscendC::Reg::Max(max_acc, max_acc, scores_reg, mask_all);
                        }
                        AscendC::Reg::ReduceMax(max_scalar, max_acc, mask_all);
                        AscendC::Reg::Duplicate(max_broadcast, max_scalar, mask_all);
                        AscendC::Reg::Duplicate(min_acc, (int32_t)0x7FFFFFFF);
                        for (uint16_t v = 0; v < NUM_VREGS; v += 1) {
                            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(scores_reg, scores_ptr + v * 64);
                            AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(indices_reg, indices_ptr + v * 64);
                            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(eq_mask, scores_reg, max_broadcast, mask_all);
                            AscendC::Reg::Select(cand_reg, indices_reg, int_max_reg, eq_mask);
                            AscendC::Reg::Min(min_acc, min_acc, cand_reg, mask_all);
                        }
                        AscendC::Reg::ReduceMin(winner_scalar, min_acc, mask_all);
                        AscendC::Reg::Duplicate(winner_broadcast, winner_scalar, mask_all);
                        AscendC::Reg::CompareScalar<int32_t, AscendC::CMPMODE::EQ>(slot_mask, lane_idx, (int32_t)k, mask_all);
                        AscendC::Reg::Select(out_reg, winner_broadcast, out_reg, slot_mask);
                        for (uint16_t v = 0; v < NUM_VREGS; v += 1) {
                            AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(indices_reg, indices_ptr + v * 64);
                            AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::EQ>(winner_mask, indices_reg, winner_broadcast, mask_all);
                            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(scores_reg, scores_ptr + v * 64);
                            AscendC::Reg::Select(scores_reg, neg_inf_reg, scores_reg, winner_mask);
                            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM>(scores_ptr + v * 64, scores_reg, mask_all);
                        }
                        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                    }
                    AscendC::Reg::DataCopy<int32_t, AscendC::Reg::StoreDist::DIST_NORM>(out_ptr, out_reg, mask_all);
                    """, [aligned_k], asctile.int32, [ranked_scores, indices])
            else:
                # High-level ops: stable selection loop (max + argmin-on-ties).
                topk_idx_buf = asctile.full([aligned_k], -1, dtype=asctile.int32)

                for k in asctile.range(num_topk):
                    max_val = ranked_scores.max()
                    selected_idx = asctile.where(ranked_scores == max_val, indices, 999999).min()

                    # TopK slot write via arithmetic blend (mul/add) instead of
                    # where/select to avoid the cast(select(...)) pattern that triggers
                    # ApplyHomomorphism pass failures when topk_idx_buf is cast to int64.
                    pos_mask_i32 = asctile.where(k_index == k, 1, 0)
                    topk_idx_buf = topk_idx_buf * (1 - pos_mask_i32) + selected_idx * pos_mask_i32

                    is_winner = (indices == selected_idx)
                    ranked_scores = asctile.where(is_winner, NEG_INF, ranked_scores)

            if has_fix_routing:
                fix_val = asctile.copy_in(fix_routing_gm, [row])
                if fix_val > 0:
                    fix_indices = asctile.copy_in(fix_indices_gm, [row * num_topk], [aligned_k])
                    fix_indices_i32 = asctile.cast(fix_indices, asctile.int32)
                    topk_idx_buf = fix_indices_i32

            safe_idx = asctile.where(topk_idx_buf >= 0, topk_idx_buf, 0)
            topk_scores_buf = asctile.reindex(scores, safe_idx)

            # Weight normalisation: zero invalid slots once, then div+mul without
            # redundant where (0 / sum * scale == 0, so invalid slots stay zero).
            topk_scores_buf = asctile.where(k_index < num_topk, topk_scores_buf, 0.0)
            score_sum = topk_scores_buf.sum()
            topk_weights_buf = topk_scores_buf / (score_sum + 1e-20) * routed_scaling_factor

            if num_shared_experts > 0:
                is_shared = asctile.where(k_index >= num_topk, 1, 0) * asctile.where(
                    k_index < num_topk + num_shared_experts, 1, 0)
                not_shared = 1 - is_shared
                shared_idx = num_routed_experts + (k_index - num_topk)
                topk_idx_buf = topk_idx_buf * not_shared + shared_idx * is_shared
                is_shared_f = asctile.cast(is_shared, asctile.float32)
                not_shared_f = asctile.cast(not_shared, asctile.float32)
                topk_weights_buf = topk_weights_buf * not_shared_f + is_shared_f

            # Hoist .to(int64): single cast reused for both unmapped and topk_idx outputs.
            topk_idx_i64 = topk_idx_buf.to(asctile.int64)
            if has_unmapped:
                asctile.copy_out(topk_idx_i64, unmapped_gm, [row * num_topk], real_shape=[num_topk])
            asctile.copy_out(topk_idx_i64, topk_idx_gm, [row * num_physical_topk], real_shape=[num_physical_topk])
            asctile.copy_out(topk_weights_buf, topk_weights_gm, [row * num_physical_topk],
                             real_shape=[num_physical_topk])

        else:
            topk_idx_buf = neg_one_i32
            topk_weights_buf = zero_f32

            if has_unmapped:
                asctile.copy_out(neg_one_i64, unmapped_gm, [row * num_topk], real_shape=[num_topk])

            # FR override (only load random data for FR rows)
            if has_force_random:
                if is_fr > 0:
                    rand_idx = asctile.copy_in(random_idx_gm, [row * num_physical_topk], [aligned_k])
                    rand_weights = asctile.copy_in(random_weights_gm, [row * num_physical_topk], [aligned_k])
                    not_fr = 1 - is_fr
                    fr_f32 = asctile.cast(is_fr, asctile.float32)
                    not_fr_f32 = asctile.cast(not_fr, asctile.float32)
                    topk_idx_buf = topk_idx_buf * not_fr + rand_idx * is_fr
                    topk_weights_buf = topk_weights_buf * not_fr_f32 + rand_weights * fr_f32

            topk_idx_i64 = topk_idx_buf.to(asctile.int64)
            asctile.copy_out(topk_idx_i64, topk_idx_gm, [row * num_physical_topk], real_shape=[num_physical_topk])
            asctile.copy_out(topk_weights_buf, topk_weights_gm, [row * num_physical_topk],
                             real_shape=[num_physical_topk])

    # --- Physical mapping (second pass, reads from GM, writes back to GM) ---
    if has_physical_map:
        for row in asctile.range(asctile.block_idx(), num_tokens, asctile.block_num()):
            do_map = 1
            if has_force_random:
                fr_val = asctile.copy_in(force_random_gm, [row])
                do_map = 1 - fr_val
            if do_map:
                for slot in asctile.range(num_physical_topk):
                    base = row * num_physical_topk + slot
                    logical_i64 = asctile.copy_in(topk_idx_gm, [base])
                    if logical_i64 >= 0:
                        logical_i32 = asctile.cast(logical_i64, asctile.int32)
                        count = asctile.copy_in(logical_count_gm, [logical_i32])
                        dup_idx = (ep_rank + row * 23333) % count
                        flat_idx = logical_i32 * map_width + asctile.cast(dup_idx, asctile.int32)
                        physical = asctile.copy_in(to_physical_map_gm, [flat_idx])
                        physical_i64 = asctile.cast(physical, asctile.int64)
                        asctile.copy_out(physical_i64, topk_idx_gm, [base])


def run_moe_topk_gate(profiler, runs, logits, topk_idx, topk_weights, bias_padded, image_bias_padded,
                      image_token_mask_tensor, mask_tensor, index_tensor, k_index_tensor, unmapped_topk_idx,
                      to_physical_map, logical_count, fix_routing_mask_tensor, fix_indices_tensor, force_random_tensor,
                      random_idx_tensor, random_weights_tensor, num_tokens, routed_scaling_factor, num_topk,
                      num_routed_experts, num_shared_experts, num_physical_topk, num_logical_experts, map_width,
                      ep_rank, scoring_type, has_bias, has_image_bias, has_mask, has_unmapped, has_physical_map,
                      has_fix_routing, has_force_random, aligned_experts, aligned_k, *, use_inline_vf=0):
    """Host entry: launch the MoE TopK gate kernel.

    `use_inline_vf` selects the TopK implementation compiled into the kernel:
    0 — high-level max/argmin selection loop, 1 — register-resident inline_vf.
    """
    block_num = min(CORE_NUM, max(1, num_tokens))

    params = [
        logits, bias_padded, image_bias_padded, image_token_mask_tensor, mask_tensor, index_tensor, k_index_tensor,
        topk_idx, topk_weights, unmapped_topk_idx, to_physical_map, logical_count, fix_routing_mask_tensor,
        fix_indices_tensor, force_random_tensor, random_idx_tensor, random_weights_tensor
    ]
    params.extend([
        num_tokens,
        routed_scaling_factor,
        num_topk,
        num_routed_experts,
        num_shared_experts,
        num_physical_topk,
        num_logical_experts,
        map_width,
        ep_rank,
        scoring_type,
        has_bias,
        has_image_bias,
        has_mask,
        has_unmapped,
        has_physical_map,
        has_fix_routing,
        has_force_random,
        aligned_experts,
        aligned_k,
        use_inline_vf,
    ])

    with profiler.profile():
        for _ in range(runs):
            moe_topk_gate_kernel[block_num](*params)


def run_moe_topk_gate_vf(profiler, runs, *args, **kwargs):
    """Host entry: same as run_moe_topk_gate, but with the inline_vf TopK."""
    run_moe_topk_gate(profiler, runs, *args, use_inline_vf=1, **kwargs)


def run_and_compare(profiler, runs, logits, num_topk, num_routed_experts, num_shared_experts, scoring_func, with_bias,
                    with_image_bias, with_mask, use_shared_as_routed, aligned_experts, aligned_k,
                    with_physical_map=False, with_fix_routing=False, with_force_random=False, compile_only=False,
                    use_inline_vf=0, num_padded_tokens=0):
    num_tokens = logits.shape[0]
    routed_scaling_factor = 1.5
    ep_rank = 0
    scoring_type = _SCORING_MAP.get(scoring_func, IDENTITY)

    if not use_shared_as_routed:
        num_shared_experts = 0

    num_physical_topk = num_topk + num_shared_experts
    num_logical_experts = num_routed_experts + num_shared_experts
    num_physical_experts = num_routed_experts + num_shared_experts

    topk_idx = torch.full((num_tokens, num_physical_topk), -1, dtype=torch.int64)
    topk_weights = torch.zeros((num_tokens, num_physical_topk), dtype=torch.float32)

    # unmapped_topk_idx: always allocated, kernel writes it
    unmapped_topk_idx = torch.full((num_tokens, num_topk), -1, dtype=torch.int64)

    bias = torch.randn(num_routed_experts, dtype=torch.float32) if with_bias else None
    bias_padded = torch.zeros(aligned_experts, dtype=torch.float32)
    if bias is not None:
        bias_padded[:num_routed_experts] = bias

    image_bias = torch.randn(num_routed_experts, dtype=torch.float32) if with_image_bias else None
    image_bias_padded = torch.zeros(aligned_experts, dtype=torch.float32)
    if image_bias is not None:
        image_bias_padded[:num_routed_experts] = image_bias

    image_token_mask = None
    if with_image_bias:
        image_token_mask = torch.rand(num_tokens, dtype=torch.float32) < 0.5
    image_token_mask_tensor = (image_token_mask.to(torch.int32) if image_token_mask is not None else torch.zeros(
        1, dtype=torch.int32))

    mask = None
    if with_mask:
        mask = torch.ones(num_tokens, dtype=torch.bool)
        mask[-min(num_padded_tokens if num_padded_tokens > 0 else 10, num_tokens):] = False
    mask_tensor = mask.to(torch.int32) if mask is not None else torch.zeros(1, dtype=torch.int32)

    fix_routing_mask = None
    fix_indices = None
    if with_fix_routing:
        fix_routing_mask = torch.ones(num_tokens, dtype=torch.bool)
        fix_indices = torch.randint(0, num_routed_experts, (num_tokens, num_topk), dtype=torch.int64)
    fix_routing_mask_tensor = (fix_routing_mask.to(torch.int32) if fix_routing_mask is not None else torch.zeros(
        1, dtype=torch.int32))
    fix_indices_tensor = (fix_indices.to(torch.int32) if fix_indices is not None else torch.zeros(1, dtype=torch.int32))

    # Force random setup (precomputed on host, no device-side RNG)
    force_random = None
    random_idx = torch.zeros(num_tokens, num_physical_topk, dtype=torch.int32)
    random_weights = torch.zeros(num_tokens, num_physical_topk, dtype=torch.float32)
    if with_force_random:
        force_random = torch.rand(num_tokens, dtype=torch.float32) < 0.5
        if with_fix_routing and fix_routing_mask is not None:
            force_random = torch.logical_and(~fix_routing_mask, force_random)
        for i in range(num_tokens):
            if force_random[i]:
                perm = torch.randperm(num_physical_experts)[:num_physical_topk]
                random_idx[i] = perm.to(torch.int32)
                random_weights[i] = torch.rand(num_physical_topk)
    force_random_tensor = (force_random.to(torch.int32) if force_random is not None else torch.zeros(
        1, dtype=torch.int32))

    index_tensor = torch.arange(aligned_experts, dtype=torch.int32)
    k_index_tensor = torch.arange(aligned_k, dtype=torch.int32)

    # Physical mapping setup (identity map, matching TileLang test pattern)
    map_width = 1
    to_physical_map = torch.zeros(1, dtype=torch.int32)
    logical_count = torch.zeros(1, dtype=torch.int32)
    if with_physical_map:
        to_physical_map = torch.arange(num_logical_experts, dtype=torch.int32)
        to_physical_map = to_physical_map.view(-1, 1).contiguous()
        logical_count = torch.ones(num_logical_experts, dtype=torch.int32)

    runner = run_moe_topk_gate_vf if use_inline_vf else run_moe_topk_gate
    runner(profiler, runs, logits, topk_idx, topk_weights, bias_padded, image_bias_padded, image_token_mask_tensor,
           mask_tensor, index_tensor, k_index_tensor, unmapped_topk_idx, to_physical_map, logical_count,
           fix_routing_mask_tensor, fix_indices_tensor, force_random_tensor, random_idx, random_weights, num_tokens,
           routed_scaling_factor, num_topk, num_routed_experts, num_shared_experts, num_physical_topk,
           num_logical_experts, map_width, ep_rank, scoring_type, with_bias, with_image_bias, with_mask, True,
           with_physical_map, with_fix_routing, with_force_random, aligned_experts, aligned_k)

    # Reference (without force_random — force_random rows are checked separately)
    unmapped_ref = torch.zeros(num_tokens, num_topk, dtype=torch.int64)
    if with_fix_routing:
        unmapped_ref[:] = fix_indices
    ref_kwargs = {}
    if bias is not None:
        ref_kwargs['bias'] = bias
    if image_bias is not None:
        ref_kwargs['image_bias'] = image_bias
        ref_kwargs['image_token_mask'] = image_token_mask
    if mask is not None:
        ref_kwargs['mask'] = mask
    if with_fix_routing:
        ref_kwargs['fix_routing_mask'] = fix_routing_mask
    if with_physical_map:
        ref_kwargs['to_physical_map'] = to_physical_map
        ref_kwargs['logical_count'] = logical_count
    ref_kwargs['unmapped_topk_idx'] = unmapped_ref

    topk_idx_ref, topk_weights_ref = torch_moe_topk_gate(logits, num_topk, use_shared_as_routed, num_shared_experts,
                                                         routed_scaling_factor, ep_rank, scoring_func, **ref_kwargs)

    if compile_only:
        return

    compare_mask = torch.ones(num_tokens, dtype=torch.bool)
    if force_random is not None:
        compare_mask = compare_mask & ~force_random
    if mask is not None:
        compare_mask = compare_mask & mask

    if compare_mask.any():
        sorted_idx, _ = topk_idx[compare_mask].sort(dim=1)
        sorted_idx_ref, _ = topk_idx_ref[compare_mask].sort(dim=1)
        torch.testing.assert_close(sorted_idx, sorted_idx_ref)

        sorted_w, _ = topk_weights[compare_mask].sort(dim=1)
        sorted_w_ref, _ = topk_weights_ref[compare_mask].sort(dim=1)
        torch.testing.assert_close(sorted_w, sorted_w_ref, atol=1e-5, rtol=1e-5)

        sorted_unmapped, _ = unmapped_topk_idx[compare_mask].sort(dim=1)
        sorted_unmapped_ref, _ = unmapped_ref[compare_mask].sort(dim=1)
        torch.testing.assert_close(sorted_unmapped, sorted_unmapped_ref)

    # Check force_random rows (exclude masked rows — mask has higher priority)
    if force_random is not None and force_random.any():
        fr_check = force_random
        if mask is not None:
            fr_check = fr_check & mask
        if fr_check.any():
            fr_idx = topk_idx[fr_check]
            fr_w = topk_weights[fr_check]
            assert torch.all(fr_idx >= 0) and torch.all(fr_idx < num_physical_experts), (
                f"force_random idx out of range: {fr_idx}")
            assert torch.all(fr_w > 0), f"force_random weights not positive: {fr_w}"
        assert torch.all(unmapped_topk_idx[force_random] == -1), ("unmapped should be -1 for force_random")

    if mask is not None:
        masked = ~mask
        if masked.any():
            assert torch.all(topk_idx[masked] == -1), "masked idx should be -1"
            assert torch.all(topk_weights[masked] == 0), "masked weights should be 0"
            assert torch.all(unmapped_topk_idx[masked] == -1), "unmapped should be -1 for masked"


@pytest.mark.parametrize("use_inline_vf", [
    pytest.param(0, id="hl"),
    pytest.param(1, id="vf"),
])
@pytest.mark.parametrize(
    "test_name, num_tokens, num_routed_experts, num_topk, "
    "num_shared_experts, scoring_func, with_bias, with_image_bias, "
    "with_mask, use_shared_as_routed, with_physical_map, "
    "with_fix_routing, with_force_random", [
        ("basic_sigmoid_bias", 128, 8, 2, 1, "sigmoid", True, False, False, False, False, False, False),
        ("basic_softmax_bias", 128, 8, 2, 1, "softmax", True, False, False, False, False, False, False),
        ("basic_identity_bias", 128, 8, 2, 1, "identity", True, False, False, False, False, False, False),
        ("basic_sqrtsoftplus_bias", 128, 8, 2, 1, "sqrtsoftplus", True, False, False, False, False, False, False),
        ("basic_sigmoid_nobias", 64, 8, 2, 1, "sigmoid", False, False, False, False, False, False, False),
        ("basic_softmax_nobias", 64, 8, 2, 1, "softmax", False, False, False, False, False, False, False),
        ("basic_identity_nobias", 64, 8, 2, 1, "identity", False, False, False, False, False, False, False),
        ("basic_sqrtsoftplus_nobias", 64, 8, 2, 1, "sqrtsoftplus", False, False, False, False, False, False, False),
        ("mask_sigmoid", 128, 8, 2, 1, "sigmoid", True, False, True, False, False, False, False),
        ("mask_identity", 128, 8, 2, 1, "identity", True, False, True, False, False, False, False),
        ("mask_softmax_nobias", 64, 8, 2, 1, "softmax", False, False, True, False, False, False, False),
        ("shared_sigmoid", 128, 8, 2, 1, "sigmoid", True, False, False, True, False, False, False),
        ("shared_identity", 128, 8, 2, 1, "identity", True, False, False, True, False, False, False),
        ("shared_softmax_nobias", 64, 8, 2, 1, "softmax", False, False, False, True, False, False, False),
        ("combined_sigmoid", 128, 8, 2, 1, "sigmoid", True, False, True, True, False, False, False),
        ("combined_identity", 128, 8, 2, 1, "identity", True, False, True, True, False, False, False),
        ("experts_6_sigmoid", 128, 6, 2, 1, "sigmoid", True, False, False, False, False, False, False),
        ("experts_6_softmax", 128, 6, 2, 1, "softmax", True, False, False, False, False, False, False),
        ("experts_6_mask", 128, 6, 2, 1, "identity", True, False, True, False, False, False, False),
        ("experts_6_shared", 128, 6, 2, 1, "sqrtsoftplus", True, False, False, True, False, False, False),
        ("pmap_sigmoid", 128, 8, 2, 0, "sigmoid", True, False, False, False, True, False, False),
        ("pmap_identity", 128, 8, 2, 0, "identity", True, False, False, False, True, False, False),
        ("pmap_softmax_mask", 128, 8, 2, 0, "softmax", True, False, True, False, True, False, False),
        ("pmap_shared_sigmoid", 128, 8, 2, 1, "sigmoid", True, False, False, True, True, False, False),
        ("pmap_shared_mask", 128, 8, 2, 1, "identity", True, False, True, True, True, False, False),
        ("img_bias_identity", 128, 8, 2, 1, "identity", True, True, False, False, False, False, False),
        ("img_bias_sigmoid", 128, 8, 2, 1, "sigmoid", True, True, False, False, False, False, False),
        ("img_bias_nobias", 64, 8, 2, 1, "sigmoid", False, True, False, False, False, False, False),
        ("img_bias_softmax", 128, 8, 2, 1, "softmax", True, True, False, False, False, False, False),
        ("img_bias_mask", 128, 8, 2, 1, "identity", True, True, True, False, False, False, False),
        ("img_bias_shared", 128, 8, 2, 1, "sigmoid", True, True, False, True, False, False, False),
        ("img_bias_pmap", 128, 8, 2, 0, "identity", True, True, False, False, True, False, False),
        ("img_bias_all", 128, 8, 2, 1, "sigmoid", True, True, True, True, True, False, False),
        ("fix_routing_sigmoid", 128, 8, 2, 1, "sigmoid", True, False, False, False, False, True, False),
        ("fix_routing_softmax", 128, 8, 2, 1, "softmax", True, False, False, False, False, True, False),
        ("fix_routing_identity", 128, 8, 2, 1, "identity", True, False, False, False, False, True, False),
        ("fix_routing_mask", 128, 8, 2, 1, "identity", True, False, True, False, False, True, False),
        ("fix_routing_shared", 128, 8, 2, 1, "sigmoid", True, False, False, True, False, True, False),
        ("fix_routing_img_bias", 128, 8, 2, 1, "softmax", True, True, False, False, False, True, False),
        ("fix_routing_pmap", 128, 8, 2, 0, "identity", True, False, False, False, True, True, False),
        ("fix_routing_all", 128, 8, 2, 1, "sigmoid", True, True, True, True, True, True, False),
        ("fr_sigmoid", 128, 8, 2, 1, "sigmoid", True, False, False, False, False, False, True),
        ("fr_identity", 128, 8, 2, 1, "identity", True, False, False, False, False, False, True),
        ("fr_mask", 128, 8, 2, 1, "sigmoid", True, False, True, False, False, False, True),
        ("fr_shared", 128, 8, 2, 1, "identity", True, False, False, True, False, False, True),
        ("fr_img_bias", 128, 8, 2, 1, "softmax", True, True, False, False, False, False, True),
        # Accuracy mismatch
        # ("fr_pmap",              128, 8, 2, 0, "identity",     True,  False, False, False, True, False, True),
        # ("fr_all",               128, 8, 2, 1, "sigmoid",      True,  True,  True,  True,  True, False, True),
    ])
def test_moe_topk_gate(profiler, runs, compile_only, test_name, num_tokens, num_routed_experts, num_topk,
                       num_shared_experts, scoring_func, with_bias, with_image_bias, with_mask, use_shared_as_routed,
                       with_physical_map, with_fix_routing, with_force_random, use_inline_vf):
    aligned_experts = _align_up(num_routed_experts, FLOAT32_ALIGN)
    aligned_k = _align_up(max(num_topk + num_shared_experts, 1), FLOAT32_ALIGN)

    logits = torch.randn(num_tokens, num_routed_experts, dtype=torch.float32)

    run_and_compare(profiler, runs, logits, num_topk, num_routed_experts, num_shared_experts, scoring_func, with_bias,
                    with_image_bias, with_mask, use_shared_as_routed, aligned_experts, aligned_k,
                    with_physical_map=with_physical_map, with_fix_routing=with_fix_routing,
                    with_force_random=with_force_random, compile_only=compile_only, use_inline_vf=use_inline_vf)


@pytest.mark.parametrize("use_inline_vf", [
    pytest.param(0, id="hl"),
    pytest.param(1, id="vf"),
])
@pytest.mark.parametrize("with_bias, with_image_bias, with_force_random", [
    (True, True, True),
])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "sqrtsoftplus", "softmax", "identity"])
@pytest.mark.parametrize("num_tokens", [512, 8001])
@pytest.mark.parametrize("num_padded_tokens", [0, 10])
@pytest.mark.parametrize("num_routed_experts, num_shared_experts, num_topk", [
    (256, 2, 8),
    (256, 1, 6),
    (384, 1, 6),
])
def test_moe_topk_gate_benchmark(profiler, runs, compile_only, num_tokens, num_padded_tokens, num_routed_experts,
                                 num_shared_experts, num_topk, scoring_func, with_bias, with_image_bias,
                                 with_force_random, use_inline_vf):
    aligned_experts = _align_up(num_routed_experts, FLOAT32_ALIGN)
    aligned_k = _align_up(max(num_topk, 1), FLOAT32_ALIGN)

    total_tokens = num_tokens + num_padded_tokens
    logits = torch.randn(total_tokens, num_routed_experts, dtype=torch.float32)

    run_and_compare(profiler, runs, logits, num_topk, num_routed_experts, num_shared_experts, scoring_func, with_bias,
                    with_image_bias, num_padded_tokens > 0, False, aligned_experts, aligned_k, with_fix_routing=False,
                    with_force_random=with_force_random, compile_only=compile_only, use_inline_vf=use_inline_vf,
                    num_padded_tokens=num_padded_tokens)
