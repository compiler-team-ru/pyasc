# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch.nn.functional as F


def stable_topk(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    _, sorted_indices = torch.sort(scores, dim=1, descending=True, stable=True)
    topk_idx = sorted_indices[:, :num_topk].contiguous()
    return topk_idx.as_strided(topk_idx.shape, (num_topk, 1))


def moe_topk_gate(
    logits: torch.Tensor,
    num_topk: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    scoring_func: str,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    image_bias: torch.Tensor | None = None,
    image_token_mask: torch.Tensor | None = None,
    fix_routing_mask: torch.Tensor | None = None,
    to_physical_map: torch.Tensor | None = None,
    logical_count: torch.Tensor | None = None,
    unmapped_topk_idx: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for top-k expert routing."""
    num_tokens_full, num_routed_experts = logits.shape

    if not use_shared_as_routed:
        num_shared_experts = 0

    num_physical_topk = num_topk + num_shared_experts
    num_logical_experts = num_routed_experts + num_shared_experts
    device = logits.device

    if out is None:
        topk_idx_out = torch.full((num_tokens_full, num_physical_topk), -1, dtype=torch.int64, device=device)
        topk_weights_out = torch.zeros((num_tokens_full, num_physical_topk), dtype=torch.float32, device=device)
    else:
        topk_idx_out, topk_weights_out = out
        topk_idx_out.fill_(-1)
        topk_weights_out.zero_()

    if num_tokens_full == 0:
        return topk_idx_out, topk_weights_out

    active = mask if mask is not None else torch.ones(num_tokens_full, dtype=torch.bool, device=device)
    active_indices = active.nonzero(as_tuple=False).squeeze(1)
    num_tokens = active_indices.numel()

    if num_tokens == 0:
        if unmapped_topk_idx is not None:
            unmapped_topk_idx[~active] = -1
        return topk_idx_out, topk_weights_out

    logits_a = logits[active_indices]
    if bias is None:
        bias = torch.zeros(num_routed_experts, dtype=torch.float32, device=device)

    if scoring_func == 'sigmoid':
        scores_wo_bias = torch.sigmoid(logits_a)
    elif scoring_func == 'sqrtsoftplus':
        scores_wo_bias = F.softplus(logits_a).sqrt()
    elif scoring_func == 'softmax':
        scores_wo_bias = torch.softmax(logits_a, dim=-1)
    else:  # Identity
        scores_wo_bias = logits_a

    if image_token_mask is not None:
        image_token_mask_a = image_token_mask[active_indices]
        bias_for_ranking = torch.where(image_token_mask_a.unsqueeze(-1), image_bias.unsqueeze(0), bias.unsqueeze(0))
    else:
        bias_for_ranking = bias.unsqueeze(0)

    scores_biased = ((logits_a + bias_for_ranking) if scoring_func == 'softmax' else
                     (scores_wo_bias + bias_for_ranking))

    fix_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    if fix_routing_mask is not None and unmapped_topk_idx is not None:
        fix_mask = fix_routing_mask[active_indices]

    topk_idx_local = torch.full((num_tokens, num_topk), -1, dtype=torch.int64, device=device)
    topk_score_local = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device=device)

    normal_mask = ~fix_mask
    if normal_mask.any():
        normal_indices = normal_mask.nonzero(as_tuple=False).squeeze(1)
        sb = scores_biased[normal_indices]
        selected = stable_topk(sb, num_topk)
        topk_idx_local[normal_indices] = selected
        topk_score_local[normal_indices] = scores_wo_bias[normal_indices].gather(1, selected)

    if fix_mask.any() and unmapped_topk_idx is not None:
        fix_indices = fix_mask.nonzero(as_tuple=False).squeeze(1)
        pre_idx = unmapped_topk_idx[active_indices[fix_indices]]
        topk_idx_local[fix_indices] = pre_idx
        topk_score_local[fix_indices] = scores_wo_bias[fix_indices].gather(1, pre_idx.clamp(min=0))

    if unmapped_topk_idx is not None:
        unmapped_topk_idx[active_indices] = topk_idx_local
        if mask is not None:
            unmapped_topk_idx[~active] = -1

    topk_sum = topk_score_local.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights_routed = topk_score_local / topk_sum * routed_scaling_factor

    if num_shared_experts > 0:
        shared_idx = torch.arange(num_routed_experts, num_logical_experts, dtype=torch.int64, device=device)
        topk_idx_all = torch.cat([topk_idx_local, shared_idx.expand(num_tokens, -1)], dim=1)
        topk_weights_all = torch.cat(
            [
                topk_weights_routed,
                torch.ones((num_tokens, num_shared_experts), dtype=torch.float32, device=device),
            ],
            dim=1,
        )
    else:
        topk_idx_all, topk_weights_all = topk_idx_local, topk_weights_routed

    if to_physical_map is not None and logical_count is not None:
        physical_map_flat = to_physical_map.reshape(-1)
        map_width = to_physical_map.size(1)
        global_idx = active_indices.to(torch.int64)
        for lane in range(num_physical_topk):
            logical = topk_idx_all[:, lane]
            valid = logical >= 0
            logical_safe = logical.clamp(min=0)
            count = torch.index_select(logical_count, 0, logical_safe).to(torch.int64)
            dup_idx = (ep_rank + global_idx * 23333) % count
            flat_idx = logical_safe * map_width + dup_idx
            physical = torch.index_select(physical_map_flat, 0, flat_idx).to(torch.int64)
            topk_idx_all[:, lane] = torch.where(valid, physical, logical)

    topk_idx_out[active_indices] = topk_idx_all
    topk_weights_out[active_indices] = topk_weights_all

    return topk_idx_out, topk_weights_out
