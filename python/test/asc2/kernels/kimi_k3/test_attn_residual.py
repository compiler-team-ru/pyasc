# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference implementation: https://github.com/sgl-project/sgl-kernel-npu/blob/c28ea2a940a53c00f3a0322d9576210a7f5ae92f/python/sgl_kernel_npu/sgl_kernel_npu/kimi_k3/attn_residual.py

import asc2
import pytest
import torch


@asc2.jit(always_compile=True)
def attn_residual_kernel(prefix_ptr: asc2.GlobalAddress, bank_ptr: asc2.GlobalAddress,
                         combined_weight_ptr: asc2.GlobalAddress, indices_ptr: asc2.GlobalAddress,
                         output_ptr: asc2.GlobalAddress, num_tokens, num_valid_blocks, hidden_size: asc2.ConstExpr,
                         variance_epsilon: asc2.ConstExpr, num_candidates_aligned: asc2.ConstExpr,
                         unroll_factor: asc2.ConstExpr):
    """
    Ascend Kimi-K3 attention-residual score and combine pipeline.

    Keep scoring, softmax, and mixing in one persistent vector-core kernel to
    avoid materializing the score matrix and launching a second kernel.  N and
    B stay dynamic so prefill/decode shapes reuse the same compilation.
    """
    # Global memory descriptors
    prefix_gm = asc2.global_tensor(prefix_ptr, [num_tokens, hidden_size])
    bank_gm = asc2.global_tensor(bank_ptr, [num_tokens * num_valid_blocks, hidden_size])
    combined_weight_gm = asc2.global_tensor(combined_weight_ptr, [hidden_size])
    indices_gm = asc2.global_tensor(indices_ptr, [num_candidates_aligned])
    output_gm = asc2.global_tensor(output_ptr, [num_tokens, hidden_size])
    token = asc2.block_idx()
    if token < num_tokens:
        # Load combined weight once per token
        combined_weight = asc2.copy_in(combined_weight_gm, [0], [hidden_size])
        # Initialize indices and scores
        indices_2d = asc2.copy_in(indices_gm, [0], [num_candidates_aligned])
        scores_2d = asc2.full([num_candidates_aligned], -1e9, dtype=asc2.float32)
        # Score each candidate (bank rows + prefix)
        for row in asc2.range(num_valid_blocks + 1, unroll_factor=unroll_factor):
            value = asc2.full([hidden_size], 0, dtype=asc2.float32)
            if row < num_valid_blocks:
                value = asc2.copy_in(bank_gm, [token * num_valid_blocks + row, 0], [hidden_size])
            else:
                value = asc2.copy_in(prefix_gm, [token, 0], [hidden_size])
            # Compute RMSNorm score
            sum_sq = asc2.reduce_sum(value * value)
            inverse_rms = 1.0 / asc2.sqrt(sum_sq / hidden_size + variance_epsilon)
            score = asc2.reduce_sum(value * inverse_rms * combined_weight)
            score_2d = asc2.full([num_candidates_aligned], score, dtype=asc2.float32)
            mask = indices_2d == asc2.full([num_candidates_aligned], row, dtype=asc2.int32)
            scores_2d = asc2.where(mask, score_2d, scores_2d)
        # Softmax over scores
        probabilities_2d = asc2.softmax(scores_2d)
        # Weighted sum of candidates
        output_vec = asc2.zeros([hidden_size], dtype=asc2.float32)
        for row in asc2.range(num_valid_blocks + 1, unroll_factor=unroll_factor):
            value = asc2.full([hidden_size], 0, dtype=asc2.float32)
            if row < num_valid_blocks:
                value = asc2.copy_in(bank_gm, [token * num_valid_blocks + row, 0], [hidden_size])
            else:
                value = asc2.copy_in(prefix_gm, [token, 0], [hidden_size])
            mask = indices_2d == asc2.full([num_candidates_aligned], row, dtype=asc2.int32)
            probability = asc2.reduce_sum(
                asc2.where(mask, probabilities_2d, asc2.zeros([num_candidates_aligned], dtype=asc2.float32)))
            output_vec = output_vec + value * probability
        asc2.copy_out(output_vec, output_gm, [token, 0])


def reference_attn_residual(prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon):
    num_tokens, hidden_size = prefix_sum.shape
    output = torch.zeros_like(prefix_sum)
    for token in range(num_tokens):
        candidates = []
        for row in range(num_valid_blocks):
            candidates.append(bank[token, row, :])
        candidates.append(prefix_sum[token, :])
        candidates = torch.stack(candidates)
        sum_sq = (candidates * candidates).sum(dim=1)
        inverse_rms = torch.rsqrt(sum_sq / hidden_size + variance_epsilon)
        scores = (candidates * inverse_rms.unsqueeze(1) * combined_weight.unsqueeze(0)).sum(dim=1)
        probabilities = torch.softmax(scores, dim=0)
        out = (probabilities.unsqueeze(1) * candidates).sum(dim=0)
        output[token] = out
    return output


@pytest.mark.parametrize("num_tokens, num_valid_blocks, hidden_size, unroll_factor", [
    (4, 2, 128, 2),
    (8, 2, 128, 2),
    (2, 4, 128, 2),
    (16, 1, 256, 1),
    # fla (L, B, T, D) → (num_tokens=B*T, num_valid_blocks=L-1, hidden_size=D)
    # (1000, 2, 4096, 2),      # L=3 UB overflow
    # (15, 14, 4096, 2),       # T=15 UB overflow
    # (1000, 6, 1000, 2),      # D=1000
    # (1000, 6, 2000, 2),      # D=2000
    # (5000, 28, 4096, 2),     # L=29 + B=5 UB overflow
    # (5000, 14, 7186, 2),     # B=5 + D=7186 UB overflow
    # (189, 28, 7186, 2),      # L=29 + D=7186 + T=63 UB overflow
    # (16000, 9, 4096, 2),     # L=10, B=2, T=8000 UB overflow
    # pypto: b=2, t=4096, n=25, d=512 → num_tokens = 2*4096 = 8192
    # (8192, 25, 512, 2),
    # pypto: b=1, t=1023, n=32, d=512 → num_tokens = 1*1023 = 1023
    # (1023, 32, 512, 2),
])
def test_attn_residual(num_tokens, num_valid_blocks, hidden_size, unroll_factor):
    block_num = num_tokens
    variance_epsilon = 1e-5
    num_candidates = num_valid_blocks + 1
    num_candidates_aligned = max(8, 2**(num_candidates - 1).bit_length())
    prefix_sum = torch.randn(num_tokens, hidden_size, dtype=torch.float32)
    bank = torch.randn(num_tokens, num_valid_blocks, hidden_size, dtype=torch.float32)
    combined_weight = torch.randn(hidden_size, dtype=torch.float32)
    bank_2d = bank.reshape(num_tokens * num_valid_blocks, hidden_size)
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    indices = torch.arange(num_candidates_aligned, dtype=torch.int32)
    attn_residual_kernel[block_num](prefix_sum, bank_2d, combined_weight, indices, output, num_tokens, num_valid_blocks,
                                    hidden_size, variance_epsilon, num_candidates_aligned, unroll_factor)
    expected = reference_attn_residual(prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
