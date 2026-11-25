# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Target benchmark + correctness test for the AscTile ``topk_gate`` rewrite.
"""

# Reference implementation: https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/moe/topk_gate_asc.py#L9

from asc.experimental import asctile
import pytest
import torch

# Finite sentinel because some copy_in paths reject -inf.
NEG_INF = -1e30
BIG_INT = 999999
# A 256-byte Ascend C vector register holds 64 float32/int32 elements.
VREG_ELEMS = 64
# BATCH must divide the host-padded token count.
BATCH = 8


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def topk_gate_kernel(scores_ptr: asctile.GlobalAddress, idx_ptr: asctile.GlobalAddress,
                     k_idx_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, num_tokens: int,
                     num_experts: asctile.ConstExpr[int], N_PAD: asctile.ConstExpr[int], K: asctile.ConstExpr[int],
                     K_PAD: asctile.ConstExpr[int]):
    """TopK-gate kernel using high-level batched AscTile operations.

    ``BATCH`` rows are loaded as a single 2D ``[BATCH, N_PAD]`` UB tensor and
    processed simultaneously with ``reduce_max``, ``reduce_min``, ``where``,
    ``broadcast_to``. The kernel computes indices in int32 then casts to int64
    before the batched store, so the host receives int64 directly. The
    ``idx_ptr``/``k_idx_ptr`` auxiliary index buffers provide the expert and
    output-slot indices.
    """
    scores_gm = asctile.global_tensor(scores_ptr, [num_tokens, num_experts])
    idx_gm = asctile.global_tensor(idx_ptr, [N_PAD])
    k_idx_gm = asctile.global_tensor(k_idx_ptr, [K_PAD])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens, K])

    indices = asctile.copy_in(idx_gm, [0], [N_PAD])
    k_index = asctile.copy_in(k_idx_gm, [0], [K_PAD])

    _batch = int(BATCH)
    _n_pad = int(N_PAD)
    _k_pad = int(K_PAD)
    _num_experts = int(num_experts)

    for batch_idx in range(asctile.block_idx(), num_tokens // _batch, asctile.block_num()):
        base_row = batch_idx * _batch
        # DMA pads the expert tail with NEG_INF so padding lanes cannot win.
        ranked_scores = asctile.copy_in(scores_gm, [base_row, 0], [_batch, _n_pad], real_shape=[_batch, _num_experts],
                                        pad_value=NEG_INF)

        indices_2d = indices.expand_dims(0).broadcast_to(_batch, _n_pad)
        topk_out = asctile.full([_batch, _k_pad], -1, dtype=asctile.int32)

        for k in range(K, unroll_factor=2):
            max_val = asctile.reduce_max(ranked_scores, 1, keep_dims=True)
            max_broadcast = max_val.broadcast_to(_batch, _n_pad)
            eq_mask = ranked_scores == max_broadcast
            # Taking the minimum candidate index breaks equal-score ties.
            cand = asctile.where(eq_mask, indices_2d, BIG_INT)
            winner = asctile.reduce_min(cand, 1, keep_dims=True)
            winner_kpad = winner.broadcast_to(_batch, _k_pad)
            # Broadcast int32 slot indices before comparison: expand_dims is unsupported on int1.
            k_index_2d = k_index.expand_dims(0).broadcast_to(_batch, _k_pad)
            k_mask_2d = k_index_2d == k
            topk_out = asctile.where(k_mask_2d, winner_kpad, topk_out)
            winner_npad = winner.broadcast_to(_batch, _n_pad)
            winner_mask = (indices_2d == winner_npad)
            ranked_scores = asctile.where(winner_mask, NEG_INF, ranked_scores)

        # real_shape stores only K entries per row, excluding K_PAD lanes.
        topk_out_i64 = asctile.cast(topk_out, asctile.int64)
        asctile.copy_out(topk_out_i64, out_gm, [base_row, 0], real_shape=[_batch, int(K)])


@asctile.jit(reuse_alloc=0, vf_fusion=True, always_compile=True)
def topk_gate_vf_kernel(scores_ptr: asctile.GlobalAddress, idx_ptr: asctile.GlobalAddress,
                        k_idx_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, num_tokens: int,
                        num_experts: asctile.ConstExpr[int], N_PAD: asctile.ConstExpr[int], K: asctile.ConstExpr[int],
                        K_PAD: asctile.ConstExpr[int]):
    """TopK-gate kernel using a register-resident, fully batched ``inline_vf``.

    The whole batch is processed by a single ``inline_vf`` call whose row loop
    lives inside the ``__VEC_SCOPE__`` block (one VF scope per batch instead
    of one per row). Scores/indices are loaded into named register variables
    (one per vreg) and stay resident across all K iterations; winner masking
    is done in-register via ``Select``. The winner index is written directly
    as int32 into the lower 4 bytes of each int64 output slot via
    ``DataCopy<..., DIST_FIRST_ELEMENT_B32>`` -- eliminating the separate
    int32->int64 cast op -- exactly TileKernels' ``T.view(topk_idx, (.., K*2),
    int32)`` + ``vsts(.., ONEPT_B32)`` trick. ``RegTensor`` cannot be placed
    in C arrays, so the ``inline_vf`` code is generated at JIT time with one
    named variable per vreg. The ``idx_ptr``/``k_idx_ptr`` parameters are
    unused by this kernel.
    """
    scores_gm = asctile.global_tensor(scores_ptr, [num_tokens, num_experts])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens, K])

    _BATCH = 8
    _num_vregs = int(N_PAD // VREG_ELEMS)
    _k = int(K)
    _n_pad = int(N_PAD)
    _k_pad = int(K_PAD)
    _batch = _BATCH
    _num_experts = int(num_experts)
    # RegTensor values cannot be C arrays, so unroll vregs into named variables at JIT time.
    # $0 reinterprets int64 output as int32 pairs; VL1 selects lane-0 reduction results.
    _vf_prefix = (
        f'constexpr uint16_t NUM_TOPK = {_k};\n'
        f'constexpr int32_t N_PAD_C = {_n_pad};\n'
        f'constexpr int32_t K_PAD_C = {_k_pad};\n'
        f'constexpr int32_t BATCH_C = {_batch};\n'
        f'constexpr int32_t NUM_EXPERTS = {_num_experts};\n'
        '__ubuf__ int32_t* out_ptr = reinterpret_cast<__ubuf__ int32_t*>($0.GetPhyAddr());\n'
        '__ubuf__ float* scores_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());\n'
        'constexpr int32_t OUT_STRIDE = 2 * K_PAD_C;\n'
        'AscendC::Reg::MaskReg mask_all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();\n'
        'AscendC::Reg::MaskReg mask_vl1 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();\n'
        'AscendC::Reg::RegTensor<float> neg_inf_reg;\n'
        'AscendC::Reg::Duplicate(neg_inf_reg, (float)(-1e30));\n'
        'AscendC::Reg::RegTensor<int32_t> int_max_reg;\n'
        'AscendC::Reg::Duplicate(int_max_reg, (int32_t)0x7FFFFFFF);\n'
        'AscendC::Reg::RegTensor<int32_t> zero_reg;\n'
        'AscendC::Reg::Duplicate(zero_reg, (int32_t)0);\n'
        'AscendC::Reg::RegTensor<int32_t> num_exp_reg;\n'
        'AscendC::Reg::Duplicate(num_exp_reg, (int32_t)NUM_EXPERTS);\n'
        'AscendC::Reg::MaskReg pad_mask;\n')
    _lines = []
    for _v in asctile.static_range(_num_vregs):
        _lines.append(f'AscendC::Reg::RegTensor<float> scores_v{_v};')
        _lines.append(f'AscendC::Reg::RegTensor<int32_t> indices_v{_v};')
    _lines += [
        'AscendC::Reg::RegTensor<float> max_acc, max_scalar, max_broadcast;',
        'AscendC::Reg::RegTensor<int32_t> cand_reg, min_acc, winner_scalar, winner_broadcast;',
        'AscendC::Reg::MaskReg eq_mask, winner_mask;',
        'for (uint16_t row = 0; row < (uint16_t)BATCH_C; row += 1) {',
    ]
    # Reapply NEG_INF to padding lanes to guard against DMA padding artefacts.
    for _v in asctile.static_range(_num_vregs):
        _voff = _v * VREG_ELEMS
        _lines.append(f'    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(scores_v{_v}, '
                      f'scores_ptr + row * N_PAD_C + {_voff});')
        _lines.append(f'    AscendC::Reg::Arange<int32_t, AscendC::Reg::IndexOrder::INCREASE_ORDER>(indices_v{_v}, '
                      f'(int32_t){_voff});')
        _lines.append(f'    AscendC::Reg::CompareScalar<int32_t, AscendC::CMPMODE::LT>(pad_mask, indices_v{_v}, '
                      f'(int32_t)NUM_EXPERTS, mask_all);')
        _lines.append(f'    AscendC::Reg::Select(scores_v{_v}, scores_v{_v}, neg_inf_reg, pad_mask);')
    _lines += [
        '    for (uint16_t k = 0; k < NUM_TOPK; k += 1) {',
        '        AscendC::Reg::Duplicate(max_acc, (float)(-1e30));',
    ]
    for _v in asctile.static_range(_num_vregs):
        _lines.append(f'        AscendC::Reg::Max(max_acc, max_acc, scores_v{_v}, mask_all);')
    _lines += [
        '        AscendC::Reg::ReduceMax(max_scalar, max_acc, mask_all);',
        '        AscendC::Reg::Duplicate(max_broadcast, max_scalar, mask_all);',
        '        AscendC::Reg::Duplicate(min_acc, (int32_t)0x7FFFFFFF);',
    ]
    for _v in asctile.static_range(_num_vregs):
        _lines.append(
            f'        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(eq_mask, scores_v{_v}, max_broadcast, mask_all);'
        )
        _lines.append(f'        AscendC::Reg::Select(cand_reg, indices_v{_v}, int_max_reg, eq_mask);')
        _lines.append('        AscendC::Reg::Min(min_acc, min_acc, cand_reg, mask_all);')
    _lines += [
        '        AscendC::Reg::ReduceMin(winner_scalar, min_acc, mask_all);',
        '        AscendC::Reg::Duplicate(winner_broadcast, winner_scalar, mask_all);',
        # Store lane 0 in the low int32 half of slot k; zero the high half because UB may hold stale data.
        '        AscendC::Reg::DataCopy<int32_t, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>('
        'out_ptr + row * OUT_STRIDE + 2 * k, winner_scalar, mask_vl1);',
        '        AscendC::Reg::DataCopy<int32_t, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>('
        'out_ptr + row * OUT_STRIDE + 2 * k + 1, zero_reg, mask_vl1);',
    ]
    for _v in asctile.static_range(_num_vregs):
        _lines.append(f'        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::EQ>(winner_mask, indices_v{_v}, '
                      f'winner_broadcast, mask_all);')
        _lines.append(f'        AscendC::Reg::Select(scores_v{_v}, neg_inf_reg, scores_v{_v}, winner_mask);')
    _lines += [
        '    }',
        '}',
    ]
    _vf_code = _vf_prefix + '\n'.join(_lines)

    for batch_idx in range(asctile.block_idx(), num_tokens // _batch, asctile.block_num(), unroll_factor=2):
        base_row = batch_idx * _batch
        scores_batch = asctile.copy_in(scores_gm, [base_row, 0], [_batch, _n_pad], real_shape=[_batch, _num_experts],
                                       pad_value=NEG_INF)
        scores_batch = asctile.reshape(scores_batch, _batch * _n_pad)
        out_batch = asctile.inline_vf(_vf_code, [_batch, _k_pad], asctile.int64, [scores_batch])
        # real_shape stores only K entries per row, excluding K_PAD lanes.
        asctile.copy_out(out_batch, out_gm, [base_row, 0], real_shape=[_batch, int(K)])


def run_topk_gate(profiler, runs, scores_padded, num_tokens_padded, num_experts, n_pad, num_topk, k_pad, block_num):
    """Host entry: launch the TopK kernel with high-level ops (no ``inline_vf``).

    Allocates the auxiliary index buffers (``idx_ptr``/``k_idx_ptr``, used by
    this kernel only) and the int64 output. Returns ``out`` (int64
    ``[num_tokens_padded, num_topk]``); the caller slices ``[:num_tokens]``.
    """
    idx_buf = torch.arange(n_pad, dtype=torch.int32)
    k_idx_buf = torch.arange(k_pad, dtype=torch.int32)
    out = torch.zeros(num_tokens_padded, num_topk, dtype=torch.int64)
    with profiler.profile():
        for _ in range(runs):
            topk_gate_kernel[block_num](scores_padded, idx_buf, k_idx_buf, out, num_tokens_padded,
                                        num_experts=num_experts, N_PAD=n_pad, K=num_topk, K_PAD=k_pad)
    return out


def run_topk_gate_vf(profiler, runs, scores_padded, num_tokens_padded, num_experts, n_pad, num_topk, k_pad, block_num):
    """Host entry: launch the TopK kernel with the register-resident ``inline_vf``.

    The auxiliary index buffers are still allocated and passed to retain the
    common launch interface, but the inline_vf kernel ignores them: expert
    indices are generated in-register via ``Arange``. Returns
    ``out`` (int64 ``[num_tokens_padded, num_topk]``); the caller slices
    ``[:num_tokens]``.
    """
    idx_buf = torch.arange(n_pad, dtype=torch.int32)
    k_idx_buf = torch.arange(k_pad, dtype=torch.int32)
    out = torch.zeros(num_tokens_padded, num_topk, dtype=torch.int64)
    with profiler.profile():
        for _ in range(runs):
            topk_gate_vf_kernel[block_num](scores_padded, idx_buf, k_idx_buf, out, num_tokens_padded,
                                           num_experts=num_experts, N_PAD=n_pad, K=num_topk, K_PAD=k_pad)
    return out


def _next_mul64(n: int) -> int:
    return ((n + VREG_ELEMS - 1) // VREG_ELEMS) * VREG_ELEMS


def _ref_topk(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    _, indices = torch.sort(scores, dim=1, descending=True, stable=True)
    return indices[:, :num_topk].contiguous()


# yapf: disable
@pytest.mark.parametrize("use_inline_vf", [
    pytest.param(0, id="hl"),
    pytest.param(1, id="vf"),
])
@pytest.mark.parametrize("_, num_tokens, num_experts, num_topk, block_num", [
# PYASC_TESTS_BEGIN
    pytest.param(None, 512,  16,  6, 72, id="t512-e16-k6"),
    pytest.param(None, 8001, 16,  6, 72, id="t8001-e16-k6"),
    pytest.param(None, 512,  32,  6, 72, id="t512-e32-k6"),
    pytest.param(None, 8001, 32,  6, 72, id="t8001-e32-k6"),
    pytest.param(None, 512,  36,  6, 72, id="t512-e36-k6"),
    pytest.param(None, 8001, 36,  6, 72, id="t8001-e36-k6"),
    pytest.param(None, 512,  64,  6, 72, id="t512-e64-k6"),
    pytest.param(None, 8001, 64,  6, 72, id="t8001-e64-k6"),
    pytest.param(None, 512,  72,  6, 72, id="t512-e72-k6"),
    pytest.param(None, 8001, 72,  6, 72, id="t8001-e72-k6"),
    pytest.param(None, 512,  96,  6, 72, id="t512-e96-k6"),
    pytest.param(None, 8001, 96,  6, 72, id="t8001-e96-k6"),
    pytest.param(None, 512,  108, 6, 72, id="t512-e108-k6"),
    pytest.param(None, 8001, 108, 6, 72, id="t8001-e108-k6"),
    pytest.param(None, 512,  128, 6, 72, id="t512-e128-k6"),
    pytest.param(None, 8001, 128, 6, 72, id="t8001-e128-k6"),
    pytest.param(None, 512,  144, 6, 72, id="t512-e144-k6"),
    pytest.param(None, 8001, 144, 6, 72, id="t8001-e144-k6"),
    pytest.param(None, 512,  256, 8, 72, id="t512-e256-k8"),
    pytest.param(None, 8001, 256, 8, 72, id="t8001-e256-k8"),
# PYASC_TESTS_END
])
# yapf: enable
def test_topk_gate(profiler, runs, _, num_tokens, num_experts, num_topk, block_num, compile_only, use_inline_vf):
    n_pad = _next_mul64(num_experts)
    k_pad = _next_mul64(num_topk)
    # Pad the token count for batched DMA; copy_in pads the expert tail in hardware.
    num_tokens_padded = ((num_tokens + BATCH - 1) // BATCH) * BATCH
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    scores_padded = torch.zeros(num_tokens_padded, num_experts, dtype=torch.float32)
    scores_padded[:num_tokens, :num_experts] = scores

    runner = run_topk_gate_vf if use_inline_vf else run_topk_gate
    out = runner(profiler, runs, scores_padded, num_tokens_padded, num_experts, n_pad, num_topk, k_pad, block_num)
    if compile_only:
        return

    ref = _ref_topk(scores, num_topk)
    torch.testing.assert_close(out[:num_tokens, :num_topk], ref)
