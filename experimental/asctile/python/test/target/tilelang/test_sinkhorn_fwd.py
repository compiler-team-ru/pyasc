# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Target benchmark + correctness test for the AscTile ``sinkhorn`` forward.
"""

# Reference implementation: https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/mhc/sinkhorn_asc.py#L28

from asc.experimental import asctile
import pytest
import torch

MHC = 4
# Eight fp32 lanes meet the 32-byte UB last-dimension alignment requirement.
MHC_PAD = 8
BATCH = 16
VR_ELEMS = 64
TOKENS_PER_VR = 4
REPEAT = 10
EPS = 1e-6


def _build_reindex_index(batch: int = BATCH) -> torch.Tensor:
    """Build the constant ``[batch*MHC*MHC_PAD]`` int32 forward index for ``reindex``.

    For ``batch > MHC`` the [batch*MHC, MHC_PAD] <-> [MHC, batch*MHC_PAD] permutation is
    NOT self-inverse (the token and row dimensions have different sizes), so forward
    and inverse indices differ. This returns the forward index ([batch*MHC, MHC_PAD] ->
    [MHC, batch*MHC_PAD]); use ``_build_reindex_index_inv`` for the reverse.
    """
    idx = torch.empty(batch * MHC * MHC_PAD, dtype=torch.int32)
    for row in range(MHC):
        for t in range(batch):
            for c in range(MHC_PAD):
                idx[row * batch * MHC_PAD + t * MHC_PAD + c] = t * (MHC * MHC_PAD) + row * MHC_PAD + c
    return idx


def _build_reindex_index_inv(batch: int = BATCH) -> torch.Tensor:
    """Build the inverse index ([MHC, batch*MHC_PAD] -> [batch*MHC, MHC_PAD])."""
    idx = torch.empty(batch * MHC * MHC_PAD, dtype=torch.int32)
    for t in range(batch):
        for row in range(MHC):
            for c in range(MHC_PAD):
                idx[t * (MHC * MHC_PAD) + row * MHC_PAD + c] = row * (batch * MHC_PAD) + t * MHC_PAD + c
    return idx


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def sinkhorn_fwd_kernel(x_ptr: asctile.GlobalAddress, idx_f_ptr: asctile.GlobalAddress,
                        idx_inv_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, num_tokens: int,
                        REPEAT: asctile.ConstExpr[int], EPS: asctile.ConstExpr[float],
                        MHC: asctile.ConstExpr[int] = MHC, MHC_PAD: asctile.ConstExpr[int] = MHC_PAD,
                        BATCH: asctile.ConstExpr[int] = BATCH):
    r"""Forward Sinkhorn normalisation using packed BATCH with hybrid layout.

    Packs ``BATCH`` tokens into one ``[BATCH*MHC, MHC_PAD]`` 2D UB tile.
    Row-normalise uses RB pattern (reduce dim 1) directly; col-normalise
    reindexes to ``[MHC, BATCH*MHC_PAD]`` for RA pattern (reduce dim 0), then
    reindexes back. This cuts all ops by ~BATCH x vs. the single-token
    version. For ``BATCH > MHC`` the forward and inverse permutations differ
    (token and row dimensions have different sizes), so two index tensors are
    used: ``idx_f`` (``[BATCH*MHC, MHC_PAD]`` -> ``[MHC, BATCH*MHC_PAD]``)
    and ``idx_inv`` (the reverse). See ``test_sinkhorn_bwd.py`` for the
    ``-88`` sentinel and reindex rationale.
    """
    x_gm = asctile.global_tensor(x_ptr, [num_tokens * MHC, MHC])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens * MHC, MHC_PAD])
    idx_f_gm = asctile.global_tensor(idx_f_ptr, [BATCH * MHC * MHC_PAD])
    idx_inv_gm = asctile.global_tensor(idx_inv_ptr, [BATCH * MHC * MHC_PAD])
    n_groups = num_tokens // BATCH
    TILE_ROWS = BATCH * MHC

    idx_f_1d = asctile.copy_in(idx_f_gm, [0], [BATCH * MHC * MHC_PAD])
    idx_inv_1d = asctile.copy_in(idx_inv_gm, [0], [BATCH * MHC * MHC_PAD])
    idx_432 = asctile.reshape(idx_f_1d, [MHC, BATCH * MHC_PAD])
    idx_168 = asctile.reshape(idx_inv_1d, [TILE_ROWS, MHC_PAD])

    for grp in range(asctile.block_idx(), n_groups, asctile.block_num()):
        base = grp * TILE_ROWS

        # -88.0 padding underflows to zero in softmax; see the backward test for sentinel rationale.
        x = asctile.copy_in(x_gm, [base, 0], [TILE_ROWS, MHC_PAD], real_shape=[TILE_ROWS, MHC], pad_value=-88.0)
        cur = asctile.softmax(x)

        cur_432 = asctile.reindex(cur, idx_432)
        col_sum = asctile.reduce_sum(cur_432, 0, keep_dims=True)
        cur_432 = cur_432 * (1.0 / (col_sum + EPS))
        cur = asctile.reindex(cur_432, idx_168)

        # Runtime range avoids unrolling REPEAT - 1 copies of the loop body.
        for _ in range(REPEAT - 1):
            r_sum = asctile.reduce_sum(cur, 1, keep_dims=True)
            cur = cur * (1.0 / (r_sum + EPS))
            cur_432 = asctile.reindex(cur, idx_432)
            c_sum = asctile.reduce_sum(cur_432, 0, keep_dims=True)
            cur_432 = cur_432 * (1.0 / (c_sum + EPS))
            cur = asctile.reindex(cur_432, idx_168)

        asctile.copy_out(cur, out_gm, [base, 0])


@asctile.jit(reuse_alloc=0, vf_fusion=True)
def sinkhorn_fwd_vf_kernel(x_ptr: asctile.GlobalAddress, idx_f_ptr: asctile.GlobalAddress,
                           idx_inv_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, num_tokens: int,
                           REPEAT: asctile.ConstExpr[int], EPS: asctile.ConstExpr[float],
                           MHC: asctile.ConstExpr[int] = MHC, MHC_PAD: asctile.ConstExpr[int] = MHC_PAD,
                           BATCH: asctile.ConstExpr[int] = BATCH):
    r"""Forward Sinkhorn normalisation using register-resident ``inline_vf``.

    Packs 4 tokens x 16 elements (4x4 matrices) = 64 fp32 into one vector
    register and performs the entire Sinkhorn normalisation chain in-register
    via ``inline_vf``, mirroring the TileKernels ``mhc_sinkhorn_fwd_asc``
    SimdVF block. The 1D global tensors ([num_tokens * 16] fp32) store each
    token's 4x4 matrix as 16 contiguous elements. Each kernel iteration loads
    4 tokens (64 fp32), runs the VF block, and stores 64 fp32 back. The
    ``idx_f_ptr``/``idx_inv_ptr``/``MHC_PAD``/``BATCH`` parameters are unused
    by this kernel (no reindex is needed).
    """
    x_gm = asctile.global_tensor(x_ptr, [num_tokens * MHC * MHC])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens * MHC * MHC])
    _vf_code = (f'__ubuf__ float* in_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());\n'
                f'__ubuf__ float* out_ptr = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());\n'
                f"""
        AscendC::Reg::MaskReg mask_full = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::RegTensor<int32_t> idx;
        AscendC::Reg::Arange<int32_t, AscendC::Reg::IndexOrder::INCREASE_ORDER>(idx, (int32_t)0);

        AscendC::Reg::RegTensor<int32_t> const_7;
        AscendC::Reg::Duplicate(const_7, (int32_t)7, mask_full);
        AscendC::Reg::RegTensor<int32_t> mod8;
        AscendC::Reg::And(mod8, idx, const_7, mask_full);
        AscendC::Reg::MaskReg mask_lo;
        AscendC::Reg::CompareScalar<int32_t, AscendC::CMPMODE::LT>(mask_lo, mod8, (int32_t)4, mask_full);
        AscendC::Reg::MaskReg mask_hi;
        AscendC::Reg::CompareScalar<int32_t, AscendC::CMPMODE::GE>(mask_hi, mod8, (int32_t)4, mask_full);

        AscendC::Reg::RegTensor<int32_t> const_15;
        AscendC::Reg::Duplicate(const_15, (int32_t)15, mask_full);
        AscendC::Reg::RegTensor<int32_t> token2;
        AscendC::Reg::ShiftRights(token2, idx, (int16_t)4, mask_full);
        AscendC::Reg::ShiftLefts(token2, token2, (int16_t)1, mask_full);
        AscendC::Reg::RegTensor<int32_t> local;
        AscendC::Reg::And(local, idx, const_15, mask_full);
        AscendC::Reg::RegTensor<int32_t> local_div8;
        AscendC::Reg::ShiftRights(local_div8, local, (int16_t)3, mask_full);
        AscendC::Reg::RegTensor<int32_t> bcast_idx2;
        AscendC::Reg::Add(bcast_idx2, token2, local_div8, mask_full);

        AscendC::Reg::RegTensor<int32_t> token_base_tidx;
        AscendC::Reg::ShiftRights(token_base_tidx, idx, (int16_t)4, mask_full);
        AscendC::Reg::ShiftLefts(token_base_tidx, token_base_tidx, (int16_t)4, mask_full);
        AscendC::Reg::RegTensor<int32_t> const_3;
        AscendC::Reg::Duplicate(const_3, (int32_t)3, mask_full);
        AscendC::Reg::RegTensor<int32_t> col;
        AscendC::Reg::And(col, idx, const_3, mask_full);
        AscendC::Reg::RegTensor<int32_t> col4;
        AscendC::Reg::ShiftLefts(col4, col, (int16_t)2, mask_full);
        AscendC::Reg::RegTensor<int32_t> row;
        AscendC::Reg::ShiftRights(row, local, (int16_t)2, mask_full);
        AscendC::Reg::RegTensor<int32_t> t_idx_tmp;
        AscendC::Reg::Add(t_idx_tmp, token_base_tidx, col4, mask_full);
        AscendC::Reg::RegTensor<int32_t> t_idx;
        AscendC::Reg::Add(t_idx, t_idx_tmp, row, mask_full);

        AscendC::Reg::RegTensor<float> mat_0;
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(mat_0, in_ptr);

        // Row max broadcast: vsel(vselr(vcgmax(lo), bcast), vselr(vcgmax(hi), bcast), mask_lo)
        AscendC::Reg::RegTensor<float> rmax_lo, rmax_hi, rmax_lo_b, rmax_hi_b, rmax_b_0;
        AscendC::Reg::ReduceMaxWithDataBlock<float>(rmax_lo, mat_0, mask_lo);
        AscendC::Reg::ReduceMaxWithDataBlock<float>(rmax_hi, mat_0, mask_hi);
        AscendC::Reg::Gather(rmax_lo_b, rmax_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
        AscendC::Reg::Gather(rmax_hi_b, rmax_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
        AscendC::Reg::Select<float>(rmax_b_0, rmax_lo_b, rmax_hi_b, mask_lo);

        AscendC::Reg::RegTensor<float> mat_1;
        AscendC::Reg::FusedExpSub(mat_1, mat_0, rmax_b_0, mask_full);

        // Row sum broadcast
        AscendC::Reg::RegTensor<float> rsum_lo, rsum_hi, rsum_lo_b, rsum_hi_b, rsum_b_0;
        AscendC::Reg::ReduceSumWithDataBlock<float>(rsum_lo, mat_1, mask_lo);
        AscendC::Reg::ReduceSumWithDataBlock<float>(rsum_hi, mat_1, mask_hi);
        AscendC::Reg::Gather(rsum_lo_b, rsum_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
        AscendC::Reg::Gather(rsum_hi_b, rsum_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
        AscendC::Reg::Select<float>(rsum_b_0, rsum_lo_b, rsum_hi_b, mask_lo);

        AscendC::Reg::RegTensor<float> mat_2, mat_3;
        AscendC::Reg::Div<float>(mat_2, mat_1, rsum_b_0, mask_full);
        AscendC::Reg::Adds<float>(mat_3, mat_2, (float){EPS}, mask_full);

        // Col normalize: transpose, reduce, normalize, transpose back
        AscendC::Reg::RegTensor<float> tr_0;
        AscendC::Reg::Gather(tr_0, mat_3, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
        AscendC::Reg::RegTensor<float> csum_lo, csum_hi, csum_lo_b, csum_hi_b, csum_b_0;
        AscendC::Reg::ReduceSumWithDataBlock<float>(csum_lo, tr_0, mask_lo);
        AscendC::Reg::ReduceSumWithDataBlock<float>(csum_hi, tr_0, mask_hi);
        AscendC::Reg::Gather(csum_lo_b, csum_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
        AscendC::Reg::Gather(csum_hi_b, csum_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
        AscendC::Reg::Select<float>(csum_b_0, csum_lo_b, csum_hi_b, mask_lo);
        AscendC::Reg::RegTensor<float> csum_eps_0, tr_1, mat_4;
        AscendC::Reg::Adds<float>(csum_eps_0, csum_b_0, (float){EPS}, mask_full);
        AscendC::Reg::Div<float>(tr_1, tr_0, csum_eps_0, mask_full);
        AscendC::Reg::Gather(mat_4, tr_1, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);

        // repeat-1 iterations of row + col normalize
        AscendC::Reg::RegTensor<float> cur;
        AscendC::Reg::Adds<float>(cur, mat_4, (float)0.0, mask_full);
        for (uint16_t step = 0; step < (uint16_t){REPEAT} - 1; step++) {{
            AscendC::Reg::RegTensor<float> rs_lo, rs_hi, rs_lo_b, rs_hi_b, rsum_b;
            AscendC::Reg::ReduceSumWithDataBlock<float>(rs_lo, cur, mask_lo);
            AscendC::Reg::ReduceSumWithDataBlock<float>(rs_hi, cur, mask_hi);
            AscendC::Reg::Gather(rs_lo_b, rs_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
            AscendC::Reg::Gather(rs_hi_b, rs_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
            AscendC::Reg::Select<float>(rsum_b, rs_lo_b, rs_hi_b, mask_lo);
            AscendC::Reg::RegTensor<float> rsum_eps;
            AscendC::Reg::Adds<float>(rsum_eps, rsum_b, (float){EPS}, mask_full);
            AscendC::Reg::Div<float>(cur, cur, rsum_eps, mask_full);
            AscendC::Reg::RegTensor<float> tr;
            AscendC::Reg::Gather(tr, cur, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
            AscendC::Reg::RegTensor<float> cs_lo, cs_hi, cs_lo_b, cs_hi_b, csum_b;
            AscendC::Reg::ReduceSumWithDataBlock<float>(cs_lo, tr, mask_lo);
            AscendC::Reg::ReduceSumWithDataBlock<float>(cs_hi, tr, mask_hi);
            AscendC::Reg::Gather(cs_lo_b, cs_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
            AscendC::Reg::Gather(cs_hi_b, cs_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
            AscendC::Reg::Select<float>(csum_b, cs_lo_b, cs_hi_b, mask_lo);
            AscendC::Reg::RegTensor<float> csum_eps;
            AscendC::Reg::Adds<float>(csum_eps, csum_b, (float){EPS}, mask_full);
            AscendC::Reg::Div<float>(tr, tr, csum_eps, mask_full);
            AscendC::Reg::Gather(cur, tr, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
        }}

        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(out_ptr, cur, mask_full);
        """)

    num_groups = num_tokens // TOKENS_PER_VR
    for grp in range(asctile.block_idx(), num_groups, asctile.block_num(), unroll_factor=2):
        offset = grp * TOKENS_PER_VR * MHC * MHC
        x_tile = asctile.copy_in(x_gm, [offset], [VR_ELEMS])
        out_tile = asctile.inline_vf(_vf_code, [VR_ELEMS], asctile.float32, [x_tile])
        asctile.copy_out(out_tile, out_gm, [offset])


def _extract_mhc(t_padded: torch.Tensor, view_shape: torch.Size) -> torch.Tensor:
    """Extract [MHC, MHC] from the [MHC, MHC_PAD] result and restore shape."""
    extracted = t_padded[:, :, :MHC].contiguous()
    return extracted.view(view_shape)


def _sinkhorn_ref(x: torch.Tensor, repeat: int = REPEAT, eps: float = EPS) -> torch.Tensor:
    """Torch reference matching the TileKernels formula (eps on dividend + divisor)."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def run_sinkhorn_fwd(profiler, runs, x, num_tokens, mhc, batch, block_num, idx_f, idx_inv):
    """Host entry: launch the forward kernel with high-level ops (no ``inline_vf``).

    Pads ``num_tokens`` up to a multiple of ``batch`` (``batch`` tokens packed
    per outer iteration). No host-side column padding: the kernel DMA-loads
    each ``[batch*MHC, MHC]`` tile and pads the trailing columns to
    ``[batch*MHC, MHC_PAD]`` with ``-88.0`` via ``copy_in(pad_value=...)``.
    Returns the output as ``[num_tokens, mhc, mhc]`` (real tokens only).
    """
    num_tokens_padded = ((num_tokens + batch - 1) // batch) * batch
    x_v = x.view(num_tokens, mhc, mhc).contiguous()
    if num_tokens_padded != num_tokens:
        x_pad = torch.zeros(num_tokens_padded, mhc, mhc, dtype=torch.float32)
        x_pad[:num_tokens] = x_v
        x_v = x_pad

    x_flat = x_v.view(num_tokens_padded * MHC, MHC).contiguous()
    out_flat = torch.zeros(num_tokens_padded * MHC, MHC_PAD, dtype=torch.float32)
    with profiler.profile():
        for _ in range(runs):
            sinkhorn_fwd_kernel[block_num](x_flat, idx_f, idx_inv, out_flat, num_tokens_padded, REPEAT=REPEAT, EPS=EPS,
                                           MHC=mhc, MHC_PAD=MHC_PAD, BATCH=batch)
    return _extract_mhc(out_flat.view(num_tokens_padded, MHC, MHC_PAD)[:num_tokens], (num_tokens, mhc, mhc))


def run_sinkhorn_fwd_vf(profiler, runs, x, num_tokens, mhc, batch, block_num, idx_f, idx_inv):
    """Host entry: launch the forward kernel with the 1-VR ``inline_vf`` SIMD block.

    ``batch``/``idx_f``/``idx_inv`` are accepted for interface parity with
    ``run_sinkhorn_fwd`` but unused: the inline_vf form packs a fixed 4 tokens
    per VR and needs no reindex indices (the pointers are still passed to the
    kernel, which ignores them). Returns the output as
    ``[num_tokens, mhc, mhc]``.
    """
    x_flat = x.view(num_tokens * mhc * mhc).contiguous()

    num_tokens_padded = ((num_tokens + TOKENS_PER_VR - 1) // TOKENS_PER_VR) * TOKENS_PER_VR
    if num_tokens_padded != num_tokens:
        x_padded = torch.zeros(num_tokens_padded * mhc * mhc, dtype=torch.float32)
        x_padded[:num_tokens * mhc * mhc] = x_flat
        x_flat = x_padded

    out_flat = torch.empty(num_tokens_padded * mhc * mhc, dtype=torch.float32)
    with profiler.profile():
        for _ in range(runs):
            sinkhorn_fwd_vf_kernel[block_num](x_flat, idx_f, idx_inv, out_flat, num_tokens_padded, REPEAT=REPEAT,
                                              EPS=EPS, MHC=mhc)
    return out_flat[:num_tokens * mhc * mhc].view(num_tokens, mhc, mhc)


# yapf: disable
@pytest.mark.parametrize("use_inline_vf", [
    pytest.param(0, id="hl"),
    pytest.param(1, id="vf"),
])
@pytest.mark.parametrize("_, n0, n1, mhc, block_num, batch", [
# PYASC_TESTS_BEGIN
    pytest.param(None, 1, 512,  4, 72, 8,  id="n1-n512-mhc4-fwd"),
    pytest.param(None, 1, 8192, 4, 72, 32, id="n1-n8192-mhc4-fwd"),
# PYASC_TESTS_END
])
# yapf: enable
def test_sinkhorn_fwd(profiler, runs, _, n0, n1, mhc, block_num, batch, compile_only, use_inline_vf):
    num_tokens = n0 * n1
    x = torch.randn(num_tokens, mhc, mhc, dtype=torch.float32)

    idx_f = _build_reindex_index(batch)
    idx_inv = _build_reindex_index_inv(batch)

    runner = run_sinkhorn_fwd_vf if use_inline_vf else run_sinkhorn_fwd
    out = runner(profiler, runs, x, num_tokens, mhc, batch, block_num, idx_f, idx_inv)
    if compile_only:
        return
    ref = _sinkhorn_ref(x)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
