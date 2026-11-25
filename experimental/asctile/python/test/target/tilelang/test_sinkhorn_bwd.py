# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Target benchmark + correctness test for the AscTile ``sinkhorn`` backward.
"""

# Reference implementation: https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/mhc/sinkhorn_asc.py#L170

from asc.experimental import asctile
import pytest
import torch

MHC = 4
# Eight fp32 lanes meet the 32-byte UB last-dimension alignment requirement.
MHC_PAD = 8
BATCH = 16
VR_ELEMS = 64
TOKENS_PER_BLOCK = 8
BLOCK_ELEMS = 2 * VR_ELEMS
REPEAT = 10
EPS = 1e-6


def _build_reindex_index(batch: int = BATCH) -> torch.Tensor:
    """Build the constant forward index ([batch*MHC, MHC_PAD] -> [MHC, batch*MHC_PAD]).

    For ``batch > MHC`` the forward and inverse permutations differ (token and
    row dimensions have different sizes). Use ``_build_reindex_index_inv`` for
    the reverse direction. See ``test_sinkhorn_fwd.py`` for details.
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


@asctile.jit(static_alloc=True, vf_fusion=True)
def sinkhorn_bwd_vf_kernel(grad_ptr: asctile.GlobalAddress, x_ptr: asctile.GlobalAddress,
                           mask_ptr: asctile.GlobalAddress, idx_f_ptr: asctile.GlobalAddress,
                           idx_inv_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, num_tokens: int,
                           REPEAT: asctile.ConstExpr[int], EPS: asctile.ConstExpr[float],
                           MHC: asctile.ConstExpr[int] = MHC, MHC_PAD: asctile.ConstExpr[int] = MHC_PAD,
                           BATCH: asctile.ConstExpr[int] = BATCH):
    r"""Sinkhorn backward via forward recompute + reverse iteration.

    Packs 8 tokens x 16 elements (4x4 matrices) = 128 fp32 into two vector
    registers (2-VR/8-token block) and performs the entire forward-recompute +
    backward-reverse chain in-register via ``inline_vf``, mirroring the
    TileKernels ``mhc_sinkhorn_bwd_asc`` SimdVF block (2-VR interleaved
    variant). VR0 and VR1 operations are interleaved to hide vld/vsts latency.
    The 1D global tensors ([num_tokens * 16] fp32) store each token's 4x4
    matrix as 16 contiguous elements; each iteration loads 8 tokens (128 fp32
    for x, 128 fp32 for grad), runs the VF block, and stores 128 fp32 back. UB
    scratchpad (4 sections: xs0, xs1, sums0, sums1, each ``[2*REPEAT,
    VR_ELEMS]`` fp32) holds the forward intermediate states for the reverse
    phase, matching the TileKernels ``alloc_shared`` approach. The
    ``mask_ptr``/``idx_f_ptr``/``idx_inv_ptr``/``MHC_PAD``/``BATCH``
    parameters are unused by this kernel (no reindex or padded-column mask is
    needed). The VF code is a single triple-quoted f-string literal (same
    pattern as ``test_moe_topk_gate.py`` / ``test_post_fwd.py``): VR is fixed
    = 2, so v=0 and v=1 are spelled out explicitly, and the ``REPEAT``/
    ``EPS`` ConstExprs (plus the ``n_states``/``n_states*_vr`` offsets) are
    interpolated into the f-string.
    """
    n_states = 2 * REPEAT
    x_gm = asctile.global_tensor(x_ptr, [num_tokens * MHC * MHC])
    grad_gm = asctile.global_tensor(grad_ptr, [num_tokens * MHC * MHC])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens * MHC * MHC])

    # Persistent scratch lets TPipe pipeline DMA loads without aliasing; each
    # inline_vf call consumes its scratch before the next group.
    # Layout: xs0 | xs1 | sums0 | sums1, each n_states * VR_ELEMS fp32.
    scratch = asctile.zeros([4 * n_states * VR_ELEMS], asctile.float32)

    _vr = VR_ELEMS
    _vf_code = f"""
__ubuf__ float* out_ptr = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
__ubuf__ float* x_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
__ubuf__ float* grad_ptr = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
__ubuf__ float* xs_ptr = reinterpret_cast<__ubuf__ float*>($3.GetPhyAddr());
__ubuf__ float* xs0_ptr = xs_ptr;
__ubuf__ float* xs1_ptr = xs_ptr + {n_states * _vr};
__ubuf__ float* sums0_ptr = xs_ptr + {2 * n_states * _vr};
__ubuf__ float* sums1_ptr = xs_ptr + {3 * n_states * _vr};

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

AscendC::Reg::RegTensor<float> x0, x1;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x0, x_ptr);
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x1, x_ptr + {_vr});
AscendC::Reg::RegTensor<float> g0, g1;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(g0, grad_ptr);
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(g1, grad_ptr + {_vr});

// State 0: softmax(x) — VR0
AscendC::Reg::RegTensor<float> rmax0_lo, rmax0_hi, rmax0_lo_b, rmax0_hi_b, rmax0_b;
AscendC::Reg::ReduceMaxWithDataBlock<float>(rmax0_lo, x0, mask_lo);
AscendC::Reg::ReduceMaxWithDataBlock<float>(rmax0_hi, x0, mask_hi);
AscendC::Reg::Gather(rmax0_lo_b, rmax0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(rmax0_hi_b, rmax0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(rmax0_b, rmax0_lo_b, rmax0_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> ex0;
AscendC::Reg::FusedExpSub(ex0, x0, rmax0_b, mask_full);
AscendC::Reg::RegTensor<float> rs0_lo, rs0_hi, rs0_lo_b, rs0_hi_b, rs0_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(rs0_lo, ex0, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(rs0_hi, ex0, mask_hi);
AscendC::Reg::Gather(rs0_lo_b, rs0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(rs0_hi_b, rs0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(rs0_b, rs0_lo_b, rs0_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> sm0;
AscendC::Reg::Div<float>(sm0, ex0, rs0_b, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(xs0_ptr + 0*{_vr}, sm0, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums0_ptr + 0*{_vr}, rs0_b, mask_full);

// State 0: softmax(x) — VR1
AscendC::Reg::RegTensor<float> rmax1_lo, rmax1_hi, rmax1_lo_b, rmax1_hi_b, rmax1_b;
AscendC::Reg::ReduceMaxWithDataBlock<float>(rmax1_lo, x1, mask_lo);
AscendC::Reg::ReduceMaxWithDataBlock<float>(rmax1_hi, x1, mask_hi);
AscendC::Reg::Gather(rmax1_lo_b, rmax1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(rmax1_hi_b, rmax1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(rmax1_b, rmax1_lo_b, rmax1_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> ex1;
AscendC::Reg::FusedExpSub(ex1, x1, rmax1_b, mask_full);
AscendC::Reg::RegTensor<float> rs1_lo, rs1_hi, rs1_lo_b, rs1_hi_b, rs1_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(rs1_lo, ex1, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(rs1_hi, ex1, mask_hi);
AscendC::Reg::Gather(rs1_lo_b, rs1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(rs1_hi_b, rs1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(rs1_b, rs1_lo_b, rs1_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> sm1;
AscendC::Reg::Div<float>(sm1, ex1, rs1_b, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(xs1_ptr + 0*{_vr}, sm1, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums1_ptr + 0*{_vr}, rs1_b, mask_full);

// State 1: col normalise — VR0
AscendC::Reg::RegTensor<float> sm_eps0;
AscendC::Reg::Adds<float>(sm_eps0, sm0, (float){EPS}, mask_full);
AscendC::Reg::RegTensor<float> tr0_0;
AscendC::Reg::Gather(tr0_0, sm_eps0, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::RegTensor<float> cs0_lo, cs0_hi, cs0_lo_b, cs0_hi_b, cs0_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(cs0_lo, tr0_0, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(cs0_hi, tr0_0, mask_hi);
AscendC::Reg::Gather(cs0_lo_b, cs0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(cs0_hi_b, cs0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(cs0_b, cs0_lo_b, cs0_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> cs0_b_t;
AscendC::Reg::Gather(cs0_b_t, cs0_b, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(xs0_ptr + 1*{_vr}, sm_eps0, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums0_ptr + 1*{_vr}, cs0_b_t, mask_full);
AscendC::Reg::RegTensor<float> cs_eps0, tr1_0, cur0;
AscendC::Reg::Adds<float>(cs_eps0, cs0_b, (float){EPS}, mask_full);
AscendC::Reg::Div<float>(tr1_0, tr0_0, cs_eps0, mask_full);
AscendC::Reg::Gather(cur0, tr1_0, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);

// State 1: col normalise — VR1
AscendC::Reg::RegTensor<float> sm_eps1;
AscendC::Reg::Adds<float>(sm_eps1, sm1, (float){EPS}, mask_full);
AscendC::Reg::RegTensor<float> tr0_1;
AscendC::Reg::Gather(tr0_1, sm_eps1, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::RegTensor<float> cs1_lo, cs1_hi, cs1_lo_b, cs1_hi_b, cs1_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(cs1_lo, tr0_1, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(cs1_hi, tr0_1, mask_hi);
AscendC::Reg::Gather(cs1_lo_b, cs1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(cs1_hi_b, cs1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(cs1_b, cs1_lo_b, cs1_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> cs1_b_t;
AscendC::Reg::Gather(cs1_b_t, cs1_b, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(xs1_ptr + 1*{_vr}, sm_eps1, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums1_ptr + 1*{_vr}, cs1_b_t, mask_full);
AscendC::Reg::RegTensor<float> cs_eps1, tr1_1, cur1;
AscendC::Reg::Adds<float>(cs_eps1, cs1_b, (float){EPS}, mask_full);
AscendC::Reg::Div<float>(tr1_1, tr0_1, cs_eps1, mask_full);
AscendC::Reg::Gather(cur1, tr1_1, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);

// States 2..n_states-1: row/col normalise
for (uint16_t step = 2; step < (uint16_t){n_states}; step++) {{
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(xs0_ptr + step*{_vr}, cur0, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(xs1_ptr + step*{_vr}, cur1, mask_full);
    if (step % 2 == 0) {{
AscendC::Reg::RegTensor<float> s0_lo, s0_hi, s0_lo_b, s0_hi_b, s0_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(s0_lo, cur0, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(s0_hi, cur0, mask_hi);
AscendC::Reg::Gather(s0_lo_b, s0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(s0_hi_b, s0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(s0_b, s0_lo_b, s0_hi_b, mask_lo);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums0_ptr + step*{_vr}, s0_b, mask_full);
        AscendC::Reg::RegTensor<float> s_eps0;
        AscendC::Reg::Adds<float>(s_eps0, s0_b, (float){EPS}, mask_full);
        AscendC::Reg::Div<float>(cur0, cur0, s_eps0, mask_full);
AscendC::Reg::RegTensor<float> s1_lo, s1_hi, s1_lo_b, s1_hi_b, s1_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(s1_lo, cur1, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(s1_hi, cur1, mask_hi);
AscendC::Reg::Gather(s1_lo_b, s1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(s1_hi_b, s1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(s1_b, s1_lo_b, s1_hi_b, mask_lo);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums1_ptr + step*{_vr}, s1_b, mask_full);
        AscendC::Reg::RegTensor<float> s_eps1;
        AscendC::Reg::Adds<float>(s_eps1, s1_b, (float){EPS}, mask_full);
        AscendC::Reg::Div<float>(cur1, cur1, s_eps1, mask_full);
    }} else {{
AscendC::Reg::RegTensor<float> tr_s0;
AscendC::Reg::Gather(tr_s0, cur0, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::RegTensor<float> s0_lo, s0_hi, s0_lo_b, s0_hi_b, s0_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(s0_lo, tr_s0, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(s0_hi, tr_s0, mask_hi);
AscendC::Reg::Gather(s0_lo_b, s0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(s0_hi_b, s0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(s0_b, s0_lo_b, s0_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> s0_b_t;
AscendC::Reg::Gather(s0_b_t, s0_b, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
        AscendC::Reg::RegTensor<float> s_eps0;
        AscendC::Reg::Adds<float>(s_eps0, s0_b, (float){EPS}, mask_full);
        AscendC::Reg::RegTensor<float> tr_d0;
        AscendC::Reg::Div<float>(tr_d0, tr_s0, s_eps0, mask_full);
        AscendC::Reg::Gather(cur0, tr_d0, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums0_ptr + step*{_vr}, s0_b_t, mask_full);
AscendC::Reg::RegTensor<float> tr_s1;
AscendC::Reg::Gather(tr_s1, cur1, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::RegTensor<float> s1_lo, s1_hi, s1_lo_b, s1_hi_b, s1_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(s1_lo, tr_s1, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(s1_hi, tr_s1, mask_hi);
AscendC::Reg::Gather(s1_lo_b, s1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(s1_hi_b, s1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(s1_b, s1_lo_b, s1_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> s1_b_t;
AscendC::Reg::Gather(s1_b_t, s1_b, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
        AscendC::Reg::RegTensor<float> s_eps1;
        AscendC::Reg::Adds<float>(s_eps1, s1_b, (float){EPS}, mask_full);
        AscendC::Reg::RegTensor<float> tr_d1;
        AscendC::Reg::Div<float>(tr_d1, tr_s1, s_eps1, mask_full);
        AscendC::Reg::Gather(cur1, tr_d1, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sums1_ptr + step*{_vr}, s1_b_t, mask_full);
    }}
}}

// mem_bar(VST_VLD) between Phase 1 vsts and Phase 2 vld
AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
// Phase 2: reverse the normalise steps
for (uint16_t step = (uint16_t){n_states} - 1; step > 0; step--) {{
AscendC::Reg::RegTensor<float> xi0;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(xi0, xs0_ptr + step*{_vr});
AscendC::Reg::RegTensor<float> sm0;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(sm0, sums0_ptr + step*{_vr});
    AscendC::Reg::RegTensor<float> denom0;
    AscendC::Reg::Adds<float>(denom0, sm0, (float){EPS}, mask_full);
    AscendC::Reg::RegTensor<float> tmp0;
    AscendC::Reg::Mul<float>(tmp0, g0, xi0, mask_full);
AscendC::Reg::RegTensor<float> xi1;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(xi1, xs1_ptr + step*{_vr});
AscendC::Reg::RegTensor<float> sm1;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(sm1, sums1_ptr + step*{_vr});
    AscendC::Reg::RegTensor<float> denom1;
    AscendC::Reg::Adds<float>(denom1, sm1, (float){EPS}, mask_full);
    AscendC::Reg::RegTensor<float> tmp1;
    AscendC::Reg::Mul<float>(tmp1, g1, xi1, mask_full);
    AscendC::Reg::RegTensor<float> t_b0, t_b1;
    if (step % 2 != 0) {{
AscendC::Reg::RegTensor<float> tr_t0;
AscendC::Reg::Gather(tr_t0, tmp0, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::RegTensor<float> t0_lo, t0_hi, t0_lo_b, t0_hi_b, t0_b_raw;
AscendC::Reg::ReduceSumWithDataBlock<float>(t0_lo, tr_t0, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(t0_hi, tr_t0, mask_hi);
AscendC::Reg::Gather(t0_lo_b, t0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(t0_hi_b, t0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(t0_b_raw, t0_lo_b, t0_hi_b, mask_lo);
AscendC::Reg::Gather(t_b0, t0_b_raw, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::RegTensor<float> tr_t1;
AscendC::Reg::Gather(tr_t1, tmp1, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
AscendC::Reg::RegTensor<float> t1_lo, t1_hi, t1_lo_b, t1_hi_b, t1_b_raw;
AscendC::Reg::ReduceSumWithDataBlock<float>(t1_lo, tr_t1, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(t1_hi, tr_t1, mask_hi);
AscendC::Reg::Gather(t1_lo_b, t1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(t1_hi_b, t1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(t1_b_raw, t1_lo_b, t1_hi_b, mask_lo);
AscendC::Reg::Gather(t_b1, t1_b_raw, (AscendC::Reg::RegTensor<uint32_t>&)t_idx);
    }} else {{
AscendC::Reg::RegTensor<float> t0_lo, t0_hi, t0_lo_b, t0_hi_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(t0_lo, tmp0, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(t0_hi, tmp0, mask_hi);
AscendC::Reg::Gather(t0_lo_b, t0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(t0_hi_b, t0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(t_b0, t0_lo_b, t0_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> t1_lo, t1_hi, t1_lo_b, t1_hi_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(t1_lo, tmp1, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(t1_hi, tmp1, mask_hi);
AscendC::Reg::Gather(t1_lo_b, t1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(t1_hi_b, t1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(t_b1, t1_lo_b, t1_hi_b, mask_lo);
    }}
    AscendC::Reg::RegTensor<float> t_div0, g_sub0;
    AscendC::Reg::Div<float>(t_div0, t_b0, denom0, mask_full);
    AscendC::Reg::Sub<float>(g_sub0, g0, t_div0, mask_full);
    AscendC::Reg::Div<float>(g0, g_sub0, denom0, mask_full);
    AscendC::Reg::RegTensor<float> t_div1, g_sub1;
    AscendC::Reg::Div<float>(t_div1, t_b1, denom1, mask_full);
    AscendC::Reg::Sub<float>(g_sub1, g1, t_div1, mask_full);
    AscendC::Reg::Div<float>(g1, g_sub1, denom1, mask_full);
}}

// Final softmax backward — VR0
AscendC::Reg::RegTensor<float> xi_f0;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(xi_f0, xs0_ptr + 0*{_vr});
AscendC::Reg::RegTensor<float> tmp_f0;
AscendC::Reg::Mul<float>(tmp_f0, g0, xi_f0, mask_full);
AscendC::Reg::RegTensor<float> f0_lo, f0_hi, f0_lo_b, f0_hi_b, f0_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(f0_lo, tmp_f0, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(f0_hi, tmp_f0, mask_hi);
AscendC::Reg::Gather(f0_lo_b, f0_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(f0_hi_b, f0_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(f0_b, f0_lo_b, f0_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> g_sub_f0;
AscendC::Reg::Sub<float>(g_sub_f0, g0, f0_b, mask_full);
AscendC::Reg::Mul<float>(g0, g_sub_f0, xi_f0, mask_full);

// Final softmax backward — VR1
AscendC::Reg::RegTensor<float> xi_f1;
AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(xi_f1, xs1_ptr + 0*{_vr});
AscendC::Reg::RegTensor<float> tmp_f1;
AscendC::Reg::Mul<float>(tmp_f1, g1, xi_f1, mask_full);
AscendC::Reg::RegTensor<float> f1_lo, f1_hi, f1_lo_b, f1_hi_b, f1_b;
AscendC::Reg::ReduceSumWithDataBlock<float>(f1_lo, tmp_f1, mask_lo);
AscendC::Reg::ReduceSumWithDataBlock<float>(f1_hi, tmp_f1, mask_hi);
AscendC::Reg::Gather(f1_lo_b, f1_lo, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Gather(f1_hi_b, f1_hi, (AscendC::Reg::RegTensor<uint32_t>&)bcast_idx2);
AscendC::Reg::Select<float>(f1_b, f1_lo_b, f1_hi_b, mask_lo);
AscendC::Reg::RegTensor<float> g_sub_f1;
AscendC::Reg::Sub<float>(g_sub_f1, g1, f1_b, mask_full);
AscendC::Reg::Mul<float>(g1, g_sub_f1, xi_f1, mask_full);

AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(out_ptr, g0, mask_full);
AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(out_ptr + {_vr}, g1, mask_full);
"""

    num_groups = num_tokens // TOKENS_PER_BLOCK
    for grp in range(asctile.block_idx(), num_groups, asctile.block_num(), unroll_factor=2):
        offset = grp * TOKENS_PER_BLOCK * MHC * MHC
        x_tile = asctile.copy_in(x_gm, [offset], [BLOCK_ELEMS])
        grad_tile = asctile.copy_in(grad_gm, [offset], [BLOCK_ELEMS])
        out_tile = asctile.inline_vf(_vf_code, [BLOCK_ELEMS], asctile.float32, [x_tile, grad_tile, scratch])
        asctile.copy_out(out_tile, out_gm, [offset])


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def sinkhorn_bwd_kernel(grad_ptr: asctile.GlobalAddress, x_ptr: asctile.GlobalAddress, mask_ptr: asctile.GlobalAddress,
                        idx_f_ptr: asctile.GlobalAddress, idx_inv_ptr: asctile.GlobalAddress,
                        out_ptr: asctile.GlobalAddress, num_tokens: int, REPEAT: asctile.ConstExpr[int],
                        EPS: asctile.ConstExpr[float], MHC: asctile.ConstExpr[int] = MHC,
                        MHC_PAD: asctile.ConstExpr[int] = MHC_PAD, BATCH: asctile.ConstExpr[int] = BATCH):
    r"""Sinkhorn backward via forward recompute + reverse iteration.

    Packs ``BATCH`` tokens into one ``[BATCH*MHC, MHC_PAD]`` 2D UB tile so
    that every op processes all ``BATCH`` tokens in a single instruction. Uses
    a hybrid layout (``[BATCH*MHC, MHC_PAD]`` for row-normalise, reindex to
    ``[MHC, BATCH*MHC_PAD]`` for col-normalise) via ``asctile.reindex``
    (gather). For ``BATCH > MHC`` forward and inverse permutations differ, so
    two index tensors are used. See ``test_sinkhorn_fwd.py`` for the reindex
    rationale. State 0 softmax uses the fused ``asctile.softmax`` hardware
    intrinsic (``AscendC::SoftMax``) instead of separate max+sub+exp+sum+div.
    The forward and backward normalise loops use runtime ``range()`` (not
    ``static_range``) to keep code size small. xs[step] is saved in the layout
    matching the normalise direction so the backward phase skips reindexing
    x_pre. See the module docstring for the ``-88`` sentinel and
    constant-mask padded-column reset rationale.
    """
    grad_gm = asctile.global_tensor(grad_ptr, [num_tokens * MHC, MHC])
    x_gm = asctile.global_tensor(x_ptr, [num_tokens * MHC, MHC])
    mask_gm = asctile.global_tensor(mask_ptr, [MHC_PAD])
    idx_f_gm = asctile.global_tensor(idx_f_ptr, [BATCH * MHC * MHC_PAD])
    idx_inv_gm = asctile.global_tensor(idx_inv_ptr, [BATCH * MHC * MHC_PAD])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens * MHC, MHC_PAD])
    n_states = 2 * REPEAT
    n_groups = num_tokens // BATCH
    TILE_ROWS = BATCH * MHC

    idx_f_1d = asctile.copy_in(idx_f_gm, [0], [BATCH * MHC * MHC_PAD])
    idx_inv_1d = asctile.copy_in(idx_inv_gm, [0], [BATCH * MHC * MHC_PAD])
    idx_432 = asctile.reshape(idx_f_1d, [MHC, BATCH * MHC_PAD])
    idx_168 = asctile.reshape(idx_inv_1d, [TILE_ROWS, MHC_PAD])
    mask = asctile.copy_in(mask_gm, [0], [1, MHC_PAD])

    for grp in range(asctile.block_idx(), n_groups, asctile.block_num()):
        base = grp * TILE_ROWS

        # -88.0 x padding underflows to zero in softmax; grad padding stays zero.
        x_in = asctile.copy_in(x_gm, [base, 0], [TILE_ROWS, MHC_PAD], real_shape=[TILE_ROWS, MHC], pad_value=-88.0)
        g = asctile.copy_in(grad_gm, [base, 0], [TILE_ROWS, MHC_PAD], real_shape=[TILE_ROWS, MHC], pad_value=0.0)

        xs = [None] * n_states
        sums = [None] * n_states

        sm = asctile.softmax(x_in)
        xs[0] = sm
        # Softmax backward uses only xs[0].
        sums[0] = None
        cur = sm

        cur_432 = asctile.reindex(cur, idx_432)
        xs[1] = cur_432
        c_sum = asctile.reduce_sum(cur_432, 0, keep_dims=True)
        sums[1] = c_sum
        cur_432 = cur_432 * (1.0 / (c_sum + EPS))
        cur = asctile.reindex(cur_432, idx_168)

        # static_range is required for Python-list indexing by step.
        for step in asctile.static_range(2, n_states):
            if step % 2 == 0:
                xs[step] = cur
                s = asctile.reduce_sum(cur, 1, keep_dims=True)
                sums[step] = s
                cur = cur * (1.0 / (s + EPS))
            else:
                cur_432 = asctile.reindex(cur, idx_432)
                xs[step] = cur_432
                s = asctile.reduce_sum(cur_432, 0, keep_dims=True)
                sums[step] = s
                cur_432 = cur_432 * (1.0 / (s + EPS))
                cur = asctile.reindex(cur_432, idx_168)

        # y = x / (s + eps) => gx = (gy - sum_d(gy*x)*inv) * inv, where inv=1/(s+eps).
        for step in asctile.static_range(n_states - 1, 0, -1):
            x_pre = xs[step]
            s = sums[step]
            if step % 2 == 0:
                inv = 1.0 / (s + EPS)
                tmp = g * x_pre
                t_sum = asctile.reduce_sum(tmp, 1, keep_dims=True)
                g = (g - t_sum * inv) * inv
            else:
                g_432 = asctile.reindex(g, idx_432)
                inv = 1.0 / (s + EPS)
                tmp = g_432 * x_pre
                t_sum = asctile.reduce_sum(tmp, 0, keep_dims=True)
                g_432 = (g_432 - t_sum * inv) * inv
                g = asctile.reindex(g_432, idx_168)
                g = g * mask

        # Final softmax backward: gx = (gy - row_sum(gy * x)) * x   (x = softmax)
        x_pre = xs[0]
        tmp = g * x_pre
        t_sum = asctile.reduce_sum(tmp, 1, keep_dims=True)
        g = (g - t_sum) * x_pre

        asctile.copy_out(g, out_gm, [base, 0])


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


def run_sinkhorn_bwd(profiler, runs, grad, x, num_tokens, mhc, batch, block_num, mask, idx_f, idx_inv):
    """Host entry: launch the backward kernel with high-level ops (no ``inline_vf``).

    Pads ``num_tokens`` up to a multiple of ``batch`` (``batch`` tokens packed
    per outer iteration). No host-side column padding: the kernel DMA-loads
    each ``[batch*MHC, MHC]`` tile and pads the trailing columns to
    ``[batch*MHC, MHC_PAD]`` with ``-88.0`` (x) / ``0.0`` (grad) via
    ``copy_in(pad_value=...)``. Returns the gradient as ``[num_tokens, mhc,
    mhc]`` (real tokens only).
    """
    num_tokens_padded = ((num_tokens + batch - 1) // batch) * batch
    x_v = x.view(num_tokens, mhc, mhc).contiguous()
    grad_v = grad.view(num_tokens, mhc, mhc).contiguous()
    if num_tokens_padded != num_tokens:
        x_pad = torch.zeros(num_tokens_padded, mhc, mhc, dtype=torch.float32)
        x_pad[:num_tokens] = x_v
        x_v = x_pad
        grad_pad = torch.zeros(num_tokens_padded, mhc, mhc, dtype=torch.float32)
        grad_pad[:num_tokens] = grad_v
        grad_v = grad_pad

    x_flat = x_v.view(num_tokens_padded * MHC, MHC).contiguous()
    g_flat = grad_v.view(num_tokens_padded * MHC, MHC).contiguous()
    out_flat = torch.zeros(num_tokens_padded * MHC, MHC_PAD, dtype=torch.float32)
    with profiler.profile():
        for _ in range(runs):
            sinkhorn_bwd_kernel[block_num](g_flat, x_flat, mask, idx_f, idx_inv, out_flat, num_tokens_padded,
                                           REPEAT=REPEAT, EPS=EPS, MHC=mhc, MHC_PAD=MHC_PAD, BATCH=batch)
    return _extract_mhc(out_flat.view(num_tokens_padded, MHC, MHC_PAD)[:num_tokens], (num_tokens, mhc, mhc))


def run_sinkhorn_bwd_vf(profiler, runs, grad, x, num_tokens, mhc, batch, block_num, mask, idx_f, idx_inv):
    """Host entry: launch the backward kernel with the 2-VR ``inline_vf`` SIMD block.

    ``batch``/``mask``/``idx_f``/``idx_inv`` are accepted for interface parity
    with ``run_sinkhorn_bwd`` but unused: the inline_vf form packs a fixed 8
    tokens per 2-VR block and needs no reindex indices or padded-column mask
    (the pointers are still passed to the kernel, which ignores them). Returns
    the gradient as ``[num_tokens, mhc, mhc]``.
    """
    x_flat = x.view(num_tokens * mhc * mhc).contiguous()
    grad_flat = grad.view(num_tokens * mhc * mhc).contiguous()

    num_tokens_padded = ((num_tokens + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK
    if num_tokens_padded != num_tokens:
        x_pad = torch.zeros(num_tokens_padded * mhc * mhc, dtype=torch.float32)
        x_pad[:num_tokens * mhc * mhc] = x_flat
        x_flat = x_pad
        grad_pad = torch.zeros(num_tokens_padded * mhc * mhc, dtype=torch.float32)
        grad_pad[:num_tokens * mhc * mhc] = grad_flat
        grad_flat = grad_pad

    out_flat = torch.empty(num_tokens_padded * mhc * mhc, dtype=torch.float32)
    with profiler.profile():
        for _ in range(runs):
            sinkhorn_bwd_vf_kernel[block_num](grad_flat, x_flat, mask, idx_f, idx_inv, out_flat, num_tokens_padded,
                                              REPEAT=REPEAT, EPS=EPS, MHC=mhc)
    return out_flat[:num_tokens * mhc * mhc].view(num_tokens, mhc, mhc)


# yapf: disable
@pytest.mark.parametrize("use_inline_vf", [
    pytest.param(0, id="hl"),
    pytest.param(1, id="vf"),
])
@pytest.mark.parametrize("_, n0, n1, mhc, block_num, batch", [
# PYASC_TESTS_BEGIN
    pytest.param(None, 1, 512,  4, 72, 8,  id="n1-n512-mhc4-bwd"),
    pytest.param(None, 1, 8192, 4, 72, 16, id="n1-n8192-mhc4-bwd"),
# PYASC_TESTS_END
])
# yapf: enable
def test_sinkhorn_bwd(profiler, runs, _, n0, n1, mhc, block_num, batch, compile_only, use_inline_vf):
    num_tokens = n0 * n1
    x = torch.randn(num_tokens, mhc, mhc, dtype=torch.float32)
    grad = torch.randn(num_tokens, mhc, mhc, dtype=torch.float32)

    # Mask padded columns between reverse steps to prevent 1/EPS amplification
    # from overflowing to NaN; inline_vf ignores the mask and reindex indices.
    mask = torch.tensor([1.0] * MHC + [0.0] * (MHC_PAD - MHC), dtype=torch.float32)
    idx_f = _build_reindex_index(batch)
    idx_inv = _build_reindex_index_inv(batch)

    runner = run_sinkhorn_bwd_vf if use_inline_vf else run_sinkhorn_bwd
    out = runner(profiler, runs, grad, x, num_tokens, mhc, batch, block_num, mask, idx_f, idx_inv)
    if compile_only:
        return
    xr = x.view(n0, n1, mhc, mhc).clone().requires_grad_(True)
    out_ref = _sinkhorn_ref(xr)
    torch.autograd.backward([out_ref], [grad.view(n0, n1, mhc, mhc)])
    torch.testing.assert_close(out, xr.grad.view(num_tokens, mhc, mhc), rtol=1e-3, atol=1e-3)
