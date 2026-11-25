# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of the CANN Open Software License Agreement Version
# 2.0 (the "License").  Please refer to the License for details. You may not use
# this file except in compliance with the License.  THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
# A PARTICULAR PURPOSE.  See LICENSE in the root of the software repository for
# the full text of the License.
"""
Target benchmark + correctness test for the AscTile ``post`` backward.

Math (backward of ``out[n,mo,h] = post[n,mo]*x[n,h]
                              + sum_mi comb[n,mi,mo]*res[n,mi,h]``)::

    d_x[n,h]          = sum_mo  d_o[n,mo,h] * post[n,mo]
    d_post[n,mo]      = sum_h   d_o[n,mo,h] * x[n,h]
    d_residual[n,mi,h] = sum_mo d_o[n,mo,h] * comb[n,mi,mo]
    d_comb[n,mi,mo]   = sum_h   d_o[n,mo,h] * res[n,mi,h]
"""

# Reference implementation: https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/mhc/post_asc.py#L79

from asc.experimental import asctile
import pytest
import torch

MHC = 4
MHC_PAD = 8  # GM tensors padded to MHC_PAD for 32-byte col alignment (inline_vf form)
VEC = 64  # vector register width (for H_TILE alignment)


def _h_blk(h: int) -> int:
    """Per-case H_TILE for the backward (close to TileKernels ``h_blk``).

    Both forms arrive at the same bound. High-level form: per-tile UB
    footprint at peak is d_o [MHC, H_TILE] fp32 (64 KB) + 1 broadcast
    [MHC, H_TILE] fp32 (64 KB) + res_row [H_TILE] fp32 (16 KB) = 144 KB; UB
    capacity 253952 B -> H_TILE=4096 fits with ~104 KB headroom. inline_vf
    form: 3 tile buffers (d_o, x, res) + d_x_ub + d_residual output +
    d_post_lanes [MHC_PAD, VEC] ~2 KB + d_comb_lanes [MHC, MHC_PAD, VEC] ~8 KB;
    peak (main VF, all tile buffers live) ~147 KB at H_TILE=4096, and with
    reuse_alloc=2 the freed d_x_ub+d_residual (40 KB) cover the d_post/d_comb
    lanes (10 KB). So h<=4096 uses h directly (single tile); h>4096 splits
    into 2 tiles capped at 4096.

    The cap is chosen as ``min(ceil_div(h, 2), 4096)`` rather than the absolute
    UB-max (4480, the largest multiple of 64 fitting UB) because bwd is
    compute-bound: total work is proportional to the padded hidden size, and
    choosing H_TILE = h/2 (exactly dividing h) avoids padding entirely. For
    each benchmark h the chosen H_TILE is the largest divisor of h that fits
    UB:
      h=4096 -> 4096 (1 tile, no pad)
      h=5120 -> 2560 = h/2 (2 tiles, no pad)
      h=7168 -> 3584 = h/2 (2 tiles, no pad)
    """
    if h <= 4096:
        blk = h
    else:
        blk = min((h + 1) // 2, 4096)
    blk = (blk + 63) & ~63  # align to 64
    assert blk % VEC == 0, f"h_blk {blk} must be a multiple of VEC {VEC}"
    return blk


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def post_bwd_kernel(d_o_ptr: asctile.GlobalAddress, x_ptr: asctile.GlobalAddress, residual_ptr: asctile.GlobalAddress,
                    post_mix_ptr: asctile.GlobalAddress, comb_mix_ptr: asctile.GlobalAddress,
                    d_x_ptr: asctile.GlobalAddress, d_residual_ptr: asctile.GlobalAddress,
                    d_post_ptr: asctile.GlobalAddress, d_comb_ptr: asctile.GlobalAddress, num_tokens: int, hidden: int,
                    H_TILE: asctile.ConstExpr[int], MHC: asctile.ConstExpr[int] = MHC):
    # Unpadded [MHC] tensors have disjoint GM blocks per token, so parallel blocks do not race.
    d_o_gm = asctile.global_tensor(d_o_ptr, [num_tokens * MHC, hidden])
    x_gm = asctile.global_tensor(x_ptr, [num_tokens, hidden])
    res_gm = asctile.global_tensor(residual_ptr, [num_tokens * MHC, hidden])
    post_gm = asctile.global_tensor(post_mix_ptr, [num_tokens, MHC])
    comb_gm = asctile.global_tensor(comb_mix_ptr, [num_tokens * MHC, MHC])
    d_x_gm = asctile.global_tensor(d_x_ptr, [num_tokens, hidden])
    d_res_gm = asctile.global_tensor(d_residual_ptr, [num_tokens * MHC, hidden])
    d_post_gm = asctile.global_tensor(d_post_ptr, [num_tokens, MHC])
    d_comb_gm = asctile.global_tensor(d_comb_ptr, [num_tokens * MHC, MHC])

    for tok in range(asctile.block_idx(), num_tokens, asctile.block_num()):
        cbase = tok * MHC
        post = asctile.copy_in(post_gm, [tok, 0], [MHC])  # [MHC] fp32
        post_2d = asctile.expand_dims(post, 1)  # [MHC, 1]

        # LocalTensor has no row assignment, and codegen tracks reassigned names rather than list mutation;
        # use named d_comb row accumulators with static_range to unroll and fold the branches.
        d_post_acc = asctile.zeros([MHC], asctile.float32)
        dc0 = asctile.zeros([MHC], asctile.float32)
        dc1 = asctile.zeros([MHC], asctile.float32)
        dc2 = asctile.zeros([MHC], asctile.float32)
        dc3 = asctile.zeros([MHC], asctile.float32)

        for h_off in range(0, hidden, H_TILE):
            d_o = asctile.cast(asctile.copy_in(d_o_gm, [cbase, h_off], [MHC, H_TILE]),
                               asctile.float32)  # [MHC, H_TILE] fp32
            x = asctile.cast(asctile.copy_in(x_gm, [tok, h_off], [H_TILE]), asctile.float32)  # [H_TILE] fp32

            d_x_tile = asctile.reduce_sum(d_o * post_2d, 0)  # [H_TILE] fp32
            asctile.copy_out(asctile.cast(d_x_tile, asctile.bfloat16), d_x_gm, [tok, h_off])

            d_post_acc = d_post_acc + asctile.reduce_sum(d_o * asctile.expand_dims(x, 0), 1)  # [MHC]

            # Share each res_row load between d_residual and d_comb to overlap MTE2 with V-pipe compute.
            # The reductions consume their broadcast temps sequentially, allowing reuse_alloc=2 to share UB.
            for mi in asctile.static_range(MHC):
                comb_row = asctile.copy_in(comb_gm, [cbase + mi, 0], [MHC])  # [MHC] fp32
                res_row = asctile.cast(asctile.copy_in(res_gm, [cbase + mi, h_off], [H_TILE]),
                                       asctile.float32)  # [H_TILE] fp32

                d_res_row = asctile.reduce_sum(d_o * asctile.expand_dims(comb_row, 1), 0)  # [H_TILE]
                asctile.copy_out(asctile.cast(d_res_row, asctile.bfloat16), d_res_gm, [cbase + mi, h_off])

                d_comb_row = asctile.reduce_sum(d_o * asctile.expand_dims(res_row, 0), 1)  # [MHC]
                if mi == 0:
                    dc0 = dc0 + d_comb_row
                elif mi == 1:
                    dc1 = dc1 + d_comb_row
                elif mi == 2:
                    dc2 = dc2 + d_comb_row
                else:
                    dc3 = dc3 + d_comb_row

        asctile.copy_out(d_post_acc, d_post_gm, [tok, 0])
        asctile.copy_out(dc0, d_comb_gm, [cbase + 0, 0])
        asctile.copy_out(dc1, d_comb_gm, [cbase + 1, 0])
        asctile.copy_out(dc2, d_comb_gm, [cbase + 2, 0])
        asctile.copy_out(dc3, d_comb_gm, [cbase + 3, 0])


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def post_bwd_vf_kernel(d_o_ptr: asctile.GlobalAddress, x_ptr: asctile.GlobalAddress,
                       residual_ptr: asctile.GlobalAddress, post_mix_ptr: asctile.GlobalAddress,
                       comb_mix_ptr: asctile.GlobalAddress, d_x_ptr: asctile.GlobalAddress,
                       d_residual_ptr: asctile.GlobalAddress, d_post_ptr: asctile.GlobalAddress,
                       d_comb_ptr: asctile.GlobalAddress, num_tokens: int, hidden: int, H_TILE: asctile.ConstExpr[int],
                       MHC: asctile.ConstExpr[int] = MHC, MHC_PAD: asctile.ConstExpr[int] = MHC_PAD):
    # post/comb use MHC_PAD columns for 32-byte alignment; d_x/d_residual outputs are unpadded and disjoint.
    # Keep d_o at MHC rows: the VF blocks read only real rows, and padding would waste 50% of its GM traffic.
    d_o_gm = asctile.global_tensor(d_o_ptr, [num_tokens * MHC, hidden])
    x_gm = asctile.global_tensor(x_ptr, [num_tokens, hidden])
    res_gm = asctile.global_tensor(residual_ptr, [num_tokens * MHC, hidden])
    post_gm = asctile.global_tensor(post_mix_ptr, [num_tokens, MHC_PAD])
    comb_gm = asctile.global_tensor(comb_mix_ptr, [num_tokens * MHC, MHC_PAD])
    d_x_gm = asctile.global_tensor(d_x_ptr, [num_tokens, hidden])
    d_res_gm = asctile.global_tensor(d_residual_ptr, [num_tokens * MHC, hidden])
    d_post_gm = asctile.global_tensor(d_post_ptr, [num_tokens, MHC_PAD])
    d_comb_gm = asctile.global_tensor(d_comb_ptr, [num_tokens * MHC, MHC_PAD])

    # MHC=4 is expanded into named registers because RegTensor cannot be stored in C arrays.
    _num_chunks = H_TILE // VEC

    # Main VF UB operands: $1 comb, $2 post, $3 d_o, $4 d_x_ub (in-place); $0 is d_residual.
    _vf_code = f"""
__ubuf__ float* comb_p      = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
__ubuf__ float* post_p      = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
__ubuf__ bfloat16_t* d_o_p  = reinterpret_cast<__ubuf__ bfloat16_t*>($3.GetPhyAddr());
__ubuf__ bfloat16_t* d_x_p  = reinterpret_cast<__ubuf__ bfloat16_t*>($4.GetPhyAddr());
__ubuf__ bfloat16_t* out_p  = reinterpret_cast<__ubuf__ bfloat16_t*>($0.GetPhyAddr());

static constexpr AscendC::Reg::CastTrait bf16ToF32 = {{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING}};
static constexpr AscendC::Reg::CastTrait f32ToBf16 = {{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT}};

uint32_t vec_len = {VEC};
uint16_t num_chunks = {_num_chunks};
AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(vec_len);

AscendC::Reg::RegTensor<bfloat16_t> bf16_reg;
AscendC::Reg::RegTensor<float> d_x_reg, d_res_reg;

AscendC::Reg::RegTensor<float> comb_0_0, comb_0_1, comb_0_2, comb_0_3, comb_1_0, comb_1_1, comb_1_2, comb_1_3, comb_2_0, comb_2_1, comb_2_2, comb_2_3, comb_3_0, comb_3_1, comb_3_2, comb_3_3, post_0, post_1, post_2, post_3, do_0, do_1, do_2, do_3;

// Pre-load comb[mi,mo] and post[mo] into registers (once per h-tile)
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_0_0, comb_p + 0 * {MHC_PAD} + 0);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_0_1, comb_p + 0 * {MHC_PAD} + 1);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_0_2, comb_p + 0 * {MHC_PAD} + 2);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_0_3, comb_p + 0 * {MHC_PAD} + 3);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_1_0, comb_p + 1 * {MHC_PAD} + 0);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_1_1, comb_p + 1 * {MHC_PAD} + 1);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_1_2, comb_p + 1 * {MHC_PAD} + 2);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_1_3, comb_p + 1 * {MHC_PAD} + 3);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_2_0, comb_p + 2 * {MHC_PAD} + 0);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_2_1, comb_p + 2 * {MHC_PAD} + 1);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_2_2, comb_p + 2 * {MHC_PAD} + 2);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_2_3, comb_p + 2 * {MHC_PAD} + 3);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_3_0, comb_p + 3 * {MHC_PAD} + 0);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_3_1, comb_p + 3 * {MHC_PAD} + 1);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_3_2, comb_p + 3 * {MHC_PAD} + 2);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(comb_3_3, comb_p + 3 * {MHC_PAD} + 3);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(post_0, post_p + 0);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(post_1, post_p + 1);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(post_2, post_p + 2);
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(post_3, post_p + 3);

for (uint16_t c = 0; c < num_chunks; ++c) {{
    uint32_t col = (uint32_t)c * {VEC};
    // Load d_o[mo] into registers (once per chunk)
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_0, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_1, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_2, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_3, bf16_reg, mask);
    // d_x = sum_mo d_o[mo] * post[mo]  (FMA)
    AscendC::Reg::Duplicate<float>(d_x_reg, 0.0f);
    AscendC::Reg::MulAddDst(d_x_reg, do_0, post_0, mask);
    AscendC::Reg::MulAddDst(d_x_reg, do_1, post_1, mask);
    AscendC::Reg::MulAddDst(d_x_reg, do_2, post_2, mask);
    AscendC::Reg::MulAddDst(d_x_reg, do_3, post_3, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, d_x_reg, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(d_x_p + col, bf16_reg, mask);
    // for mi: d_residual[mi] = sum_mo d_o[mo]*comb[mi,mo]  (FMA)
    AscendC::Reg::Duplicate<float>(d_res_reg, 0.0f);
    AscendC::Reg::MulAddDst(d_res_reg, do_0, comb_0_0, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_1, comb_0_1, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_2, comb_0_2, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_3, comb_0_3, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, d_res_reg, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 0 * {H_TILE} + col, bf16_reg, mask);
    AscendC::Reg::Duplicate<float>(d_res_reg, 0.0f);
    AscendC::Reg::MulAddDst(d_res_reg, do_0, comb_1_0, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_1, comb_1_1, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_2, comb_1_2, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_3, comb_1_3, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, d_res_reg, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 1 * {H_TILE} + col, bf16_reg, mask);
    AscendC::Reg::Duplicate<float>(d_res_reg, 0.0f);
    AscendC::Reg::MulAddDst(d_res_reg, do_0, comb_2_0, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_1, comb_2_1, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_2, comb_2_2, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_3, comb_2_3, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, d_res_reg, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 2 * {H_TILE} + col, bf16_reg, mask);
    AscendC::Reg::Duplicate<float>(d_res_reg, 0.0f);
    AscendC::Reg::MulAddDst(d_res_reg, do_0, comb_3_0, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_1, comb_3_1, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_2, comb_3_2, mask);
    AscendC::Reg::MulAddDst(d_res_reg, do_3, comb_3_3, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, d_res_reg, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 3 * {H_TILE} + col, bf16_reg, mask);
}}
"""

    # Post VF reduces d_o*x into $0 [MHC_PAD,VEC] fp32; only the first MHC rows are valid.
    _loop_c1_post = f"""
for (uint16_t c = 1; c < num_chunks; ++c) {{
    uint32_t col = (uint32_t)c * {VEC};
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, x_p + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(x_reg, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_0, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_1, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_2, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_3, bf16_reg, mask);
    AscendC::Reg::MulAddDst(post_acc_0, do_0, x_reg, mask);
    AscendC::Reg::MulAddDst(post_acc_1, do_1, x_reg, mask);
    AscendC::Reg::MulAddDst(post_acc_2, do_2, x_reg, mask);
    AscendC::Reg::MulAddDst(post_acc_3, do_3, x_reg, mask);
}}

""" if _num_chunks > 1 else ""
    _vf_post_code = f"""
__ubuf__ float* out_p   = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
__ubuf__ bfloat16_t* d_o_p = reinterpret_cast<__ubuf__ bfloat16_t*>($1.GetPhyAddr());
__ubuf__ bfloat16_t* x_p   = reinterpret_cast<__ubuf__ bfloat16_t*>($2.GetPhyAddr());

static constexpr AscendC::Reg::CastTrait bf16ToF32 = {{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING}};

uint32_t vec_len = {VEC};
uint16_t num_chunks = {_num_chunks};
AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(vec_len);

AscendC::Reg::RegTensor<bfloat16_t> bf16_reg;
AscendC::Reg::RegTensor<float> x_reg;
AscendC::Reg::RegTensor<float> post_acc_0, post_acc_1, post_acc_2, post_acc_3, do_0, do_1, do_2, do_3;

// === First chunk (c=0): init accumulators via Mul ===
{{
    uint32_t col = 0;
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, x_p + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(x_reg, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_0, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_1, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_2, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_3, bf16_reg, mask);
    AscendC::Reg::Mul(post_acc_0, do_0, x_reg, mask);
    AscendC::Reg::Mul(post_acc_1, do_1, x_reg, mask);
    AscendC::Reg::Mul(post_acc_2, do_2, x_reg, mask);
    AscendC::Reg::Mul(post_acc_3, do_3, x_reg, mask);
}}
{_loop_c1_post}// Store d_post_acc [MHC_PAD, VEC] to $0 (write-only)
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 0 * {VEC}, post_acc_0, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 1 * {VEC}, post_acc_1, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 2 * {VEC}, post_acc_2, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 3 * {VEC}, post_acc_3, mask);
"""

    # Comb VF reduces d_o*res into $0 [MHC,MHC_PAD,VEC] fp32.
    _loop_c1_comb = f"""
for (uint16_t c = 1; c < num_chunks; ++c) {{
    uint32_t col = (uint32_t)c * {VEC};
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_0, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_1, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_2, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_3, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_0_0, do_0, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_0_1, do_1, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_0_2, do_2, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_0_3, do_3, res_mi_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_1_0, do_0, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_1_1, do_1, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_1_2, do_2, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_1_3, do_3, res_mi_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_2_0, do_0, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_2_1, do_1, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_2_2, do_2, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_2_3, do_3, res_mi_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_3_0, do_0, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_3_1, do_1, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_3_2, do_2, res_mi_reg, mask);
    AscendC::Reg::MulAddDst(comb_acc_3_3, do_3, res_mi_reg, mask);
}}

""" if _num_chunks > 1 else ""
    _vf_comb_code = f"""
__ubuf__ float* out_p   = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
__ubuf__ bfloat16_t* d_o_p = reinterpret_cast<__ubuf__ bfloat16_t*>($1.GetPhyAddr());
__ubuf__ bfloat16_t* res_p = reinterpret_cast<__ubuf__ bfloat16_t*>($2.GetPhyAddr());

static constexpr AscendC::Reg::CastTrait bf16ToF32 = {{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING}};

uint32_t vec_len = {VEC};
uint16_t num_chunks = {_num_chunks};
AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(vec_len);

AscendC::Reg::RegTensor<bfloat16_t> bf16_reg;
AscendC::Reg::RegTensor<float> res_mi_reg;
AscendC::Reg::RegTensor<float> comb_acc_0_0, comb_acc_0_1, comb_acc_0_2, comb_acc_0_3, comb_acc_1_0, comb_acc_1_1, comb_acc_1_2, comb_acc_1_3, comb_acc_2_0, comb_acc_2_1, comb_acc_2_2, comb_acc_2_3, comb_acc_3_0, comb_acc_3_1, comb_acc_3_2, comb_acc_3_3, do_0, do_1, do_2, do_3;

// === First chunk (c=0): init accumulators via Mul ===
{{
    uint32_t col = 0;
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_0, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_1, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_2, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, d_o_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(do_3, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::Mul(comb_acc_0_0, do_0, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_0_1, do_1, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_0_2, do_2, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_0_3, do_3, res_mi_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::Mul(comb_acc_1_0, do_0, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_1_1, do_1, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_1_2, do_2, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_1_3, do_3, res_mi_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::Mul(comb_acc_2_0, do_0, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_2_1, do_1, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_2_2, do_2, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_2_3, do_3, res_mi_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_mi_reg, bf16_reg, mask);
    AscendC::Reg::Mul(comb_acc_3_0, do_0, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_3_1, do_1, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_3_2, do_2, res_mi_reg, mask);
    AscendC::Reg::Mul(comb_acc_3_3, do_3, res_mi_reg, mask);
}}
{_loop_c1_comb}// Store d_comb_acc [MHC, MHC_PAD, VEC] to $0 (write-only)
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 0 * {VEC}, comb_acc_0_0, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 1 * {VEC}, comb_acc_0_1, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 2 * {VEC}, comb_acc_0_2, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 3 * {VEC}, comb_acc_0_3, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 8 * {VEC}, comb_acc_1_0, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 9 * {VEC}, comb_acc_1_1, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 10 * {VEC}, comb_acc_1_2, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 11 * {VEC}, comb_acc_1_3, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 16 * {VEC}, comb_acc_2_0, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 17 * {VEC}, comb_acc_2_1, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 18 * {VEC}, comb_acc_2_2, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 19 * {VEC}, comb_acc_2_3, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 24 * {VEC}, comb_acc_3_0, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 25 * {VEC}, comb_acc_3_1, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 26 * {VEC}, comb_acc_3_2, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(out_p + 27 * {VEC}, comb_acc_3_3, mask);
"""

    for tok in range(asctile.block_idx(), num_tokens, asctile.block_num()):
        cbase = tok * MHC
        obase = tok * MHC
        comb = asctile.copy_in(comb_gm, [cbase, 0], [MHC, MHC_PAD])  # [mi, mo_pad] fp32
        post = asctile.copy_in(post_gm, [tok, 0], [MHC_PAD])  # [MHC_PAD] fp32

        d_post_acc = asctile.zeros([MHC_PAD], asctile.float32)
        d_comb_acc = asctile.zeros([MHC, MHC_PAD], asctile.float32)

        # Separate VF calls limit register pressure and avoid spills; reduce_sum folds VEC lanes into the accumulators.
        for h_off in asctile.range(0, hidden, H_TILE):
            d_o_bf = asctile.copy_in(d_o_gm, [cbase, h_off], [MHC, H_TILE])
            x = asctile.copy_in(x_gm, [tok, h_off], [H_TILE])
            res = asctile.copy_in(res_gm, [cbase, h_off], [MHC, H_TILE])
            d_x_ub = asctile.zeros([H_TILE], asctile.bfloat16)
            d_res_tile = asctile.inline_vf(_vf_code, [MHC, H_TILE], asctile.bfloat16, [comb, post, d_o_bf, d_x_ub])
            asctile.copy_out(d_res_tile, d_res_gm, [obase, h_off])
            asctile.copy_out(d_x_ub, d_x_gm, [tok, h_off])
            d_post_lanes = asctile.inline_vf(_vf_post_code, [MHC_PAD, VEC], asctile.float32, [d_o_bf, x])
            d_post_acc = d_post_acc + asctile.reduce_sum(d_post_lanes, 1)  # [MHC_PAD]
            d_comb_lanes = asctile.inline_vf(_vf_comb_code, [MHC, MHC_PAD, VEC], asctile.float32, [d_o_bf, res])
            d_comb_acc = d_comb_acc + asctile.reduce_sum(d_comb_lanes, 2)  # [MHC, MHC_PAD]

        asctile.copy_out(d_post_acc, d_post_gm, [tok, 0])
        asctile.copy_out(d_comb_acc, d_comb_gm, [cbase, 0])


def _padMHC(t: torch.Tensor, mhc_pad: int = MHC_PAD) -> torch.Tensor:
    """Pad the last dim [MHC] -> [MHC_PAD] with 0."""
    n = t.shape[:-1]
    padded = torch.zeros(*n, mhc_pad, dtype=t.dtype)
    padded[..., :MHC] = t
    return padded


def _post_fwd_ref(x, residual, post_mix, comb_mix):
    # comb_mix[n,mi,mo] and residual[n,mi,h] produce mix[n,mo,h].
    x_f = x.float()
    res_f = residual.float()
    mix = torch.einsum('nmM,nmh->nMh', comb_mix, res_f)
    out = post_mix[:, :, None].float() * x_f[:, None, :] + mix
    return out.to(torch.bfloat16)


def run_post_bwd(profiler, runs, d_o_flat, x, res_flat, post_mix, comb_mix, num_tokens, h_pad, h_tile, block_num):
    """Host entry: launch the backward kernel with high-level ops (no inline_vf).

    Uses the unpadded [MHC] layout for post/comb/d_post/d_comb. Returns the
    output tensors (d_x, d_residual, d_post, d_comb).
    """
    comb_flat = comb_mix.view(num_tokens * MHC, MHC)
    d_x = torch.empty(num_tokens, h_pad, dtype=torch.bfloat16)
    d_residual = torch.empty(num_tokens * MHC, h_pad, dtype=torch.bfloat16)
    d_post = torch.empty(num_tokens, MHC, dtype=torch.float32)
    d_comb = torch.empty(num_tokens * MHC, MHC, dtype=torch.float32)
    with profiler.profile():
        for _ in range(runs):
            post_bwd_kernel[block_num](d_o_flat, x, res_flat, post_mix, comb_flat, d_x, d_residual, d_post, d_comb,
                                       num_tokens, h_pad, H_TILE=h_tile, MHC=MHC)
    return d_x, d_residual, d_post, d_comb


def run_post_bwd_vf(profiler, runs, d_o_flat, x, res_flat, post_mix, comb_mix, num_tokens, h_pad, h_tile, block_num):
    """Host entry: launch the backward kernel with the three inline_vf blocks.

    Pads post/comb to [MHC_PAD] (fp32 [MHC] = 16 B violates the 32-byte
    last-dim alignment required by the VF LoadAlign broadcast loads) and
    allocates [MHC_PAD] d_post/d_comb outputs. Returns the output tensors
    (d_x, d_residual, d_post, d_comb).
    """
    post_padded = _padMHC(post_mix)
    comb_flat_padded = _padMHC(comb_mix.view(num_tokens * MHC, MHC))
    d_x = torch.empty(num_tokens, h_pad, dtype=torch.bfloat16)
    d_residual = torch.empty(num_tokens * MHC, h_pad, dtype=torch.bfloat16)
    d_post = torch.empty(num_tokens, MHC_PAD, dtype=torch.float32)
    d_comb = torch.empty(num_tokens * MHC, MHC_PAD, dtype=torch.float32)
    with profiler.profile():
        for _ in range(runs):
            post_bwd_vf_kernel[block_num](d_o_flat, x, res_flat, post_padded, comb_flat_padded, d_x, d_residual, d_post,
                                          d_comb, num_tokens, h_pad, H_TILE=h_tile, MHC=MHC, MHC_PAD=MHC_PAD)
    return d_x, d_residual, d_post, d_comb


# yapf: disable
# Column order: (None, n0, n1, h, block_num)
@pytest.mark.parametrize("use_inline_vf", [
    pytest.param(0, id="hl"),
    pytest.param(1, id="vf"),
])
@pytest.mark.parametrize("_, n0, n1, h, block_num", [
# PYASC_TESTS_BEGIN
    pytest.param(None, 1, 512,  7168, 72, id="n1-n512-h7168-bwd"),
    pytest.param(None, 1, 512,  5120, 72, id="n1-n512-h5120-bwd"),
    pytest.param(None, 1, 512,  4096, 72, id="n1-n512-h4096-bwd"),
    pytest.param(None, 1, 8192, 7168, 72, id="n1-n8192-h7168-bwd"),
    pytest.param(None, 1, 8192, 5120, 72, id="n1-n8192-h5120-bwd"),
    pytest.param(None, 1, 8192, 4096, 72, id="n1-n8192-h4096-bwd"),
# PYASC_TESTS_END
])
# yapf: enable
def test_post_bwd(profiler, runs, _, n0, n1, h, block_num, compile_only, use_inline_vf):
    num_tokens = n0 * n1
    x = torch.randn(num_tokens, h, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, MHC, h, dtype=torch.bfloat16)
    post_mix = torch.randn(num_tokens, MHC, dtype=torch.float32)
    comb_mix = torch.randn(num_tokens, MHC, MHC, dtype=torch.float32)

    h_tile = _h_blk(h)
    # Full H_TILE loads require hidden padding; zero tails do not affect reductions and outputs are sliced back.
    h_pad = ((h + h_tile - 1) // h_tile) * h_tile
    if h_pad != h:
        x = torch.nn.functional.pad(x, (0, h_pad - h))
        residual = torch.nn.functional.pad(residual, (0, h_pad - h))
        d_o_pad = torch.nn.functional.pad(torch.randn(num_tokens, MHC, h, dtype=torch.bfloat16), (0, h_pad - h))
    else:
        d_o_pad = torch.randn(num_tokens, MHC, h, dtype=torch.bfloat16)

    # Flatten 3D inputs to 2D for the kernel (3D copy_in has a codegen bug).
    res_flat = residual.view(num_tokens * MHC, h_pad)
    d_o_flat = d_o_pad.view(num_tokens * MHC, h_pad)

    runner = run_post_bwd_vf if use_inline_vf else run_post_bwd
    d_x, d_residual, d_post, d_comb = runner(profiler, runs, d_o_flat, x, res_flat, post_mix, comb_mix, num_tokens,
                                             h_pad, h_tile, block_num)
    if compile_only:
        return
    d_x = d_x[:, :h]
    d_residual = d_residual.view(num_tokens, MHC, h_pad)[:, :, :h]
    # Match the reference gradient to the real columns seen by the kernel.
    x_r = x[:, :h].clone().requires_grad_(True)
    res_r = residual[:, :, :h].clone().requires_grad_(True)
    post_r = post_mix.clone().requires_grad_(True)
    comb_r = comb_mix.clone().requires_grad_(True)
    out_ref = _post_fwd_ref(x_r, res_r, post_r, comb_r)
    torch.autograd.backward([out_ref], [d_o_pad[:, :, :h]])
    torch.testing.assert_close(d_x, x_r.grad, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(d_residual, res_r.grad, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(d_post[:, :MHC], post_r.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d_comb.view(num_tokens, MHC, -1)[:, :, :MHC], comb_r.grad, rtol=1e-2, atol=1e-2)
