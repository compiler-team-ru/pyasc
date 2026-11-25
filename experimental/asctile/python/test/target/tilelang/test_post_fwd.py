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
Target benchmark + correctness test for the AscTile ``post`` forward.

Math: ``out[n, mo, h] = post_layer_mix[n, mo] * x[n, h]
                       + sum_mi comb_res_mix[n, mi, mo] * residual[n, mi, h]``
"""

# Reference implementation: https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/mhc/post_asc.py#L11

from asc.experimental import asctile
import pytest
import torch

MHC = 4
MHC_PAD = 8  # 8 fp32 = 32 bytes, satisfies 32-byte last-dim alignment
# H_TILE must be a multiple of the 64-lane register width so LoadAlign stays within a row.
VEC = 64

# The forms have different UB live sets and therefore different tile caps.
VF_H_TILE_CAP = 4096
# 3008 is the largest multiple of 64 that fits the high-level form's fp32 broadcast temps in UB.
HL_H_TILE_CAP = 3008


def _h_blk(h: int, cap: int = VF_H_TILE_CAP) -> int:
    """TileKernels h_blk strategy: process as much of hidden as possible in one tile.

    ``blk = h if h <= cap else min(ceil_div(h, 2), cap)``, aligned to 64.
    With H_TILE=1024 AscTile did 4-7x more h-iterations than TileKernels, which
    dominated the GM->UB overhead even in the inline_vf version (the VF block
    itself is fast, but there are too many tiles). The per-form caps:
      - inline_vf (4096): h=4096 -> 4096 (1 tile), h=5120 -> 2560,
        h=7168 -> 3584 -- every benchmark hidden size divides the tile, so no
        host-side hidden padding is needed.
      - high-level (3008): h=4096 -> 2048 (2 tiles), h=5120 -> 2560 (2 tiles),
        h=7168 -> 3008 (3 tiles, host pads hidden to 9024).
    """
    blk = h if h <= cap else min((h + 1) // 2, cap)
    blk = (blk + 63) & ~63  # align to 64
    assert blk % VEC == 0, f"h_blk {blk} must be a multiple of VEC {VEC}"
    return blk


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def post_fwd_kernel(x_ptr: asctile.GlobalAddress, residual_ptr: asctile.GlobalAddress,
                    post_mix_ptr: asctile.GlobalAddress, comb_mix_ptr: asctile.GlobalAddress,
                    out_ptr: asctile.GlobalAddress, num_tokens: int, hidden: int, H_TILE: asctile.ConstExpr[int],
                    MHC: asctile.ConstExpr[int] = MHC, MHC_PAD: asctile.ConstExpr[int] = MHC_PAD):
    # Flatten to 2D because 3D copy_in has a blockCount codegen bug.
    # post/comb use MHC_PAD columns for 32-byte alignment; this form reads only the MHC real elements.
    # Unpadded output rows are disjoint per token; MHC_PAD rows would double GM-store traffic.
    x_gm = asctile.global_tensor(x_ptr, [num_tokens, hidden])
    res_gm = asctile.global_tensor(residual_ptr, [num_tokens * MHC, hidden])
    post_gm = asctile.global_tensor(post_mix_ptr, [num_tokens, MHC_PAD])
    comb_gm = asctile.global_tensor(comb_mix_ptr, [num_tokens * MHC, MHC_PAD])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens * MHC, hidden])

    for tok in range(asctile.block_idx(), num_tokens, asctile.block_num()):
        cbase = tok * MHC
        obase = tok * MHC
        post = asctile.copy_in(post_gm, [tok, 0], [MHC])  # [MHC] fp32
        post_2d = asctile.expand_dims(post, 1)  # [MHC, 1]

        # Preload the next tile before computing the current one to overlap MTE2 and V-pipe work.
        # Initializing out_tile with post*x avoids a separate mix accumulator and final addition.
        # Named per-mi rows are required because codegen does not track list mutation across iterations;
        # static_range unrolls the loop and folds each if/elif branch.
        # H_TILE=3008 is the UB limit; 3072 overflows.
        x_cur = asctile.copy_in(x_gm, [tok, 0], [H_TILE])  # [H_TILE] bf16
        res_cur_0 = asctile.copy_in(res_gm, [cbase + 0, 0], [H_TILE])
        res_cur_1 = asctile.copy_in(res_gm, [cbase + 1, 0], [H_TILE])
        res_cur_2 = asctile.copy_in(res_gm, [cbase + 2, 0], [H_TILE])
        res_cur_3 = asctile.copy_in(res_gm, [cbase + 3, 0], [H_TILE])

        for h_off in asctile.range(0, hidden - H_TILE, H_TILE):
            x_next = asctile.copy_in(x_gm, [tok, h_off + H_TILE], [H_TILE])
            res_next_0 = asctile.copy_in(res_gm, [cbase + 0, h_off + H_TILE], [H_TILE])
            res_next_1 = asctile.copy_in(res_gm, [cbase + 1, h_off + H_TILE], [H_TILE])
            res_next_2 = asctile.copy_in(res_gm, [cbase + 2, h_off + H_TILE], [H_TILE])
            res_next_3 = asctile.copy_in(res_gm, [cbase + 3, h_off + H_TILE], [H_TILE])

            x_fp32 = asctile.cast(x_cur, asctile.float32)
            out_tile = post_2d * asctile.expand_dims(x_fp32, 0)  # [MHC, H_TILE] fp32

            for mi in asctile.static_range(MHC):
                if mi == 0:
                    res_row = res_cur_0
                elif mi == 1:
                    res_row = res_cur_1
                elif mi == 2:
                    res_row = res_cur_2
                else:
                    res_row = res_cur_3
                comb_row = asctile.copy_in(comb_gm, [cbase + mi, 0], [MHC])  # [MHC] fp32
                res_row_fp32 = asctile.cast(res_row, asctile.float32)
                # [MHC, 1] * [1, H_TILE] -> [MHC, H_TILE]
                out_tile = out_tile + asctile.expand_dims(comb_row, 1) * asctile.expand_dims(res_row_fp32, 0)

            asctile.copy_out(asctile.cast(out_tile, asctile.bfloat16), out_gm, [obase, h_off])

            x_cur = x_next
            res_cur_0 = res_next_0
            res_cur_1 = res_next_1
            res_cur_2 = res_next_2
            res_cur_3 = res_next_3

        h_off_last = hidden - H_TILE
        x_fp32 = asctile.cast(x_cur, asctile.float32)
        out_tile = post_2d * asctile.expand_dims(x_fp32, 0)  # [MHC, H_TILE] fp32
        for mi in asctile.static_range(MHC):
            if mi == 0:
                res_row = res_cur_0
            elif mi == 1:
                res_row = res_cur_1
            elif mi == 2:
                res_row = res_cur_2
            else:
                res_row = res_cur_3
            comb_row = asctile.copy_in(comb_gm, [cbase + mi, 0], [MHC])
            res_row_fp32 = asctile.cast(res_row, asctile.float32)
            out_tile = out_tile + asctile.expand_dims(comb_row, 1) * asctile.expand_dims(res_row_fp32, 0)
        asctile.copy_out(asctile.cast(out_tile, asctile.bfloat16), out_gm, [obase, h_off_last])


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def post_fwd_vf_kernel(x_ptr: asctile.GlobalAddress, residual_ptr: asctile.GlobalAddress,
                       post_mix_ptr: asctile.GlobalAddress, comb_mix_ptr: asctile.GlobalAddress,
                       out_ptr: asctile.GlobalAddress, num_tokens: int, hidden: int, H_TILE: asctile.ConstExpr[int],
                       MHC: asctile.ConstExpr[int] = MHC, MHC_PAD: asctile.ConstExpr[int] = MHC_PAD):
    # Flatten to 2D because 3D copy_in has a blockCount codegen bug.
    # The VF block loads full MHC_PAD rows for 32-byte-aligned post/comb inputs.
    # Unpadded output rows are disjoint per token; MHC_PAD rows would double GM-store traffic.
    x_gm = asctile.global_tensor(x_ptr, [num_tokens, hidden])
    res_gm = asctile.global_tensor(residual_ptr, [num_tokens * MHC, hidden])
    post_gm = asctile.global_tensor(post_mix_ptr, [num_tokens, MHC_PAD])
    comb_gm = asctile.global_tensor(comb_mix_ptr, [num_tokens * MHC, MHC_PAD])
    out_gm = asctile.global_tensor(out_ptr, [num_tokens * MHC, hidden])

    # The VF block computes the full mix in registers, including bf16/fp32 conversion.
    # MHC=4 is expanded into named registers because RegTensor cannot be stored in C arrays.
    # UB inputs: $1 comb [MHC,MHC_PAD], $2 res [MHC,H_TILE], $3 post [MHC_PAD], $4 x [H_TILE].
    # UB output: $0 out [MHC,H_TILE] bf16; H_TILE is a multiple of VEC.
    _num_chunks = H_TILE // VEC
    _vf_code = f"""
__ubuf__ float* comb_p      = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
__ubuf__ bfloat16_t* res_p  = reinterpret_cast<__ubuf__ bfloat16_t*>($2.GetPhyAddr());
__ubuf__ float* post_p      = reinterpret_cast<__ubuf__ float*>($3.GetPhyAddr());
__ubuf__ bfloat16_t* x_p    = reinterpret_cast<__ubuf__ bfloat16_t*>($4.GetPhyAddr());
__ubuf__ bfloat16_t* out_p  = reinterpret_cast<__ubuf__ bfloat16_t*>($0.GetPhyAddr());

static constexpr AscendC::Reg::CastTrait bf16ToF32 = {{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING}};
static constexpr AscendC::Reg::CastTrait f32ToBf16 = {{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT}};

uint32_t vec_len = {VEC};
uint16_t num_chunks = {_num_chunks};
AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(vec_len);

AscendC::Reg::RegTensor<bfloat16_t> bf16_reg;
AscendC::Reg::RegTensor<float> x_reg, tmp;

AscendC::Reg::RegTensor<float> comb_0_0, comb_0_1, comb_0_2, comb_0_3, comb_1_0, comb_1_1, comb_1_2, comb_1_3, comb_2_0, comb_2_1, comb_2_2, comb_2_3, comb_3_0, comb_3_1, comb_3_2, comb_3_3, post_0, post_1, post_2, post_3, res_0, res_1, res_2, res_3, out_0, out_1, out_2, out_3;

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
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, x_p + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(x_reg, bf16_reg, mask);
    // Load res[mi] into registers (once per chunk)
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 0 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_0, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 1 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_1, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 2 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_2, bf16_reg, mask);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bf16_reg, res_p + 3 * {H_TILE} + col);
    AscendC::Reg::Cast<float, bfloat16_t, bf16ToF32>(res_3, bf16_reg, mask);
    // out[mo] = x * post[mo]
    AscendC::Reg::Mul(out_0, x_reg, post_0, mask);
    AscendC::Reg::Mul(out_1, x_reg, post_1, mask);
    AscendC::Reg::Mul(out_2, x_reg, post_2, mask);
    AscendC::Reg::Mul(out_3, x_reg, post_3, mask);
    // out[mo] += res[mi] * comb[mi,mo]  (FMA via MulAddDst)
    AscendC::Reg::MulAddDst(out_0, res_0, comb_0_0, mask);
    AscendC::Reg::MulAddDst(out_1, res_0, comb_0_1, mask);
    AscendC::Reg::MulAddDst(out_2, res_0, comb_0_2, mask);
    AscendC::Reg::MulAddDst(out_3, res_0, comb_0_3, mask);
    AscendC::Reg::MulAddDst(out_0, res_1, comb_1_0, mask);
    AscendC::Reg::MulAddDst(out_1, res_1, comb_1_1, mask);
    AscendC::Reg::MulAddDst(out_2, res_1, comb_1_2, mask);
    AscendC::Reg::MulAddDst(out_3, res_1, comb_1_3, mask);
    AscendC::Reg::MulAddDst(out_0, res_2, comb_2_0, mask);
    AscendC::Reg::MulAddDst(out_1, res_2, comb_2_1, mask);
    AscendC::Reg::MulAddDst(out_2, res_2, comb_2_2, mask);
    AscendC::Reg::MulAddDst(out_3, res_2, comb_2_3, mask);
    AscendC::Reg::MulAddDst(out_0, res_3, comb_3_0, mask);
    AscendC::Reg::MulAddDst(out_1, res_3, comb_3_1, mask);
    AscendC::Reg::MulAddDst(out_2, res_3, comb_3_2, mask);
    AscendC::Reg::MulAddDst(out_3, res_3, comb_3_3, mask);
    // Store out[mo] as bf16
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, out_0, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 0 * {H_TILE} + col, bf16_reg, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, out_1, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 1 * {H_TILE} + col, bf16_reg, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, out_2, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 2 * {H_TILE} + col, bf16_reg, mask);
    AscendC::Reg::Cast<bfloat16_t, float, f32ToBf16>(bf16_reg, out_3, mask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(out_p + 3 * {H_TILE} + col, bf16_reg, mask);
}}
"""

    for tok in range(asctile.block_idx(), num_tokens, asctile.block_num(), unroll_factor=2):
        cbase = tok * MHC
        obase = tok * MHC
        comb = asctile.copy_in(comb_gm, [cbase, 0], [MHC, MHC_PAD])  # [mi, mo_pad] fp32
        post = asctile.copy_in(post_gm, [tok, 0], [MHC_PAD])  # [mo_pad] fp32

        # Preload the next tile before computing the current one to overlap MTE2 and V-pipe work.
        # Five live tile buffers need 36*H_TILE*2 bytes plus overhead, so large tiles use the other path.
        # ConstExpr folds the branch, avoiding scf.if stripping local UB attributes from yielded tensors.
        if H_TILE <= 2560:
            res_cur = asctile.copy_in(res_gm, [cbase, 0], [MHC, H_TILE])  # [mi, h] bf16
            x_cur = asctile.copy_in(x_gm, [tok, 0], [H_TILE])  # [h] bf16

            for h_off in asctile.range(0, hidden - H_TILE, H_TILE):
                res_next = asctile.copy_in(res_gm, [cbase, h_off + H_TILE], [MHC, H_TILE])
                x_next = asctile.copy_in(x_gm, [tok, h_off + H_TILE], [H_TILE])
                out_tile = asctile.inline_vf(_vf_code, [MHC, H_TILE], asctile.bfloat16, [comb, res_cur, post, x_cur])
                asctile.copy_out(out_tile, out_gm, [obase, h_off])
                res_cur = res_next
                x_cur = x_next

            h_off_last = hidden - H_TILE
            out_tile = asctile.inline_vf(_vf_code, [MHC, H_TILE], asctile.bfloat16, [comb, res_cur, post, x_cur])
            asctile.copy_out(out_tile, out_gm, [obase, h_off_last])
        else:
            # Larger tiles fit only three live buffers, so do not pipeline them.
            for h_off in asctile.range(0, hidden, H_TILE, unroll_factor=2):
                res = asctile.copy_in(res_gm, [cbase, h_off], [MHC, H_TILE])  # [mi, h] bf16
                x = asctile.copy_in(x_gm, [tok, h_off], [H_TILE])  # [h] bf16
                out_tile = asctile.inline_vf(_vf_code, [MHC, H_TILE], asctile.bfloat16, [comb, res, post, x])
                asctile.copy_out(out_tile, out_gm, [obase, h_off])


def _pad_mhc(t: torch.Tensor, mhc_pad: int = MHC_PAD) -> torch.Tensor:
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


def run_post_fwd(profiler, runs, x, res_flat, post_mix, comb_mix, num_tokens, hidden, block_num):
    """Host entry: launch the forward kernel with high-level ops (no inline_vf).

    Pads hidden up to a multiple of the hl h_tile (the hl cap of 3008 does not
    divide every benchmark hidden size, e.g. 7168 -> 3x3008 -> padded to 9024).
    Returns (out_flat, h_pad) -- the caller slices out_flat[:, :hidden].
    """
    h_tile = _h_blk(hidden, HL_H_TILE_CAP)
    h_pad = ((hidden + h_tile - 1) // h_tile) * h_tile
    if h_pad != hidden:
        x = torch.nn.functional.pad(x, (0, h_pad - hidden))
        res_flat = torch.nn.functional.pad(res_flat, (0, h_pad - hidden))
    post_padded = _pad_mhc(post_mix)
    comb_flat_padded = _pad_mhc(comb_mix.view(num_tokens * MHC, MHC))
    out_flat = torch.zeros(num_tokens * MHC, h_pad, dtype=torch.bfloat16)
    with profiler.profile():
        for _ in range(runs):
            post_fwd_kernel[block_num](x, res_flat, post_padded, comb_flat_padded, out_flat, num_tokens, h_pad,
                                       H_TILE=h_tile, MHC=MHC, MHC_PAD=MHC_PAD)
    return out_flat, h_pad


def run_post_fwd_vf(profiler, runs, x, res_flat, post_mix, comb_mix, num_tokens, hidden, block_num):
    """Host entry: launch the forward kernel with the register-level inline_vf block.

    No hidden padding is needed: the vf h_tile (h if h <= 4096 else h/2,
    aligned to 64) divides every benchmark hidden size, so h_pad == hidden.
    Returns (out_flat, h_pad) for interface parity with run_post_fwd.
    """
    h_tile = _h_blk(hidden)
    post_padded = _pad_mhc(post_mix)
    comb_flat_padded = _pad_mhc(comb_mix.view(num_tokens * MHC, MHC))
    out_flat = torch.zeros(num_tokens * MHC, hidden, dtype=torch.bfloat16)
    with profiler.profile():
        for _ in range(runs):
            post_fwd_vf_kernel[block_num](x, res_flat, post_padded, comb_flat_padded, out_flat, num_tokens, hidden,
                                          H_TILE=h_tile, MHC=MHC, MHC_PAD=MHC_PAD)
    return out_flat, hidden


# yapf: disable
# Column order: (None, n0, n1, h, block_num)
@pytest.mark.parametrize("use_inline_vf", [
    pytest.param(0, id="hl"),
    pytest.param(1, id="vf"),
])
@pytest.mark.parametrize("_, n0, n1, h, block_num", [
# PYASC_TESTS_BEGIN
    pytest.param(None, 1, 512,  7168, 72, id="n1-n512-h7168-fwd"),
    pytest.param(None, 1, 512,  5120, 72, id="n1-n512-h5120-fwd"),
    pytest.param(None, 1, 512,  4096, 72, id="n1-n512-h4096-fwd"),
    pytest.param(None, 1, 8192, 7168, 72, id="n1-n8192-h7168-fwd"),
    pytest.param(None, 1, 8192, 5120, 72, id="n1-n8192-h5120-fwd"),
    pytest.param(None, 1, 8192, 4096, 72, id="n1-n8192-h4096-fwd"),
# PYASC_TESTS_END
])
# yapf: enable
def test_post_fwd(profiler, runs, _, n0, n1, h, block_num, compile_only, use_inline_vf):
    num_tokens = n0 * n1
    x = torch.randn(num_tokens, h, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, MHC, h, dtype=torch.bfloat16)
    post_mix = torch.randn(num_tokens, MHC, dtype=torch.float32)
    comb_mix = torch.randn(num_tokens, MHC, MHC, dtype=torch.float32)

    # Flatten 3D inputs to 2D for the kernel (3D copy_in has a codegen bug).
    res_flat = residual.view(num_tokens * MHC, h)

    runner = run_post_fwd_vf if use_inline_vf else run_post_fwd
    out_flat, h_pad = runner(profiler, runs, x, res_flat, post_mix, comb_mix, num_tokens, h, block_num)
    if compile_only:
        return
    out = out_flat[:, :h].view(num_tokens, MHC, h)
    ref = _post_fwd_ref(x[:, :h], residual[:, :, :h], post_mix, comb_mix)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
