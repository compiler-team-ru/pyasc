# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile
import torch


@asctile.jit(always_compile=True)
def vmuladd_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress, z_ptr: asctile.GlobalAddress,
                   out_ptr: asctile.GlobalAddress, size: int, tile_size: asctile.ConstExpr[int],
                   tile_per_block: asctile.ConstExpr[int], buffer_factor: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    z_gm = asctile.global_tensor(z_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    base_offset = asctile.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=buffer_factor):
        tile_offset = base_offset + i * tile_size
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size])
        y = asctile.copy_in(y_gm, [tile_offset], [tile_size])
        z = asctile.copy_in(z_gm, [tile_offset], [tile_size])
        out = asctile.inline_vf(
            """
            __ubuf__ float* v47 = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
            __ubuf__ float* v48 = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
            __ubuf__ float* v49 = reinterpret_cast<__ubuf__ float*>($3.GetPhyAddr());
            __ubuf__ float* v50 = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
            AscendC::Reg::RegTensor<float> v51;
            AscendC::Reg::RegTensor<float> v52;
            AscendC::Reg::RegTensor<float> v53;
            AscendC::Reg::RegTensor<float> v55;
            AscendC::Reg::RegTensor<float> v56;
            uint32_t v57 = 32;
            for (uint16_t v58 = 0; v58 < static_cast<uint16_t>(1); v58 += 1) {
              uint32_t v59 = v58 * 64;
              AscendC::Reg::MaskReg v60 = AscendC::Reg::UpdateMask<float>(v57);
              __ubuf__ float* v61 = v47 + v59;
              AscendC::Reg::DataCopy(v51, v61);
              __ubuf__ float* v62 = v48 + v59;
              AscendC::Reg::DataCopy(v52, v62);
              AscendC::Reg::Mul(v53, v51, v52, v60);
              __ubuf__ float* v63 = v49 + v59;
              AscendC::Reg::DataCopy(v55, v63);
              AscendC::Reg::Add(v56, v53, v55, v60);
              __ubuf__ float* v64 = v50 + v59;
              AscendC::Reg::DataCopy(v64, v56, v60);
            }
            """, x.shape, x.dtype, [x, y, z])
        asctile.copy_out(out, out_gm, [tile_offset])


def test_vmuladd_inline_vf():
    size = 8192
    x = torch.rand(size, dtype=torch.float32) * 10
    y = torch.rand(size, dtype=torch.float32) * 10
    z = torch.rand(size, dtype=torch.float32) * 10
    out = torch.empty_like(x)
    core_num = 16
    tile_size = 32
    num_tiles = asctile.ceildiv(size, tile_size)
    vmuladd_kernel[core_num](x, y, z, out, size, tile_size, asctile.ceildiv(num_tiles, core_num), buffer_factor=2)
    torch.testing.assert_close(out, x * y + z)
