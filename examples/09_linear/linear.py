# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import logging
import weakref

import torch

import asc
import asc.lib.host as host
import asc.lib.runtime as rt
import asc.runtime.config as config
from asc.lib.runtime.support import DeviceInfoType, DeviceModuleType

logging.basicConfig(level=logging.INFO)

# A 128x128x128 block fits L0A/L0B/L0C and is the Matmul API baseline.
BASE_M = 128
BASE_N = 128
BASE_K = 128
# A 64-row direct block creates enough tasks when 128-row blocks underfill AICs.
DIRECT_M_TILE = 64
# The direct path advances K by 512 elements to amortize L1/L0 transfers.
DIRECT_K_TILE = 512
# Each Matmul kernel launch uses this CANN-recommended 16 MiB workspace budget.
WORKSPACE_BYTES = 16 * 1024 * 1024
# A 1024-row static tile reuses narrow projection weights across a long M range.
LARGE_M_TILE = 1024
# Keep up to eight live weights prepacked without retaining their source tensors.
MAX_NZ_WEIGHT_CACHE_SIZE = 8


# Return the number of AICs available on the current device.
# Inputs: none; reads runtime device information.
# Output: a positive AIC core count.
def get_max_aic_core_num() -> int:
    return max(1, int(rt.device_info(
        DeviceModuleType.RT_MODULE_TYPE_AICORE,
        DeviceInfoType.INFO_TYPE_CORE_NUM,
    )))


# Map one output tile to input/output offsets and tail sizes. B is stored in NZ
# layout, so its N-stride is singleCoreN*16 instead of Kb*singleCoreN.
# Inputs: runtime Matmul tiling and a logical output tile ID.
# Output: A/B/C offsets and valid M/N sizes for that tile.
@asc.jit
def calc_offsets(tiling: asc.adv.TCubeTiling, tile_id):
    m_blocks = tiling.m.ceildiv(tiling.single_core_m)
    m_index = tile_id % m_blocks
    n_index = tile_id // m_blocks
    offset_a = m_index * tiling.k_a * tiling.single_core_m
    offset_b = n_index * tiling.single_core_n * 16
    offset_c = m_index * tiling.n * tiling.single_core_m
    offset_c += n_index * tiling.single_core_n
    tail_m = tiling.m - m_index * tiling.single_core_m
    tail_n = tiling.n - n_index * tiling.single_core_n
    if tail_m >= tiling.single_core_m:
        tail_m = tiling.single_core_m
    if tail_n >= tiling.single_core_n:
        tail_n = tiling.single_core_n
    return offset_a, offset_b, offset_c, tail_m, tail_n


# Compute C=A*B^T with a runtime tiling. Each AIC processes output tiles in
# round-robin order; B is read from a host-prepacked NZ buffer.
# Inputs: GM addresses for A/B/C, runtime tiling, and Matmul workspace.
# Output: C is written in ND layout; the kernel has no Python return value.
@asc.jit(matmul_cube_only=True)
def linear_kernel(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress, tiling: asc.adv.TCubeTiling,
                  workspace: asc.GlobalAddress):
    if asc.ascend_is_aic():
        block_idx = asc.get_block_idx()
        m_blocks = tiling.m.ceildiv(tiling.single_core_m)
        n_blocks = tiling.n.ceildiv(tiling.single_core_n)
        total_blocks = m_blocks * n_blocks
        a_gm = asc.GlobalTensor()
        b_gm = asc.GlobalTensor()
        c_gm = asc.GlobalTensor()
        pipe = asc.TPipe()
        matmul = asc.adv.Matmul(
            a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16, False),
            b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.NZ, asc.float16, True),
            c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16),
            bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float32),
        )
        asc.adv.register_matmul(pipe, workspace, matmul, tiling)
        for tile_id in asc.range(block_idx, total_blocks, tiling.used_core_num):
            offsets = calc_offsets(tiling, tile_id)
            offset_a, offset_b, offset_c, tail_m, tail_n = offsets
            a_gm.set_global_buffer(a + offset_a)
            b_gm.set_global_buffer(b + offset_b)
            c_gm.set_global_buffer(c + offset_c)
            matmul.set_org_shape(tiling.m, tiling.n, tiling.k_a, tiling.k_b)
            matmul.set_tensor_a(a_gm, False)
            matmul.set_tensor_b(b_gm, True)
            matmul.set_tail(tail_m, tail_n)
            matmul.iterate_all(c_gm)
        matmul.end()
        asc.pipe_barrier(asc.PipeID.PIPE_ALL)


# The MDL and NORM variants stay separate because the Matmul schedule must be
# selected at compile time.
# Compute a compile-time-tiled projection with the MDL Matmul schedule.
# Inputs: A/B/C/workspace addresses and compile-time matrix/block dimensions.
# Output: C is written in ND layout; the kernel has no Python return value.
@asc.jit(matmul_cube_only=True, kernel_type=config.KernelType.AIC_ONLY)
def linear_kernel_static_projection(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                                    workspace: asc.GlobalAddress, m: asc.ConstExpr[int], n: asc.ConstExpr[int],
                                    k_len: asc.ConstExpr[int], m_tile: asc.ConstExpr[int], n_tile: asc.ConstExpr[int],
                                    base_m: asc.ConstExpr[int], base_n: asc.ConstExpr[int]):
    if asc.ascend_is_aic():
        pipe = asc.TPipe()
        mm_cfg = asc.adv.get_mm_config(asc.adv.MatmulShapeParams(m_tile, n_tile, k_len, base_m, base_n, BASE_K),
                                       asc.adv.MatmulFuncParams(enable_kdim_reorder_load=(k_len == 512)),
                                       asc.MatmulConfigMode.CONFIG_MDL)
        matmul = asc.adv.Matmul(a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16, False),
                                b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.NZ, asc.float16,
                                                     True), c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND,
                                                                                 asc.float16),
                                bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND,
                                                        asc.float32), matmul_config=mm_cfg)
        asc.adv.register_matmul(pipe, workspace, matmul)
        a_gm = asc.GlobalTensor()
        b_gm = asc.GlobalTensor()
        c_gm = asc.GlobalTensor()
        m_blocks = (m + m_tile - 1) // m_tile
        total_blocks = m_blocks * ((n + n_tile - 1) // n_tile)
        for tile_id in asc.range(asc.get_block_idx(), total_blocks, asc.get_block_num()):
            m_index, n_index = tile_id % m_blocks, tile_id // m_blocks
            tail_m, tail_n = m - m_index * m_tile, n - n_index * n_tile
            if tail_m >= m_tile:
                tail_m = m_tile + tile_id * 0
            if tail_n >= n_tile:
                tail_n = n_tile + tile_id * 0
            a_gm.set_global_buffer(a + m_index * m_tile * k_len)
            b_gm.set_global_buffer(b + n_index * n_tile * 16)
            c_gm.set_global_buffer(c + m_index * m_tile * n + n_index * n_tile)
            matmul.set_org_shape(m, n, k_len, k_len)
            matmul.set_tensor_a(a_gm, False)
            matmul.set_tensor_b(b_gm, True)
            matmul.set_tail(tail_m, tail_n)
            matmul.iterate_all(c_gm)
        matmul.end()
        asc.pipe_barrier(asc.PipeID.PIPE_ALL)


# Compute a reduction-heavy projection with the NORM Matmul schedule.
# Inputs: A/B/C/workspace addresses and compile-time matrix/block dimensions.
# Output: C is written in ND layout; the kernel has no Python return value.
@asc.jit(matmul_cube_only=True, kernel_type=config.KernelType.AIC_ONLY)
def linear_kernel_static_projection_norm(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                                         workspace: asc.GlobalAddress, m: asc.ConstExpr[int], n: asc.ConstExpr[int],
                                         k_len: asc.ConstExpr[int], m_tile: asc.ConstExpr[int],
                                         n_tile: asc.ConstExpr[int], base_m: asc.ConstExpr[int],
                                         base_n: asc.ConstExpr[int]):
    if asc.ascend_is_aic():
        pipe = asc.TPipe()
        mm_cfg = asc.adv.get_mm_config(asc.adv.MatmulShapeParams(m_tile, n_tile, k_len, base_m, base_n, BASE_K),
                                       asc.adv.MatmulFuncParams(enable_kdim_reorder_load=(k_len == 512)),
                                       asc.MatmulConfigMode.CONFIG_NORM)
        matmul = asc.adv.Matmul(a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16, False),
                                b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.NZ, asc.float16,
                                                     True), c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND,
                                                                                 asc.float16),
                                bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND,
                                                        asc.float32), matmul_config=mm_cfg)
        asc.adv.register_matmul(pipe, workspace, matmul)
        a_gm = asc.GlobalTensor()
        b_gm = asc.GlobalTensor()
        c_gm = asc.GlobalTensor()
        m_blocks = (m + m_tile - 1) // m_tile
        total_blocks = m_blocks * ((n + n_tile - 1) // n_tile)
        for tile_id in asc.range(asc.get_block_idx(), total_blocks, asc.get_block_num()):
            m_index, n_index = tile_id % m_blocks, tile_id // m_blocks
            tail_m, tail_n = m - m_index * m_tile, n - n_index * n_tile
            if tail_m >= m_tile:
                tail_m = m_tile + tile_id * 0
            if tail_n >= n_tile:
                tail_n = n_tile + tile_id * 0
            a_gm.set_global_buffer(a + m_index * m_tile * k_len)
            b_gm.set_global_buffer(b + n_index * n_tile * 16)
            c_gm.set_global_buffer(c + m_index * m_tile * n + n_index * n_tile)
            matmul.set_org_shape(m, n, k_len, k_len)
            matmul.set_tensor_a(a_gm, False)
            matmul.set_tensor_b(b_gm, True)
            matmul.set_tail(tail_m, tail_n)
            matmul.iterate_all(c_gm)
        matmul.end()
        asc.pipe_barrier(asc.PipeID.PIPE_ALL)


# Move one direct-Cube output tile through L1/L0 and accumulate it in L0C.
# Inputs: GM/L1/L0 tensors, tile geometry, and full matrix dimensions.
# Output: the accumulated float32 tile is left in c_l0.
@asc.jit
def compute_direct_tile(a_gm, b_gm, a_l1, b_l1, a_l0, b_l0, c_l0, m_base, n_base, m_actual, n_actual, m, n, k, n_tile):
    for k_base in range(0, k, DIRECT_K_TILE):
        asc.data_copy(a_l1, a_gm[m_base * k + k_base:],
                      asc.Nd2NzParams(1, m_actual, DIRECT_K_TILE, 0, k, DIRECT_M_TILE, 1, 0))
        asc.data_copy(b_l1, b_gm[(k_base // 16) * n * 16 + n_base * 16:],
                      asc.DataCopyParams(DIRECT_K_TILE // 16, n_actual, n - n_actual, n_tile - n_actual))
        asc.pipe_barrier(asc.PipeID.PIPE_ALL)
        for row in range(DIRECT_M_TILE // 16):
            asc.load_data(a_l0[row * (DIRECT_K_TILE // 16) * 256:], a_l1[row * 256:],
                          asc.LoadData2DParams(0, DIRECT_K_TILE // 16, DIRECT_M_TILE // 16, 0, 0, False, 0))
        asc.load_data(b_l0, b_l1, asc.LoadData2DParams(0, DIRECT_K_TILE * n_tile // 256, 1, 0, 0, False, 0))
        asc.pipe_barrier(asc.PipeID.PIPE_ALL)
        asc.mmad(
            c_l0, a_l0, b_l0,
            asc.MmadParams(m=m_actual, n=n_actual, k=DIRECT_K_TILE, unit_flag=0, cmatrix_source=False,
                           cmatrix_init_val=(k_base == 0)))
        asc.pipe_barrier(asc.PipeID.PIPE_ALL)


# Compute aligned short matrices directly with Cube instructions.
# Inputs: A/B/C addresses and compile-time M/N/K plus N tile size.
# Output: C is written in ND layout; the kernel has no Python return value.
@asc.jit(matmul_cube_only=True, kernel_type=config.KernelType.AIC_ONLY)
def linear_kernel_direct(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress, m: asc.ConstExpr[int],
                         n: asc.ConstExpr[int], k: asc.ConstExpr[int], n_tile: asc.ConstExpr[int]):
    if asc.ascend_is_aic():
        a_gm, b_gm, c_gm = asc.GlobalTensor(), asc.GlobalTensor(), asc.GlobalTensor()
        a_gm.set_global_buffer(a)
        b_gm.set_global_buffer(b)
        c_gm.set_global_buffer(c)
        cb = asc.LocalTensor(asc.uint8, asc.TPosition.A1, 0, 1024 * 1024)
        l0a = asc.LocalTensor(asc.uint8, asc.TPosition.A2, 0, 64 * 1024)
        l0b = asc.LocalTensor(asc.uint8, asc.TPosition.B2, 0, 64 * 1024)
        l0c = asc.LocalTensor(asc.uint8, asc.TPosition.CO1, 0, 128 * 1024)
        a_l1 = cb.reinterpret_cast(asc.float16)
        b_offset = DIRECT_M_TILE * DIRECT_K_TILE * asc.float16.sizeof()
        b_l1 = cb[b_offset:].reinterpret_cast(asc.float16)
        a_l0 = l0a.reinterpret_cast(asc.float16)
        b_l0 = l0b.reinterpret_cast(asc.float16)
        c_l0 = l0c.reinterpret_cast(asc.float32)
        m_blocks = (m + DIRECT_M_TILE - 1) // DIRECT_M_TILE
        total_blocks = m_blocks * ((n + n_tile - 1) // n_tile)
        for tile_id in asc.range(asc.get_block_idx(), total_blocks, asc.get_block_num()):
            m_base = (tile_id % m_blocks) * DIRECT_M_TILE
            n_base = (tile_id // m_blocks) * n_tile
            m_actual = m - m_base
            n_actual = n - n_base
            if m_actual >= DIRECT_M_TILE:
                m_actual = DIRECT_M_TILE + tile_id * 0
            if n_actual >= n_tile:
                n_actual = n_tile + tile_id * 0
            compute_direct_tile(a_gm, b_gm, a_l1, b_l1, a_l0, b_l0, c_l0, m_base, n_base, m_actual, n_actual, m, n, k,
                                n_tile)
            asc.fixpipe(
                c_gm[m_base * n + n_base:], c_l0,
                asc.FixpipeParamsV220(n_size=n_actual, m_size=m_actual, src_stride=DIRECT_M_TILE, dst_stride=n,
                                      quant_pre=asc.QuantModes.F322F16), asc.FixpipeConfig.cfg_row_major())
            asc.pipe_barrier(asc.PipeID.PIPE_ALL)


# Reorder an ND [n,k] weight into the contiguous NZ buffer consumed by Matmul:
# element (row, col) lands at (col//16)*(n*16) + row*16 + col%16. The reordering
# runs on the host and is memoized per tensor
# so it never launches a device op inside the measured kernel loop.
_NZ_WEIGHT_CACHE: dict[int, tuple] = {}


# Input: one ND float16 weight tensor shaped [N,K].
# Output: the equivalent contiguous NZ tensor on the original device.
def nd_to_nz_weight(weight: torch.Tensor) -> torch.Tensor:
    device = weight.device
    key = id(weight)
    entry = _NZ_WEIGHT_CACHE.get(key)
    if entry is not None:
        source_ref, cached = entry
        if source_ref() is weight and cached.device == device:
            return cached
        del _NZ_WEIGHT_CACHE[key]
    n, k = weight.shape
    nz = weight.cpu().reshape(n, k // 16, 16).permute(1, 0, 2).contiguous().reshape(-1)
    nz = nz.to(device)
    if len(_NZ_WEIGHT_CACHE) >= MAX_NZ_WEIGHT_CACHE_SIZE:
        del _NZ_WEIGHT_CACHE[next(iter(_NZ_WEIGHT_CACHE))]
    _NZ_WEIGHT_CACHE[key] = (weakref.ref(weight), nz)
    return nz


# Shrink output blocks only when the default split cannot occupy all AICs.
# Inputs: M/N dimensions and available AIC count.
# Output: selected M/N base blocks and resulting tile count.
def select_tiling_blocks(m: int, n: int, core_num: int):
    base_m, base_n = BASE_M, BASE_N
    tile_count = ((m + base_m - 1) // base_m) \
        * ((n + base_n - 1) // base_n)
    # Split only while the current blocks occupy at most half of the AICs, so
    # the smaller block meaningfully increases parallelism.
    if tile_count * 2 <= core_num:
        base_m //= 2
        tile_count = ((m + base_m - 1) // base_m) \
            * ((n + base_n - 1) // base_n)
    if tile_count * 2 <= core_num:
        base_n //= 2
        tile_count = ((m + base_m - 1) // base_m) \
            * ((n + base_n - 1) // base_n)
    return base_m, base_n, tile_count


# Build one runtime Matmul tiling for ND activations and NZ weights.
# Inputs: M/N/K dimensions and whether to request a fixed basic block.
# Output: a valid TCubeTiling, or None when CANN tiling fails.
def try_build_tiling(m: int, n: int, k: int, fixed_split: bool):
    max_cores = get_max_aic_core_num()
    base_m, base_n, tile_count = select_tiling_blocks(m, n, max_cores)
    core_num = min(max_cores, tile_count)
    mm = host.MultiCoreMatmulTiling(host.get_ascendc_platform())
    mm.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False)
    mm.set_b_type(host.TPosition.GM, host.CubeFormat.NZ, host.DataType.DT_FLOAT16, True)
    mm.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
    mm.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    mm.set_dim(core_num)
    mm.set_org_shape(m, n, k)
    mm.set_shape(m, n, k)
    mm.set_traverse(host.MatrixTraverse.FIRSTM)
    mm.enable_bias(False)
    mm.set_buffer_space(-1, -1, -1)
    # Keep the selected 64/128 base block stable across compilation paths.
    # Unsupported shapes retry with CANN automatic tiling in the caller.
    if fixed_split:
        mm.set_fix_split(base_m, base_n, BASE_K)
    tiling = asc.adv.TCubeTiling()
    ret = mm.get_tiling(tiling)
    return tiling if ret == 0 and int(tiling.used_core_num) > 0 else None


# Group base blocks into one larger per-core output region to reuse A/B data,
# reducing repeated input loads for large-M generic shapes.
# Inputs: M/N dimensions, available cores, and mutable runtime tiling.
# Output: None; updates the per-core shape and used-core count in tiling.
def group_output_tiles(m: int, n: int, core_num: int, tiling: asc.adv.TCubeTiling):
    base_tiles = ((m + BASE_M - 1) // BASE_M) \
        * ((n + BASE_N - 1) // BASE_N)
    if base_tiles <= 2 * core_num:
        return
    n_blocks = (n + BASE_N - 1) // BASE_N
    max_n_groups = 4 if n_blocks <= 4 else 2
    n_groups = min(n_blocks, max_n_groups, core_num)
    m_groups = (core_num + n_groups - 1) // n_groups
    single_m = (m + m_groups - 1) // m_groups
    single_m = (single_m + 15) // 16 * 16
    single_n = (n + n_groups - 1) // n_groups
    single_n = (single_n + 15) // 16 * 16
    tiling.single_core_m = single_m
    tiling.single_core_n = single_n
    total_groups = ((m + single_m - 1) // single_m) * ((n + single_n - 1) // single_n)
    tiling.used_core_num = min(core_num, total_groups)


# Prefer the common fixed basic block and fall back to CANN automatic tiling.
# Inputs: M/N/K dimensions.
# Output: a valid runtime TCubeTiling or raises RuntimeError.
def build_matmul_tiling(m: int, n: int, k: int):
    tiling = try_build_tiling(m, n, k, True)
    if tiling is None:
        tiling = try_build_tiling(m, n, k, False)
    if tiling is None:
        raise RuntimeError("Matmul tiling failed")
    group_output_tiles(m, n, get_max_aic_core_num(), tiling)
    return tiling


# Prepack weight and launch the selected static Matmul configuration.
# Inputs: tensors, matrix/block dimensions, block count, and schedule selector.
# Output: the provided output tensor after asynchronous kernel submission.
def _launch_static(x, weight, out, m, n, k_len, m_tile, n_tile, blocks, use_norm=False):
    weight_nz = nd_to_nz_weight(weight)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=x.device)
    kernel = linear_kernel_static_projection_norm if use_norm \
        else linear_kernel_static_projection
    kernel[blocks, rt.current_stream()](x, weight_nz, out, workspace, m, n, k_len, m_tile, n_tile, min(m_tile, BASE_M),
                                        min(n_tile, BASE_N))
    return out


# Choose a direct-Cube tile from alignment and available parallelism.
# Inputs: M/N/K dimensions and available AIC count.
# Output: (N tile size, tile count), or None when the direct path is unsuitable.
def select_direct_config(m: int, n: int, k: int, core_num: int):
    if m % 16 != 0 or n % 16 != 0 or k % DIRECT_K_TILE != 0:
        return None
    base_tiles = ((m + BASE_M - 1) // BASE_M) \
        * ((n + BASE_N - 1) // BASE_N)
    if base_tiles * 2 > core_num:
        return None
    n_tile = BASE_N
    tile_count = ((m + DIRECT_M_TILE - 1) // DIRECT_M_TILE) \
        * ((n + n_tile - 1) // n_tile)
    if tile_count < core_num:
        n_tile = BASE_N // 2
        tile_count = ((m + DIRECT_M_TILE - 1) // DIRECT_M_TILE) \
            * ((n + n_tile - 1) // n_tile)
    return n_tile, tile_count


# Validate the public Linear tensor contract before allocating intermediates.
# Inputs: activation x and weight tensors.
# Output: None; raises ValueError or TypeError for unsupported inputs.
def _validate_linear_inputs(x: torch.Tensor, weight: torch.Tensor):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("require x[M,K] and weight[N,K] with the same K")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("x and weight must use float16")
    if x.device != weight.device or x.device.type not in ("cpu", "npu"):
        raise ValueError("x and weight must be on the same CPU or NPU device")
    if x.shape[0] <= 0 or weight.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("M, N, and K must be positive")
    if weight.shape[0] % 16 != 0 or x.shape[1] % 16 != 0:
        raise ValueError("N and K must be multiples of 16 for NZ weight layout")


# Allocate the output and dispatch the selected kernel for this shape.
# Inputs: float16 x[M,K] and weight[N,K] tensors on the same device.
# Output: float16 output[M,N].
def linear_launch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _validate_linear_inputs(x, weight)
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float16, device=x.device)

    max_core_num = get_max_aic_core_num()
    direct_config = select_direct_config(m, n, k, max_core_num)
    base_tiles = ((m + BASE_M - 1) // BASE_M) \
        * ((n + BASE_N - 1) // BASE_N)
    n_blocks = (n + BASE_N - 1) // BASE_N
    supported_static_k = k in (DIRECT_K_TILE, 2 * DIRECT_K_TILE)
    use_reduction_static = supported_static_k \
        and direct_config is not None and k > DIRECT_K_TILE
    use_large_static = supported_static_k \
        and base_tiles > 2 * max_core_num and n_blocks <= 4
    large_tiles = ((m + LARGE_M_TILE - 1) // LARGE_M_TILE) * n_blocks
    use_large_static = use_large_static \
        and large_tiles * 2 >= max_core_num
    if use_reduction_static:
        n_tile = BASE_N // 2
        tiles = ((m + DIRECT_M_TILE - 1) // DIRECT_M_TILE) \
            * ((n + n_tile - 1) // n_tile)
        return _launch_static(x, weight, out, m, n, k, DIRECT_M_TILE, n_tile, min(max_core_num, tiles), use_norm=True)
    if use_large_static:
        return _launch_static(x, weight, out, m, n, k, LARGE_M_TILE, BASE_N, min(max_core_num, large_tiles))
    if direct_config is not None:
        n_tile, tile_count = direct_config
        blocks = min(max_core_num, tile_count)
        weight_nz = nd_to_nz_weight(weight)
        linear_kernel_direct[blocks, rt.current_stream()](x, weight_nz, out, m, n, k, n_tile)
        return out

    tiling = build_matmul_tiling(m, n, k)
    weight_nz = nd_to_nz_weight(weight)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=x.device)
    linear_kernel[tiling.used_core_num, rt.current_stream()](x, weight_nz, out, tiling, workspace)
    return out


# Run the sample correctness check for the selected backend and platform.
# Inputs: PyAsc backend and target platform.
# Output: None; raises AssertionError when the result is incorrect.
def linear_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    torch.manual_seed(42)
    x = torch.randn(128, 256, dtype=torch.float16, device=device)
    weight = torch.randn(256, 256, dtype=torch.float16, device=device)
    actual = linear_launch(x, weight)
    if device == "npu":
        torch.npu.synchronize()
    expected = (x.cpu().float() @ weight.cpu().float().T).to(torch.float16)
    assert torch.allclose(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", default="Model")
    parser.add_argument("-v", default=None)
    args = parser.parse_args()
    backend = config.Backend(args.r)
    platform = config.Platform(args.v) if args.v else None
    logging.info("[INFO] start process sample linear.")
    linear_custom(backend, platform)
    logging.info("[INFO] Sample linear run success.")
