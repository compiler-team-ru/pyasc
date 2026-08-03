# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import argparse
import torch
try:
    import torch_npu
except ModuleNotFoundError:
    pass

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt
from asc.lib.runtime.support import DeviceModuleType, DeviceInfoType

logging.basicConfig(level=logging.INFO)

BUFFER_NUM = 2
RMSNORM_ROW_ALIGN = 16


def _rows_per_call(hidden_size: int) -> int:
    if hidden_size <= 512:
        return 8
    if hidden_size <= 1024:
        return 4
    return 2


def get_max_core_num() -> int:
    return max(1, int(rt.device_info(DeviceModuleType.RT_MODULE_TYPE_VECTOR_CORE,
                                      DeviceInfoType.INFO_TYPE_CORE_NUM)))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_rmsnorm_launch_params(total_length, hidden_size, max_core_num=None):
    if max_core_num is None:
        max_core_num = get_max_core_num()
    total_rows = total_length // hidden_size
    effective_cores = max(1, min(total_rows, max_core_num))
    rows_per_core = ceil_div(total_rows, effective_cores)
    block_length = rows_per_core * hidden_size
    return effective_cores, block_length, rows_per_core


def pad_flat_tensor(x: torch.Tensor, padded_len: int) -> torch.Tensor:
    flat = x.reshape(-1)
    if padded_len <= flat.numel():
        return flat
    padded = flat.new_empty(padded_len)
    padded[:flat.numel()] = flat
    return padded


def empty_padded_like(x: torch.Tensor, padded_len: int) -> torch.Tensor:
    flat = x.reshape(-1)
    if padded_len <= flat.numel():
        return torch.empty_like(flat)
    return torch.empty(padded_len, dtype=x.dtype, device=x.device)


@asc.jit(kernel_type=config.KernelType.AIV_ONLY)
def rmsnorm_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, rms: asc.GlobalAddress,
                   gamma: asc.GlobalAddress,
                   block_length: asc.ConstExpr[int], hidden_size: asc.ConstExpr[int],
                   eps: asc.ConstExpr[float], rows_per_core: asc.ConstExpr[int],
                   max_rows: asc.ConstExpr[int]):

    offset = asc.get_block_idx() * block_length
    x_gm, y_gm, rms_gm, gamma_gm = asc.GlobalTensor(), asc.GlobalTensor(), asc.GlobalTensor(), asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)
    rms_gm.set_global_buffer(rms + asc.get_block_idx() * rows_per_core * 4)
    gamma_gm.set_global_buffer(gamma)

    pipe = asc.TPipe()
    in_queue = asc.TQue(asc.TPosition.VECIN, 1)
    out_queue = asc.TQue(asc.TPosition.VECOUT, 1)
    gamma_buf = asc.TBuf(asc.TPosition.VECCALC)
    aligned_rows = ((max_rows + RMSNORM_ROW_ALIGN - 1) // RMSNORM_ROW_ALIGN) * RMSNORM_ROW_ALIGN
    chunk_size, dtype_size = max_rows * hidden_size, x_gm.dtype.sizeof()
    buf_len = max_rows * hidden_size * dtype_size
    rmsnorm_init_buffers(pipe, in_queue, out_queue, gamma_buf, buf_len,
                         hidden_size * dtype_size)

    gamma_local = gamma_buf.get(gamma_gm.dtype, len=hidden_size)
    rmsnorm_load_gamma(gamma_gm, gamma_local, hidden_size)

    full_groups, rem_rows = rows_per_core // max_rows, rows_per_core % max_rows
    for i in asc.range(full_groups):
        off = i * chunk_size
        tiling = rmsnorm_make_tiling(max_rows, hidden_size, chunk_size, aligned_rows)
        rmsnorm_process_block(in_queue, out_queue, x_gm, y_gm, gamma_local, eps, tiling, off, chunk_size)

    if rem_rows > 0:
        rem_bsh = rem_rows * hidden_size
        off = full_groups * chunk_size
        rem_aligned = ((rem_rows + RMSNORM_ROW_ALIGN - 1) // RMSNORM_ROW_ALIGN) * RMSNORM_ROW_ALIGN
        tiling = rmsnorm_make_tiling(rem_rows, hidden_size, rem_bsh, rem_aligned)
        rmsnorm_process_block(in_queue, out_queue, x_gm, y_gm, gamma_local, eps, tiling, off, rem_bsh)

    # Per-row rms = x[0] * gamma[0] / (y[0] + eps)
    rms_buf = asc.TBuf(asc.TPosition.VECCALC)
    rms_compute_buf(pipe, rms_buf, rows_per_core)
    rms_compute_kernel(rms_gm, x_gm, y_gm, gamma_gm, rms_buf, rows_per_core, hidden_size, eps)


@asc.jit
def rmsnorm_process_block(in_queue: asc.TQue, out_queue: asc.TQue,
                           x_gm: asc.GlobalTensor, y_gm: asc.GlobalTensor,
                           gamma_local, eps: asc.ConstExpr[float],
                           tiling, off, count: asc.ConstExpr[int]):
    x_local = in_queue.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm[off:], count=count)
    in_queue.enque(x_local)
    x_local = in_queue.deque(x_gm.dtype)
    y_local = out_queue.alloc_tensor(y_gm.dtype)
    asc.adv.rmsnorm(y_local, x_local, gamma_local, eps, tiling, basic_block=True)
    in_queue.free_tensor(x_local)
    out_queue.enque(y_local)
    y_local = out_queue.deque(y_gm.dtype)
    asc.data_copy(y_gm[off:], y_local, count=count)
    out_queue.free_tensor(y_local)


@asc.jit
def rmsnorm_init_buffers(pipe: asc.TPipe, in_queue: asc.TQue, out_queue: asc.TQue,
                         gamma_buf: asc.TBuf, buf_len: asc.ConstExpr[int], gamma_len: asc.ConstExpr[int]):
    pipe.init_buffer(que=in_queue, num=BUFFER_NUM, len=buf_len)
    pipe.init_buffer(que=out_queue, num=BUFFER_NUM, len=buf_len)
    pipe.init_buffer(buf=gamma_buf, len=gamma_len)


@asc.jit
def rmsnorm_make_tiling(row_count: asc.ConstExpr[int], hidden_size: asc.ConstExpr[int],
                        bsh_length: asc.ConstExpr[int], aligned_rows: asc.ConstExpr[int]):
    return asc.adv.RmsNormTiling(
        b_length=1, s_length=row_count, h_length=hidden_size, original_h_length=hidden_size,
        reciprocal_of_h_length=1.0 / float(hidden_size), main_bsh_length=bsh_length,
        main_bs_length=row_count, main_bs_length_align=aligned_rows, loop_round=1,
        input_tail_pos=bsh_length, tail_bsh_length=0, tail_bs_length=0)


@asc.jit
def rmsnorm_load_gamma(gamma_gm: asc.GlobalTensor, gamma_local,
                       hidden_size: asc.ConstExpr[int]):
    asc.data_copy(gamma_local, gamma_gm, count=hidden_size)
    asc.set_flag(asc.HardEvent.MTE2_V)
    asc.wait_flag(asc.HardEvent.MTE2_V)


@asc.jit
def rms_compute_buf(pipe, rms_buf, rows_per_core):
    pipe.init_buffer(buf=rms_buf, len=rows_per_core * 4 + 16)


@asc.jit
def rms_compute_kernel(rms_gm, x_gm, y_gm, gamma_gm, rms_buf, rows_per_core, hidden_size, eps):
    rms_local = rms_buf.get(rms_gm.dtype, len=rows_per_core)
    g_tmp = rms_buf.get(rms_gm.dtype, len=1)
    asc.data_copy(g_tmp, gamma_gm, count=1)
    g0 = g_tmp.get_value(0)
    for row in asc.range(rows_per_core):
        off = row * hidden_size
        x_tmp = rms_buf.get(rms_gm.dtype, len=1)
        y_tmp = rms_buf.get(rms_gm.dtype, len=1)
        asc.data_copy(x_tmp, x_gm[off:], count=1)
        asc.data_copy(y_tmp, y_gm[off:], count=1)
        rms_local.set_value(row, x_tmp.get_value(0) * g0 / (y_tmp.get_value(0) + eps))
    asc.data_copy(rms_gm, rms_local, count=rows_per_core)


def rmsnorm_launch(x, gamma, eps=1e-6):
    total_length = x.numel()
    hidden_size = gamma.shape[0]
    effective_cores, block_length, rows_per_core = compute_rmsnorm_launch_params(
        total_length, hidden_size)
    padded_len = effective_cores * block_length
    x_pad = pad_flat_tensor(x, padded_len)
    y_pad = empty_padded_like(x, padded_len)
    max_rows = _rows_per_call(hidden_size)
    total_rows = total_length // hidden_size
    rms_pad = torch.empty(total_rows, dtype=x.dtype, device=x.device)
    rmsnorm_kernel[effective_cores, rt.current_stream()](
        x_pad, y_pad, rms_pad, gamma, block_length, hidden_size, eps,
        rows_per_core, max_rows)
    if x.device.type == "npu":
        torch.npu.synchronize()
    return y_pad[:total_length].reshape_as(x), rms_pad


def rmsnorm_reference(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() / rms * gamma.float()).to(x.dtype)


def rmsnorm_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    hidden_size = 256
    x = torch.randn(2, 8, hidden_size, dtype=torch.float32, device=device)
    gamma = torch.ones(hidden_size, dtype=torch.float32, device=device)
    y, _ = rmsnorm_launch(x, gamma)
    expected = rmsnorm_reference(x, gamma)
    assert torch.allclose(y, expected, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend to run")
    parser.add_argument("-v", type=str, default=None, help="platform to run")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError(f"Unsupported Backend! Supported: {list(config.Backend.__members__.keys())}")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [platform.value for platform in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] start process sample rmsnorm.")
    rmsnorm_custom(backend, platform)
    logging.info("[INFO] Sample rmsnorm run success.")
