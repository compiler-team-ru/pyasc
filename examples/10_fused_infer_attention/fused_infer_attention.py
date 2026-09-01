# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import logging
import math

import torch

import asc
import asc.lib.host as host
import asc.lib.runtime as rt
import asc.runtime.config as config
from asc.lib.runtime.support import DeviceInfoType, DeviceModuleType

logging.basicConfig(level=logging.INFO)

# A 512-column tile keeps the QK/PV Cube work large enough while fitting the
# score, mask, probability, and online-softmax state in AIV UB.
KV_TILE = 512
# SoftmaxFlashV2 handles eight rows per call to keep its temporary UB bounded.
SOFTMAX_ROWS = 8
# Two slots overlap QK/Softmax(inner) with PV(inner - 1).
PIPE_SLOTS = 2
# Vector instructions allow at most 255 repeats on the target architecture.
MAX_VECTOR_REPEATS = 255
# Two slots use eight distinct event IDs, all within the A2 range [0, 10].
FLAGS_PER_SLOT = 2
QK_READY_FLAG_BASE = 0
PROB_READY_FLAG_BASE = 1
PV_READY_FLAG_BASE = PIPE_SLOTS * FLAGS_PER_SLOT
PV_DONE_FLAG_BASE = PV_READY_FLAG_BASE + 1
# Per-slot AIC/AIV handshake:
# AIC QK -> QK_READY -> AIV Softmax -> PROB_READY -> AIC PV
#        -> PV_READY -> AIV accumulation -> PV_DONE -> AIC reuses the slot.
# CANN's minimum-size query returns 18,688 bytes for an 8x512 float16 tile.
SOFTMAX_TMP_BYTES = 18688
# Two Matmul instances each receive half of this workspace.
WORKSPACE_BYTES = 16 * 1024 * 1024
# Two 128-row Cube blocks balance task overhead and AIC/AIV pipeline overlap.
PREFERRED_QUERY_BLOCK_ROWS = 256
# One native 128-row Cube block is the minimum useful Query task.
MIN_QUERY_BLOCK_ROWS = 128
# Matmul and contiguous GM copies require Query starts to be 16-row aligned.
QUERY_ROW_ALIGN = 16
# Padding S to 64 keeps Matmul tails and workspace offsets aligned.
KERNEL_SEQ_ALIGN = 64
# SoftmaxFlashV2 stores eight float reduction values for each score row.
SOFTMAX_REDUCE_LANES = 8
# Vector division broadcasts sixteen half denominators for each output row.
NORMALIZE_BROADCAST_LANES = 16
# Two float broadcast regions plus one half denominator region use 96 bytes.
NORM_BYTES_PER_SUB_ROW = (2 * SOFTMAX_REDUCE_LANES * asc.float32.sizeof() +
                          NORMALIZE_BROADCAST_LANES * asc.float16.sizeof())


# Return ceil(value / divisor); both inputs are positive integers.
# Inputs: integer value and positive divisor.
# Output: integer ceiling quotient.
def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


# Return value rounded up to a positive alignment.
# Inputs: integer value and positive alignment.
# Output: the smallest aligned integer not less than value.
def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


# Return the number of AICs available on the current device.
# Inputs: none; reads runtime device information.
# Output: a positive AIC core count.
def get_max_aic_core_num() -> int:
    return max(1, int(rt.device_info(
        DeviceModuleType.RT_MODULE_TYPE_AICORE,
        DeviceInfoType.INFO_TYPE_CORE_NUM,
    )))


# Merge one PV tile into acc_state using the online-softmax rescale factor.
# Inputs: PV queue/tensor, online state, slot geometry, and first-tile flag.
# Output: None; updates acc_state in UB.
@asc.jit
def _accumulate_partial(partial_queue, partial_gm, acc_state, exp_max_state, slot, max_sub_rows, row_offset, sub_rows,
                        head_dim, is_first):
    partial_local = partial_queue.alloc_tensor(partial_gm.dtype)
    asc.data_copy(
        partial_local,
        partial_gm[row_offset * head_dim:],
        count=sub_rows * head_dim,
    )
    partial_queue.enque(partial_local)
    partial_local = partial_queue.deque(partial_gm.dtype)
    if is_first:
        asc.adds(acc_state, partial_local, 0.0, count=sub_rows * head_dim)
    else:
        params = asc.BinaryRepeatParams(
            dst_blk_stride=1,
            src0_blk_stride=0,
            src1_blk_stride=1,
            dst_rep_stride=head_dim // 16,
            src0_rep_stride=1,
            src1_rep_stride=head_dim // 16,
        )
        pv_exp = exp_max_state[slot * max_sub_rows * 16:]
        asc.mul(acc_state, pv_exp, acc_state, head_dim, sub_rows, params)
        asc.pipe_barrier(asc.PipeID.PIPE_V)
        asc.add(acc_state, partial_local, acc_state, count=sub_rows * head_dim)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    partial_queue.free_tensor(partial_local)


# Normalize accumulated rows with sum_state and write the final output to GM.
# Inputs: pipeline/buffer state, output address, row range, and head dimension.
# Output: None; writes normalized rows to out_gm.
@asc.jit
def _normalize_and_store(pipe, norm_buf, sum_state, acc_state, out_gm, out_offset, sub_rows, head_dim):
    reduce_count = sub_rows * SOFTMAX_REDUCE_LANES
    norm_float = norm_buf.get(asc.float32)
    # Two f32 broadcast lanes occupy reduce_count*2 f32 elements. Reinterpret
    # the following bytes as f16 denominators without overlapping that region.
    norm_half = norm_float[reduce_count * 2:].reinterpret_cast(asc.float16)
    copy_params = asc.CopyRepeatParams(dst_stride=2, src_stride=1, dst_repeat_size=16, src_repeat_size=8)
    repeats = (reduce_count + 63) // 64
    asc.copy(norm_float, sum_state, 64, repeats, copy_params)
    # Offset eight f32 values (32 bytes) to form the second broadcast lane.
    asc.copy(norm_float[8:], sum_state, 64, repeats, copy_params)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.cast(norm_half, norm_float, asc.RoundMode.CAST_ROUND, reduce_count * 2)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    div_params = asc.BinaryRepeatParams(
        dst_blk_stride=1,
        src0_blk_stride=1,
        src1_blk_stride=0,
        dst_rep_stride=head_dim // 16,
        src0_rep_stride=head_dim // 16,
        src1_rep_stride=1,
    )
    for row_start in asc.range(0, sub_rows, MAX_VECTOR_REPEATS):
        rows = sub_rows - row_start
        if rows > MAX_VECTOR_REPEATS:
            rows = MAX_VECTOR_REPEATS + row_start * 0
        acc_chunk = acc_state[row_start * head_dim:]
        # Each row uses 16 f16 denominator values for vector broadcast.
        norm_chunk = norm_half[row_start * NORMALIZE_BROADCAST_LANES:]
        asc.div(acc_chunk, acc_chunk, norm_chunk, head_dim, rows, div_params)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    event_id = pipe.fetch_event_id(event=asc.HardEvent.V_MTE3)
    asc.set_flag(asc.HardEvent.V_MTE3, event_id)
    asc.wait_flag(asc.HardEvent.V_MTE3, event_id)
    asc.data_copy(out_gm[out_offset:], acc_state, count=sub_rows * head_dim)


# Apply the first or update form of online Softmax to one score tile.
# Inputs: local score/state tensors, tilings, tile shape, and inner-tile index.
# Output: None; writes probabilities and updates online Softmax state.
@asc.jit
def _online_softmax(prob_local, score_local, tile_sum, tile_max, tile_exp, shared_tmp, first_tiling, update_tiling,
                    full_config, rows, kv_rows, inner):
    shape = asc.adv.SoftMaxShapeInfo(rows, kv_rows, rows, kv_rows)
    is_full_tile = rows == SOFTMAX_ROWS and kv_rows == KV_TILE
    if inner == 0:
        if is_full_tile:
            asc.adv.softmax_flash_v2(prob_local, tile_sum, tile_max, score_local, tile_exp, tile_sum, tile_max,
                                     first_tiling, shape, shared_tmp_buffer=shared_tmp, is_basic_block=True,
                                     config=full_config)
        else:
            asc.adv.softmax_flash_v2(prob_local, tile_sum, tile_max, score_local, tile_exp, tile_sum, tile_max,
                                     first_tiling, shape, shared_tmp_buffer=shared_tmp, is_basic_block=True)
    else:
        if is_full_tile:
            asc.adv.softmax_flash_v2(prob_local, tile_sum, tile_max, score_local, tile_exp, tile_sum, tile_max,
                                     update_tiling, shape, shared_tmp_buffer=shared_tmp, is_update=True,
                                     is_basic_block=True, config=full_config)
        else:
            asc.adv.softmax_flash_v2(prob_local, tile_sum, tile_max, score_local, tile_exp, tile_sum, tile_max,
                                     update_tiling, shape, shared_tmp_buffer=shared_tmp, is_update=True,
                                     is_basic_block=True)


# Scale and mask up to eight score rows, then write probabilities to GM.
# Inputs: queues, GM slices, online state, tilings, shape, scale, and mask flag.
# Output: None; writes one probability row group to prob_dst.
@asc.jit
def _run_softmax_rows(score_queue, mask_queue, prob_queue, score_src, prob_dst, mask_src, tile_sum, tile_max, tile_exp,
                      shared_tmp, first_tiling, update_tiling, full_config, rows, kv_rows, seq_len, scale, inner,
                      use_mask):
    elem_count = rows * kv_rows
    score_local = score_queue.alloc_tensor(asc.float16)
    asc.data_copy(score_local, score_src, count=elem_count)
    score_queue.enque(score_local)
    score_local = score_queue.deque(asc.float16)
    asc.muls(score_local, score_local, scale, count=elem_count)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    if use_mask:
        mask_local = mask_queue.alloc_tensor(asc.float16)
        mask_params = asc.DataCopyParams(block_count=rows, block_len=kv_rows * asc.float16.sizeof() // 32,
                                         src_stride=(seq_len - kv_rows) * asc.float16.sizeof() // 32, dst_stride=0)
        asc.data_copy(mask_local, mask_src, mask_params)
        mask_queue.enque(mask_local)
        mask_local = mask_queue.deque(asc.float16)
        asc.add(score_local, score_local, mask_local, count=elem_count)
        asc.pipe_barrier(asc.PipeID.PIPE_V)
        mask_queue.free_tensor(mask_local)
    prob_local = prob_queue.alloc_tensor(asc.float16)
    _online_softmax(prob_local, score_local, tile_sum, tile_max, tile_exp, shared_tmp, first_tiling, update_tiling,
                    full_config, rows, kv_rows, inner)
    score_queue.free_tensor(score_local)
    prob_queue.enque(prob_local)
    prob_local = prob_queue.deque(asc.float16)
    asc.data_copy(prob_dst, prob_local, count=elem_count)
    prob_queue.free_tensor(prob_local)


# Run AIV Softmax for one KV tile and consume the previous PV result.
# Inputs: AIV queues/state, GM tensors, task geometry, and launch constants.
# Output: None; publishes probabilities and updates accumulated output state.
@asc.jit
def _process_aiv_tile(score_queue, mask_queue, partial_queue, prob_queue, score_gm, prob_gm, partial_gm, mask_gm,
                      sum_state, max_state, exp_max_state, acc_state, shared_tmp, first_tiling, update_tiling,
                      full_config, core_idx, inner, effective_kv, chunk_row_start, row_offset, sub_rows, seq_len,
                      head_dim, scale, block_rows, max_sub_rows):
    slot, kv_start = inner % PIPE_SLOTS, inner * KV_TILE
    kv_rows = effective_kv - kv_start
    if kv_rows > KV_TILE:
        kv_rows = KV_TILE + inner * 0
    slot_base = (core_idx * PIPE_SLOTS + slot) * block_rows
    asc.cross_core_wait_flag(QK_READY_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
    for done in asc.range(0, sub_rows, SOFTMAX_ROWS):
        rows = sub_rows - done
        if rows > SOFTMAX_ROWS:
            rows = SOFTMAX_ROWS + done * 0
        gm_row = row_offset + done
        mask_offset = (chunk_row_start + gm_row) * seq_len + kv_start
        _run_softmax_rows(score_queue, mask_queue, prob_queue, score_gm[slot_base * KV_TILE + gm_row * kv_rows:],
                          prob_gm[slot_base * KV_TILE + gm_row * kv_rows:], mask_gm[mask_offset:], sum_state[done * 8:],
                          max_state[done * 8:], exp_max_state[(slot * max_sub_rows + done) * 16:], shared_tmp,
                          first_tiling, update_tiling, full_config, rows, kv_rows, seq_len, scale, inner,
                          kv_start + kv_rows > chunk_row_start)
    asc.cross_core_set_flag(PROB_READY_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_MTE3)
    if inner >= 1:
        pv_inner, pv_slot = inner - 1, (inner - 1) % PIPE_SLOTS
        pv_base = (core_idx * PIPE_SLOTS + pv_slot) * block_rows
        asc.cross_core_wait_flag(PV_READY_FLAG_BASE + pv_slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
        _accumulate_partial(partial_queue, partial_gm[pv_base * head_dim:], acc_state, exp_max_state, pv_slot,
                            max_sub_rows, row_offset, sub_rows, head_dim, pv_inner == 0)
        asc.cross_core_set_flag(PV_DONE_FLAG_BASE + pv_slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_MTE3)


# Drain the final PV tiles, normalize the task output, and release its last slot.
# Inputs: AIV pipeline state, GM tensors, task geometry, and inner-block count.
# Output: None; writes final output rows and releases the rotating slot.
@asc.jit
def _finish_aiv_task(pipe, partial_queue, partial_gm, exp_max_state, acc_state, norm_buf, sum_state, out_gm, core_idx,
                     q_row_start, row_offset, sub_rows, head_dim, block_rows, max_sub_rows, inner_blocks):
    flush_start = inner_blocks - 1
    # Keep the initializer as a runtime PlainValue, matching pv_slot below.
    last_slot = inner_blocks * 0
    for pv_inner in asc.range(flush_start, inner_blocks):
        pv_slot = pv_inner % PIPE_SLOTS
        pv_base = (core_idx * PIPE_SLOTS + pv_slot) * block_rows
        asc.cross_core_wait_flag(PV_READY_FLAG_BASE + pv_slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
        _accumulate_partial(partial_queue, partial_gm[pv_base * head_dim:], acc_state, exp_max_state, pv_slot,
                            max_sub_rows, row_offset, sub_rows, head_dim, pv_inner == 0)
        if pv_inner + 1 < inner_blocks:
            asc.cross_core_set_flag(PV_DONE_FLAG_BASE + pv_slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_MTE3)
        last_slot = pv_slot
    _normalize_and_store(pipe, norm_buf, sum_state, acc_state, out_gm, (q_row_start + row_offset) * head_dim, sub_rows,
                         head_dim)
    asc.cross_core_set_flag(PV_DONE_FLAG_BASE + last_slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_MTE3)


# Execute all online-softmax and accumulation stages for one AIV task.
# Inputs: AIV resources, GM tensors, task index, and launch geometry.
# Output: None; completes the Vector-side work for one Query task.
@asc.jit
def _run_aiv_task(pipe, score_queue, mask_queue, partial_queue, prob_queue, score_gm, prob_gm, partial_gm, mask_gm,
                  out_gm, sum_state, max_state, exp_max_state, acc_state, shared_tmp, norm_buf, first_tiling,
                  update_tiling, full_config, core_idx, task_idx, seq_len, head_dim, scale, block_rows, chunks_per_head,
                  max_sub_rows, sub_idx):
    head_idx = task_idx // chunks_per_head
    chunk_idx = task_idx - head_idx * chunks_per_head
    chunk_row_start = chunk_idx * block_rows
    q_rows = seq_len - chunk_row_start
    if q_rows > block_rows:
        q_rows = block_rows + task_idx * 0
    effective_kv = chunk_row_start + q_rows
    q_row_start = head_idx * seq_len + chunk_row_start
    first_rows = (q_rows + 1) // 2
    row_offset = sub_idx * first_rows
    sub_rows = first_rows + sub_idx * (q_rows - first_rows * 2)
    inner_blocks = (effective_kv + KV_TILE - 1) // KV_TILE
    for inner in asc.range(inner_blocks):
        _process_aiv_tile(score_queue, mask_queue, partial_queue, prob_queue, score_gm, prob_gm, partial_gm, mask_gm,
                          sum_state, max_state, exp_max_state, acc_state, shared_tmp, first_tiling, update_tiling,
                          full_config, core_idx, inner, effective_kv, chunk_row_start, row_offset, sub_rows, seq_len,
                          head_dim, scale, block_rows, max_sub_rows)
    _finish_aiv_task(pipe, partial_queue, partial_gm, exp_max_state, acc_state, norm_buf, sum_state, out_gm, core_idx,
                     q_row_start, row_offset, sub_rows, head_dim, block_rows, max_sub_rows, inner_blocks)


# Initialize AIV queues and buffers, then process this core's assigned tasks.
# Inputs: pipeline, GM addresses, tilings, core task range, and launch geometry.
# Output: None; writes this AIV sub-core's output rows.
@asc.jit
def _aiv_path(pipe, mask, out, score_addr, prob_addr, partial_addr, first_tiling, update_tiling, core_idx, task_begin,
              task_end, seq_len, head_dim, scale, block_rows, chunks_per_head):
    score_queue, mask_queue = (asc.TQue(asc.TPosition.VECIN, 1), asc.TQue(asc.TPosition.VECIN, 1))
    partial_queue, prob_queue = asc.TQue(asc.TPosition.VECIN, 1), asc.TQue(asc.TPosition.VECOUT, 1)
    sum_buf, max_buf = (asc.TBuf(asc.TPosition.VECCALC), asc.TBuf(asc.TPosition.VECCALC))
    exp_max_buf, acc_buf = (asc.TBuf(asc.TPosition.VECCALC), asc.TBuf(asc.TPosition.VECCALC))
    norm_buf, tmp_buf = (asc.TBuf(asc.TPosition.VECCALC), asc.TBuf(asc.TPosition.VECCALC))
    max_sub_rows = (block_rows + 1) // 2
    tile_bytes = SOFTMAX_ROWS * KV_TILE * asc.float16.sizeof()
    pipe.init_buffer(que=score_queue, num=1, len=tile_bytes)
    pipe.init_buffer(que=mask_queue, num=1, len=tile_bytes)
    pipe.init_buffer(que=partial_queue, num=1, len=max_sub_rows * head_dim * asc.float16.sizeof())
    pipe.init_buffer(que=prob_queue, num=1, len=tile_bytes)
    pipe.init_buffer(buf=sum_buf, len=max_sub_rows * 8 * asc.float32.sizeof())
    pipe.init_buffer(buf=max_buf, len=max_sub_rows * 8 * asc.float32.sizeof())
    pipe.init_buffer(buf=exp_max_buf, len=PIPE_SLOTS * max_sub_rows * 16 * asc.float16.sizeof())
    pipe.init_buffer(buf=acc_buf, len=max_sub_rows * head_dim * asc.float16.sizeof())
    pipe.init_buffer(buf=norm_buf, len=max_sub_rows * NORM_BYTES_PER_SUB_ROW)
    pipe.init_buffer(buf=tmp_buf, len=SOFTMAX_TMP_BYTES)
    sum_state, max_state = sum_buf.get(asc.float32), max_buf.get(asc.float32)
    exp_max_state = exp_max_buf.get(asc.float16)
    acc_state, shared_tmp = acc_buf.get(asc.float16), tmp_buf.get(asc.uint8)
    score_gm, prob_gm, partial_gm = (asc.GlobalTensor(), asc.GlobalTensor(), asc.GlobalTensor())
    mask_gm, out_gm = asc.GlobalTensor(), asc.GlobalTensor()
    score_gm.set_global_buffer(score_addr)
    prob_gm.set_global_buffer(prob_addr)
    partial_gm.set_global_buffer(partial_addr)
    mask_gm.set_global_buffer(mask)
    out_gm.set_global_buffer(out)
    full_config = asc.adv.SoftmaxConfig(False, SOFTMAX_ROWS, KV_TILE)
    sub_idx = asc.get_sub_block_idx()
    for task_idx in asc.range(task_begin, task_end):
        _run_aiv_task(pipe, score_queue, mask_queue, partial_queue, prob_queue, score_gm, prob_gm, partial_gm, mask_gm,
                      out_gm, sum_state, max_state, exp_max_state, acc_state, shared_tmp, norm_buf, first_tiling,
                      update_tiling, full_config, core_idx, task_idx, seq_len, head_dim, scale, block_rows,
                      chunks_per_head, max_sub_rows, sub_idx)


# Process one short-sequence QK result and publish its probability tile.
# Inputs: AIV resources, QK/probability/mask tensors, task/slot, and geometry.
# Output: None; writes probability data and signals the AIC consumer.
@asc.jit
def _short_softmax_task(score_queue, mask_queue, prob_queue, score_gm, prob_gm, mask_gm, sum_state, max_state,
                        exp_max_state, shared_tmp, first_tiling, update_tiling, full_config, core_idx, task_idx, slot,
                        seq_len, scale, block_rows, chunks_per_head, max_sub_rows, sub_idx):
    head_idx = task_idx // chunks_per_head
    chunk_idx = task_idx - head_idx * chunks_per_head
    chunk_row_start = chunk_idx * block_rows
    q_rows = seq_len - chunk_row_start
    if q_rows > block_rows:
        q_rows = block_rows + task_idx * 0
    kv_rows = chunk_row_start + q_rows
    first_rows = (q_rows + 1) // 2
    row_offset = sub_idx * first_rows
    sub_rows = first_rows + sub_idx * (q_rows - first_rows * 2)
    slot_base = (core_idx * PIPE_SLOTS + slot) * block_rows
    asc.cross_core_wait_flag(QK_READY_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
    for done in asc.range(0, sub_rows, SOFTMAX_ROWS):
        rows = sub_rows - done
        if rows > SOFTMAX_ROWS:
            rows = SOFTMAX_ROWS + done * 0
        gm_row = row_offset + done
        mask_offset = (chunk_row_start + gm_row) * seq_len
        _run_softmax_rows(score_queue, mask_queue, prob_queue, score_gm[slot_base * KV_TILE + gm_row * kv_rows:],
                          prob_gm[slot_base * KV_TILE + gm_row * kv_rows:], mask_gm[mask_offset:],
                          sum_state[(slot * max_sub_rows + done) * 8:], max_state[done * 8:], exp_max_state[done * 16:],
                          shared_tmp, first_tiling, update_tiling, full_config, rows, kv_rows, seq_len, scale, 0,
                          kv_rows > chunk_row_start)
    asc.cross_core_set_flag(PROB_READY_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_MTE3)


# Normalize one short-sequence PV result and release its rotating slot.
# Inputs: AIV resources, PV/output tensors, task/slot, and launch geometry.
# Output: None; writes normalized output rows and releases the slot.
@asc.jit
def _short_finish_task(pipe, partial_queue, partial_gm, sum_state, norm_buf, out_gm, core_idx, task_idx, slot, seq_len,
                       head_dim, block_rows, chunks_per_head, max_sub_rows, sub_idx):
    head_idx = task_idx // chunks_per_head
    chunk_idx = task_idx - head_idx * chunks_per_head
    chunk_row_start = chunk_idx * block_rows
    q_rows = seq_len - chunk_row_start
    if q_rows > block_rows:
        q_rows = block_rows + task_idx * 0
    q_row_start = head_idx * seq_len + chunk_row_start
    first_rows = (q_rows + 1) // 2
    row_offset = sub_idx * first_rows
    sub_rows = first_rows + sub_idx * (q_rows - first_rows * 2)
    slot_base = (core_idx * PIPE_SLOTS + slot) * block_rows
    asc.cross_core_wait_flag(PV_READY_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
    partial_local = partial_queue.alloc_tensor(asc.float16)
    asc.data_copy(partial_local, partial_gm[slot_base * head_dim + row_offset * head_dim:], count=sub_rows * head_dim)
    partial_queue.enque(partial_local)
    partial_local = partial_queue.deque(asc.float16)
    _normalize_and_store(pipe, norm_buf, sum_state[slot * max_sub_rows * 8:], partial_local, out_gm,
                         (q_row_start + row_offset) * head_dim, sub_rows, head_dim)
    partial_queue.free_tensor(partial_local)
    asc.cross_core_set_flag(PV_DONE_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_MTE3)


# Overlap QK/Softmax for task n with PV/normalization for task n - 1.
# Inputs: short-path AIV resources, task range, and launch geometry.
# Output: None; completes all assigned short-sequence tasks.
@asc.jit
def _run_short_aiv_pipeline(pipe, score_queue, mask_queue, partial_queue, prob_queue, score_gm, prob_gm, partial_gm,
                            mask_gm, out_gm, sum_state, max_state, exp_max_state, shared_tmp, norm_buf, first_tiling,
                            update_tiling, full_config, core_idx, task_begin, task_end, seq_len, head_dim, scale,
                            block_rows, chunks_per_head, max_sub_rows, sub_idx):
    task_count = task_end - task_begin
    for step in asc.range(task_count + 1):
        if step < task_count:
            _short_softmax_task(score_queue, mask_queue, prob_queue, score_gm, prob_gm, mask_gm, sum_state, max_state,
                                exp_max_state, shared_tmp, first_tiling, update_tiling, full_config, core_idx,
                                task_begin + step, step % PIPE_SLOTS, seq_len, scale, block_rows, chunks_per_head,
                                max_sub_rows, sub_idx)
        if step >= 1:
            pv_step = step - 1
            if pv_step < task_count:
                _short_finish_task(pipe, partial_queue, partial_gm, sum_state, norm_buf, out_gm, core_idx,
                                   task_begin + pv_step, pv_step % PIPE_SLOTS, seq_len, head_dim, block_rows,
                                   chunks_per_head, max_sub_rows, sub_idx)


# Initialize queues and buffers used by the short-sequence AIV pipeline.
# Inputs: pipeline, block dimensions, head dimension, and Softmax temp bytes.
# Output: initialized queues, buffers, and local state tensors.
@asc.jit
def _init_short_aiv_buffers(pipe, score_queue, mask_queue, partial_queue, prob_queue, sum_buf, max_buf, exp_max_buf,
                            tmp_buf, norm_buf, max_sub_rows, head_dim):
    tile_bytes = SOFTMAX_ROWS * KV_TILE * asc.float16.sizeof()
    pipe.init_buffer(que=score_queue, num=1, len=tile_bytes)
    pipe.init_buffer(que=mask_queue, num=1, len=tile_bytes)
    pipe.init_buffer(que=partial_queue, num=1, len=max_sub_rows * head_dim * asc.float16.sizeof())
    pipe.init_buffer(que=prob_queue, num=1, len=tile_bytes)
    pipe.init_buffer(buf=sum_buf, len=PIPE_SLOTS * max_sub_rows * 8 * asc.float32.sizeof())
    pipe.init_buffer(buf=max_buf, len=max_sub_rows * 8 * asc.float32.sizeof())
    pipe.init_buffer(buf=exp_max_buf, len=max_sub_rows * 16 * asc.float16.sizeof())
    pipe.init_buffer(buf=tmp_buf, len=SOFTMAX_TMP_BYTES)
    pipe.init_buffer(buf=norm_buf, len=max_sub_rows * NORM_BYTES_PER_SUB_ROW)


# Allocate short-sequence AIV state and run the cross-task pipeline.
# Inputs: pipeline, GM addresses, tilings, core task range, and launch geometry.
# Output: None; writes this AIV sub-core's short-sequence output rows.
@asc.jit
def _short_aiv_path(pipe, mask, out, score_addr, prob_addr, partial_addr, first_tiling, update_tiling, core_idx,
                    task_begin, task_end, seq_len, head_dim, scale, block_rows, chunks_per_head):
    score_queue = asc.TQue(asc.TPosition.VECIN, 1)
    mask_queue = asc.TQue(asc.TPosition.VECIN, 1)
    partial_queue = asc.TQue(asc.TPosition.VECIN, 1)
    prob_queue = asc.TQue(asc.TPosition.VECOUT, 1)
    sum_buf = asc.TBuf(asc.TPosition.VECCALC)
    max_buf = asc.TBuf(asc.TPosition.VECCALC)
    exp_max_buf = asc.TBuf(asc.TPosition.VECCALC)
    tmp_buf = asc.TBuf(asc.TPosition.VECCALC)
    norm_buf = asc.TBuf(asc.TPosition.VECCALC)
    max_sub_rows = (block_rows + 1) // 2
    _init_short_aiv_buffers(pipe, score_queue, mask_queue, partial_queue, prob_queue, sum_buf, max_buf, exp_max_buf,
                            tmp_buf, norm_buf, max_sub_rows, head_dim)
    score_gm, prob_gm = asc.GlobalTensor(), asc.GlobalTensor()
    partial_gm, mask_gm = asc.GlobalTensor(), asc.GlobalTensor()
    out_gm = asc.GlobalTensor()
    score_gm.set_global_buffer(score_addr)
    prob_gm.set_global_buffer(prob_addr)
    partial_gm.set_global_buffer(partial_addr)
    mask_gm.set_global_buffer(mask)
    out_gm.set_global_buffer(out)
    full_config = asc.adv.SoftmaxConfig(False, SOFTMAX_ROWS, KV_TILE)
    _run_short_aiv_pipeline(pipe, score_queue, mask_queue,
                            partial_queue, prob_queue, score_gm, prob_gm, partial_gm, mask_gm, out_gm,
                            sum_buf.get(asc.float32), max_buf.get(asc.float32), exp_max_buf.get(asc.float16),
                            tmp_buf.get(asc.uint8), norm_buf, first_tiling, update_tiling, full_config, core_idx,
                            task_begin, task_end, seq_len, head_dim, scale, block_rows, chunks_per_head, max_sub_rows,
                            asc.get_sub_block_idx())


# Run one Probability x Value tile and signal the AIV consumer.
# Inputs: PV Matmul, probability/value/output slices, tile shape, and slot.
# Output: None; writes one partial output tile and publishes its event.
@asc.jit
def _run_pv(mm_pv, prob_gm, value_gm, partial_gm, q_rows, head_dim, kv_rows, slot):
    asc.cross_core_wait_flag(PROB_READY_FLAG_BASE + (slot % PIPE_SLOTS) * FLAGS_PER_SLOT, mode_id=0,
                             pipe=asc.PipeID.PIPE_S)
    mm_pv.set_org_shape(q_rows, head_dim, kv_rows, head_dim)
    mm_pv.set_tensor_a(prob_gm, False)
    mm_pv.set_tensor_b(value_gm, False)
    mm_pv.set_tail(q_rows, head_dim, kv_rows)
    mm_pv.iterate_all(partial_gm, en_atomic=0, sync=True, en_sequential_write=False, wait_iterate_all=False,
                      fake_msg=False)
    asc.pipe_barrier(asc.PipeID.PIPE_ALL)
    asc.cross_core_set_flag(PV_READY_FLAG_BASE + (slot % PIPE_SLOTS) * FLAGS_PER_SLOT, mode_id=2,
                            pipe=asc.PipeID.PIPE_FIX)


# Drain pending PV tiles after the main QK loop; update pending in place.
# Inputs: PV Matmul/state, GM tensors, task geometry, and inner-block count.
# Output: None; writes remaining PV tiles and clears all pending slot flags.
@asc.jit
def _flush_pv_tiles(mm_pv, pending, prob_gm, v_gm, partial_gm, core_idx, head_row_start, q_rows, effective_kv, head_dim,
                    block_rows, inner_blocks):
    flush_start = inner_blocks - 1
    for flush in asc.range(flush_start, inner_blocks):
        slot, kv_start = flush % PIPE_SLOTS, flush * KV_TILE
        kv_rows = effective_kv - kv_start
        if kv_rows > KV_TILE:
            kv_rows = KV_TILE + flush * 0
        slot_base = (core_idx * PIPE_SLOTS + slot) * block_rows
        if pending[slot] != 0:
            asc.cross_core_wait_flag(PV_DONE_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
            pending[slot] = 0
        _run_pv(mm_pv, prob_gm[slot_base * KV_TILE:], v_gm[(head_row_start + kv_start) * head_dim:],
                partial_gm[slot_base * head_dim:], q_rows, head_dim, kv_rows, flush)
        pending[slot] = 1
    for slot in asc.static_range(PIPE_SLOTS):
        if pending[slot] != 0:
            asc.cross_core_wait_flag(PV_DONE_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
            pending[slot] = 0


# Execute QK tiles while overlapping PV work from the previous iteration.
# Inputs: QK/PV Matmul state, GM tensors, task index, and launch geometry.
# Output: None; writes QK/PV intermediates and drains the final PV tile.
@asc.jit
def _run_aic_task(mm_qk, mm_pv, pending, q_gm, k_gm, v_gm, score_gm, prob_gm, partial_gm, core_idx, task_idx, seq_len,
                  head_dim, block_rows, chunks_per_head):
    head_idx = task_idx // chunks_per_head
    chunk_idx = task_idx - head_idx * chunks_per_head
    chunk_row_start = chunk_idx * block_rows
    q_rows = seq_len - chunk_row_start
    if q_rows > block_rows:
        q_rows = block_rows + task_idx * 0
    effective_kv = chunk_row_start + q_rows
    head_row_start = head_idx * seq_len
    q_row_start = head_row_start + chunk_row_start
    inner_blocks = (effective_kv + KV_TILE - 1) // KV_TILE
    for inner in asc.range(inner_blocks):
        slot, kv_start = inner % PIPE_SLOTS, inner * KV_TILE
        kv_rows = effective_kv - kv_start
        if kv_rows > KV_TILE:
            kv_rows = KV_TILE + inner * 0
        slot_base = (core_idx * PIPE_SLOTS + slot) * block_rows
        mm_qk.set_org_shape(q_rows, kv_rows, head_dim, head_dim)
        mm_qk.set_tensor_a(q_gm[q_row_start * head_dim:], False)
        mm_qk.set_tensor_b(k_gm[(head_row_start + kv_start) * head_dim:], True)
        mm_qk.set_tail(q_rows, kv_rows, -1)
        mm_qk.iterate_all(score_gm[slot_base * KV_TILE:], en_atomic=0, sync=True, en_sequential_write=False,
                          wait_iterate_all=False, fake_msg=False)
        asc.pipe_barrier(asc.PipeID.PIPE_ALL)
        asc.cross_core_set_flag(QK_READY_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_FIX)
        if inner >= 1:
            prev_start, prev_slot = (inner - 1) * KV_TILE, (inner - 1) % PIPE_SLOTS
            prev_rows = effective_kv - prev_start
            if prev_rows > KV_TILE:
                prev_rows = KV_TILE + inner * 0
            prev_base = (core_idx * PIPE_SLOTS + prev_slot) * block_rows
            if pending[prev_slot] != 0:
                asc.cross_core_wait_flag(PV_DONE_FLAG_BASE + prev_slot * FLAGS_PER_SLOT, mode_id=0,
                                         pipe=asc.PipeID.PIPE_S)
                pending[prev_slot] = 0
            _run_pv(mm_pv, prob_gm[prev_base * KV_TILE:], v_gm[(head_row_start + prev_start) * head_dim:],
                    partial_gm[prev_base * head_dim:], q_rows, head_dim, prev_rows, inner - 1)
            pending[prev_slot] = 1
    _flush_pv_tiles(mm_pv, pending, prob_gm, v_gm, partial_gm, core_idx, head_row_start, q_rows, effective_kv, head_dim,
                    block_rows, inner_blocks)


# Produce one compact QK tile for a short-sequence Query task.
# Inputs: QK Matmul, Q/K/score tensors, task/slot, and launch geometry.
# Output: None; writes one score tile and signals the AIV consumer.
@asc.jit
def _short_qk_task(mm_qk, q_gm, k_gm, score_gm, core_idx, task_idx, slot, seq_len, head_dim, block_rows,
                   chunks_per_head):
    head_idx = task_idx // chunks_per_head
    chunk_idx = task_idx - head_idx * chunks_per_head
    chunk_row_start = chunk_idx * block_rows
    q_rows = seq_len - chunk_row_start
    if q_rows > block_rows:
        q_rows = block_rows + task_idx * 0
    kv_rows = chunk_row_start + q_rows
    head_row_start = head_idx * seq_len
    slot_base = (core_idx * PIPE_SLOTS + slot) * block_rows
    mm_qk.set_org_shape(q_rows, kv_rows, head_dim, head_dim)
    mm_qk.set_tensor_a(q_gm[(head_row_start + chunk_row_start) * head_dim:], False)
    mm_qk.set_tensor_b(k_gm[head_row_start * head_dim:], True)
    mm_qk.set_tail(q_rows, kv_rows, -1)
    mm_qk.iterate_all(score_gm[slot_base * KV_TILE:], en_atomic=0, sync=True, en_sequential_write=False,
                      wait_iterate_all=False, fake_msg=False)
    asc.pipe_barrier(asc.PipeID.PIPE_ALL)
    asc.cross_core_set_flag(QK_READY_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=2, pipe=asc.PipeID.PIPE_FIX)


# Consume one short-sequence probability tile and publish its PV result.
# Inputs: PV Matmul, V/probability/output tensors, task/slot, and geometry.
# Output: None; writes one partial output tile and signals the AIV consumer.
@asc.jit
def _short_pv_task(mm_pv, v_gm, prob_gm, partial_gm, core_idx, task_idx, slot, seq_len, head_dim, block_rows,
                   chunks_per_head):
    head_idx = task_idx // chunks_per_head
    chunk_idx = task_idx - head_idx * chunks_per_head
    chunk_row_start = chunk_idx * block_rows
    q_rows = seq_len - chunk_row_start
    if q_rows > block_rows:
        q_rows = block_rows + task_idx * 0
    kv_rows = chunk_row_start + q_rows
    head_row_start = head_idx * seq_len
    slot_base = (core_idx * PIPE_SLOTS + slot) * block_rows
    _run_pv(mm_pv, prob_gm[slot_base * KV_TILE:], v_gm[head_row_start * head_dim:], partial_gm[slot_base * head_dim:],
            q_rows, head_dim, kv_rows, slot)


# Rotate two workspace slots across independent short-sequence tasks.
# Inputs: QK/PV state, GM tensors, core task range, and launch geometry.
# Output: None; submits all short-sequence Cube work and tracks pending slots.
@asc.jit
def _run_short_aic_pipeline(mm_qk, mm_pv, pending, q_gm, k_gm, v_gm, score_gm, prob_gm, partial_gm, core_idx,
                            task_begin, task_end, seq_len, head_dim, block_rows, chunks_per_head):
    task_count = task_end - task_begin
    for step in asc.range(task_count + 1):
        if step < task_count:
            slot = step % PIPE_SLOTS
            if pending[slot] != 0:
                asc.cross_core_wait_flag(PV_DONE_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
                pending[slot] = 0
            _short_qk_task(mm_qk, q_gm, k_gm, score_gm, core_idx, task_begin + step, slot, seq_len, head_dim,
                           block_rows, chunks_per_head)
        if step >= 1:
            pv_step = step - 1
            if pv_step < task_count:
                slot = pv_step % PIPE_SLOTS
                _short_pv_task(mm_pv, v_gm, prob_gm, partial_gm, core_idx, task_begin + pv_step, slot, seq_len,
                               head_dim, block_rows, chunks_per_head)
                pending[slot] = 1


# Wait for outstanding AIV consumers and release the two Matmul objects.
# Inputs: QK/PV Matmul objects and pending slot flags.
# Output: None; all slot dependencies and Matmul operations are completed.
@asc.jit
def _finish_aic(mm_qk, mm_pv, pending):
    for slot in asc.static_range(PIPE_SLOTS):
        if pending[slot] != 0:
            asc.cross_core_wait_flag(PV_DONE_FLAG_BASE + slot * FLAGS_PER_SLOT, mode_id=0, pipe=asc.PipeID.PIPE_S)
    mm_qk.end()
    mm_pv.end()


# Initialize QK/PV Matmul objects and process this core's assigned tasks.
# Inputs: pipeline, Q/K/V and intermediate addresses, tilings, and task range.
# Output: None; writes Cube-side score and partial-output intermediates.
@asc.jit
def _aic_path(pipe, q, k, v, score_addr, prob_addr, partial_addr, workspace, qk_tiling, pv_tiling, core_idx, task_begin,
              task_end, seq_len, head_dim, block_rows, chunks_per_head):
    q_gm, k_gm, v_gm = asc.GlobalTensor(), asc.GlobalTensor(), asc.GlobalTensor()
    score_gm, prob_gm, partial_gm = (asc.GlobalTensor(), asc.GlobalTensor(), asc.GlobalTensor())
    q_gm.set_global_buffer(q)
    k_gm.set_global_buffer(k)
    v_gm.set_global_buffer(v)
    score_gm.set_global_buffer(score_addr)
    prob_gm.set_global_buffer(prob_addr)
    partial_gm.set_global_buffer(partial_addr)
    mm_config = asc.adv.MatmulConfig(enable_get_tensor_c=False, enable_set_bias=False, enable_quant_vector=False,
                                     enable_set_define_data=False, iterate_mode=asc.IterateMode.ITERATE_MODE_ALL,
                                     enable_mix_dual_master=True)
    a_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16, False)
    c_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16, False)
    bias_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float32)
    mm_qk = asc.adv.Matmul(a=a_type, b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16, True),
                           c=c_type, bias=bias_type, matmul_config=mm_config)
    mm_pv = asc.adv.Matmul(a=a_type, b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float16, False),
                           c=c_type, bias=bias_type, matmul_config=mm_config)
    asc.adv.register_matmul(pipe, workspace, mm_qk, qk_tiling)
    asc.adv.register_matmul(pipe, workspace + WORKSPACE_BYTES // 2, mm_pv, pv_tiling)
    pending = asc.array(asc.int32, PIPE_SLOTS, fill_value=0)
    if seq_len <= KV_TILE:
        _run_short_aic_pipeline(mm_qk, mm_pv, pending, q_gm, k_gm, v_gm, score_gm, prob_gm, partial_gm, core_idx,
                                task_begin, task_end, seq_len, head_dim, block_rows, chunks_per_head)
    else:
        for task_idx in asc.range(task_begin, task_end):
            _run_aic_task(mm_qk, mm_pv, pending, q_gm, k_gm, v_gm, score_gm, prob_gm, partial_gm, core_idx, task_idx,
                          seq_len, head_dim, block_rows, chunks_per_head)
    _finish_aic(mm_qk, mm_pv, pending)


# Dispatch the MIX entry to the AIC path or one of the two AIV paths.
# Inputs: BNSD Q/K/V, causal mask, intermediates, tilings, and task metadata.
# Output: BNSD float16 attention values written through out.
@asc.jit(
    matmul_cube_only=True,
    kernel_type=config.KernelType.MIX_AIC_1_2,
)
def fused_infer_attention_kernel(
        q: asc.GlobalAddress, k: asc.GlobalAddress, v: asc.GlobalAddress, mask: asc.GlobalAddress,
        out: asc.GlobalAddress, score_gm: asc.GlobalAddress, prob_gm: asc.GlobalAddress, partial_gm: asc.GlobalAddress,
        workspace: asc.GlobalAddress, qk_tiling: asc.adv.TCubeTiling, pv_tiling: asc.adv.TCubeTiling,
        first_tiling: asc.adv.SoftmaxTiling, update_tiling: asc.adv.SoftmaxTiling, task_starts: asc.GlobalAddress,
        task_ends: asc.GlobalAddress, seq_len: asc.ConstExpr[int], head_dim: asc.ConstExpr[int],
        scale: asc.ConstExpr[float], block_rows: asc.ConstExpr[int], chunks_per_head: asc.ConstExpr[int]):
    core_idx = asc.get_block_idx() // asc.get_task_ratio()
    starts_gm, ends_gm = asc.GlobalTensor(), asc.GlobalTensor()
    starts_gm.set_global_buffer(task_starts)
    ends_gm.set_global_buffer(task_ends)
    task_begin = starts_gm.get_value(core_idx)
    task_end = ends_gm.get_value(core_idx)
    pipe = asc.TPipe()
    if asc.ascend_is_aiv():
        if task_begin < task_end:
            if seq_len <= KV_TILE:
                _short_aiv_path(pipe, mask, out, score_gm, prob_gm, partial_gm, first_tiling, update_tiling, core_idx,
                                task_begin, task_end, seq_len, head_dim, scale, block_rows, chunks_per_head)
            else:
                _aiv_path(pipe, mask, out, score_gm, prob_gm, partial_gm, first_tiling, update_tiling, core_idx,
                          task_begin, task_end, seq_len, head_dim, scale, block_rows, chunks_per_head)
    if asc.ascend_is_aic():
        if task_begin < task_end:
            _aic_path(pipe, q, k, v, score_gm, prob_gm, partial_gm, workspace, qk_tiling, pv_tiling, core_idx,
                      task_begin, task_end, seq_len, head_dim, block_rows, chunks_per_head)


# Build a single-core QK or PV tiling object for the requested matrix shape.
# Inputs: M/N/K dimensions and whether B is transposed.
# Output: a valid TCubeTiling or raises RuntimeError.
def _build_matmul_tiling(m: int, n: int, k: int, transpose_b: bool):
    mm = host.MultiCoreMatmulTiling(host.get_ascendc_platform())
    mm.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False)
    mm.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, transpose_b)
    mm.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
    mm.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    mm.set_dim(1)
    mm.set_org_shape(m, n, k)
    mm.set_shape(m, n, k)
    mm.set_traverse(host.MatrixTraverse.FIRSTM)
    mm.enable_bias(False)
    mm.set_buffer_space(-1, -1, -1)
    mm.set_fix_split(128, min(128, n), 128)
    tiling = asc.adv.TCubeTiling()
    if mm.get_tiling(tiling) != 0 or int(tiling.used_core_num) <= 0:
        raise RuntimeError(f"Matmul tiling failed for ({m}, {n}, {k})")
    return tiling


# Build SoftmaxFlashV2 tiling for one 8 x 512 score tile.
# Inputs: none; dimensions are fixed by SOFTMAX_ROWS and KV_TILE.
# Output: one Softmax tiling object usable by a first or update call.
def _build_softmax_tiling():
    return asc.adv.SoftmaxTiling(
        src_m=8,
        src_k=512,
        src_size=4096,
        out_max_m=8,
        out_max_k=8,
        out_max_size=64,
        split_m=8,
        split_k=512,
        split_size=4096,
        reduce_m=8,
        reduce_k=8,
        reduce_size=64,
        range_m=1,
        tail_m=0,
        tail_split_size=0,
        tail_reduce_size=0,
    )


# Choose block rows and weighted task ranges; return all launch metadata.
# Inputs: batch, head count, sequence length, and head dimension.
# Output: core count, block rows, chunks per head, and per-core task ranges.
def _launch_config(batch: int, heads: int, seq_len: int, head_dim: int):
    max_cores = get_max_aic_core_num()
    total_heads = batch * heads
    # One-KV-tile tasks favor the native 128-row Cube block.
    if seq_len <= KV_TILE:
        block_rows = min(seq_len, MIN_QUERY_BLOCK_ROWS)
    else:
        tasks_per_head = max(1, _ceil_div(max_cores, total_heads))
        rows_for_parallelism = _align_up(_ceil_div(seq_len, tasks_per_head), QUERY_ROW_ALIGN)
        min_rows = min(seq_len, MIN_QUERY_BLOCK_ROWS)
        block_rows = max(
            min_rows,
            min(seq_len, PREFERRED_QUERY_BLOCK_ROWS, rows_for_parallelism),
        )
    chunks_per_head = _ceil_div(seq_len, block_rows)
    total_tasks = total_heads * chunks_per_head
    cores = max(1, min(max_cores, total_tasks))

    weights = []
    for task in range(total_tasks):
        chunk = task % chunks_per_head
        row_start = chunk * block_rows
        query_rows = min(block_rows, seq_len - row_start)
        effective_kv = row_start + query_rows
        weights.append(query_rows * _ceil_div(effective_kv, KV_TILE))
    total_weight = sum(weights)
    starts = [0] * (cores + 1)
    task, weight = 0, 0
    for core in range(1, cores):
        target = total_weight * core // cores
        while task < total_tasks and weight + weights[task] <= target:
            weight += weights[task]
            task += 1
        starts[core] = task
    starts[cores] = total_tasks
    return cores, block_rows, chunks_per_head, starts[:-1], starts[1:]


# Validate public tensor shapes, dtypes, devices, mask form, and supported D/S.
# Inputs: Q/K/V tensors and broadcastable causal mask.
# Output: None; raises ValueError or TypeError for unsupported inputs.
def _validate_inputs(q, k, v, mask):
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have the same BNSD shape")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must use float16")
    if (q.device.type not in ("cpu", "npu") or k.device != q.device or v.device != q.device):
        raise ValueError("q, k, and v must be on the same CPU or NPU device")
    seq_len, head_dim = q.shape[-2:]
    if seq_len <= 0 or head_dim not in (64, 128):
        raise ValueError("seq_len must be positive and head_dim must be 64 or 128")
    if mask.dtype != torch.bool:
        raise TypeError("mask must use bool")
    if mask.device != q.device:
        raise ValueError("mask must be on the same device as q")
    if mask.shape[-2:] != (seq_len, seq_len) or mask.numel() != seq_len * seq_len:
        raise ValueError("mask must contain one broadcastable [seq_len, seq_len] matrix")


# Pad the internal sequence to the Matmul K-tail alignment.
# Inputs: Q/K/V tensors and causal mask.
# Output: padded tensors/mask and aligned kernel sequence length.
def _pad_attention_inputs(q, k, v, mask):
    batch, heads, seq_len, head_dim = q.shape
    kernel_seq_len = _align_up(seq_len, KERNEL_SEQ_ALIGN)
    if kernel_seq_len == seq_len:
        return q, k, v, mask.reshape(seq_len, seq_len), seq_len
    kernel_shape = (batch, heads, kernel_seq_len, head_dim)
    padded = [torch.zeros(kernel_shape, dtype=q.dtype, device=q.device) for _ in range(3)]
    for dst, src in zip(padded, (q, k, v)):
        dst[:, :, :seq_len, :].copy_(src)
    padded_mask = torch.triu(
        torch.ones(kernel_seq_len, kernel_seq_len, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    padded_mask[:seq_len, :seq_len].copy_(mask.reshape(seq_len, seq_len))
    return padded[0], padded[1], padded[2], padded_mask, kernel_seq_len


# Allocate intermediates, launch the fused kernel, and return attention output.
# Inputs: BNSD float16 Q/K/V, causal mask, and optional scale.
# Output: BNSD float16 attention tensor with padding removed.
def fused_infer_attention_launch(q, k, v, mask, scale=None):
    _validate_inputs(q, k, v, mask)
    batch, heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    q_kernel, k_kernel, v_kernel, mask_kernel, kernel_seq_len = (_pad_attention_inputs(q, k, v, mask))
    cores, block_rows, chunks_per_head, starts, ends = _launch_config(batch, heads, kernel_seq_len, head_dim)
    mask_half = torch.zeros((kernel_seq_len, kernel_seq_len), dtype=torch.float16, device=q.device)
    mask_half.masked_fill_(mask_kernel, float("-inf"))
    out = torch.empty_like(q_kernel)
    score = torch.empty(cores * PIPE_SLOTS * block_rows * KV_TILE, dtype=torch.float16, device=q.device)
    prob = torch.empty_like(score)
    partial = torch.empty(cores * PIPE_SLOTS * block_rows * head_dim, dtype=torch.float16, device=q.device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
    starts_dev = torch.tensor(starts, dtype=torch.int32, device=q.device)
    ends_dev = torch.tensor(ends, dtype=torch.int32, device=q.device)
    tiling_rows = max(block_rows, 128)
    qk_tiling = _build_matmul_tiling(tiling_rows, KV_TILE, head_dim, True)
    pv_tiling = _build_matmul_tiling(tiling_rows, head_dim, KV_TILE, False)
    first_tiling = _build_softmax_tiling()
    update_tiling = _build_softmax_tiling()
    fused_infer_attention_kernel[cores, rt.current_stream()](q_kernel, k_kernel, v_kernel, mask_half, out, score, prob,
                                                             partial, workspace, qk_tiling, pv_tiling, first_tiling,
                                                             update_tiling, starts_dev, ends_dev, kernel_seq_len,
                                                             head_dim, scale, block_rows, chunks_per_head)
    return out[:, :, :seq_len, :]


# Launch the PyAsc backend repeatedly for benchmark warmup and measurement.
# Inputs: Q/K/V tensors, mask, scale, and warmup/iteration counts.
# Output: None; synchronizes after all launches.
def run_pyasc(q, k, v, mask, scale, warmup, iters):
    for _ in range(warmup + iters):
        fused_infer_attention_launch(q, k, v, mask, scale)
    torch.npu.synchronize()


# Compute the float32 CPU causal-attention reference output.
# Inputs: Q/K/V tensors, causal mask, and scale.
# Output: float32 CPU attention reference tensor.
def _attention_reference(q, k, v, mask, scale):
    q_cpu = q.cpu().float()
    k_cpu = k.cpu().float()
    v_cpu = v.cpu().float()
    scores = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
    scores.masked_fill_(mask.cpu(), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v_cpu)


# Run the sample correctness check for the selected backend and platform.
# Inputs: PyAsc backend and target platform.
# Output: None; raises AssertionError when the result is incorrect.
def fused_infer_attention_custom(backend, platform):
    config.set_platform(backend, platform)
    device = "npu" if backend == config.Backend.NPU else "cpu"
    shape = (1, 1, 64, 64)
    torch.manual_seed(42)
    q = torch.randn(shape, dtype=torch.float16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.triu(
        torch.ones(1, 1, shape[2], shape[2], dtype=torch.bool, device=device),
        diagonal=1,
    )
    scale = 1.0 / math.sqrt(shape[-1])
    actual = fused_infer_attention_launch(q, k, v, mask, scale)
    if device == "npu":
        torch.npu.synchronize()
    expected = _attention_reference(q, k, v, mask, scale)
    assert torch.allclose(actual.cpu().float(), expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", default="Model", help="backend to run")
    parser.add_argument("-v", default=None, help="platform to run")
    args = parser.parse_args()
    backend = config.Backend(args.r)
    platform = config.Platform(args.v) if args.v else None
    logging.info("[INFO] start process sample fused_infer_attention.")
    fused_infer_attention_custom(backend, platform)
    logging.info("[INFO] Sample fused_infer_attention run success.")
