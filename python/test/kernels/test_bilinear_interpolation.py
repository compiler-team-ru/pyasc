# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt


@asc.jit
def vbilinear_kernel(x: asc.GlobalAddress, x0: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, \
                    src0_len: asc.ConstExpr[int], src0_offset_len: asc.ConstExpr[int], \
                    src1_len: asc.ConstExpr[int], dst_len: asc.ConstExpr[int], \
                    h_repeat: asc.ConstExpr[int], repeat_mode: asc.ConstExpr[bool], \
                    dst_blk_stride: asc.ConstExpr[int], v_r_offset: asc.ConstExpr[int], v_repeat: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    x0_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x)
    x0_gm.set_global_buffer(x0)
    y_gm.set_global_buffer(y)
    z_gm.set_global_buffer(z)
    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, 1)
    in_queue_x0 = asc.TQue(asc.TPosition.VECIN, 1)
    in_queue_y = asc.TQue(asc.TPosition.VECIN, 1)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, 1)
    tmp_buf = asc.TBuf(asc.TPosition.VECCALC)
    pipe.init_buffer(que=in_queue_x, num=1, len=src0_len * x.dtype.sizeof())
    pipe.init_buffer(que=in_queue_x0, num=1, len=src0_offset_len * x0.dtype.sizeof())
    pipe.init_buffer(que=in_queue_y, num=1, len=src1_len * y.dtype.sizeof())
    pipe.init_buffer(que=out_queue_z, num=1, len=dst_len * z.dtype.sizeof())
    pipe.init_buffer(buf=tmp_buf, len=(src0_len + src1_len) * x.dtype.sizeof())
    copy_in(x_gm, x0_gm, y_gm, in_queue_x, in_queue_x0, in_queue_y, src0_len, src0_offset_len, src1_len)
    compute(in_queue_x, in_queue_x0, in_queue_y, out_queue_z, tmp_buf, h_repeat, repeat_mode, dst_blk_stride,
            v_r_offset, v_repeat)
    copy_out(z_gm, out_queue_z, dst_len)


@asc.jit
def copy_in(x_gm: asc.GlobalAddress, x0_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress, \
            in_queue_x: asc.TQue, in_queue_x0: asc.TQue, in_queue_y: asc.TQue, \
            src0_len: asc.ConstExpr[int], src0_offset_len: asc.ConstExpr[int], src1_len: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    x0_local = in_queue_x0.alloc_tensor(x0_gm.dtype)
    y_local = in_queue_y.alloc_tensor(y_gm.dtype)
    asc.data_copy(x_local, x_gm, count=src0_len)
    asc.data_copy(x0_local, x0_gm, count=src0_offset_len)
    asc.data_copy(y_local, y_gm, count=src1_len)
    in_queue_x.enque(x_local)
    in_queue_x0.enque(x0_local)
    in_queue_y.enque(y_local)


@asc.jit
def compute(in_queue_x: asc.TQue, in_queue_x0: asc.TQue, in_queue_y: asc.TQue, out_queue_z: asc.TQue, \
            tmp_buf: asc.TBuf, h_repeat: asc.ConstExpr[int], repeat_mode: asc.ConstExpr[bool], \
            dst_blk_stride: asc.ConstExpr[int], v_r_offset: asc.ConstExpr[int], v_repeat: asc.ConstExpr[int]):
    x_local = in_queue_x.deque(asc.half)
    x0_local = in_queue_x0.deque(asc.uint32)
    y_local = in_queue_y.deque(asc.half)
    z_local = out_queue_z.alloc_tensor(asc.half)
    tmp = tmp_buf.alloc_tensor(asc.uint8)
    mask = 128
    asc.bilinear_interpolation(z_local, x_local, x0_local, y_local, mask, h_repeat, repeat_mode, dst_blk_stride,
                               v_r_offset, v_repeat, tmp)
    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)
    in_queue_x0.free_tensor(x0_local)
    in_queue_y.free_tensor(y_local)
    tmp_buf.free_tensor(tmp)


@asc.jit
def copy_out(z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, dst_len: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm, z_local, count=dst_len)
    out_queue_z.free_tensor(z_local)


def vbilinear_launch(x: np.ndarray, x0: np.ndarray, y: np.ndarray, \
                src0_len: asc.ConstExpr[int], src0_offset_len: asc.ConstExpr[int], \
                src1_len: asc.ConstExpr[int], dst_len: asc.ConstExpr[int], \
                h_repeat: asc.ConstExpr[int], repeat_mode: asc.ConstExpr[int], \
                dst_blk_stride: asc.ConstExpr[int], v_r_offset: asc.ConstExpr[int], \
                v_repeat: asc.ConstExpr) -> np.ndarray:
    z = np.array([0] * dst_len, dtype=np.int32)
    z = z.astype(np.float16)
    vbilinear_kernel[1, rt.current_stream()](x, x0, y, z, src0_len, src0_offset_len, src1_len, dst_len, h_repeat,
                                             repeat_mode, dst_blk_stride, v_r_offset, v_repeat)
    return z


def gen_golden_data(src0_len, src0_offset_len, src1_len, dst_len, h_repeat, repeat_mode, v_repeat):
    src0 = np.random.uniform(-10, 10, src0_len).astype('half')
    src0_offset = np.arange(0, src0_offset_len, 1, 'uint32')
    src0_offset = src0_offset * src0_offset_len
    src1 = np.random.uniform(-10, 10, src1_len).astype('half')
    gather_data = np.zeros(src0_len).astype('half')
    blk_index = 0
    loop_num = h_repeat * v_repeat * 8
    for _ in range(loop_num):
        for j in range(16):
            gather_data[blk_index * 16 + j] = src0[blk_index * 16 + j]
        blk_index += 1
    for i in range(gather_data.size // 128):
        for j in range(128):
            if repeat_mode:
                gather_data[i * 128 + j] = gather_data[i * 128 + j] * src1[i * 8 + j // 16]
            else:
                gather_data[i * 128 + j] = gather_data[i * 128 + j] * src1[i]
    dst = np.zeros(dst_len).astype('half')
    for i in range(0, v_repeat):
        for j in range(0, h_repeat):
            gather_data_start_index = i * h_repeat * 128 + j * 128
            gather_data_end_index = gather_data_start_index + 128
            dst[i * 128: (i + 1) * 128] = dst[i * 128: (i + 1) * 128] + \
                    gather_data[gather_data_start_index: gather_data_end_index]
    return src0, src0_offset, src1, dst


def test_vbilinear(backend: config.Backend):
    config.set_platform(backend)
    h_repeat = 2
    repeat_mode = False
    dst_blk_stride = 1
    v_r_offset = 128
    v_repeat = 2
    src0_len = 512
    src0_offset_len = 32
    src1_len = 16
    dst_len = 256
    src0, src0_offset, src1, dst = gen_golden_data(src0_len, src0_offset_len, src1_len, dst_len, h_repeat, repeat_mode,
                                                   v_repeat)
    z = vbilinear_launch(src0, src0_offset, src1, src0_len, src0_offset_len, src1_len, dst_len, h_repeat, repeat_mode,
                         dst_blk_stride, v_r_offset, v_repeat)
    np.testing.assert_allclose(z, dst, atol=1e-5)


if __name__ == "__main__":
    test_vbilinear(config.Backend.Model)
