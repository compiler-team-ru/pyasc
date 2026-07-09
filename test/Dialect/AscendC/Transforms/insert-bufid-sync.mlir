// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-insert-bufid-sync %s | FileCheck %s

// CHECK-LABEL: func.func @tbuf_get_tensor_basic
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg0 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.add_l2 %0, %0, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @tbuf_get_tensor_basic(%arg0: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.add_l2 %tensor, %tensor, %tensor, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @tbuf_multiple_tensors
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg0 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg1 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.add_l2 %0, %0, %1, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @tbuf_multiple_tensors(%arg0: !ascendc.tbuf<vecin>, %arg1: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %tensor0 = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %tensor1 = ascendc.tbuf.get_tensor %arg1 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.add_l2 %tensor0, %tensor0, %tensor1, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @data_copy_gm_to_tbuf
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg1 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @data_copy_gm_to_tbuf(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg1 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %tensor, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @data_copy_tbuf_to_gm
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg1 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @data_copy_tbuf_to_gm(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg1 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %arg0, %tensor, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @mmad_tbuf_tensors
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg0 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg1 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<a1>, !ascendc.local_tensor<*xf16>
// CHECK-NEXT:  %2 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 2 : i32} : !ascendc.tbuf<b1>, !ascendc.local_tensor<*xf16>
// CHECK-NEXT:  ascendc.get_buf pipe_m, 0
// CHECK-NEXT:  ascendc.get_buf pipe_m, 1
// CHECK-NEXT:  ascendc.get_buf pipe_m, 2
// CHECK-NEXT:  ascendc.mmad %0, %1, %2, %arg3 {ascendc.buf_ids = [0 : i32, 1 : i32, 2 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NEXT:  ascendc.rls_buf pipe_m, 2
// CHECK-NEXT:  ascendc.rls_buf pipe_m, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_m, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @mmad_tbuf_tensors(%arg0: !ascendc.tbuf<co1>, %arg1: !ascendc.tbuf<a1>, %arg2: !ascendc.tbuf<b1>, %arg3: !ascendc.mmad_params) {
  %dst = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
  %src0 = ascendc.tbuf.get_tensor %arg1 : !ascendc.tbuf<a1>, !ascendc.local_tensor<*xf16>
  %src1 = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<b1>, !ascendc.local_tensor<*xf16>
  ascendc.mmad %dst, %src0, %src1, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
  return
}

// CHECK-LABEL: func.func @load_data_g2l_tbuf
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg0 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<a1>, !ascendc.local_tensor<*xf16>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg1 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<a2>, !ascendc.local_tensor<*xf16>
// CHECK-NEXT:  ascendc.get_buf pipe_mte1, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte1, 1
// CHECK-NEXT:  ascendc.load_data_g2l %0, %1, %arg2 {ascendc.buf_ids = [0 : i32, 1 : i32]} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:  ascendc.rls_buf pipe_mte1, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_mte1, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @load_data_g2l_tbuf(%arg0: !ascendc.tbuf<a1>, %arg1: !ascendc.tbuf<a2>, %arg2: !ascendc.load_data_2d_params) {
  %dst = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<a1>, !ascendc.local_tensor<*xf16>
  %src = ascendc.tbuf.get_tensor %arg1 : !ascendc.tbuf<a2>, !ascendc.local_tensor<*xf16>
  ascendc.load_data_g2l %dst, %src, %arg2 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_params
  return
}

// CHECK-LABEL: func.func @fixpipe_tbuf_tensor
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg1 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_fix, 0
// CHECK-NEXT:  ascendc.fixpipe %arg0, %0, %arg2, %arg3 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  ascendc.rls_buf pipe_fix, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @fixpipe_tbuf_tensor(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.tbuf<co1>, %arg2: !ascendc.fixpipe_params_v220, %arg3: !ascendc.fixpipe_config) {
  %src = ascendc.tbuf.get_tensor %arg1 : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
  ascendc.fixpipe %arg0, %src, %arg2, %arg3 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
  return
}

// CHECK-LABEL: func.func @for_loop_sync
// CHECK:       scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    ascendc.add_l2 %arg0, %arg0, %arg0, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.set_flag mte3_mte2, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.set_flag mte3_mte1, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.wait_flag mte3_mte2, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.wait_flag mte3_mte1, %c0_i32_0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @for_loop_sync(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  scf.for %arg1 = %c0 to %c10 step %c1 : i32 {
    ascendc.add_l2 %arg0, %arg0, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  }
  return
}

// CHECK-LABEL: func.func @reinterpret_cast_tbuf
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg0 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<*xf32> to !ascendc.local_tensor<*xf16>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<*xf16> to !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.add_l2 %2, %2, %2, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reinterpret_cast_tbuf(%arg0: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %cast1 = ascendc.reinterpret_cast %tensor : !ascendc.local_tensor<*xf32> to !ascendc.local_tensor<*xf16>
  %cast2 = ascendc.reinterpret_cast %cast1 : !ascendc.local_tensor<*xf16> to !ascendc.local_tensor<*xf32>
  ascendc.add_l2 %cast2, %cast2, %cast2, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @subindex_tbuf
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg0 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%arg1] : !ascendc.local_tensor<*xf32>, i32, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.add_l2 %1, %1, %1, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @subindex_tbuf(%arg0: !ascendc.tbuf<vecin>, %arg1: i32) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %sub = ascendc.local_tensor.subindex %tensor[%arg1] : !ascendc.local_tensor<*xf32>, i32, !ascendc.local_tensor<*xf32>
  ascendc.add_l2 %sub, %sub, %sub, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @mixed_operations
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.add_l2 %0, %0, %arg1, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @mixed_operations(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %tensor, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.add_l2 %tensor, %tensor, %arg1, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg2, %tensor, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_mte2_vec
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.add_l2 %1, %0, %0, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 1
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %1, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 1
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_mte2_vec(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %tensor2 = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %tensor, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.add_l2 %tensor2, %tensor, %tensor, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %tensor2, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_mte2_mte3
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_mte2_mte3(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %tensor, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %tensor, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_vec_vec
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %2 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 2 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %3 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 3 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.get_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.add_l2 %2, %0, %1, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32, 2 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_v, 3
// CHECK-NEXT:  ascendc.get_buf pipe_v, 2
// CHECK-NEXT:  ascendc.add_l2 %3, %2, %2, %c256_i32 {ascendc.buf_ids = [2 : i32, 3 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 2
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 3
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 3
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %3, %c256_i32 {ascendc.buf_ids = [3 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 3
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_vec_vec(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>, %arg3: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %x_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %x2_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  %z2_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %x2_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.add_l2 %z_local, %x_local, %x2_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.add_l2 %z2_local, %z_local, %z_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %z2_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_vec_mte3
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %2 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 2 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.get_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.add_l2 %2, %0, %1, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32, 2 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 2
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %2, %c256_i32 {ascendc.buf_ids = [2 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 2
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_vec_mte3(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>, %arg3: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %x_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %x2_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %x2_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.add_l2 %z_local, %x_local, %x2_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_mte3_vec
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg4 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.add_l2 %1, %0, %0, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 1
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %1, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 1
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_mte3_vec(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: !ascendc.tbuf<vecin>, %arg4: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %x_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg4 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %x_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.add_l2 %z_local, %x_local, %x_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg2, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_nested_for
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %2 = ascendc.tbuf.get_tensor %arg4 {ascendc.buf_id = 2 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    scf.for %arg6 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:      ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:      ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:      ascendc.get_buf pipe_mte2, 1
// CHECK-NEXT:      ascendc.data_copy_l2 %1, %arg1, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:      ascendc.rls_buf pipe_mte2, 1
// CHECK-NEXT:      ascendc.get_buf pipe_v, 2
// CHECK-NEXT:      ascendc.get_buf pipe_v, 0
// CHECK-NEXT:      ascendc.get_buf pipe_v, 1
// CHECK-NEXT:      ascendc.add_l2 %2, %0, %1, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32, 2 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:      ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:      ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:      ascendc.rls_buf pipe_v, 2
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.set_flag mte3_mte2, %c0_i32_1 : i32
// CHECK-NEXT:      ascendc.set_flag mte3_mte1, %c0_i32_1 : i32
// CHECK-NEXT:      ascendc.wait_flag mte3_mte2, %c0_i32_1 : i32
// CHECK-NEXT:      ascendc.wait_flag mte3_mte1, %c0_i32_1 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.set_flag mte3_mte2, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.set_flag mte3_mte1, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.wait_flag mte3_mte2, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.wait_flag mte3_mte1, %c0_i32_0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 2
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %2, %c256_i32 {ascendc.buf_ids = [2 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 2
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_nested_for(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: !ascendc.tbuf<vecin>, %arg4: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %x_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %y_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg4 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  scf.for %arg5 = %c0 to %c2 step %c1 : i32 {
    scf.for %arg6 = %c0 to %c4 step %c1 : i32 {
      ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
      ascendc.data_copy_l2 %y_local, %arg1, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
      ascendc.add_l2 %z_local, %x_local, %y_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    }
  }
  ascendc.data_copy_l2 %arg2, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_if_no_for
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    ascendc.get_buf pipe_v, 1
// CHECK-NEXT:    ascendc.get_buf pipe_v, 0
// CHECK-NEXT:    ascendc.add_l2 %1, %0, %0, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:    ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:    ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:    ascendc.get_buf pipe_mte3, 1
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %1, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:    ascendc.rls_buf pipe_mte3, 1
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_if_no_for(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>, %arg3: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %true = arith.constant true
  %x_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  scf.if %true {
    ascendc.add_l2 %z_local, %x_local, %x_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.data_copy_l2 %arg1, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  }
  return
}

// CHECK-LABEL: func.func @sync_for_if
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  scf.for %arg4 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:    ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:    scf.if %true {
// CHECK-NEXT:      ascendc.get_buf pipe_v, 1
// CHECK-NEXT:      ascendc.get_buf pipe_v, 0
// CHECK-NEXT:      ascendc.add_l2 %1, %0, %0, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:      ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:      ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:      ascendc.get_buf pipe_mte3, 1
// CHECK-NEXT:      ascendc.data_copy_l2 %arg1, %1, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:      ascendc.rls_buf pipe_mte3, 1
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.set_flag mte3_mte2, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.set_flag mte3_mte1, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.wait_flag mte3_mte2, %c0_i32_0 : i32
// CHECK-NEXT:    ascendc.wait_flag mte3_mte1, %c0_i32_0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_for_if(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>, %arg3: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  %x_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  scf.for %arg4 = %c0 to %c4 step %c1 : i32 {
    ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
    scf.if %true {
      ascendc.add_l2 %z_local, %x_local, %x_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
      ascendc.data_copy_l2 %arg1, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    }
  }
  return
}

// CHECK-LABEL: func.func @sync_mte3_mte2
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg1, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_mte3_mte2(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %tensor, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %tensor, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %tensor, %arg1, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %tensor, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_mte2_scalar
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_s, 0
// CHECK-NEXT:  ascendc.local_tensor.set_value %0, %c0_i32, %cst {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, i32, f32
// CHECK-NEXT:  ascendc.rls_buf pipe_s, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_mte2_scalar(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c55 = arith.constant 55.0 : f32
  %tensor = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %tensor, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.local_tensor.set_value %tensor, %c0, %c55 : !ascendc.local_tensor<*xf32>, i32, f32
  ascendc.data_copy_l2 %arg1, %tensor, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_vec_scalar
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %2 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 2 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.get_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.add_l2 %2, %0, %1, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32, 2 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_s, 2
// CHECK-NEXT:  ascendc.local_tensor.set_value %2, %c0_i32, %cst {ascendc.buf_ids = [2 : i32]} : !ascendc.local_tensor<*xf32>, i32, f32
// CHECK-NEXT:  ascendc.rls_buf pipe_s, 2
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 2
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %2, %c256_i32 {ascendc.buf_ids = [2 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 2
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_vec_scalar(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>, %arg3: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c55 = arith.constant 55.0 : f32
  %x_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %x2_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %x2_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.add_l2 %z_local, %x_local, %x2_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.local_tensor.set_value %z_local, %c0, %c55 : !ascendc.local_tensor<*xf32>, i32, f32
  ascendc.data_copy_l2 %arg1, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_scalar_vec
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %2 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 2 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.get_buf pipe_s, 0
// CHECK-NEXT:  ascendc.local_tensor.set_value %0, %c0_i32, %cst {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, i32, f32
// CHECK-NEXT:  ascendc.rls_buf pipe_s, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.add_l2 %2, %0, %1, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32, 2 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 2
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %2, %c256_i32 {ascendc.buf_ids = [2 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 2
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_scalar_vec(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>, %arg3: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c55 = arith.constant 55.0 : f32
  %x_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %x2_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %x2_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.local_tensor.set_value %x_local, %c0, %c55 : !ascendc.local_tensor<*xf32>, i32, f32
  ascendc.add_l2 %z_local, %x_local, %x2_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_scalar_scalar
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 1 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %2 = ascendc.tbuf.get_tensor %arg3 {ascendc.buf_id = 2 : i32} : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c256_i32 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 1
// CHECK-NEXT:  ascendc.get_buf pipe_s, 0
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %0, %c0_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, i32, f32
// CHECK-NEXT:  ascendc.rls_buf pipe_s, 0
// CHECK-NEXT:  ascendc.get_buf pipe_s, 0
// CHECK-NEXT:  ascendc.local_tensor.set_value %0, %c1_i32, %3 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, i32, f32
// CHECK-NEXT:  ascendc.rls_buf pipe_s, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_v, 0
// CHECK-NEXT:  ascendc.get_buf pipe_v, 1
// CHECK-NEXT:  ascendc.add_l2 %2, %0, %1, %c256_i32 {ascendc.buf_ids = [0 : i32, 1 : i32, 2 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 1
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 0
// CHECK-NEXT:  ascendc.rls_buf pipe_v, 2
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 2
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %2, %c256_i32 {ascendc.buf_ids = [2 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 2
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_scalar_scalar(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>, %arg3: !ascendc.tbuf<vecout>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %x_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %x2_local = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  %z_local = ascendc.tbuf.get_tensor %arg3 : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %x_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.data_copy_l2 %x2_local, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  %z = ascendc.local_tensor.get_value %x_local, %c0 : !ascendc.local_tensor<*xf32>, i32, f32
  ascendc.local_tensor.set_value %x_local, %c1, %z : !ascendc.local_tensor<*xf32>, i32, f32
  ascendc.add_l2 %z_local, %x_local, %x2_local, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.data_copy_l2 %arg1, %z_local, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @sync_scalar_mte3
// CHECK:       %0 = ascendc.tbuf.get_tensor %arg2 {ascendc.buf_id = 0 : i32} : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT:  ascendc.get_buf pipe_s, 0
// CHECK-NEXT:  ascendc.local_tensor.set_value %0, %c0_i32, %cst {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, i32, f32
// CHECK-NEXT:  ascendc.rls_buf pipe_s, 0
// CHECK-NEXT:  ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c256_i32 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:  ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @sync_scalar_mte3(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c55 = arith.constant 55.0 : f32
  %tensor = ascendc.tbuf.get_tensor %arg2 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.data_copy_l2 %tensor, %arg0, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.local_tensor.set_value %tensor, %c0, %c55 : !ascendc.local_tensor<*xf32>, i32, f32
  ascendc.data_copy_l2 %arg1, %tensor, %c256 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}
