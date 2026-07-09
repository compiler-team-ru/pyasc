// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-fuse-bufid-sync %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_same_bufid
// CHECK: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.add_l2
// CHECK-NOT: ascendc.rls_buf pipe_v, 0
// CHECK-NOT: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.mul_l2
// CHECK-NOT: ascendc.rls_buf pipe_v, 0
// CHECK-NOT: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.sub_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @fuse_same_bufid(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_v, 0
  ascendc.add_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  ascendc.get_buf pipe_v, 0
  ascendc.mul_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  ascendc.get_buf pipe_v, 0
  ascendc.sub_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  return
}

// CHECK-LABEL: func.func @different_pipe_no_fuse
// CHECK: ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NEXT: ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT: ascendc.get_buf pipe_v, 1
// CHECK-NEXT: ascendc.add_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 1
// CHECK-NEXT: ascendc.get_buf pipe_mte3, 2
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NEXT: ascendc.rls_buf pipe_mte3, 2
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @different_pipe_no_fuse(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_mte2, 0
  ascendc.data_copy_l2 %arg0, %arg1, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte2, 0
  ascendc.get_buf pipe_v, 1
  ascendc.add_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 1
  ascendc.get_buf pipe_mte3, 2
  ascendc.data_copy_l2 %arg2, %arg0, %c256 {ascendc.buf_ids = [2 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte3, 2
  return
}

// CHECK-LABEL: func.func @different_bufid_no_fuse
// CHECK: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.get_buf pipe_v, 1
// CHECK-NEXT: ascendc.add_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 1
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK-NEXT: ascendc.get_buf pipe_v, 2
// CHECK-NEXT: ascendc.get_buf pipe_v, 3
// CHECK-NEXT: ascendc.mul_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 3
// CHECK-NEXT: ascendc.rls_buf pipe_v, 2
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @different_bufid_no_fuse(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_v, 0
  ascendc.get_buf pipe_v, 1
  ascendc.add_l2 %arg0, %arg0, %arg1, %c256 {ascendc.buf_ids = [0, 1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 1
  ascendc.rls_buf pipe_v, 0
  ascendc.get_buf pipe_v, 2
  ascendc.get_buf pipe_v, 3
  ascendc.mul_l2 %arg1, %arg0, %arg1, %c256 {ascendc.buf_ids = [2, 3 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 3
  ascendc.rls_buf pipe_v, 2
  return
}

// CHECK-LABEL: func.func @for_loop_breaks_fusion
// CHECK: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.add_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK: scf.for
// CHECK: ascendc.get_buf pipe_v, 1
// CHECK-NEXT: ascendc.mul_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 1
// CHECK: return
// CHECK:}
func.func @for_loop_breaks_fusion(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  ascendc.get_buf pipe_v, 0
  ascendc.add_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  scf.for %arg1 = %c0 to %c10 step %c1 : i32 {
    ascendc.get_buf pipe_v, 1
    ascendc.mul_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.rls_buf pipe_v, 1
  }
  return
}

// CHECK-LABEL: func.func @if_op_breaks_fusion
// CHECK: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.add_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK: scf.if
// CHECK: ascendc.get_buf pipe_v, 1
// CHECK-NEXT: ascendc.mul_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 1
// CHECK: return
// CHECK:}
func.func @if_op_breaks_fusion(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  %true = arith.constant true
  ascendc.get_buf pipe_v, 0
  ascendc.add_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  scf.if %true {
    ascendc.get_buf pipe_v, 1
    ascendc.mul_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.rls_buf pipe_v, 1
  }
  return
}

// CHECK-LABEL: func.func @single_op_no_fusion
// CHECK: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.add_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @single_op_no_fusion(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_v, 0
  ascendc.add_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  return
}

// CHECK-LABEL: func.func @fuse_data_copy_mte2
// CHECK: ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NOT: ascendc.rls_buf pipe_mte2, 0
// CHECK-NOT: ascendc.get_buf pipe_mte2
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NEXT: ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @fuse_data_copy_mte2(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: !ascendc.global_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_mte2, 0
  ascendc.data_copy_l2 %arg0, %arg2, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte2, 0
  ascendc.get_buf pipe_mte2, 0
  ascendc.data_copy_l2 %arg1, %arg3, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte2, 0
  return
}

// CHECK-LABEL: func.func @fuse_data_copy_mte3
// CHECK: ascendc.get_buf pipe_mte3, 0
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NOT: ascendc.rls_buf pipe_mte3, 0
// CHECK-NOT: ascendc.get_buf pipe_mte3
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NEXT: ascendc.rls_buf pipe_mte3, 0
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @fuse_data_copy_mte3(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.local_tensor<*xf32>, %arg3: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_mte3, 0
  ascendc.data_copy_l2 %arg0, %arg2, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte3, 0
  ascendc.get_buf pipe_mte3, 0
  ascendc.data_copy_l2 %arg1, %arg3, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte3, 0
  return
}

// CHECK-LABEL: func.func @complex_sequence
// CHECK: ascendc.get_buf pipe_mte2, 0
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NEXT: ascendc.rls_buf pipe_mte2, 0
// CHECK-NEXT: ascendc.get_buf pipe_v, 1
// CHECK-NEXT: ascendc.add_l2
// CHECK-NOT: ascendc.rls_buf pipe_v, 1
// CHECK-NOT: ascendc.get_buf pipe_v
// CHECK-NEXT: ascendc.mul_l2
// CHECK-NOT: ascendc.rls_buf pipe_v, 1
// CHECK-NOT: ascendc.get_buf pipe_v
// CHECK-NEXT: ascendc.sub_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 1
// CHECK-NEXT: ascendc.get_buf pipe_mte3, 2
// CHECK-NEXT: ascendc.data_copy_l2
// CHECK-NEXT: ascendc.rls_buf pipe_mte3, 2
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @complex_sequence(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: !ascendc.global_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_mte2, 0
  ascendc.data_copy_l2 %arg0, %arg2, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte2, 0
  ascendc.get_buf pipe_v, 1
  ascendc.add_l2 %arg0, %arg0, %arg1, %c256 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 1
  ascendc.get_buf pipe_v, 1
  ascendc.mul_l2 %arg0, %arg0, %arg1, %c256 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 1
  ascendc.get_buf pipe_v, 1
  ascendc.sub_l2 %arg0, %arg0, %arg1, %c256 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 1
  ascendc.get_buf pipe_mte3, 2
  ascendc.data_copy_l2 %arg3, %arg0, %c256 {ascendc.buf_ids = [2 : i32]} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_mte3, 2
  return
}

// CHECK-LABEL: func.func @yield_breaks_fusion
// CHECK: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.add_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK: scf.for
// CHECK: ascendc.get_buf pipe_v, 1
// CHECK-NEXT: ascendc.mul_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 1
// CHECK: return
// CHECK:}
func.func @yield_breaks_fusion(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  ascendc.get_buf pipe_v, 0
  ascendc.add_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  scf.for %arg1 = %c0 to %c10 step %c1 : i32 {
    ascendc.get_buf pipe_v, 1
    ascendc.mul_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [1 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.rls_buf pipe_v, 1
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @return_breaks_fusion
// CHECK: ascendc.get_buf pipe_v, 0
// CHECK-NEXT: ascendc.add_l2
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @return_breaks_fusion(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.get_buf pipe_v, 0
  ascendc.add_l2 %arg0, %arg0, %arg0, %c256 {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.rls_buf pipe_v, 0
  return
}
