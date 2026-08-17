// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-insert-bias-bufid-sync %s | FileCheck %s

// CHECK-LABEL: func.func @basic_bias_sync
// CHECK: %[[BIAS:[0-9]+]] = ascendc.local_tensor_v3 c2, 0, 64 {ascendc.buf_id = 0 : i32} : !ascendc.local_tensor<64xf32>
// CHECK: ascendc.data_copy_l0 %[[BIAS]], %arg1, %{{[0-9]+}} {ascendc.buf_ids = [0 : i32]}
// CHECK: %[[PARAMS:[0-9]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c64_i32 : i32, "n" = %c64_i32 : i32, "k" = %c128_i32 : i32, "cmatrixInitVal" = %false : i1, "cmatrixSource" = %true : i1)
// CHECK-NEXT: ascendc.get_buf pipe_m, 0
// CHECK-NEXT: ascendc.mmad %arg0, %arg1, %arg2, %[[PARAMS]] : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NEXT: ascendc.rls_buf pipe_m, 0
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @basic_bias_sync(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %true = arith.constant true
  %false = arith.constant false
  %bias = ascendc.local_tensor_v3 c2, 0, 64 {ascendc.buf_id = 0 : i32} : !ascendc.local_tensor<64xf32>
  %params = ascendc.construct !ascendc.data_copy_params(%c256) : i32
  ascendc.data_copy_l0 %bias, %arg1, %params {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<*xf16>, !ascendc.data_copy_params
  %mmad_params = emitasc.init_struct !ascendc.mmad_params("m" = %c64 : i32, "n" = %c64 : i32, "k" = %c128 : i32, "cmatrixInitVal" = %false : i1, "cmatrixSource" = %true : i1)
  ascendc.mmad %arg0, %arg1, %arg2, %mmad_params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
  return
}

// CHECK-LABEL: func.func @no_bias_copy
// CHECK: %[[PARAMS:[0-9]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c64_i32 : i32, "n" = %c64_i32 : i32, "k" = %c128_i32 : i32, "cmatrixInitVal" = %false : i1, "cmatrixSource" = %true : i1)
// CHECK-NEXT: ascendc.mmad %arg0, %arg1, %arg2, %[[PARAMS]] : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NOT: ascendc.get_buf pipe_m
// CHECK-NOT: ascendc.rls_buf pipe_m
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @no_bias_copy(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %true = arith.constant true
  %false = arith.constant false
  %mmad_params = emitasc.init_struct !ascendc.mmad_params("m" = %c64 : i32, "n" = %c64 : i32, "k" = %c128 : i32, "cmatrixInitVal" = %false : i1, "cmatrixSource" = %true : i1)
  ascendc.mmad %arg0, %arg1, %arg2, %mmad_params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
  return
}

// CHECK-LABEL: func.func @no_cmatrix_source
// CHECK: %[[BIAS:[0-9]+]] = ascendc.local_tensor_v3 c2, 0, 64 {ascendc.buf_id = 0 : i32} : !ascendc.local_tensor<64xf32>
// CHECK: ascendc.data_copy_l0 %[[BIAS]], %arg1, %{{[0-9]+}} {ascendc.buf_ids = [0 : i32]}
// CHECK: %[[PARAMS:[0-9]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c64_i32 : i32, "n" = %c64_i32 : i32, "k" = %c128_i32 : i32, "cmatrixInitVal" = %true : i1)
// CHECK-NEXT: ascendc.mmad %arg0, %arg1, %arg2, %[[PARAMS]] : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NOT: ascendc.get_buf pipe_m
// CHECK-NOT: ascendc.rls_buf pipe_m
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @no_cmatrix_source(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %true = arith.constant true
  %false = arith.constant false
  %bias = ascendc.local_tensor_v3 c2, 0, 64 {ascendc.buf_id = 0 : i32} : !ascendc.local_tensor<64xf32>
  %params = ascendc.construct !ascendc.data_copy_params(%c256) : i32
  ascendc.data_copy_l0 %bias, %arg1, %params {ascendc.buf_ids = [0 : i32]} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<*xf16>, !ascendc.data_copy_params
  %mmad_params = emitasc.init_struct !ascendc.mmad_params("m" = %c64 : i32, "n" = %c64 : i32, "k" = %c128 : i32, "cmatrixInitVal" = %true : i1)
  ascendc.mmad %arg0, %arg1, %arg2, %mmad_params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
  return
}
