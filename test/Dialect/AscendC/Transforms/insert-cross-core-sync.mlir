// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-insert-cross-core-sync %s | FileCheck %s

// CHECK-LABEL: func.func @aic_mmad_then_aiv(
// CHECK:       %0 = ascendc.if_aic(%arg0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    ascendc.mmad %2, %arg0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_fix : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c16_i32, 4, pipe_fix : i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte1 : i32
// CHECK-NEXT:    %c16_i32_1 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c16_i32_1, 4, pipe_mte1 : i32
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = ascendc.if_aiv(%0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<32xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_s : i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_s : i32
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:    ascendc.relu_l2 %2, %0, %arg3 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %1 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:}
func.func @aic_mmad_then_aiv(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params, %arg3: i64) -> !ascendc.local_tensor<32xf32> {
  %0 = ascendc.if_aic(%arg0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %1, %arg0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aiv(%0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<32xf32> {
    %2 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
    ascendc.relu_l2 %2, %0, %arg3: !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<16x16xf32>, i64
    ascendc.yield %2 : !ascendc.local_tensor<32xf32>
  }
  return %1 : !ascendc.local_tensor<32xf32>
}

// CHECK-LABEL: func.func @aiv_relu_then_aic(
// CHECK:       %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:    ascendc.relu_l2 %2, %arg0, %arg3 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = ascendc.if_aic(%arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_s : i32
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    ascendc.mmad %2, %arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_relu_then_aic(%arg0: !ascendc.local_tensor<32xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params, %arg3: i64) -> !ascendc.local_tensor<16x16xf32> {
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
    %1 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
    ascendc.relu_l2 %1, %arg0, %arg3 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
    ascendc.yield %1 : !ascendc.local_tensor<32xf32>
  }
  %1 = ascendc.if_aic(%arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %2, %arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aic_then_aic(
// CHECK:       ascendc.if_aic
// CHECK:       ascendc.if_aic
// CHECK-NOT:   cross_core
func.func @aic_then_aic(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %0 = ascendc.if_aic(%arg0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %1, %arg0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aic(%0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %2, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aiv_then_aiv(
// CHECK:       ascendc.if_aiv
// CHECK:       ascendc.if_aiv
// CHECK-NOT:   cross_core
func.func @aiv_then_aiv(%arg0: !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
  %c32_i64 = arith.constant 32 : i64
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
    %1 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
    ascendc.relu_l2 %1, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
    ascendc.yield %1 : !ascendc.local_tensor<32xf32>
  }
  %1 = ascendc.if_aiv(%0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
    %2 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
    ascendc.relu_l2 %2, %0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
    ascendc.yield %2 : !ascendc.local_tensor<32xf32>
  }
  return %1 : !ascendc.local_tensor<32xf32>
}

// CHECK-LABEL: func.func @aiv_local_to_local_then_aic(
// CHECK:       %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = ascendc.if_aic(%arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_s : i32
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    ascendc.mmad %2, %arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_local_to_local_then_aic(%arg0: !ascendc.local_tensor<32xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %c256 = arith.constant 256 : i32
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
    %1 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<32xf32>
    ascendc.data_copy_l2 %1, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i32
    ascendc.yield %1 : !ascendc.local_tensor<32xf32>
  }
  %1 = ascendc.if_aic(%arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %2, %arg1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}
