// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascendc-insert-cross-core-sync %s | FileCheck %s

// CHECK-LABEL: func.func @aiv_trigger_no_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 0 : i32} {
// CHECK-NOT: cross_core
func.func @aiv_trigger_no_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
    %c256 = arith.constant 256 : i32
    ascendc.data_copy_l2 %1, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aic(%0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %2, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @same_core_no_trigger(%arg0: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 0 : i32} {
// CHECK-NOT: cross_core
func.func @same_core_no_trigger(%arg0: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
  %c32 = arith.constant 32 : i64
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
    ascendc.relu_l2 %1, %arg0, %c32 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i64
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aic(%0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    ascendc.yield %0 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aiv_trigger_aic_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_trigger_aic_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %c256 = arith.constant 256 : i32
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @loop_forward_backward_sync(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aic {
// CHECK-NEXT:  %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_s : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %arg1) -> (!ascendc.local_tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:      ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_mte3 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @loop_forward_backward_sync(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c256 = arith.constant 256 : i32
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %arg1) -> !ascendc.local_tensor<16x16xf32> : i32 {
    ascendc.if_aiv {
      ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
      ascendc.yield
    }
    %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
      ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
    }
    scf.yield %0 : !ascendc.local_tensor<16x16xf32>
  }
  return %result : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aiv_trigger_two_aic_consumer_groups(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.mmad %1, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %0, %c256_i32 {direction = #ascendc.copy_direction<a1, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %5 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    ascendc.mmad %2, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %5 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_trigger_two_aic_consumer_groups(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %co2 = ascendc.local_tensor_v3 co1, 1024, 1024 : !ascendc.local_tensor<16x16xf32>
  %ub = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %c256 = arith.constant 256 : i32
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co1, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  ascendc.if_aiv {
    ascendc.data_copy_l2 %ub, %dst, %c256 {direction = #ascendc.copy_direction<a1, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %1 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co2, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @same_core_triggers(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 0 : i32} {
// CHECK-NOT: cross_core
func.func @same_core_triggers(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
  %c256 = arith.constant 256 : i32
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
    ascendc.data_copy_l2 %1, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aiv(%arg1 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %2 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
    ascendc.data_copy_l2 %2, %arg1, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aiv_trigger_two_consumers_same_group(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.mmad %1, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.mmad %2, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_trigger_two_consumers_same_group(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %co2 = ascendc.local_tensor_v3 co1, 1024, 1024 : !ascendc.local_tensor<16x16xf32>
  %c256 = arith.constant 256 : i32
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co1, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.mmad %co2, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}
