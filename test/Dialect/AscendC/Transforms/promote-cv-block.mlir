// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s --ascendc-promote-cv-block --split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {asc.kernel_type = "cube"} {
// CHECK-NEXT: func.func @test_only_if_aic(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) {
// CHECK-NEXT: %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT: %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT: ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_only_if_aic(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) {
    %c16_i32 = arith.constant 16 : i32
    %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
    ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
      %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
      ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "vector"} {
// CHECK-NEXT: func.func @test_only_if_aiv(%arg0: !ascendc.local_tensor<32xf32>) {
// CHECK-NEXT: %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: ascendc.relu_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_only_if_aiv(%arg0: !ascendc.local_tensor<32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    ascendc.if_aiv(%arg0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
      %0 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
      ascendc.relu_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
      ascendc.yield %0 : !ascendc.local_tensor<32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "mixed"} {
// CHECK-NEXT: func.func @test_mixed(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.local_tensor<32xf32>) {
// CHECK-NEXT: %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT: %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT: %1 = ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT: %3 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT: ascendc.mmad %3, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT: ascendc.yield %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT: %2 = ascendc.if_aiv(%arg2 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
// CHECK-NEXT: %3 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: ascendc.relu_l2 %3, %arg2, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT: ascendc.yield %3 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_mixed(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.local_tensor<32xf32>) {
    %c16_i32 = arith.constant 16 : i32
    %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
    ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
      %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
      ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
    }
    %c32_i64 = arith.constant 32 : i64
    ascendc.if_aiv(%arg2 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<32xf32> {
      %1 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
      ascendc.relu_l2 %1, %arg2, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
      ascendc.yield %1 : !ascendc.local_tensor<32xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: module {
// CHECK-NEXT: func.func @test_no_if_ops(%arg0: !ascendc.local_tensor<32xf32>) {
// CHECK-NEXT: %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT: ascendc.relu_l2 %arg0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_no_if_ops(%arg0: !ascendc.local_tensor<32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    ascendc.relu_l2 %arg0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "cube"} {
// CHECK-NEXT: func.func @test_multiple_if_aic(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) {
// CHECK-NEXT: %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT: %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT: ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT: %2 = ascendc.local_tensor_v3 co1, 0, 2048 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT: ascendc.mmad %2, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_multiple_if_aic(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) {
    %c16_i32 = arith.constant 16 : i32
    %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
    ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
      %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
      ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
    }
    ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
      %1 = ascendc.local_tensor_v3 co1, 0, 2048 : !ascendc.local_tensor<16x16xf32>
      ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "cube"} {
// CHECK-NEXT: func.func @test_if_aic_no_result(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) {
// CHECK-NEXT: %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT: %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT: ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_if_aic_no_result(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) {
    %c16_i32 = arith.constant 16 : i32
    %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
    ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) {
      %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
      ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield
    }
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "vector"} {
// CHECK-NEXT: func.func @test_if_aiv_no_operand(%arg0: !ascendc.local_tensor<32xf32>) {
// CHECK-NEXT: %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: ascendc.relu_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_if_aiv_no_operand(%arg0: !ascendc.local_tensor<32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    ascendc.if_aiv -> !ascendc.local_tensor<32xf32> {
      %0 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
      ascendc.relu_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
      ascendc.yield %0 : !ascendc.local_tensor<32xf32>
    }
    return
  }
}
