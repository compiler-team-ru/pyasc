// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s --ascendc-compute-memory-consumption --split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {asc.memory_consumed = {UB = 1536 : i64}} {
// CHECK-NEXT: func.func @test_ub() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 veccalc, 1024, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_ub() {
    %0 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<64xf32>
    %1 = ascendc.local_tensor_v3 veccalc, 1024, 128 : !ascendc.local_tensor<32xf32>
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.memory_consumed = {L0A = 256 : i64, L1 = 512 : i64, UB = 1024 : i64}} {
// CHECK-NEXT: func.func @test_mixed() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: %2 = ascendc.local_tensor_v3 a2, 0, 64 : !ascendc.local_tensor<16xf32>
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_mixed() {
    %0 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<64xf32>
    %1 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<32xf32>
    %2 = ascendc.local_tensor_v3 a2, 0, 64 : !ascendc.local_tensor<16xf32>
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.memory_consumed = {}} {
// CHECK-NEXT: func.func @test_empty() {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_empty() {
    return
  }
}
