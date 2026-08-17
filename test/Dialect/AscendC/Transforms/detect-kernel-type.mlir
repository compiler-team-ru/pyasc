// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s --ascendc-detect-kernel-type --split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {asc.kernel_type = "vector"} {
// CHECK-NEXT: func.func @test_vector(%arg0: !ascendc.local_tensor<*xf32>, %arg1: i32) {
// CHECK-NEXT: ascendc.add_l2 %arg0, %arg0, %arg0, %arg1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_vector(%arg0: !ascendc.local_tensor<*xf32>, %arg1: i32) {
    ascendc.add_l2 %arg0, %arg0, %arg0, %arg1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "cube"} {
// CHECK-NEXT: func.func @test_cube(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params) {
// CHECK-NEXT: ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_cube(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params) {
    ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "mixed"} {
// CHECK-NEXT: func.func @test_mixed(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params, %arg4: i32) {
// CHECK-NEXT: ascendc.add_l2 %arg0, %arg0, %arg0, %arg4 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT: ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_mixed(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params, %arg4: i32) {
    ascendc.add_l2 %arg0, %arg0, %arg0, %arg4 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
    return
  }
}
