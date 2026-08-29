// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s --ascendc-detect-enable-debug --split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {asc.enable_debug} {
// CHECK-NEXT: func.func @test_printf(%arg0: i32) {
// CHECK-NEXT: ascendc.printf %arg0 {desc = "test print %d"} : i32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_printf(%arg0: i32) {
    ascendc.printf %arg0 {desc = "test print %d"} : i32
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.enable_debug} {
// CHECK-NEXT: func.func @test_dump_tensor(%arg0: !ascendc.global_tensor<*xf32>, %arg1: ui32, %arg2: ui32) {
// CHECK-NEXT: ascendc.dump_tensor %arg0, %arg1, %arg2 : !ascendc.global_tensor<*xf32>, ui32, ui32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_dump_tensor(%arg0: !ascendc.global_tensor<*xf32>, %arg1: ui32, %arg2: ui32) {
    ascendc.dump_tensor %arg0, %arg1, %arg2: !ascendc.global_tensor<*xf32>, ui32, ui32
    return
  }
}

// -----

// CHECK-LABEL: module {
// CHECK-NEXT: func.func @test_no_debug(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_no_debug(%arg0: !ascendc.local_tensor<*xf32>) {
    return
  }
}
