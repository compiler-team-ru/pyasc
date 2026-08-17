// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s -ascendc-dispatch-alloc -split-input-file -debug-only=ascendc-dispatch-alloc 2>&1 | FileCheck %s

// CHECK: 'asc.static_alloc' is set to 1
// CHECK-NEXT: static tensor allocation is selected
module attributes {asc.static_alloc = true} {
  func.func @test_static_alloc_true() {
    return
  }
}

// -----

// CHECK: 'asc.static_alloc' is set to 0
// CHECK-NEXT: TPipe-backed tensor allocation is selected (alwaysBuf=0)
module attributes {asc.static_alloc = false} {
  func.func @test_static_alloc_false() {
    return
  }
}

// -----

// CHECK: 'asc.static_alloc' is not set
// CHECK-NEXT: TPipe-backed tensor allocation is selected (alwaysBuf=0)
module {
  func.func @test_no_attr_non_c310() {
    return
  }
}

// -----

// CHECK: 'asc.static_alloc' is not set
// CHECK-NEXT: static tensor allocation is selected
module attributes {asc.compilation_arch = "c310"} {
  func.func @test_c310_no_broadcast() {
    return
  }
}

// -----

// CHECK: 'asc.static_alloc' is not set
// CHECK-NEXT: TPipe-backed tensor allocation is selected (alwaysBuf=1)
module attributes {asc.compilation_arch = "c310"} {
  func.func @test_c310_with_broadcast(%arg0: !ascendc.local_tensor<16x32xf32>, %arg1: !ascendc.local_tensor<16x1xf32>, %arg2: i32) {
    ascendc.broadcast %arg0, %arg1, %arg2, %arg2, %arg2, %arg2, %arg2, %arg2 {operandSegmentSizes = array<i32: 1, 1, 3, 3>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x1xf32>, i32, i32, i32, i32, i32, i32
    return
  }
}
