// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-insert-init-dump --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @printf_only(%arg0: i32,
// CHECK-SAME: %arg1: memref<?xui8, 22> {emitasc.kernel_arg = #emitasc<kernel_arg dump_addr>}) {
// CHECK:        ascendc.init_dump %false, %arg1, %c1048576_i32 : i1, memref<?xui8, 22>, i32
// CHECK-NEXT:   ascendc.printf %arg0 {desc = "val %d"} : i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @printf_only(%arg0: i32) {
  ascendc.printf %arg0 {desc = "val %d"} : i32
  return
}

// CHECK-LABEL: func.func @test_dump_tensor_only(%arg0: !ascendc.local_tensor<*xf32>, %arg1: ui32, %arg2: ui32,
// CHECK-SAME: %arg3: memref<?xui8, 22> {emitasc.kernel_arg = #emitasc<kernel_arg dump_addr>}) {
// CHECK:        ascendc.init_dump %false, %arg3, %c1048576_i32 : i1, memref<?xui8, 22>, i32
// CHECK-NEXT:   ascendc.dump_tensor %arg0, %arg1, %arg2 : !ascendc.local_tensor<*xf32>, ui32, ui32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @test_dump_tensor_only(%arg0: !ascendc.local_tensor<*xf32>, %arg1: ui32, %arg2: ui32) {
  ascendc.dump_tensor %arg0, %arg1, %arg2 : !ascendc.local_tensor<*xf32>, ui32, ui32
  return
}

// CHECK-LABEL: func.func @test_both_debug_ops(%arg0: i32, %arg1: !ascendc.local_tensor<*xf32>, %arg2: ui32, %arg3: ui32,
// CHECK-SAME: %arg4: memref<?xui8, 22> {emitasc.kernel_arg = #emitasc<kernel_arg dump_addr>}) {
// CHECK:        ascendc.init_dump %false, %arg4, %c1048576_i32 : i1, memref<?xui8, 22>, i32
// CHECK-NEXT:   ascendc.printf %arg0 {desc = "test print %d"} : i32
// CHECK-NEXT:   ascendc.dump_tensor %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, ui32, ui32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @test_both_debug_ops(%arg0: i32, %arg1: !ascendc.local_tensor<*xf32>, %arg2: ui32, %arg3: ui32) {
  ascendc.printf %arg0 {desc = "test print %d"} : i32
  ascendc.dump_tensor %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, ui32, ui32
  return
}

// CHECK-LABEL: func.func @test_no_debug_ops(%arg0: i32) {
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @test_no_debug_ops(%arg0: i32) {
  return
}

// -----

// CHECK-LABEL: func.func @test_vector_kernel_type(%arg0: i32,
// CHECK-SAME: %arg1: memref<?xui8, 22> {emitasc.kernel_arg = #emitasc<kernel_arg dump_addr>}) {
// CHECK:        ascendc.init_dump %false, %arg1, %c1048576_i32 : i1, memref<?xui8, 22>, i32
// CHECK-NEXT:   ascendc.printf %arg0 {desc = "test print"} : i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
module attributes {asc.kernel_type = "vector"} {
  func.func @test_vector_kernel_type(%arg0: i32) {
    ascendc.printf %arg0 {desc = "test print"} : i32
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_mixed_kernel_type(%arg0: i32,
// CHECK-SAME: %arg1: memref<?xui8, 22> {emitasc.kernel_arg = #emitasc<kernel_arg dump_addr>}) {
// CHECK:        ascendc.init_dump %false, %arg1, %c1048576_i32 : i1, memref<?xui8, 22>, i32
// CHECK-NEXT:   ascendc.printf %arg0 {desc = "test print"} : i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
module attributes {asc.kernel_type = "cube"} {
  func.func @test_mixed_kernel_type(%arg0: i32) {
    ascendc.printf %arg0 {desc = "test print"} : i32
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_mixed_kernel_type(%arg0: i32,
// CHECK-SAME: %arg1: memref<?xui8, 22> {emitasc.kernel_arg = #emitasc<kernel_arg dump_addr>}) {
// CHECK:        ascendc.init_dump %true, %arg1, %c1048576_i32 : i1, memref<?xui8, 22>, i32
// CHECK-NEXT:   ascendc.printf %arg0 {desc = "test print"} : i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
module attributes {asc.kernel_type = "mixed"} {
  func.func @test_mixed_kernel_type(%arg0: i32) {
    ascendc.printf %arg0 {desc = "test print"} : i32
    return
  }
}

