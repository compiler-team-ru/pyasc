// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s | ascir-opt | FileCheck %s
// RUN: ascir-opt %s --mlir-print-op-generic | ascir-opt | FileCheck %s

// CHECK-LABEL: func.func @test_variable(%arg0: i32, %arg1: memref<?xf16>)
// CHECK-NEXT: emitasc.variable 123 : i32, memref<1xi32>
// CHECK-NEXT: emitasc.variable 1.230000e+00 : f32, memref<1xf32>
// CHECK-NEXT: emitasc.variable %arg0 : i32, memref<1xi32>
// CHECK-NEXT: emitasc.variable %arg1 : memref<?xf16>, memref<1xmemref<?xf16>>
func.func @test_variable(%arg0: i32, %arg1: memref<?xf16>) {
  %static_int = emitasc.variable 123 : i32, memref<1xi32>
  %static_float = emitasc.variable 1.23 : f32, memref<1xf32>
  %dynamic_int = emitasc.variable %arg0 : i32, memref<1xi32>
  %dynamic_memref = emitasc.variable %arg1 : memref<?xf16>, memref<1xmemref<?xf16>>
  return
}

// CHECK-LABEL: func.func @test_init_struct(%arg0: i32, %arg1: memref<?xf16>)
// CHECK-NEXT: emitasc.init_struct !emitc.opaque<"NoFields">()
// CHECK-NEXT: emitasc.init_struct !emitc.opaque<"OneField">("first_field" = %arg0 : i32)
// CHECK-NEXT: emitasc.init_struct !emitc.opaque<"ManyFields">("first_field" = %arg0 : i32, "second_field" = %arg1 : memref<?xf16>)
func.func @test_init_struct(%arg0: i32, %arg1: memref<?xf16>) {
  %no_fields = emitasc.init_struct !emitc.opaque<"NoFields">()
  %one_field = emitasc.init_struct !emitc.opaque<"OneField">("first_field" = %arg0 : i32)
  %many_fields = emitasc.init_struct !emitc.opaque<"ManyFields">("first_field" = %arg0 : i32, "second_field" = %arg1 : memref<?xf16>)
  return
}
