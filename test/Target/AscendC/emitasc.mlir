// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: int32_t call_opaque_test(int32_t v1, int32_t v2) {
// CHECK-NEXT:   int32_t v3 = add(v1, v2);
// CHECK-NEXT:   empty();
// CHECK-NEXT:   return v3;
// CHECK-NEXT: }
func.func @call_opaque_test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = emitasc.call_opaque "add" (%arg0, %arg1) : (i32, i32) -> i32
    emitasc.call_opaque "empty" () : () -> ()
    return %0 : i32
}

// CHECK-LABEL:void dereference_test(int32_t* v1) {
// CHECK-NEXT:  int32_t& v2 = *v1;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @dereference_test(%arg0: memref<?xi32>) {
    %0 = emitasc.dereference %arg0: memref<?xi32>, i32
    return
}

// CHECK-LABEL:void ptr_offset_test(int32_t* v1, uint32_t v2) {
// CHECK-NEXT:  int32_t* v3 = v1 + 512;
// CHECK-NEXT:  int32_t* v4 = v1 + v2;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @ptr_offset_test(%arg0: memref<?xi32>, %arg1: index) {
    %0 = emitasc.ptr_offset %arg0[ 512 ] : memref<?xi32>, memref<?xi32>
    %1 = emitasc.ptr_offset %arg0[%arg1] : memref<?xi32>, memref<123xi32>
    return
}

// CHECK-LABEL: __gm__ int32_t* reinterpret_cast_test(__gm__ float* v1) {
// CHECK-NEXT:   __gm__ int32_t* v2 = reinterpret_cast<__gm__ int32_t*>(v1);
// CHECK-NEXT:   return v2;
// CHECK-NEXT: }
func.func @reinterpret_cast_test(%arg0: memref<5x6xf32, 22>) -> memref<5x6xi32, 22> {
    %0 = emitasc.reinterpret_cast %arg0 : memref<5x6xf32, 22> to memref<5x6xi32, 22>
    return %0 : memref<5x6xi32, 22>
}

// CHECK-LABEL:void variable_test(int32_t v1) {
// CHECK-NEXT:   int32_t v2[1]{512};
// CHECK-NEXT:   int32_t v3[1]{v1};
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @variable_test(%arg0: i32) {
    %0 = emitasc.variable 512 : i32, memref<1xi32>
    %1 = emitasc.variable %arg0 : i32, memref<1xi32>
    return
}

// CHECK:#pragma pack(push, 8)
// CHECK-NEXT:struct PyStruct1 {
// CHECK-NEXT:  int32_t height;
// CHECK-NEXT:  int32_t width;
// CHECK-NEXT:};
// CHECK-NEXT:#pragma pack(pop)
// CHECK:#pragma pack(push, 8)
// CHECK-NEXT:struct PyStruct2 {
// CHECK-NEXT:  int32_t count;
// CHECK-NEXT:};
// CHECK-NEXT:#pragma pack(pop)
// CHECK:#pragma pack(push, 8)
// CHECK-NEXT:struct KernelConfig {
// CHECK-NEXT:  PyStruct1 struct1;
// CHECK-NEXT:  PyStruct2 struct2;
// CHECK-NEXT:};
// CHECK-NEXT:#pragma pack(pop)
// CHECK: __inline__ __attribute__((always_inline)) __aicore__ void copy_struct(__gm__ KernelConfig* v1, __gm__ PyStruct2* v2) {
// CHECK-NEXT: KernelConfig v3;
// CHECK-NEXT: for (size_t i = 0; i < sizeof(v3); i++) {
// CHECK-NEXT:   auto byte = reinterpret_cast<__gm__ uint8_t*>(v1)[i];
// CHECK-NEXT:   reinterpret_cast<uint8_t*>(&v3)[i] = byte;
// CHECK-NEXT: };
// CHECK-NEXT: PyStruct2 v4;
// CHECK-NEXT: for (size_t i = 0; i < sizeof(v4); i++) {
// CHECK-NEXT:   auto byte = reinterpret_cast<__gm__ uint8_t*>(v2)[i];
// CHECK-NEXT:   reinterpret_cast<uint8_t*>(&v4)[i] = byte;
// CHECK-NEXT: };
// CHECK-NEXT: return;
// CHECK-NEXT:}
module {
    emitasc.declare_py_struct !emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>
    emitasc.declare_py_struct !emitasc.py_struct<"PyStruct2", [i32], ["count"]>
    emitasc.declare_py_struct !emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>
    func.func @copy_struct(%arg0: memref<?x!emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>, 22>, %arg1: memref<?x!emitasc.py_struct<"PyStruct2", [i32], ["count"]>, 22>) {
        %0 = emitasc.copy_struct %arg0 : memref<?x!emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>, 22>, !emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>
        %1 = emitasc.copy_struct %arg1 : memref<?x!emitasc.py_struct<"PyStruct2", [i32], ["count"]>, 22>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>
        return
    }
}
