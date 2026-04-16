// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK:void emit_binary_ops(int32_t v1, int32_t v2) {
// CHECK-NEXT:  int32_t v3 = v1 + v2;
// CHECK-NEXT:  int32_t v4 = v1 - v2;
// CHECK-NEXT:  int32_t v5 = v1 * v2;
// CHECK-NEXT:  int32_t v6 = v1 / v2;
// CHECK-NEXT:  int32_t v7 = v1 % v2;
// CHECK-NEXT:  int32_t v8 = (v1 + v2 - 1) / v2;
// CHECK-NEXT:  int32_t v9 = v1 << v2;
// CHECK-NEXT:  int32_t v10 = v1 >> v2;
// CHECK-NEXT:  int32_t v11 = v1 >> v2;
// CHECK-NEXT:  int32_t v12 = v1 & v2;
// CHECK-NEXT:  int32_t v13 = v1 | v2;
// CHECK-NEXT:  int32_t v14 = v1 ^ v2;
// CHECK-NEXT:  int32_t v15 = v1 / v2;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_binary_ops(%0 : i32, %1 : i32) {
  %2 = arith.addi %0, %1 : i32
  %3 = arith.subi %0, %1 : i32
  %4 = arith.muli %0, %1 : i32
  %5 = arith.divsi %0, %1 : i32
  %6 = arith.remsi %0, %1 : i32
  %7 = arith.ceildivsi %0, %1 : i32
  %8 = arith.shli %0, %1 : i32
  %9 = arith.shrsi %0, %1 : i32
  %10 = arith.shrui %0, %1 : i32
  %11 = arith.andi %0, %1 : i32
  %12 = arith.ori %0, %1 : i32
  %13 = arith.xori %0, %1 : i32
  %14 = arith.divui %0, %1 : i32
  return
}

// CHECK:void emit_cmpi_op(int32_t v1, int32_t v2) {
// CHECK-NEXT:  bool v3 = v1 == v2;
// CHECK-NEXT:  bool v4 = v1 != v2;
// CHECK-NEXT:  bool v5 = v1 >= v2;
// CHECK-NEXT:  bool v6 = v1 > v2;
// CHECK-NEXT:  bool v7 = v1 >= v2;
// CHECK-NEXT:  bool v8 = v1 > v2;
// CHECK-NEXT:  bool v9 = v1 <= v2;
// CHECK-NEXT:  bool v10 = v1 < v2;
// CHECK-NEXT:  bool v11 = v1 <= v2;
// CHECK-NEXT:  bool v12 = v1 < v2;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_cmpi_op(%arg0: i32, %arg1: i32) {
  %0 = arith.cmpi eq, %arg0, %arg1 : i32
  %1 = arith.cmpi ne, %arg0, %arg1 : i32
  %2 = arith.cmpi sge, %arg0, %arg1 : i32
  %3 = arith.cmpi sgt, %arg0, %arg1 : i32
  %4 = arith.cmpi uge, %arg0, %arg1 : i32
  %5 = arith.cmpi ugt, %arg0, %arg1 : i32
  %6 = arith.cmpi sle, %arg0, %arg1 : i32
  %7 = arith.cmpi slt, %arg0, %arg1 : i32
  %8 = arith.cmpi ule, %arg0, %arg1 : i32
  %9 = arith.cmpi ult, %arg0, %arg1 : i32
  return
}

// CHECK:void emit_cmpf_op(float v1, float v2) {
// CHECK-NEXT:  bool v3 = v1 == v2;
// CHECK-NEXT:  bool v4 = v1 == v2;
// CHECK-NEXT:  bool v5 = v1 != v2;
// CHECK-NEXT:  bool v6 = v1 != v2;
// CHECK-NEXT:  bool v7 = v1 <= v2;
// CHECK-NEXT:  bool v8 = v1 <= v2;
// CHECK-NEXT:  bool v9 = v1 < v2;
// CHECK-NEXT:  bool v10 = v1 < v2;
// CHECK-NEXT:  bool v11 = v1 >= v2;
// CHECK-NEXT:  bool v12 = v1 >= v2;
// CHECK-NEXT:  bool v13 = v1 > v2;
// CHECK-NEXT:  bool v14 = v1 > v2;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_cmpf_op(%arg0: f32, %arg1: f32) {
  %0 = arith.cmpf oeq, %arg0, %arg1 : f32
  %1 = arith.cmpf ueq, %arg0, %arg1 : f32
  %2 = arith.cmpf one, %arg0, %arg1 : f32
  %3 = arith.cmpf une, %arg0, %arg1 : f32
  %4 = arith.cmpf ole, %arg0, %arg1 : f32
  %5 = arith.cmpf ule, %arg0, %arg1 : f32
  %6 = arith.cmpf olt, %arg0, %arg1 : f32
  %7 = arith.cmpf ult, %arg0, %arg1 : f32
  %8 = arith.cmpf oge, %arg0, %arg1 : f32
  %9 = arith.cmpf uge, %arg0, %arg1 : f32
  %10 = arith.cmpf ogt, %arg0, %arg1 : f32
  %11 = arith.cmpf ugt, %arg0, %arg1 : f32
  return
}

// CHECK:void emit_select_op(bool v1, int32_t v2, int32_t v3) {
// CHECK-NEXT:  int32_t v4 = v1 ? v2 : v3;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_select_op(%arg0: i1, %arg1: i32, %arg2: i32) {
  %0 = arith.select %arg0, %arg1, %arg2 : i1, i32
  return
}

// CHECK:void emit_casts(int16_t v1, float v2, uint32_t v3) {
// CHECK-NEXT:  int32_t v4 = static_cast<int32_t>(v1);
// CHECK-NEXT:  int64_t v5 = static_cast<int64_t>(v1);
// CHECK-NEXT:  int8_t v6 = static_cast<int8_t>(v1);
// CHECK-NEXT:  half v7 = static_cast<half>(v1);
// CHECK-NEXT:  double v8 = static_cast<double>(v2);
// CHECK-NEXT:  half v9 = static_cast<half>(v2);
// CHECK-NEXT:  int32_t v10 = static_cast<int32_t>(v2);
// CHECK-NEXT:  half v11 = *reinterpret_cast<half*>(&v1);
// CHECK-NEXT:  int32_t v12 = static_cast<int32_t>(v3);
// CHECK-NEXT:  uint32_t v13 = static_cast<uint32_t>(v1);
// CHECK-NEXT:  half v14 = static_cast<half>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_casts(%arg0: i16, %arg1: f32, %arg2: index) {
  %0 = arith.extsi %arg0 : i16 to i32
  %1 = arith.extui %arg0 : i16 to i64
  %2 = arith.trunci %arg0 : i16 to i8
  %3 = arith.sitofp %arg0 : i16 to f16
  %4 = arith.extf %arg1 : f32 to f64
  %5 = arith.truncf %arg1 : f32 to f16
  %6 = arith.fptosi %arg1 : f32 to i32
  %7 = arith.bitcast %arg0 : i16 to f16
  %8 = arith.index_cast %arg2 : index to i32
  %9 = arith.index_cast %arg0 : i16 to index
  %10 = arith.uitofp %arg0 : i16 to f16
  return
}

// CHECK:void emit_maxf(float v1, float v2) {
// CHECK-NEXT:  float v3 = ((v1 > v2) ? (v1) : (v2));
// CHECK-NEXT:  float v4 = ((v1 > v2) ? (v1) : (v2));
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_maxf(%arg0 : f32, %arg1 : f32) {
  %0 = arith.maxnumf %arg0, %arg1 : f32
  %1 = arith.maximumf %arg0, %arg1 : f32
  return
}

// CHECK:void emit_minf(float v1, float v2) {
// CHECK-NEXT:  float v3 = ((v1 < v2) ? (v1) : (v2));
// CHECK-NEXT:  float v4 = ((v1 < v2) ? (v1) : (v2));
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_minf(%arg0 : f32, %arg1 : f32) {
  %0 = arith.minnumf %arg0, %arg1 : f32
  %1 = arith.minimumf %arg0, %arg1 : f32
  return
}

// CHECK:void emit_constants() {
// CHECK-NEXT:  constexpr int8_t c8_i8 = 8;
// CHECK-NEXT:  constexpr int16_t c16_i16 = 16;
// CHECK-NEXT:  constexpr int32_t c32_i32 = 32;
// CHECK-NEXT:  constexpr int64_t c64_i64 = 64;
// CHECK-NEXT:  half c16_f16 = 1.60000e+01;
// CHECK-NEXT:  bfloat16_t c1_602_f16 = 1.6020e+00;
// CHECK-NEXT:  constexpr float c63_f32 = (float)6.300000000e+01;
// CHECK-NEXT:  constexpr double c6_2999999999999998_f64 = (double)6.29999999999999980e+00;
// CHECK-NEXT:  constexpr uint32_t c666_idx = 666;
// CHECK-NEXT:  constexpr float cpInf_f32 = __builtin_inff();
// CHECK-NEXT:  constexpr float cmInf_f32 = -__builtin_inff();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_constants() {
  %i8 = arith.constant 8 : i8
  %i16 = arith.constant 16 : i16
  %i32 = arith.constant 32 : i32
  %i64 = arith.constant 64 : i64
  %f16 = arith.constant 16.0 : f16
  %bf16 = arith.constant 1.6 : bf16
  %f32 = arith.constant 63.0 : f32
  %f64 = arith.constant 6.3 : f64
  %index = arith.constant 666 : index
  %cst_inf = arith.constant 0x7F800000 : f32
  %cst_minf = arith.constant 0xFF800000 : f32
  return
}

// CHECK:void emit_mului_extended(int32_t v1, int32_t v2, int16_t v3, int16_t v4) {
// CHECK-NEXT:  int32_t v5 = v1 * v2;
// CHECK-NEXT:  int32_t v6 = (static_cast<uint64_t>(v1) * static_cast<uint64_t>(v2)) >> 32;
// CHECK-NEXT:  int16_t v7 = v3 * v4;
// CHECK-NEXT:  int16_t v8 = (static_cast<uint32_t>(v3) * static_cast<uint32_t>(v4)) >> 16;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_mului_extended(%arg0 : i32, %arg1 : i32, %arg2 : i16, %arg3 : i16) {
  %0, %1 = arith.mului_extended %arg0, %arg1 : i32
  %2, %3 = arith.mului_extended %arg2, %arg3 : i16
  return
}
